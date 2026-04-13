"""Full evaluation for Option A: collision-aware mode selection.

Runs on all val scenes. For each scene computes per-mode FDE/ADE and per-mode
minimum distance to other-agent trajectories (both predicted and GT) in a common
scene frame. Then applies multiple selection methods and compares them.

Selection methods:
  - paper: argmax(pi)
  - filter tau:   pick argmax(pi) subject to min_dist_pred >= tau (fallback to paper)
  - soft:         argmax(pi * sigmoid((min_dist_pred - tau)/sigma))

Metrics:
  Standard (min-over-6, unaffected by selection):
      minFDE6, minADE6, MR@2m, brier-minFDE6
  Per-method:
      FDE1, ADE1,
      near-miss rate at tau in {1,2,3}m against PRED others (method-vs-pred),
      near-miss rate at tau in {1,2,3}m against GT others   (method-vs-gt),
      median clearance.

Usage (on DGX):
    python scripts/eval_collision_aware.py \
        --ckpt outputs/emp-forecast_av2/2026-02-04/12-15-30/checkpoints/last.ckpt \
        --data_root /u/student/2025/cs25mtech14019/emp/data/emp

Outputs:
    scripts/eval_collision_aware_out.npz : per-scene arrays (ade, fde, pi,
        min_dist_pred, min_dist_gt, has_others, paper_sel) for later plotting.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datamodule.av2_dataset import Av2Dataset, collate_fn
from src.model.trainer_forecast import Trainer


def to_scene_frame(traj_local, angle, center):
    """Rotate by `angle` and translate by `center`. Applied per-agent."""
    cos_a = torch.cos(angle).unsqueeze(-1).unsqueeze(-1)
    sin_a = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)
    x = traj_local[..., 0:1]
    y = traj_local[..., 1:2]
    rx = cos_a * x - sin_a * y
    ry = sin_a * x + cos_a * y
    return torch.cat([rx, ry], dim=-1) + center.unsqueeze(-2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=-1, help="cap num scenes (debug)")
    p.add_argument("--out", default="scripts/eval_collision_aware_out.npz")
    p.add_argument("--miss_threshold", type=float, default=2.0,
                   help="MR threshold (standard AV2 = 2.0m)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")
    print(f"[info] loading: {args.ckpt}")

    model = Trainer.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval().to(device)
    H, F = model.history_steps, model.future_steps
    print(f"[info] history={H}, future={F}")

    ds = Av2Dataset(data_root=Path(args.data_root), cached_split=args.split)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    K = 6
    ade_all = []
    fde_all = []
    pi_all = []
    min_dist_pred_all = []
    min_dist_gt_all = []
    has_others_all = []

    start = time.time()
    total_seen = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

            out = model(batch)
            y_hat = out["y_hat"]                  # (B, 6, F, 2) ego local
            pi = out["pi"]                        # (B, 6)
            y_hat_others = out["y_hat_others"]    # (B, N-1, F, 2) per-agent local

            y = batch["y"]                                             # (B, N, F, 2)
            others_future_pad = batch["x_padding_mask"][:, 1:, H:]     # (B, N-1, F)
            others_valid = ~others_future_pad

            ego_angle = batch["x_angles"][:, 0, H - 1]                 # (B,)
            ego_center = batch["x_centers"][:, 0]                      # (B, 2)
            others_angle = batch["x_angles"][:, 1:, H - 1]             # (B, N-1)
            others_center = batch["x_centers"][:, 1:]                  # (B, N-1, 2)

            B = y_hat.shape[0]

            # --- FDE/ADE in ego-local frame (y_hat and y[:,0] share this frame;
            #     this is what training validation uses) ---
            err_local = torch.norm(y_hat - y[:, 0:1], dim=-1)      # (B, 6, F)
            ade = err_local.mean(dim=-1)                           # (B, 6)
            fde = err_local[..., -1]                               # (B, 6)

            # --- Scene-frame transforms (for clearance only) ---
            y_hat_scene = to_scene_frame(
                y_hat.reshape(B * K, F, 2),
                ego_angle.unsqueeze(1).expand(B, K).reshape(B * K),
                ego_center.unsqueeze(1).expand(B, K, 2).reshape(B * K, 2),
            ).reshape(B, K, F, 2)
            y_hat_others_scene = to_scene_frame(
                y_hat_others, others_angle, others_center
            )
            y_others_gt_scene = to_scene_frame(
                y[:, 1:], others_angle, others_center
            )

            # --- min-dist per mode vs PREDICTED others ---
            diff_pred = y_hat_scene.unsqueeze(2) - y_hat_others_scene.unsqueeze(1)
            dist_pred = torch.norm(diff_pred, dim=-1)              # (B,6,N-1,F)
            valid_exp = others_valid.unsqueeze(1).expand(-1, K, -1, -1)
            dist_pred = dist_pred.masked_fill(~valid_exp, float("inf"))
            min_dist_pred = dist_pred.flatten(2).min(dim=-1).values  # (B, 6)

            # --- min-dist per mode vs GT others ---
            diff_gt = y_hat_scene.unsqueeze(2) - y_others_gt_scene.unsqueeze(1)
            dist_gt = torch.norm(diff_gt, dim=-1)
            dist_gt = dist_gt.masked_fill(~valid_exp, float("inf"))
            min_dist_gt = dist_gt.flatten(2).min(dim=-1).values    # (B, 6)

            has_any_other = others_valid.any(dim=-1).any(dim=-1)   # (B,)

            ade_all.append(ade.cpu().numpy())
            fde_all.append(fde.cpu().numpy())
            pi_all.append(pi.cpu().numpy())
            min_dist_pred_all.append(min_dist_pred.cpu().numpy())
            min_dist_gt_all.append(min_dist_gt.cpu().numpy())
            has_others_all.append(has_any_other.cpu().numpy())

            total_seen += B
            if args.limit > 0 and total_seen >= args.limit:
                break
            if batch_idx % 50 == 0:
                el = time.time() - start
                rate = total_seen / max(el, 1e-6)
                print(f"[prog] {total_seen:>6d} scenes | {el:>5.0f}s | "
                      f"{rate:>5.1f} scenes/s")

    elapsed = time.time() - start
    print(f"\n[done] {total_seen} scenes in {elapsed/60:.1f} min")

    ade = np.concatenate(ade_all, axis=0)
    fde = np.concatenate(fde_all, axis=0)
    pi = np.concatenate(pi_all, axis=0)
    mdp = np.concatenate(min_dist_pred_all, axis=0)
    mdg = np.concatenate(min_dist_gt_all, axis=0)
    has_others = np.concatenate(has_others_all, axis=0)

    # For argmax/argmin, replace inf with big finite
    mdp_f = np.where(np.isinf(mdp), 1e9, mdp)
    mdg_f = np.where(np.isinf(mdg), 1e9, mdg)

    # --- Standard metrics (unchanged by selection) ---
    minFDE6 = fde.min(axis=1)
    minADE6 = ade.min(axis=1)
    MR = (minFDE6 > args.miss_threshold).astype(float)
    best_by_fde = fde.argmin(axis=1)
    pi_best = pi[np.arange(len(pi)), best_by_fde]
    brier_minFDE6 = minFDE6 + (1.0 - pi_best) ** 2

    print("\n" + "=" * 72)
    print("STANDARD METRICS (min-over-6; independent of selection method)")
    print("=" * 72)
    print(f"  minFDE6        : {minFDE6.mean():.4f}")
    print(f"  minADE6        : {minADE6.mean():.4f}")
    print(f"  MR@2m          : {MR.mean():.4f}")
    print(f"  brier-minFDE6  : {brier_minFDE6.mean():.4f}")

    # --- Selection evaluation ---
    def eval_selection(selected, name, taus=(1.0, 2.0, 3.0)):
        idx = np.arange(len(selected))
        fde1 = fde[idx, selected]
        ade1 = ade[idx, selected]
        clear_pred = mdp_f[idx, selected]
        clear_gt = mdg_f[idx, selected]
        valid = has_others
        row = {
            "name": name,
            "fde1": fde1.mean(),
            "ade1": ade1.mean(),
            "clear_pred_med": np.median(clear_pred[valid]),
            "clear_gt_med": np.median(clear_gt[valid]),
        }
        for tau in taus:
            row[f"nm_pred_{tau}"] = (clear_pred[valid] < tau).mean() * 100
            row[f"nm_gt_{tau}"] = (clear_gt[valid] < tau).mean() * 100
        return row

    paper_sel = pi.argmax(axis=1)

    rows = [eval_selection(paper_sel, "paper (argmax pi)")]

    for tau in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        mask = mdp_f >= tau                              # (N, 6)
        masked_pi = np.where(mask, pi, -1e9)
        any_safe = mask.any(axis=1)
        sel = np.where(any_safe, masked_pi.argmax(axis=1), paper_sel)
        rows.append(eval_selection(sel, f"filter tau={tau}m"))

    tau_s, sig_s = 2.0, 0.5
    soft_w = pi * (1.0 / (1.0 + np.exp(-(mdp_f - tau_s) / sig_s)))
    soft_sel = soft_w.argmax(axis=1)
    rows.append(eval_selection(soft_sel, f"soft tau={tau_s}m sig={sig_s}"))

    # --- Table ---
    print("\n" + "=" * 118)
    header = (f"{'method':<26}  {'FDE1':>6}  {'ADE1':>6}  "
              f"{'NM<1pred':>9}  {'NM<2pred':>9}  {'NM<3pred':>9}  "
              f"{'NM<1gt':>8}  {'NM<2gt':>8}  {'NM<3gt':>8}  "
              f"{'clrP50':>7}  {'clrG50':>7}")
    print(header)
    print("-" * 118)
    for r in rows:
        print(f"{r['name']:<26}  "
              f"{r['fde1']:>6.3f}  {r['ade1']:>6.3f}  "
              f"{r['nm_pred_1.0']:>8.2f}%  {r['nm_pred_2.0']:>8.2f}%  {r['nm_pred_3.0']:>8.2f}%  "
              f"{r['nm_gt_1.0']:>7.2f}%  {r['nm_gt_2.0']:>7.2f}%  {r['nm_gt_3.0']:>7.2f}%  "
              f"{r['clear_pred_med']:>7.2f}  {r['clear_gt_med']:>7.2f}")
    print("=" * 118)
    print("\nNM<Tpred = top-1 mode within T m of any PREDICTED other trajectory")
    print("NM<Tgt   = top-1 mode within T m of any GT other trajectory (real collisions)")
    print("clrP50/clrG50 = median clearance (pred / gt)")

    np.savez(
        args.out,
        ade=ade, fde=fde, pi=pi,
        min_dist_pred=mdp, min_dist_gt=mdg,
        has_others=has_others,
        paper_sel=paper_sel,
    )
    print(f"\n[saved] per-scene arrays -> {args.out}")


if __name__ == "__main__":
    main()
