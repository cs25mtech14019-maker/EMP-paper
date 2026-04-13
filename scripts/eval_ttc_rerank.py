"""Full evaluation for Option T: TTC-aware mode selection + composed filter.

Extends eval_collision_aware.py with Time-to-Collision (TTC) as a safety signal.
For each (ego_mode, other_agent) pair we compute TTC(t) = d(t) / closing_speed(t)
at every timestep where ego is approaching; min_TTC is the minimum across agents
and time.  Re-rank modes by pi filtered/weighted on TTC, and also evaluate a
composed filter that combines distance and TTC.

Selection methods:
  - paper:            argmax(pi)
  - dist-filter:      argmax(pi) s.t. min_dist >= tau_d (fallback paper)
  - ttc-filter:       argmax(pi) s.t. min_ttc  >= tau_t (fallback paper)
  - composed:         argmax(pi) s.t. min_dist>=tau_d AND min_ttc>=tau_t

AV2 sampling is 10 Hz (dt = 0.1 s).
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

# Run B (2026-02-04) ckpt predates Delta-Decoding cumsum.
import src.model.layers.multimodal_decoder_emp as _dec_mod
def _forward_no_cumsum(self, x, __1, __2, __3):
    B = x.shape[0]
    mode_embeds = self.mode_embed.weight.view(1, self.k, self.embed_dim).repeat(B, 1, 1)
    x = x.unsqueeze(1).repeat(1, self.k, 1) + mode_embeds
    loc = self.loc(x).view(-1, self.k, self.future_steps, 2)
    pi = self.pi(x).squeeze(-1)
    return loc, pi
_dec_mod.MultimodalDecoder.forward = _forward_no_cumsum

DT = 0.1  # AV2 sampling interval (seconds)


def to_scene_frame(traj_local, angle, center):
    cos_a = torch.cos(angle).unsqueeze(-1).unsqueeze(-1)
    sin_a = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)
    x = traj_local[..., 0:1]
    y = traj_local[..., 1:2]
    rx = cos_a * x - sin_a * y
    ry = sin_a * x + cos_a * y
    return torch.cat([rx, ry], dim=-1) + center.unsqueeze(-2)


def compute_min_ttc(ego_scene, other_scene, valid_mask):
    """
    ego_scene:    (B, 6, F, 2)
    other_scene:  (B, N-1, F, 2)
    valid_mask:   (B, N-1, F) bool, True where valid
    returns:      (B, 6) min TTC across other agents and time, inf if none approaching
    """
    diff = ego_scene.unsqueeze(2) - other_scene.unsqueeze(1)      # (B,6,N-1,F,2)
    dist = torch.norm(diff, dim=-1)                               # (B,6,N-1,F)

    # finite-difference closing speed on the distance curve
    dd = (dist[..., 1:] - dist[..., :-1]) / DT                    # (B,6,N-1,F-1)
    closing = torch.clamp(-dd, min=1e-3)                          # >0 only when approaching
    d_mid = 0.5 * (dist[..., 1:] + dist[..., :-1])                # midpoint distance

    ttc = d_mid / closing                                         # (B,6,N-1,F-1)

    # mask: need BOTH endpoints valid and ego must be approaching
    v = valid_mask.unsqueeze(1).expand(-1, ego_scene.shape[1], -1, -1)  # (B,6,N-1,F)
    v_pair = v[..., 1:] & v[..., :-1]
    approaching = dd < 0
    keep = v_pair & approaching

    ttc = ttc.masked_fill(~keep, float("inf"))
    return ttc.flatten(2).min(dim=-1).values                      # (B, 6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--out", default="scripts/eval_ttc_rerank_out.npz")
    p.add_argument("--miss_threshold", type=float, default=2.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")
    print(f"[info] loading: {args.ckpt}")

    model = Trainer.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval().to(device)
    H, F = model.history_steps, model.future_steps
    print(f"[info] history={H}, future={F}, dt={DT}")

    ds = Av2Dataset(data_root=Path(args.data_root), cached_split=args.split)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    K = 6
    ade_all, fde_all, pi_all = [], [], []
    mdp_all, mdg_all = [], []
    mtp_all, mtg_all = [], []
    has_others_all = []

    start = time.time()
    total_seen = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

            out = model(batch)
            y_hat = out["y_hat"]
            pi = out["pi"]
            y_hat_others = out["y_hat_others"]

            y = batch["y"]
            others_future_pad = batch["x_padding_mask"][:, 1:, H:]
            others_valid = ~others_future_pad

            ego_angle = batch["x_angles"][:, 0, H - 1]
            ego_center = batch["x_centers"][:, 0]
            others_angle = batch["x_angles"][:, 1:, H - 1]
            others_center = batch["x_centers"][:, 1:]

            B = y_hat.shape[0]

            err_local = torch.norm(y_hat - y[:, 0:1], dim=-1)
            ade = err_local.mean(dim=-1)
            fde = err_local[..., -1]

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

            # --- min distance per mode ---
            diff_pred = y_hat_scene.unsqueeze(2) - y_hat_others_scene.unsqueeze(1)
            dist_pred = torch.norm(diff_pred, dim=-1)
            valid_exp = others_valid.unsqueeze(1).expand(-1, K, -1, -1)
            dist_pred = dist_pred.masked_fill(~valid_exp, float("inf"))
            min_dist_pred = dist_pred.flatten(2).min(dim=-1).values

            diff_gt = y_hat_scene.unsqueeze(2) - y_others_gt_scene.unsqueeze(1)
            dist_gt = torch.norm(diff_gt, dim=-1)
            dist_gt = dist_gt.masked_fill(~valid_exp, float("inf"))
            min_dist_gt = dist_gt.flatten(2).min(dim=-1).values

            # --- min TTC per mode (pred and GT others) ---
            min_ttc_pred = compute_min_ttc(y_hat_scene, y_hat_others_scene, others_valid)
            min_ttc_gt   = compute_min_ttc(y_hat_scene, y_others_gt_scene,   others_valid)

            has_any_other = others_valid.any(dim=-1).any(dim=-1)

            ade_all.append(ade.cpu().numpy())
            fde_all.append(fde.cpu().numpy())
            pi_all.append(pi.cpu().numpy())
            mdp_all.append(min_dist_pred.cpu().numpy())
            mdg_all.append(min_dist_gt.cpu().numpy())
            mtp_all.append(min_ttc_pred.cpu().numpy())
            mtg_all.append(min_ttc_gt.cpu().numpy())
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

    ade = np.concatenate(ade_all)
    fde = np.concatenate(fde_all)
    pi = np.concatenate(pi_all)
    mdp = np.concatenate(mdp_all)
    mdg = np.concatenate(mdg_all)
    mtp = np.concatenate(mtp_all)
    mtg = np.concatenate(mtg_all)
    has_others = np.concatenate(has_others_all)

    mdp_f = np.where(np.isinf(mdp), 1e9, mdp)
    mdg_f = np.where(np.isinf(mdg), 1e9, mdg)
    mtp_f = np.where(np.isinf(mtp), 1e9, mtp)
    mtg_f = np.where(np.isinf(mtg), 1e9, mtg)

    # --- Standard metrics ---
    minFDE6 = fde.min(axis=1)
    minADE6 = ade.min(axis=1)
    MR = (minFDE6 > args.miss_threshold).astype(float)
    best_by_fde = fde.argmin(axis=1)
    pi_best = pi[np.arange(len(pi)), best_by_fde]
    brier_minFDE6 = minFDE6 + (1.0 - pi_best) ** 2

    print("\n" + "=" * 72)
    print("STANDARD METRICS (min-over-6)")
    print("=" * 72)
    print(f"  minFDE6        : {minFDE6.mean():.4f}")
    print(f"  minADE6        : {minADE6.mean():.4f}")
    print(f"  MR@2m          : {MR.mean():.4f}")
    print(f"  brier-minFDE6  : {brier_minFDE6.mean():.4f}")

    # --- TTC distribution diagnostic ---
    valid_ttc = mtp_f[has_others]
    finite_ttc = valid_ttc[valid_ttc < 1e8]
    if finite_ttc.size:
        print("\nTTC(pred) distribution over (scene x mode) pairs:")
        for q in [10, 25, 50, 75, 90]:
            print(f"  p{q:02d}: {np.percentile(finite_ttc, q):6.2f} s")
        print(f"  scenes with any finite TTC (any mode approaches): "
              f"{(mtp_f.min(axis=1) < 1e8)[has_others].mean()*100:.1f}%")

    # --- Selection evaluation ---
    paper_sel = pi.argmax(axis=1)

    def eval_selection(selected, name):
        idx = np.arange(len(selected))
        fde1 = fde[idx, selected]
        ade1 = ade[idx, selected]
        clear_pred = mdp_f[idx, selected]
        clear_gt = mdg_f[idx, selected]
        ttc_pred = mtp_f[idx, selected]
        ttc_gt = mtg_f[idx, selected]
        v = has_others
        ttc_gt_finite = ttc_gt[v][ttc_gt[v] < 1e8]
        ttc_gt_med = np.median(ttc_gt_finite) if ttc_gt_finite.size else np.inf
        return {
            "name": name,
            "fde1": fde1.mean(),
            "ade1": ade1.mean(),
            "nm_pred_2": (clear_pred[v] < 2.0).mean() * 100,
            "nm_gt_2":   (clear_gt[v]   < 2.0).mean() * 100,
            "lowttc_pred_2s": ((ttc_pred[v] < 2.0)).mean() * 100,
            "lowttc_gt_2s":   ((ttc_gt[v]   < 2.0)).mean() * 100,
            "lowttc_gt_3s":   ((ttc_gt[v]   < 3.0)).mean() * 100,
            "clear_gt_med": np.median(clear_gt[v]),
            "ttc_gt_med": ttc_gt_med,
        }

    rows = [eval_selection(paper_sel, "paper (argmax pi)")]

    # distance-only filter (reference)
    for tau_d in [1.5, 2.0, 2.5]:
        mask = mdp_f >= tau_d
        any_safe = mask.any(axis=1)
        masked_pi = np.where(mask, pi, -1e9)
        sel = np.where(any_safe, masked_pi.argmax(axis=1), paper_sel)
        rows.append(eval_selection(sel, f"dist tau={tau_d}m"))

    # TTC-only filter
    for tau_t in [1.5, 2.0, 3.0, 4.0]:
        mask = mtp_f >= tau_t
        any_safe = mask.any(axis=1)
        masked_pi = np.where(mask, pi, -1e9)
        sel = np.where(any_safe, masked_pi.argmax(axis=1), paper_sel)
        rows.append(eval_selection(sel, f"ttc  tau={tau_t}s"))

    # Composed filter
    for tau_d, tau_t in [(2.0, 2.0), (2.0, 3.0), (1.5, 2.0)]:
        mask = (mdp_f >= tau_d) & (mtp_f >= tau_t)
        any_safe = mask.any(axis=1)
        masked_pi = np.where(mask, pi, -1e9)
        sel = np.where(any_safe, masked_pi.argmax(axis=1), paper_sel)
        rows.append(eval_selection(sel, f"comp d>={tau_d}m ttc>={tau_t}s"))

    # --- Table ---
    print("\n" + "=" * 130)
    header = (f"{'method':<30}  {'FDE1':>6}  {'ADE1':>6}  "
              f"{'NM<2p':>6}  {'NM<2g':>6}  "
              f"{'TTC<2p':>7}  {'TTC<2g':>7}  {'TTC<3g':>7}  "
              f"{'clrG50':>7}  {'ttcG50':>7}")
    print(header)
    print("-" * 130)
    for r in rows:
        ttc_med_s = f"{r['ttc_gt_med']:>7.2f}" if r['ttc_gt_med'] < 1e8 else "    inf"
        print(f"{r['name']:<30}  "
              f"{r['fde1']:>6.3f}  {r['ade1']:>6.3f}  "
              f"{r['nm_pred_2']:>5.2f}%  {r['nm_gt_2']:>5.2f}%  "
              f"{r['lowttc_pred_2s']:>6.2f}%  {r['lowttc_gt_2s']:>6.2f}%  {r['lowttc_gt_3s']:>6.2f}%  "
              f"{r['clear_gt_med']:>7.2f}  {ttc_med_s}")
    print("=" * 130)
    print("\nNM<Tp   = top-1 mode within Tm of any PRED other")
    print("NM<Tg   = top-1 mode within Tm of any GT   other")
    print("TTC<Ts  = top-1 mode has TTC<T s (approaching collision within T s)")
    print("clrG50  = median clearance to GT other (m)")
    print("ttcG50  = median (finite) TTC to GT other (s)")

    np.savez(
        args.out,
        ade=ade, fde=fde, pi=pi,
        min_dist_pred=mdp, min_dist_gt=mdg,
        min_ttc_pred=mtp, min_ttc_gt=mtg,
        has_others=has_others,
        paper_sel=paper_sel,
    )
    print(f"\n[saved] per-scene arrays -> {args.out}")


if __name__ == "__main__":
    main()
