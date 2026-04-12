"""Diagnostic for Option A (collision-aware mode selection).

Answers two questions before we commit to a full eval:
  1. Is y_hat_others good enough to use as a collision signal?
     -> report minADE of other-agent predicted trajectories vs ground truth
  2. Does the paper's top-1 selection ever produce tight misses / collisions
     with other agents' own predicted trajectories?
     -> report histogram of min-distance from paper-top1 ego mode to any
        predicted other-agent trajectory

Run on DGX:
    cd ~/emp
    conda activate emp_verify
    python scripts/diagnostic_collision.py \
        --ckpt outputs/emp-forecast_av2/2026-02-04/12-15-30/checkpoints/last.ckpt \
        --data_root /u/student/2025/cs25mtech14019/emp/data/emp \
        --n_scenes 500

Output: stdout report + scripts/diagnostic_out.npz (raw arrays for plotting).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datamodule.av2_dataset import Av2Dataset, collate_fn
from src.model.trainer_forecast import Trainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_root", required=True,
                   help="Parent dir containing val/ (e.g. /u/.../emp/data/emp)")
    p.add_argument("--split", default="val")
    p.add_argument("--n_scenes", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out", default="scripts/diagnostic_out.npz")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")
    print(f"[info] loading checkpoint: {args.ckpt}")

    model = Trainer.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval().to(device)
    H, F = model.history_steps, model.future_steps
    print(f"[info] history={H}, future={F}")

    ds = Av2Dataset(data_root=Path(args.data_root), cached_split=args.split)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    others_ade_all = []
    top1_min_dist_all = []
    per_mode_min_dist_all = []
    top1_idx_all = []
    gt_top1_min_dist_all = []   # sanity check: ego-GT vs other-GT (should match real collision rates)
    any_safer_count = 0
    n_scenes_seen = 0
    n_scenes_with_others = 0

    def to_scene_frame(traj_local, angle, center):
        """Rotate and translate from per-agent local frame to scene frame.
        traj_local : (..., T, 2)
        angle      : (...,)        heading of that agent at t=H-1
        center     : (..., 2)      position of that agent at t=H-1
        """
        cos_a = torch.cos(angle).unsqueeze(-1).unsqueeze(-1)
        sin_a = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)
        x = traj_local[..., 0:1]
        y = traj_local[..., 1:2]
        rx = cos_a * x - sin_a * y
        ry = sin_a * x + cos_a * y
        rot = torch.cat([rx, ry], dim=-1)
        return rot + center.unsqueeze(-2)

    with torch.no_grad():
        for batch in dl:
            if n_scenes_seen >= args.n_scenes:
                break
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

            out = model(batch)
            y_hat = out["y_hat"]                  # (B, 6, F, 2)   ego local frame
            pi = out["pi"]                        # (B, 6)
            y_hat_others = out["y_hat_others"]    # (B, N-1, F, 2) per-agent local frame

            y = batch["y"]                                          # (B, N, F, 2)
            others_future_pad = batch["x_padding_mask"][:, 1:, H:]  # (B, N-1, F)
            others_valid = ~others_future_pad                       # True = real

            # Per-agent pose at t=H-1
            ego_angle = batch["x_angles"][:, 0, H - 1]              # (B,)
            ego_center = batch["x_centers"][:, 0]                   # (B, 2)
            others_angle = batch["x_angles"][:, 1:, H - 1]          # (B, N-1)
            others_center = batch["x_centers"][:, 1:]               # (B, N-1, 2)

            B, K, _, _ = y_hat.shape

            # --- (1) y_hat_others quality: per-agent ADE on valid steps (in local frame) ---
            y_others_gt = y[:, 1:]
            err = torch.norm(y_hat_others - y_others_gt, dim=-1)    # (B, N-1, F)
            valid_f = others_valid.float()
            valid_steps = valid_f.sum(dim=-1)                       # (B, N-1)
            ade = (err * valid_f).sum(dim=-1) / valid_steps.clamp_min(1.0)
            has_any_step = valid_steps > 0                          # (B, N-1)
            others_ade_all.append(ade[has_any_step].cpu().numpy())

            # --- Transform to common scene frame ---
            # Ego: expand angle/center across 6 modes, transform (B, 6, F, 2)
            ego_angle_k = ego_angle.unsqueeze(1).expand(B, K).reshape(B * K)
            ego_center_k = ego_center.unsqueeze(1).expand(B, K, 2).reshape(B * K, 2)
            y_hat_flat = y_hat.reshape(B * K, y_hat.shape[-2], 2)
            y_hat_scene = to_scene_frame(y_hat_flat, ego_angle_k, ego_center_k)
            y_hat_scene = y_hat_scene.reshape(B, K, -1, 2)

            # Others: (B, N-1, F, 2) with per-agent angle/center
            y_hat_others_scene = to_scene_frame(
                y_hat_others, others_angle, others_center
            )

            # GT (for sanity baseline): same transforms
            y_ego_gt_scene = to_scene_frame(y[:, 0], ego_angle, ego_center)        # (B, F, 2)
            y_others_gt_scene = to_scene_frame(y[:, 1:], others_angle, others_center)

            # --- (2) min-dist: ego mode k vs PREDICTED other-agent trajectories (scene frame) ---
            diff = y_hat_scene.unsqueeze(2) - y_hat_others_scene.unsqueeze(1)  # (B,6,N-1,F,2)
            dist = torch.norm(diff, dim=-1)                                    # (B,6,N-1,F)
            valid_exp = others_valid.unsqueeze(1).expand(-1, K, -1, -1)
            dist = dist.masked_fill(~valid_exp, float("inf"))
            min_dist_per_mode = dist.flatten(2).min(dim=-1).values             # (B, 6)

            # --- (2b) Sanity: ego-GT vs other-GT min-dist (real-world collision rate) ---
            gt_diff = y_ego_gt_scene.unsqueeze(1) - y_others_gt_scene          # (B, N-1, F, 2)
            gt_dist = torch.norm(gt_diff, dim=-1)                              # (B, N-1, F)
            gt_dist = gt_dist.masked_fill(~others_valid, float("inf"))
            gt_min_dist = gt_dist.flatten(1).min(dim=-1).values                # (B,)

            top1 = pi.argmax(dim=-1)                                # (B,)
            top1_min_dist = min_dist_per_mode[torch.arange(B), top1]

            has_any_other = others_valid.any(dim=-1).any(dim=-1)    # (B,)
            valid_scenes = has_any_other & torch.isfinite(top1_min_dist)
            n_scenes_with_others += int(valid_scenes.sum())

            top1_min_dist_all.append(top1_min_dist[valid_scenes].cpu().numpy())
            per_mode_min_dist_all.append(min_dist_per_mode[valid_scenes].cpu().numpy())
            top1_idx_all.append(top1[valid_scenes].cpu().numpy())
            gt_top1_min_dist_all.append(gt_min_dist[valid_scenes].cpu().numpy())

            best_safe_mode = min_dist_per_mode.argmax(dim=-1)
            any_safer = (best_safe_mode != top1) & valid_scenes
            any_safer_count += int(any_safer.sum())

            n_scenes_seen += B

    others_ade_all = np.concatenate(others_ade_all)
    top1_min_dist_all = np.concatenate(top1_min_dist_all)
    per_mode_min_dist_all = np.concatenate(per_mode_min_dist_all, axis=0)
    top1_idx_all = np.concatenate(top1_idx_all)
    gt_top1_min_dist_all = np.concatenate(gt_top1_min_dist_all)

    N_val = len(top1_min_dist_all)

    print("\n" + "=" * 62)
    print(f"Diagnostic: {n_scenes_seen} scenes seen, "
          f"{N_val} with >=1 other agent, "
          f"{len(others_ade_all)} per-agent trajectories")
    print("=" * 62)

    print("\n[1] y_hat_others quality (ADE vs ground truth; lower = better)")
    print(f"    mean   : {np.mean(others_ade_all):.3f} m")
    print(f"    median : {np.median(others_ade_all):.3f} m")
    print(f"    p90    : {np.percentile(others_ade_all, 90):.3f} m")
    print(f"    p99    : {np.percentile(others_ade_all, 99):.3f} m")
    print(f"    max    : {np.max(others_ade_all):.3f} m")

    print("\n[2] Paper-top1 min-dist to any predicted other-agent trajectory")
    print("    (0 = through them, large = safe)")
    print(f"    mean   : {np.mean(top1_min_dist_all):.3f} m")
    print(f"    median : {np.median(top1_min_dist_all):.3f} m")
    print(f"    p10    : {np.percentile(top1_min_dist_all, 10):.3f} m")
    print(f"    p25    : {np.percentile(top1_min_dist_all, 25):.3f} m")

    pct_lt = lambda thr: 100.0 * (top1_min_dist_all < thr).sum() / max(1, N_val)
    print(f"    < 0.5m : {(top1_min_dist_all < 0.5).sum():>5d} / {N_val}  ({pct_lt(0.5):.2f}%)")
    print(f"    < 1.0m : {(top1_min_dist_all < 1.0).sum():>5d} / {N_val}  ({pct_lt(1.0):.2f}%)")
    print(f"    < 2.0m : {(top1_min_dist_all < 2.0).sum():>5d} / {N_val}  ({pct_lt(2.0):.2f}%)")
    print(f"    < 3.0m : {(top1_min_dist_all < 3.0).sum():>5d} / {N_val}  ({pct_lt(3.0):.2f}%)")

    print("\n    histogram (bar scale = relative):")
    edges = [0, 0.5, 1, 2, 3, 5, 10, 1e9]
    hist, _ = np.histogram(top1_min_dist_all, bins=edges)
    for i, h in enumerate(hist):
        lo = edges[i]
        hi = edges[i + 1]
        hi_s = f"{hi:>5.1f}" if hi < 1e8 else "  inf"
        bar = "#" * int(50 * h / max(1, hist.max()))
        print(f"      [{lo:>4.1f}, {hi_s}): {h:>6d}  {bar}")

    print("\n[2b] SANITY: ego-GT vs other-GT min-dist (real scene collisions)")
    print(f"    mean   : {np.mean(gt_top1_min_dist_all):.3f} m")
    print(f"    median : {np.median(gt_top1_min_dist_all):.3f} m")
    print(f"    p10    : {np.percentile(gt_top1_min_dist_all, 10):.3f} m")
    gt_pct_lt = lambda thr: 100.0 * (gt_top1_min_dist_all < thr).sum() / max(1, N_val)
    print(f"    < 1.0m : {(gt_top1_min_dist_all < 1.0).sum():>5d} / {N_val}  ({gt_pct_lt(1.0):.2f}%)")
    print(f"    < 2.0m : {(gt_top1_min_dist_all < 2.0).sum():>5d} / {N_val}  ({gt_pct_lt(2.0):.2f}%)")

    print("\n[3] Opportunity: scenes where a different mode has larger clearance")
    print(f"    {any_safer_count}/{N_val} ({100.0 * any_safer_count / max(1, N_val):.1f}%)")

    print("\n" + "=" * 62)
    median_ade = float(np.median(others_ade_all))
    pct_tight = pct_lt(2.0)
    if median_ade < 2.0 and pct_tight > 1.0:
        verdict = (f"GO: y_hat_others median={median_ade:.2f}m (usable) and "
                   f"{pct_tight:.2f}% of top1 are <2m (meaningful tail).")
    elif median_ade >= 2.0:
        verdict = (f"CAUTION: y_hat_others median={median_ade:.2f}m is too noisy. "
                   f"Consider analysing with GT other-agent trajectories instead, "
                   f"or pivot to Option B (kinematic filter).")
    else:
        verdict = (f"WEAK: only {pct_tight:.2f}% of top1 predictions are <2m. "
                   f"Collision filter rarely fires; underwhelming story.")
    print(verdict)
    print("=" * 62)

    np.savez(
        args.out,
        others_ade=others_ade_all,
        top1_min_dist=top1_min_dist_all,
        per_mode_min_dist=per_mode_min_dist_all,
        top1_idx=top1_idx_all,
        gt_top1_min_dist=gt_top1_min_dist_all,
    )
    print(f"\n[saved] raw arrays -> {args.out}")


if __name__ == "__main__":
    main()
