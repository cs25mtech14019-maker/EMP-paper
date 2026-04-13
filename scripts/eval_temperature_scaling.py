"""Post-hoc temperature scaling for EMP mode probability calibration.

Splits val set into calibration (80%) and test (20%).
Finds optimal temperature T on calibration set that minimises brier-minFDE6.
Reports calibrated metrics on held-out test set.

Usage (on DGX):
    python scripts/eval_temperature_scaling.py \
        --ckpt outputs/emp-forecast_av2/2026-02-04/12-15-30/checkpoints/last.ckpt \
        --data_root /u/student/2025/cs25mtech14019/emp/data/emp \
        --batch_size 64 --num_workers 4
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as Fn
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--cal_fraction", type=float, default=0.8,
                   help="Fraction of val used for calibration (rest = test)")
    p.add_argument("--out", default="scripts/eval_temperature_out.npz")
    p.add_argument("--miss_threshold", type=float, default=2.0)
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

    # Collect logits and FDE for all scenes
    logits_all = []
    fde_all = []

    start = time.time()
    total_seen = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            out = model(batch)
            y_hat = out["y_hat"]       # (B, 6, F, 2) ego-local
            pi = out["pi"]             # (B, 6) RAW LOGITS

            y = batch["y"]             # (B, N, F, 2)
            err = torch.norm(y_hat - y[:, 0:1], dim=-1)   # (B, 6, F)
            fde = err[..., -1]         # (B, 6)

            logits_all.append(pi.cpu().numpy())
            fde_all.append(fde.cpu().numpy())

            total_seen += y_hat.shape[0]
            if args.limit > 0 and total_seen >= args.limit:
                break
            if batch_idx % 50 == 0:
                el = time.time() - start
                rate = total_seen / max(el, 1e-6)
                print(f"[prog] {total_seen:>6d} scenes | {el:>5.0f}s | "
                      f"{rate:>5.1f} scenes/s")

    elapsed = time.time() - start
    print(f"\n[done] {total_seen} scenes in {elapsed/60:.1f} min")

    logits = np.concatenate(logits_all, axis=0)  # (N, 6)
    fde = np.concatenate(fde_all, axis=0)        # (N, 6)
    N = logits.shape[0]

    # --- Split into calibration / test ---
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    n_cal = int(N * args.cal_fraction)
    cal_idx = perm[:n_cal]
    test_idx = perm[n_cal:]
    print(f"\n[split] calibration: {len(cal_idx)}, test: {len(test_idx)}")

    # --- Helper: compute metrics given logits, fde, temperature ---
    def compute_metrics(logits_subset, fde_subset, T):
        """Compute brier-minFDE6 and related metrics at temperature T."""
        # Scale logits
        scaled = logits_subset / T
        # Softmax to get probabilities
        exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)

        # minFDE6 = min over 6 modes
        min_fde6 = fde_subset.min(axis=1)
        best_mode = fde_subset.argmin(axis=1)

        # Probability assigned to the best mode
        pi_best = probs[np.arange(len(probs)), best_mode]

        # brier-minFDE6
        brier = min_fde6 + (1.0 - pi_best) ** 2

        # MR
        mr = (min_fde6 > 2.0).astype(float)

        # Top-1 FDE (argmax of calibrated probabilities)
        top1 = probs.argmax(axis=1)
        fde1 = fde_subset[np.arange(len(fde_subset)), top1]

        # minADE6 not available here (we only saved FDE), report FDE-based metrics
        return {
            "T": T,
            "minFDE6": min_fde6.mean(),
            "brier": brier.mean(),
            "pi_best_mean": pi_best.mean(),
            "pi_best_median": np.median(pi_best),
            "MR": mr.mean(),
            "FDE1": fde1.mean(),
            "brier_penalty": ((1.0 - pi_best) ** 2).mean(),
        }

    # --- Sweep T on calibration set ---
    T_candidates = np.concatenate([
        np.arange(0.01, 0.1, 0.01),    # very sharp
        np.arange(0.1, 0.5, 0.02),     # sharp
        np.arange(0.5, 1.5, 0.05),     # moderate
        np.arange(1.5, 3.1, 0.1),      # broad (should be worse)
    ])

    cal_logits = logits[cal_idx]
    cal_fde = fde[cal_idx]

    print("\n" + "=" * 90)
    print("TEMPERATURE SWEEP ON CALIBRATION SET")
    print("=" * 90)
    print(f"{'T':>6}  {'brier':>8}  {'minFDE6':>8}  {'penalty':>8}  "
          f"{'pi_best':>8}  {'FDE1':>8}  {'MR':>6}")
    print("-" * 90)

    best_T = 1.0
    best_brier = float("inf")
    all_results = []

    for T in T_candidates:
        m = compute_metrics(cal_logits, cal_fde, T)
        all_results.append(m)
        if m["brier"] < best_brier:
            best_brier = m["brier"]
            best_T = T
        # Print selected rows
        if T in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0] or \
           abs(T - best_T) < 0.001:
            print(f"{T:>6.3f}  {m['brier']:>8.4f}  {m['minFDE6']:>8.4f}  "
                  f"{m['brier_penalty']:>8.4f}  {m['pi_best_mean']:>8.4f}  "
                  f"{m['FDE1']:>8.3f}  {m['MR']:>6.4f}")

    print("-" * 90)
    print(f">>> BEST T = {best_T:.3f}  (cal brier = {best_brier:.4f})")

    # --- Evaluate on HELD-OUT test set ---
    test_logits = logits[test_idx]
    test_fde = fde[test_idx]

    print("\n" + "=" * 90)
    print("RESULTS ON HELD-OUT TEST SET")
    print("=" * 90)
    print(f"{'T':>6}  {'brier':>8}  {'minFDE6':>8}  {'penalty':>8}  "
          f"{'pi_best':>8}  {'FDE1':>8}  {'MR':>6}")
    print("-" * 90)

    for T in [1.0, best_T]:
        m = compute_metrics(test_logits, test_fde, T)
        label = "baseline (T=1.0)" if T == 1.0 else f"calibrated (T={best_T:.3f})"
        print(f"{T:>6.3f}  {m['brier']:>8.4f}  {m['minFDE6']:>8.4f}  "
              f"{m['brier_penalty']:>8.4f}  {m['pi_best_mean']:>8.4f}  "
              f"{m['FDE1']:>8.3f}  {m['MR']:>6.4f}  ← {label}")

    print("=" * 90)

    # --- Also report on FULL val set for comparison with training logs ---
    print("\n" + "=" * 90)
    print("FULL VAL SET (for comparison with training val_brier-minFDE6)")
    print("=" * 90)
    for T in [1.0, best_T]:
        m = compute_metrics(logits, fde, T)
        label = "baseline" if T == 1.0 else f"calibrated T={best_T:.3f}"
        print(f"  T={T:.3f}: brier={m['brier']:.4f}, minFDE6={m['minFDE6']:.4f}, "
              f"pi_best={m['pi_best_mean']:.4f}, FDE1={m['FDE1']:.3f}  ← {label}")

    # --- Save ---
    np.savez(
        args.out,
        logits=logits, fde=fde,
        cal_idx=cal_idx, test_idx=test_idx,
        best_T=best_T,
        T_candidates=np.array([r["T"] for r in all_results]),
        brier_vs_T=np.array([r["brier"] for r in all_results]),
        penalty_vs_T=np.array([r["brier_penalty"] for r in all_results]),
        pi_best_vs_T=np.array([r["pi_best_mean"] for r in all_results]),
    )
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
