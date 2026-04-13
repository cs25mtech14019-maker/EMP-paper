"""Post-hoc mode selection strategies that don't rely on pi.

Strategies:
  1. paper:        argmax(softmax(pi))           [baseline]
  2. consensus:    mode closest to centroid of 6 endpoints
  3. w-consensus:  mode closest to pi-weighted centroid
  4. aggregated:   brier computed with clustered pi (modes within delta share prob)
  5. oracle:       mode with lowest FDE           [upper bound]

Usage (on DGX):
    python scripts/eval_posthoc_strategies.py \
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
from torch.utils.data import DataLoader

from src.datamodule.av2_dataset import Av2Dataset, collate_fn
from src.model.trainer_forecast import Trainer

# Run B ckpt predates Delta-Decoding cumsum.
import src.model.layers.multimodal_decoder_emp as _dec_mod
def _forward_no_cumsum(self, x, __1, __2, __3):
    B = x.shape[0]
    mode_embeds = self.mode_embed.weight.view(1, self.k, self.embed_dim).repeat(B, 1, 1)
    x = x.unsqueeze(1).repeat(1, self.k, 1) + mode_embeds
    loc = self.loc(x).view(-1, self.k, self.future_steps, 2)
    pi = self.pi(x).squeeze(-1)
    return loc, pi
_dec_mod.MultimodalDecoder.forward = _forward_no_cumsum


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--out", default="scripts/eval_posthoc_out.npz")
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

    # Collect per-scene data
    logits_all = []    # (N, 6)     raw logits
    fde_all = []       # (N, 6)     FDE per mode
    ade_all = []       # (N, 6)     ADE per mode
    endpoints_all = [] # (N, 6, 2)  final position (t=60) per mode, ego-local

    start = time.time()
    total_seen = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            out = model(batch)
            y_hat = out["y_hat"]       # (B, 6, F, 2)
            pi = out["pi"]             # (B, 6) raw logits

            y = batch["y"]
            err = torch.norm(y_hat - y[:, 0:1], dim=-1)   # (B, 6, F)
            fde = err[..., -1]
            ade = err.mean(dim=-1)
            endpoints = y_hat[..., -1, :]                  # (B, 6, 2)

            logits_all.append(pi.cpu().numpy())
            fde_all.append(fde.cpu().numpy())
            ade_all.append(ade.cpu().numpy())
            endpoints_all.append(endpoints.cpu().numpy())

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

    logits = np.concatenate(logits_all)      # (N, 6)
    fde = np.concatenate(fde_all)            # (N, 6)
    ade = np.concatenate(ade_all)            # (N, 6)
    endpoints = np.concatenate(endpoints_all) # (N, 6, 2)
    N = logits.shape[0]

    probs = softmax(logits)                   # (N, 6)

    # ===== MODE DIVERSITY ANALYSIS =====
    print("\n" + "=" * 70)
    print("MODE DIVERSITY ANALYSIS")
    print("=" * 70)
    # Pairwise distance between mode endpoints
    # endpoints: (N, 6, 2), compute (N, 6, 6) pairwise distances
    diff = endpoints[:, :, np.newaxis, :] - endpoints[:, np.newaxis, :, :]
    pw_dist = np.linalg.norm(diff, axis=-1)   # (N, 6, 6)
    # Mean pairwise distance (excluding diagonal)
    mask_tri = np.triu(np.ones((6, 6), dtype=bool), k=1)
    mean_pw = pw_dist[:, mask_tri].mean(axis=1)  # (N,) avg pairwise dist per scene
    print(f"  Mean pairwise endpoint distance:  {mean_pw.mean():.2f} m")
    print(f"  Median:                           {np.median(mean_pw):.2f} m")
    print(f"  p10/p25/p75/p90:                  {np.percentile(mean_pw, 10):.2f} / "
          f"{np.percentile(mean_pw, 25):.2f} / {np.percentile(mean_pw, 75):.2f} / "
          f"{np.percentile(mean_pw, 90):.2f} m")

    # How many modes are within delta of the best mode?
    best_mode_idx = fde.argmin(axis=1)  # (N,)
    best_endpoint = endpoints[np.arange(N), best_mode_idx]  # (N, 2)
    dist_to_best = np.linalg.norm(endpoints - best_endpoint[:, np.newaxis, :], axis=-1)  # (N, 6)
    for delta in [1.0, 2.0, 3.0, 5.0]:
        n_near = (dist_to_best < delta).sum(axis=1).mean()
        print(f"  Modes within {delta}m of FDE-best endpoint: {n_near:.2f} / 6")

    # ===== STANDARD METRICS =====
    minFDE6 = fde.min(axis=1)
    minADE6 = ade.min(axis=1)
    MR = (minFDE6 > 2.0).astype(float)

    print(f"\n  minFDE6 = {minFDE6.mean():.4f}")
    print(f"  minADE6 = {minADE6.mean():.4f}")
    print(f"  MR@2m   = {MR.mean():.4f}")

    # ===== SELECTION STRATEGIES =====
    def eval_strategy(sel, name, pi_for_brier=None):
        """sel: (N,) mode index per scene. pi_for_brier: (N,) probability for brier."""
        idx = np.arange(N)
        fde1 = fde[idx, sel].mean()
        ade1 = ade[idx, sel].mean()
        if pi_for_brier is None:
            pi_b = probs[idx, best_mode_idx]
        else:
            pi_b = pi_for_brier
        brier = (minFDE6 + (1.0 - pi_b) ** 2).mean()
        return {"name": name, "FDE1": fde1, "ADE1": ade1, "brier": brier,
                "pi_best": pi_b.mean()}

    rows = []

    # 1. Paper baseline: argmax(pi)
    paper_sel = probs.argmax(axis=1)
    rows.append(eval_strategy(paper_sel, "paper (argmax pi)"))

    # 2. Oracle: best mode by FDE
    rows.append(eval_strategy(best_mode_idx, "oracle (FDE-best)",
                              pi_for_brier=np.ones(N)))

    # 3. Consensus: mode closest to unweighted centroid
    centroid = endpoints.mean(axis=1)  # (N, 2)
    dist_to_centroid = np.linalg.norm(endpoints - centroid[:, np.newaxis, :], axis=-1)
    consensus_sel = dist_to_centroid.argmin(axis=1)
    rows.append(eval_strategy(consensus_sel, "consensus (centroid)"))

    # 4. Weighted consensus: mode closest to pi-weighted centroid
    w = probs[:, :, np.newaxis]  # (N, 6, 1)
    w_centroid = (endpoints * w).sum(axis=1) / w.sum(axis=1)  # (N, 2)
    dist_to_wc = np.linalg.norm(endpoints - w_centroid[:, np.newaxis, :], axis=-1)
    wconsensus_sel = dist_to_wc.argmin(axis=1)
    rows.append(eval_strategy(wconsensus_sel, "w-consensus (pi-centroid)"))

    # 5. Full-trajectory consensus: use ADE (average over time) instead of endpoint
    # approximate by: mode with minimum average distance to all other modes
    # mode_score[k] = mean_j(||endpoint_k - endpoint_j||)
    mean_dist_to_others = pw_dist.mean(axis=2)  # (N, 6): avg dist of mode k to all others
    medoid_sel = mean_dist_to_others.argmin(axis=1)  # pick mode nearest to all others
    rows.append(eval_strategy(medoid_sel, "medoid (nearest to all)"))

    # 6. Aggregated brier: cluster modes, sum probabilities of cluster containing best
    for delta in [1.0, 2.0, 3.0, 4.0, 5.0]:
        near_best = dist_to_best < delta  # (N, 6) bool
        pi_agg = (probs * near_best).sum(axis=1)  # sum prob of modes near fde-best
        brier_agg = (minFDE6 + (1.0 - pi_agg) ** 2).mean()
        rows.append({
            "name": f"agg-brier delta={delta}m",
            "FDE1": fde[np.arange(N), paper_sel].mean(),  # same selection as paper
            "ADE1": ade[np.arange(N), paper_sel].mean(),
            "brier": brier_agg,
            "pi_best": pi_agg.mean(),
        })

    # 7. Hybrid: consensus selection + aggregated brier
    for delta in [2.0, 3.0]:
        near_consensus = np.linalg.norm(
            endpoints - endpoints[np.arange(N), consensus_sel][:, np.newaxis, :], axis=-1
        ) < delta
        pi_cons_agg = (probs * near_consensus).sum(axis=1)
        # For brier: use consensus selection's fde as "minFDE1" and aggregated prob
        cons_fde = fde[np.arange(N), consensus_sel]
        # But standard brier uses minFDE6 (oracle) + (1 - p_oracle)^2
        # so we report: minFDE6 + (1 - aggregated_p_of_fde_best)^2
        near_best_c = dist_to_best < delta
        pi_agg_c = (probs * near_best_c).sum(axis=1)
        brier_hybrid = (minFDE6 + (1.0 - pi_agg_c) ** 2).mean()
        rows.append({
            "name": f"consensus+agg d={delta}m",
            "FDE1": cons_fde.mean(),
            "ADE1": ade[np.arange(N), consensus_sel].mean(),
            "brier": brier_hybrid,
            "pi_best": pi_agg_c.mean(),
        })

    # ===== TABLE =====
    print("\n" + "=" * 90)
    print(f"{'method':<30}  {'FDE1':>7}  {'ADE1':>7}  {'brier':>7}  "
          f"{'pi_best':>7}  {'ΔFDE1':>7}  {'Δbrier':>7}")
    print("-" * 90)
    base_fde = rows[0]["FDE1"]
    base_brier = rows[0]["brier"]
    for r in rows:
        d_fde = r["FDE1"] - base_fde
        d_brier = r["brier"] - base_brier
        print(f"{r['name']:<30}  {r['FDE1']:>7.3f}  {r['ADE1']:>7.3f}  "
              f"{r['brier']:>7.4f}  {r['pi_best']:>7.4f}  "
              f"{d_fde:>+7.3f}  {d_brier:>+7.4f}")
    print("=" * 90)

    # ===== SAVE =====
    np.savez(
        args.out,
        logits=logits, fde=fde, ade=ade, endpoints=endpoints,
        probs=probs, pw_dist_mean=mean_pw,
        paper_sel=paper_sel, consensus_sel=consensus_sel,
        medoid_sel=medoid_sel, best_mode_idx=best_mode_idx,
    )
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
