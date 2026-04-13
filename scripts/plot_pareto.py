import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datamodule.av2_dataset import Av2Dataset, collate_fn
from src.model.trainer_forecast import Trainer

# Monkey-patch to disable cumsum for Run B
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
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--limit", type=int, default=-1)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")
    
    model = Trainer.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval().to(device)
    H, F = model.history_steps, model.future_steps

    ds = Av2Dataset(data_root=Path(args.data_root), cached_split="val")
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )

    taus = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    
    # Storage
    all_paper_fde = []
    all_paper_nm = []
    
    # tau -> [fde array, nm array]
    all_tau_fde = {tau: [] for tau in taus}
    all_tau_nm = {tau: [] for tau in taus}

    print("[info] evaluating scenes...")
    total = 0
    start = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            
            # Forward pass
            out = model.net(batch)
            y_hat = out["y_hat"]
            pi = out["pi"]
            y_hat_others = out["y_hat_others"]
            B, K, F, _ = y_hat.shape
            pi_soft = torch.softmax(pi.double(), dim=-1).float()
            
            # Ground truth for FDE calculation
            y = batch["y"]
            err = torch.norm(y_hat - y[:, 0:1], dim=-1)
            fde = err[..., -1]  # (B, 6)

            # Paper selection (argmax pi)
            paper_sel = pi_soft.argmax(dim=1)
            
            # Calculate distance to consensus
            endpoints = y_hat[:, :, -1, :]
            w = pi_soft.unsqueeze(-1)
            w_centroid = (endpoints * w).sum(dim=1) / w.sum(dim=1)
            dist_to_wc = torch.norm(endpoints - w_centroid.unsqueeze(1), dim=-1)
            consensus_sel = dist_to_wc.argmin(dim=1)

            # Calculate clearances
            ego_angle = batch["x_angles"][:, 0, H - 1]
            ego_center = batch["x_centers"][:, 0]
            others_angle = batch["x_angles"][:, 1:, H - 1]
            others_center = batch["x_centers"][:, 1:]
            others_valid = ~batch["x_padding_mask"][:, 1:, H:]

            def to_scene(traj, angle, center):
                c = torch.cos(angle).unsqueeze(-1).unsqueeze(-1)
                s = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)
                x, yy = traj[..., 0:1], traj[..., 1:2]
                return torch.cat([c*x - s*yy, s*x + c*yy], dim=-1) + center.unsqueeze(-2)

            y_hat_scene = to_scene(
                y_hat.reshape(B*K, F, 2),
                ego_angle.unsqueeze(1).expand(B, K).reshape(B*K),
                ego_center.unsqueeze(1).expand(B, K, 2).reshape(B*K, 2),
            ).reshape(B, K, F, 2)

            y_gt_others_scene = to_scene(y[:, 1:], others_angle, others_center)

            # Clearance to GT other agents
            diff_gt = y_hat_scene.unsqueeze(2) - y_gt_others_scene.unsqueeze(1)
            dist_gt = torch.norm(diff_gt, dim=-1)
            v_exp = others_valid.unsqueeze(1).expand(-1, K, -1, -1)
            dist_gt = dist_gt.masked_fill(~v_exp, float("inf"))
            min_dist_gt = dist_gt.flatten(2).min(dim=-1).values  # (B, 6)
            
            # Store paper metrics
            idx = torch.arange(B, device=device)
            all_paper_fde.append(fde[idx, paper_sel].cpu().numpy())
            all_paper_nm.append((min_dist_gt[idx, paper_sel] < 2.0).cpu().numpy())
            
            # Evaluate for each tau
            # We filter using predicted others clearance, but for plotting, we evaluate against GT clearance!
            y_hat_others_scene = to_scene(y_hat_others, others_angle, others_center)
            diff_pred = y_hat_scene.unsqueeze(2) - y_hat_others_scene.unsqueeze(1)
            dist_pred = torch.norm(diff_pred, dim=-1)
            dist_pred = dist_pred.masked_fill(~v_exp, float("inf"))
            min_dist_pred = dist_pred.flatten(2).min(dim=-1).values  # (B, 6)

            for tau in taus:
                if tau == 0.0:
                    sel = consensus_sel
                else:
                    safe_mask = min_dist_pred >= tau
                    consensus_safe = safe_mask[idx, consensus_sel]
                    masked_pi = torch.where(safe_mask, pi_soft, torch.tensor(-1e9, device=device))
                    any_safe = safe_mask.any(dim=1)
                    safe_sel = masked_pi.argmax(dim=1)
                    sel = torch.where(consensus_safe, consensus_sel,
                            torch.where(any_safe, safe_sel, consensus_sel))
                
                all_tau_fde[tau].append(fde[idx, sel].cpu().numpy())
                all_tau_nm[tau].append((min_dist_gt[idx, sel] < 2.0).cpu().numpy())
                
            total += B
            if args.limit > 0 and total >= args.limit:
                break
            
            if bi % 20 == 0:
                print(f"[prog] {total:>6d} scenes")

    # Aggregate
    res_paper_fde = np.concatenate(all_paper_fde).mean()
    res_paper_nm = np.concatenate(all_paper_nm).mean() * 100
    
    res_tau_fde = []
    res_tau_nm = []
    for tau in taus:
        res_tau_fde.append(np.concatenate(all_tau_fde[tau]).mean())
        res_tau_nm.append(np.concatenate(all_tau_nm[tau]).mean() * 100)
    
    print("\nResults:")
    print(f"Paper: FDE={res_paper_fde:.3f}, NM={res_paper_nm:.2f}%")
    for t, f, n in zip(taus, res_tau_fde, res_tau_nm):
        print(f"Tau={t:.1f}: FDE={f:.3f}, NM={n:.2f}%")

    # Plot
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(9, 6))
    
    plt.plot(res_tau_nm, res_tau_fde, marker='o', linewidth=2, markersize=8, color='b', label='Ours (Consensus + Collision Filter)')
    
    # Annotate tau points
    for i, tau in enumerate(taus):
        offset = (0.05, 0.01) if i % 2 == 0 else (0.05, -0.02)
        if tau == 0.0:
            label = "Consensus Only"
        else:
            label = f"$\\tau$={tau}m"
        plt.annotate(label, 
                     (res_tau_nm[i], res_tau_fde[i]),
                     textcoords="offset points",
                     xytext=(10, -5 if tau == 0 else 5),
                     ha='left',
                     fontsize=10)

    # Plot paper baseline
    plt.scatter([res_paper_nm], [res_paper_fde], color='red', s=250, zorder=5, marker='*', label='Paper Baseline')
    plt.annotate('Paper (argmax $\\pi$)', 
                 (res_paper_nm, res_paper_fde),
                 textcoords="offset points",
                 xytext=(-15, 10),
                 ha='center',
                 fontsize=11, color='red', weight='bold')
    
    plt.title("Safety vs. Accuracy Trade-off (Pareto Front)", fontsize=16, pad=15)
    plt.xlabel("Near-Miss Rate (% <2m from GT) → Lower is safer", fontsize=12)
    plt.ylabel("FDE$_1$ (meters) → Lower is more accurate", fontsize=12)
    plt.legend(fontsize=12, loc='lower left')
    
    # Formatting axes
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Force axis limits to include paper star with generous padding
    ax = plt.gca()
    all_x = res_tau_nm + [res_paper_nm]
    all_y = res_tau_fde + [res_paper_fde]
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    dx = xmax - xmin
    dy = ymax - ymin
    ax.set_xlim(xmin - dx*0.15, xmax + dx*0.15)
    ax.set_ylim(ymin - dy*0.15, ymax + dy*0.15)
    
    plt.tight_layout()
    
    save_path = "pareto_plot.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[saved] {save_path}")

if __name__ == "__main__":
    main()
