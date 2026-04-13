import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection

from .emp import EMP
from src.metrics import MR, brierMinFDE, minADE, minFDE
from src.utils.optim import WarmupCosLR
from src.utils.submission_av2 import SubmissionAv2


torch.set_printoptions(sci_mode=False)


class Trainer(pl.LightningModule):
    def __init__(
        self,
        dim=128,
        historical_steps=50,
        future_steps=60,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        pretrained_weights: str = None,
        teacher_weights: str = None,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
        decoder: str = "detr"
    ) -> None:
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        self.history_steps = historical_steps
        self.future_steps = future_steps
        self.submission_handler = SubmissionAv2()

        self.net = EMP(
            embed_dim=dim,
            encoder_depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path=drop_path,
            decoder=decoder
        )  

        if pretrained_weights is not None:
            self.net.load_from_checkpoint(pretrained_weights)

        # Knowledge distillation: frozen EMP-D teacher
        self.teacher = None
        if teacher_weights is not None:
            self.teacher = EMP(
                embed_dim=dim,
                encoder_depth=encoder_depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=0.0,
                decoder="detr"
            )
            ckpt = torch.load(teacher_weights, map_location="cpu")["state_dict"]
            teacher_state = {k[len("net."):]: v for k, v in ckpt.items() if k.startswith("net.")}
            self.teacher.load_state_dict(teacher_state, strict=False)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        metrics = MetricCollection(
            {
                "minADE1": minADE(k=1),
                "minADE6": minADE(k=6),
                "minFDE1": minFDE(k=1),
                "minFDE6": minFDE(k=6),
                "MR": MR(),
                "brier-minFDE6": brierMinFDE(k=6)
            }
        )
        self.val_metrics = metrics.clone(prefix="val_")
        self.curr_ep = 0
        return


    def train(self, mode=True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def getNet(self):
        return self.net


    def forward(self, data):
        return self.net(data)


    def predict(self, data, full=False):
        with torch.no_grad():
            out = self.net(data)
        predictions, prob = self.submission_handler.format_data(
            data, out["y_hat"], out["pi"], inference=True
        )
        predictions = [predictions, out] if full else predictions
        return predictions, prob    

    def predict_improved(self, data, collision_tau=2.0):
        """Improved inference with integrated post-processing.

        Applies three improvements over the paper's argmax(pi) selection:
          1. Weighted-consensus: pick mode closest to pi-weighted centroid
          2. Collision-aware: filter modes that come within `collision_tau` meters
             of any other agent's predicted trajectory
          3. Returns aggregated probability for brier computation

        Returns:
            dict with keys:
                y_hat:          (B, 6, F, 2) all 6 mode trajectories (ego-local)
                pi:             (B, 6) raw logits
                pi_softmax:     (B, 6) softmax probabilities
                y_hat_others:   (B, N-1, F, 2) other-agent predictions
                selected:       (B,) index of selected mode per scene
                selected_traj:  (B, F, 2) selected trajectory
                method:         (B,) string label per scene ('consensus'/'collision'/'fallback')
                pi_aggregated:  (B,) aggregated probability of best-mode cluster (for brier)
        """
        with torch.no_grad():
            out = self.net(data)

        y_hat = out["y_hat"]           # (B, 6, F, 2) ego-local
        pi = out["pi"]                 # (B, 6) raw logits
        y_hat_others = out["y_hat_others"]  # (B, N-1, F, 2) per-agent-local
        B, K, F, _ = y_hat.shape

        pi_soft = torch.softmax(pi.double(), dim=-1).float()  # (B, 6)

        # --- Step 1: Weighted-consensus selection ---
        endpoints = y_hat[:, :, -1, :]                          # (B, 6, 2)
        w = pi_soft.unsqueeze(-1)                               # (B, 6, 1)
        w_centroid = (endpoints * w).sum(dim=1) / w.sum(dim=1)  # (B, 2)
        dist_to_wc = torch.norm(endpoints - w_centroid.unsqueeze(1), dim=-1)  # (B, 6)
        consensus_sel = dist_to_wc.argmin(dim=1)                # (B,)

        # --- Step 2: Collision-aware filtering ---
        # Transform to scene frame for clearance computation
        H = self.history_steps
        ego_angle = data["x_angles"][:, 0, H - 1]
        ego_center = data["x_centers"][:, 0]
        others_angle = data["x_angles"][:, 1:, H - 1]
        others_center = data["x_centers"][:, 1:]
        others_valid = ~data["x_padding_mask"][:, 1:, H:]

        def to_scene(traj, angle, center):
            c = torch.cos(angle).unsqueeze(-1).unsqueeze(-1)
            s = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)
            x, y = traj[..., 0:1], traj[..., 1:2]
            return torch.cat([c*x - s*y, s*x + c*y], dim=-1) + center.unsqueeze(-2)

        y_hat_scene = to_scene(
            y_hat.reshape(B*K, F, 2),
            ego_angle.unsqueeze(1).expand(B, K).reshape(B*K),
            ego_center.unsqueeze(1).expand(B, K, 2).reshape(B*K, 2),
        ).reshape(B, K, F, 2)

        y_hat_others_scene = to_scene(y_hat_others, others_angle, others_center)

        # Min distance per mode to any other agent
        diff = y_hat_scene.unsqueeze(2) - y_hat_others_scene.unsqueeze(1)  # (B,6,N-1,F,2)
        dist = torch.norm(diff, dim=-1)                                     # (B,6,N-1,F)
        v_exp = others_valid.unsqueeze(1).expand(-1, K, -1, -1)
        dist = dist.masked_fill(~v_exp, float("inf"))
        min_dist = dist.flatten(2).min(dim=-1).values                       # (B, 6)

        # Filter: prefer consensus mode, but if it collides, pick safest mode
        safe_mask = min_dist >= collision_tau                                # (B, 6)
        consensus_safe = safe_mask[torch.arange(B), consensus_sel]          # (B,)

        # If consensus is safe, use it. Otherwise pick highest-pi safe mode.
        # If no mode is safe, fall back to consensus anyway.
        masked_pi = torch.where(safe_mask, pi_soft, torch.tensor(-1e9, device=pi.device))
        any_safe = safe_mask.any(dim=1)
        safe_sel = masked_pi.argmax(dim=1)

        selected = torch.where(consensus_safe, consensus_sel,
                    torch.where(any_safe, safe_sel, consensus_sel))

        # --- Step 3: Mode probability aggregation ---
        # Sum probabilities of modes whose endpoints are within 3m of best mode
        best_by_pi = pi_soft.argmax(dim=1)  # paper's selection
        selected_endpoint = endpoints[torch.arange(B), selected]
        dist_to_selected = torch.norm(endpoints - selected_endpoint.unsqueeze(1), dim=-1)
        near_mask = dist_to_selected < 3.0
        pi_aggregated = (pi_soft * near_mask.float()).sum(dim=1)  # (B,)

        selected_traj = y_hat[torch.arange(B), selected]  # (B, F, 2)

        return {
            "y_hat": y_hat,
            "pi": pi,
            "pi_softmax": pi_soft,
            "y_hat_others": y_hat_others,
            "selected": selected,
            "selected_traj": selected_traj,
            "paper_selected": best_by_pi,
            "min_dist": min_dist,
            "pi_aggregated": pi_aggregated,
        }


    def cal_loss(self, out, data, batch_idx=0):
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        y, y_others = data["y"][:, 0], data["y"][:, 1:]

        loss = 0
        B = y_hat.shape[0]

        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[range(B), best_mode]

        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach(), label_smoothing=0.1)
        loss += agent_reg_loss + agent_cls_loss

        others_reg_mask = ~data["x_padding_mask"][:, 1:, self.history_steps:]
        others_reg_loss = F.smooth_l1_loss(y_hat_others[others_reg_mask], y_others[others_reg_mask])
        loss += others_reg_loss

        # Knowledge distillation losses
        kd_cls_loss = 0.0
        kd_feat_loss = 0.0
        if self.teacher is not None and self.training:
            with torch.no_grad():
                teacher_out = self.teacher(data)

            # Soft mode probability distillation (KL divergence with temperature)
            T = 2.0
            kd_cls_loss = F.kl_div(
                F.log_softmax(pi / T, dim=-1),
                F.softmax(teacher_out["pi"] / T, dim=-1),
                reduction='batchmean'
            ) * (T * T)

            # Feature-level distillation (align encoder representations)
            kd_feat_loss = F.mse_loss(out["x_agent"], teacher_out["x_agent"])

            loss += 0.5 * kd_cls_loss + 0.5 * kd_feat_loss

        return {
            "loss": loss,
            "reg_loss": agent_reg_loss.item(),
            "cls_loss": agent_cls_loss.item(),
            "others_reg_loss": others_reg_loss.item(),
            "kd_cls_loss": kd_cls_loss if isinstance(kd_cls_loss, float) else kd_cls_loss.item(),
            "kd_feat_loss": kd_feat_loss if isinstance(kd_feat_loss, float) else kd_feat_loss.item(),
        }

    def training_step(self, data, batch_idx):
        out = self(data)
        losses = self.cal_loss(out, data)
        
        for k, v in losses.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        return losses["loss"]

    def on_train_start(self):
        self.start_time = time.time()

    def on_train_end(self):
        total_time = time.time() - self.start_time
        print(f"\n[METRICS] Total Training Time: {total_time/3600:.2f} hours")

    def validation_step(self, data, batch_idx):
        start_t = time.time()
        out = self(data)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.time() - start_t) * 1000

        losses = self.cal_loss(out, data, -1)
        metrics = self.val_metrics(out, data["y"][:, 0])

        self.log(
            "val/reg_loss",
            losses["reg_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        for k in self.val_scores.keys(): self.val_scores[k].append(metrics[k].item())

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log("val/latency_ms", latency_ms, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_test_start(self) -> None:
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)

    def test_step(self, data, batch_idx) -> None:
        out = self(data)
        self.submission_handler.format_data(data, out["y_hat"], out["pi"])

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()

    def on_validation_start(self) -> None:
        self.val_scores = {"val_MR": [], "val_minADE1": [], "val_minADE6": [], "val_minFDE1": [], "val_minFDE6": [], "val_brier-minFDE6": []}

    def on_validation_end(self) -> None:      
        print( " & ".join( ["{:5.3f}".format(np.mean(self.val_scores[k])) for k in ["val_MR", "val_minADE6", "val_minFDE6", "val_brier-minFDE6"]] ) )
        self.curr_ep += 1

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
            nn.GRUCell
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
            nn.Parameter
        )
        for module_name, module in self.named_modules():
            if module_name.startswith("teacher"):
                continue
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if full_param_name.startswith("teacher"):
                    continue
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
            if not param_name.startswith("teacher")
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return [optimizer], [scheduler]
