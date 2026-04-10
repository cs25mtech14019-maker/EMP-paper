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


    def cal_loss(self, out, data, batch_idx=0):
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        y, y_others = data["y"][:, 0], data["y"][:, 1:]

        loss = 0
        B = y_hat.shape[0]
        B_range = range(B)

        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[B_range, best_mode]

        per_sample_loss = F.smooth_l1_loss(y_hat_best[..., :2], y, reduction="none")  # [B, 60, 2]
        per_sample_loss = per_sample_loss.mean(dim=(-2, -1))  # [B]

        # ---------------------------------------------------------------------
        # NEW: Kinematically-Gated HNM.
        # Calculate kinematics of ground truth to filter out sensor noise/ghosts.
        dt = 0.1
        v = torch.diff(y, dim=1) / dt                                       # [B, 59, 2]
        a_mag = torch.norm(torch.diff(v, dim=1) / dt, dim=-1)               # [B, 58]
        max_a = a_mag.max(dim=-1)[0]
        
        yaw = torch.atan2(v[..., 1], v[..., 0])                             # [B, 59]
        yaw_diff = (torch.diff(yaw, dim=1) + torch.pi) % (2 * torch.pi) - torch.pi
        max_yaw = (torch.abs(yaw_diff) / dt).max(dim=-1)[0]
        
        # Bounding limits (95th-99th percentile roughly from AV2 profiling).
        ACCEL_THRESHOLD = 15.0
        YAW_THRESHOLD = 2.0
        
        is_valid_hard_sample = (max_a < ACCEL_THRESHOLD) & (max_yaw < YAW_THRESHOLD)
        
        weights = torch.ones_like(per_sample_loss)
        valid_indices = is_valid_hard_sample.nonzero(as_tuple=True)[0]
        
        if len(valid_indices) > 0:
            valid_losses = per_sample_loss[valid_indices]
            
            # Target up to Top 25% of the WHOLE batch, but exclusively from valid subset
            num_hard = max(1, int(B * 0.25))
            num_hard = min(num_hard, len(valid_indices))
            
            if num_hard > 0:
                _, hard_sub_indices = torch.topk(valid_losses, num_hard)
                
                # Map back to the original batch indices
                true_hard_indices = valid_indices[hard_sub_indices]
                
                # Apply 2.0x HNM multiplier ONLY to valid, physically possible hard samples
                weights[true_hard_indices] = 2.0
        # ---------------------------------------------------------------------
        
        agent_reg_loss = (weights * per_sample_loss).mean()

        soft_targets = F.softmax(-l2_norm, dim=-1)
        log_pi = F.log_softmax(pi, dim=-1)
        agent_cls_loss = F.kl_div(log_pi, soft_targets.detach(), reduction='batchmean')
        
        loss += agent_reg_loss + agent_cls_loss

        others_reg_mask = ~data["x_padding_mask"][:, 1:, self.history_steps:]
        others_reg_loss = F.smooth_l1_loss(y_hat_others[others_reg_mask], y_others[others_reg_mask])
        loss += others_reg_loss

        return {
            "loss": loss,
            "reg_loss": agent_reg_loss.item(),
            "cls_loss": agent_cls_loss.item(),
            "others_reg_loss": others_reg_loss.item(),
            "mean_hnm_weight": weights.mean().item(),
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
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
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
