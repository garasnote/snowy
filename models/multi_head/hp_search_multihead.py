#!/usr/bin/env python3
"""
Optuna TPE hyperparameter search for DINOv3MultiHead.

Backbone loaded ONCE before the search loop — shared across all trials.
Only the lightweight heads (QA MLP + 2× LDecoder, ~1-2M params) are
rebuilt per trial. No re-downloading or re-materializing 86M params.

After all trials:
  - Shapley HP importances via fANOVA logged to MLflow

Usage:
    python hp_search_multihead.py --n-trials 20
    python hp_search_multihead.py --n-trials 3   # quick test
    python hp_search_multihead.py --variant dinov3_large --n-trials 20
"""

import argparse
import os
import sys
import warnings
from pathlib import Path


class _FilteredStderr:
    """Drop multiprocessing temp-dir cleanup noise from stderr."""
    _SKIP = (
        "OSError: [Errno 39] Directory not empty",
        "rmtree(tempdir",
        "_remove_temp_dir",
        "multiprocessing/util.py",
        "_run_finalizers",
        "During handling of the above exception",
    )

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def write(self, s):
        if any(pat in s for pat in self._SKIP):
            return len(s)
        return self._wrapped.write(s)

    def flush(self):
        return self._wrapped.flush()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from optuna.samplers import TPESampler
from optuna.importance import get_param_importances, FanovaImportanceEvaluator

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "models"))


# ── Shared-backbone trial model ───────────────────────────────────────────────

class TrialModel(nn.Module):
    """Wraps a pre-loaded frozen backbone + trial-specific heads.

    Backbone is NOT re-loaded per trial — caller passes the shared instance.
    Only qa_head, ground_decoder, snow_decoder are rebuilt each trial.
    """

    def __init__(self, shared_backbone, hidden_dim, n_layers,
                 dropout, proj_dim, n_patch_tokens, grid_h, grid_w,
                 input_size, extract_layers):
        super().__init__()
        from multi_head.train_multihead import LDecoder

        self.backbone       = shared_backbone   # frozen, shared
        self.hidden_dim     = hidden_dim
        self.n_patch_tokens = n_patch_tokens
        self.grid_h         = grid_h
        self.grid_w         = grid_w
        self.input_size     = input_size
        self.extract_layers = extract_layers

        self.qa_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.ground_decoder = LDecoder(hidden_dim=hidden_dim, proj_dim=proj_dim,
                                       num_layers=n_layers)
        self.snow_decoder   = LDecoder(hidden_dim=hidden_dim, proj_dim=proj_dim,
                                       num_layers=n_layers)
        self.log_var = nn.Parameter(torch.zeros(3))

    @torch.no_grad()
    def extract_features(self, pixel_values):
        outputs      = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        cls_token    = hidden_states[-1][:, 0, :]
        patch_feats  = [hidden_states[i][:, -self.n_patch_tokens:, :]
                        for i in self.extract_layers]
        return cls_token, patch_feats

    def forward_qa(self, pixel_values):
        cls, _ = self.extract_features(pixel_values)
        return self.qa_head(cls).squeeze(-1)

    def forward_seg(self, pixel_values, task, output_size=None):
        _, patch_feats = self.extract_features(pixel_values)
        decoder = self.ground_decoder if task == "ground" else self.snow_decoder
        logits  = decoder(patch_feats, self.grid_h, self.grid_w)
        if output_size is None:
            output_size = (self.input_size, self.input_size)
        if logits.shape[2:] != output_size:
            logits = F.interpolate(logits, size=output_size,
                                   mode="bilinear", align_corners=False)
        return logits

    def trainable_parameters(self):
        return (list(self.qa_head.parameters()) +
                list(self.ground_decoder.parameters()) +
                list(self.snow_decoder.parameters()) +
                [self.log_var])


# ── Lightning module for trial model ──────────────────────────────────────────

class TrialModule(torch.nn.Module):
    """Thin Lightning wrapper around TrialModel — reuses train/val logic."""
    pass


def run_trial(trial, args, shared_backbone, backbone_cfg, device):
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import CSVLogger
    from multi_head.train_multihead import (
        MultiHeadDataModule, uncertainty_loss, seg_loss,
    )
    from sklearn.metrics import roc_auc_score, f1_score
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    try:
        from lightning.pytorch.loggers import MLFlowLogger
        _mlflow_ok = True
    except ImportError:
        _mlflow_ok = False

    # ── Sample HPs ────────────────────────────────────────────────────────────
    lr         = trial.suggest_float("lr",         1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    dropout    = trial.suggest_float("dropout",    0.1,  0.5)
    proj_dim   = trial.suggest_categorical("proj_dim",   [128, 256, 512])
    patience   = trial.suggest_int("patience",     5,    20)

    trial_name = (f"trial{trial.number:03d}"
                  f"_lr{lr:.0e}_bs{batch_size}"
                  f"_do{dropout:.2f}_pd{proj_dim}_pat{patience}")
    print(f"\n{'='*60}\nTrial {trial.number}: {trial_name}\n{'='*60}")

    out_dir = _HERE / "checkpoints" / "hp_search" / trial_name
    out_dir.mkdir(parents=True, exist_ok=True)

    hidden_dim     = backbone_cfg["hidden_dim"]
    n_layers       = len(backbone_cfg["extract_layers"])
    input_size     = backbone_cfg["input_size"]
    patch_size     = backbone_cfg["patch_size"]
    extract_layers = backbone_cfg["extract_layers"]
    grid_h         = input_size // patch_size
    grid_w         = input_size // patch_size
    n_patch_tokens = grid_h * grid_w

    model = TrialModel(
        shared_backbone=shared_backbone,
        hidden_dim=hidden_dim, n_layers=n_layers,
        dropout=dropout, proj_dim=proj_dim,
        n_patch_tokens=n_patch_tokens,
        grid_h=grid_h, grid_w=grid_w,
        input_size=input_size,
        extract_layers=extract_layers,
    ).to(device)

    # ── Inline Lightning-style training loop ──────────────────────────────────
    # (avoids Lightning re-instantiating the model from scratch)
    dm = MultiHeadDataModule(args.variant, batch_size, args.num_workers)
    dm.setup()

    qa_crit = nn.BCEWithLogitsLoss()
    opt     = AdamW(model.trainable_parameters(), lr=lr, weight_decay=1e-4)
    sched   = CosineAnnealingLR(opt, T_max=args.max_epochs, eta_min=1e-5)

    # MLflow run for this trial
    mlflow_run = None
    if args.mlflow_uri and _mlflow_ok:
        try:
            import mlflow
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment(args.experiment)
            mlflow_run = mlflow.start_run(run_name=trial_name)
            mlflow.log_params({
                "trial": trial.number, "lr": lr, "batch_size": batch_size,
                "dropout": dropout, "proj_dim": proj_dim, "patience": patience,
                "variant": args.variant,
            })
        except Exception as e:
            print(f"MLflow init failed: {e}")
            mlflow_run = None

    from torch.utils.data import DataLoader
    def _dl(ds, shuffle, drop_last=False):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=drop_last)

    qa_train_dl = _dl(dm.qa_train,  shuffle=True,  drop_last=True)
    gr_train_dl = _dl(dm.gr_train,  shuffle=True)
    sn_train_dl = _dl(dm.sn_train,  shuffle=True)
    qa_val_dl   = _dl(dm.qa_val,    shuffle=False)
    gr_val_dl   = _dl(dm.gr_val,    shuffle=False)
    sn_val_dl   = _dl(dm.sn_val,    shuffle=False)

    best_score, no_improve = 0.0, 0
    best_ckpt = out_dir / "best.pt"

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        qa_iter = iter(qa_train_dl)
        gr_iter = iter(gr_train_dl)
        sn_iter = iter(sn_train_dl)
        n_steps = max(len(qa_train_dl), len(gr_train_dl), len(sn_train_dl))

        for _ in range(n_steps):
            opt.zero_grad()
            def _next(it, dl):
                try: return next(it)
                except StopIteration: return next(iter(dl))

            x_qa, y_qa = _next(qa_iter, qa_train_dl)
            x_gr, y_gr = _next(gr_iter, gr_train_dl)
            x_sn, y_sn = _next(sn_iter, sn_train_dl)

            l_qa = qa_crit(model.forward_qa(x_qa.to(device)), y_qa.to(device))
            l_gr = seg_loss(model.forward_seg(x_gr.to(device), "ground"), y_gr.to(device))
            l_sn = seg_loss(model.forward_seg(x_sn.to(device), "snow"),   y_sn.to(device))

            uncertainty_loss([l_qa, l_gr, l_sn], model.log_var).backward()
            opt.step()
        sched.step()

        # Val
        model.eval()
        qa_probs, qa_targets = [], []
        gr_inter = gr_union = sn_inter = sn_union = 0.0
        with torch.no_grad():
            for x, y in qa_val_dl:
                qa_probs.extend(torch.sigmoid(model.forward_qa(x.to(device))).cpu().tolist())
                qa_targets.extend(y.tolist())
            for x, y in gr_val_dl:
                p = (torch.sigmoid(model.forward_seg(x.to(device), "ground")) > 0.5).float()
                y = y.to(device)
                gr_inter += (p * y).sum().item(); gr_union += (p + y - p * y).sum().item()
            for x, y in sn_val_dl:
                p = (torch.sigmoid(model.forward_seg(x.to(device), "snow")) > 0.5).float()
                y = y.to(device)
                sn_inter += (p * y).sum().item(); sn_union += (p + y - p * y).sum().item()

        auroc    = roc_auc_score(qa_targets, qa_probs) if len(set(qa_targets)) > 1 else 0.0
        gr_iou   = gr_inter / (gr_union + 1e-6)
        sn_iou   = sn_inter / (sn_union + 1e-6)
        score    = (auroc + gr_iou + sn_iou) / 3.0
        sigmas   = torch.exp(model.log_var * 0.5).detach().cpu().numpy()

        print(f"  ep{epoch:3d}  qa={auroc:.4f}  gr={gr_iou:.4f}  sn={sn_iou:.4f}"
              f"  avg={score:.4f}  σ=[{sigmas[0]:.2f},{sigmas[1]:.2f},{sigmas[2]:.2f}]")

        if mlflow_run:
            try:
                import mlflow
                mlflow.log_metrics({"val/qa_auroc": auroc, "val/ground_iou": gr_iou,
                                    "val/snow_iou": sn_iou, "val/avg_score": score,
                                    "sigma/qa": float(sigmas[0]),
                                    "sigma/ground": float(sigmas[1]),
                                    "sigma/snow": float(sigmas[2])}, step=epoch)
            except Exception:
                pass

        trial.report(score, epoch)
        if trial.should_prune():
            if mlflow_run:
                import mlflow; mlflow.end_run(status="KILLED")
            raise optuna.exceptions.TrialPruned()

        if score > best_score:
            best_score = score
            no_improve = 0
            torch.save({"state_dict": model.state_dict(),
                        "score": score, "epoch": epoch}, best_ckpt)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch} (patience={patience})")
                break

    if mlflow_run:
        try:
            import mlflow; mlflow.end_run()
        except Exception:
            pass

    return best_score


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sys.stderr = _FilteredStderr(sys.stderr)

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant",     choices=["dinov3_base", "dinov3_large"],
                        default="dinov3_base")
    parser.add_argument("--n-trials",    type=int,   default=20)
    parser.add_argument("--max-epochs",  type=int,   default=100,
                        help="Epoch ceiling per trial — patience decides actual stop")
    parser.add_argument("--num-workers", type=int,   default=4)
    parser.add_argument("--mlflow-uri",  type=str,   default=None)
    parser.add_argument("--experiment",  type=str,   default=None)
    parser.add_argument("--study-name",  type=str,   default="multihead_tpe")
    parser.add_argument("--storage",     type=str,   default=None)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.experiment is None:
        args.experiment = f"multihead_{args.variant}"

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    print(f"Variant:    {args.variant}")
    print(f"Trials:     {args.n_trials}")
    print(f"Max epochs: {args.max_epochs} (patience decides early stop)")
    print(f"Experiment: {args.experiment}")
    print(f"MLflow:     {args.mlflow_uri or 'disabled'}")

    # ── Load backbone ONCE ────────────────────────────────────────────────────
    from student_ground.configs import MODEL_REGISTRY
    cfg = MODEL_REGISTRY[args.variant]
    print(f"\nLoading backbone ({args.variant}) once...")
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained(cfg["backbone"])
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    backbone = backbone.to(args.device)
    print("Backbone loaded. Starting search.\n")

    device = torch.device(args.device)
    sampler = TPESampler(seed=42, multivariate=True)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=10)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: run_trial(trial, args, backbone, cfg, device),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # ── Results ───────────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"Best trial: #{best.number}  val/avg_score={best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # ── Shapley importances ───────────────────────────────────────────────────
    print("\nComputing Shapley HP importances (fANOVA)...")
    try:
        importances = get_param_importances(
            study, evaluator=FanovaImportanceEvaluator(seed=42))
        print("\nHP Importances:")
        for hp, imp in sorted(importances.items(), key=lambda x: -x[1]):
            print(f"  {hp:20s} {imp:.4f}  {'█' * int(imp * 40)}")

        if args.mlflow_uri:
            try:
                import mlflow
                mlflow.set_tracking_uri(args.mlflow_uri)
                mlflow.set_experiment(args.experiment)
                with mlflow.start_run(run_name="shapley_importances"):
                    for hp, imp in importances.items():
                        mlflow.log_metric(f"importance/{hp}", imp)
                    mlflow.log_param("best_trial",     best.number)
                    mlflow.log_param("best_avg_score", best.value)
                    for k, v in best.params.items():
                        mlflow.log_param(f"best_{k}", v)
                print("Shapley importances logged to MLflow.")
            except Exception as e:
                print(f"MLflow shapley logging failed: {e}")
    except Exception as e:
        print(f"Shapley failed (need ≥2 completed trials): {e}")

    df = study.trials_dataframe()
    summary = _HERE / "checkpoints" / "hp_search" / "study_summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary, index=False)
    print(f"\nStudy summary → {summary}")


if __name__ == "__main__":
    main()
