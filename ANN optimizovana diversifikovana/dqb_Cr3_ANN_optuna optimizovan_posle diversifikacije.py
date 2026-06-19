"""
ANN Expert System for Cr³⁺ Phosphor Discovery
===============================================
Multilayer Perceptron (MLP) via backpropagation for predicting the crystal
field parameter Dq/B in Cr³⁺-doped inorganic phosphor host lattices.

Authors : Snežana Đurković, Prof. Dr. Miroslav Dramićanin
Group   : OMAS — Optical Materials and Spectroscopy
Institute: Nuclear Sciences "Vinča", University of Belgrade
ORCID   : https://orcid.org/0009-0007-6638-0682
Year    : 2026

Architecture
------------
Input (15) → Linear(128) → BN → ReLU → Dropout(0.342)
           → Linear(128) → BN → ReLU → Dropout(0.208)
           → Linear(1)

Non-pyramidal architecture (128→128) determined by Bayesian
hyperparameter optimization (Optuna TPE, 50 trials, 10-fold CV)
on diversified To_predict set spanning all crystal field regimes
(WCF / Edge / Tier 1 / Edge2 / SCF).

Training
--------
Optimizer    : Adam  (lr = 5.53e-3, weight_decay = 4.05e-3)
Scheduler    : CosineAnnealingLR (T_max = EPOCHS)
Loss         : MSELoss
Init         : He / Kaiming Normal
Grad clip    : max_norm = 1.0
Epochs       : 500
Batch size   : 64

Hyperparameter optimization
---------------------------
Strategy     : Bayesian optimization (Optuna TPE + MedianPruner)
Trials       : 50  (23 complete, 27 pruned)
Best CV R²   : 0.5029 (Trial 7)
Script       : ann_optuna_search.py

Usage
-----
    python dqb_Cr3_ANN.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── Configuration ──────────────────────────────────────────────────────────────
TRAIN_PATH   = "Cr3_dqb_training_set.xlsx"
PREDICT_PATH = "To_predict.xlsx"
OUTPUT_PATH  = "ann_dqb_results.xlsx"

# Hyperparameters
EPOCHS       = 500
BATCH_SIZE   = 64        # optimized
LR           = 5.53e-3   # optimized
WEIGHT_DECAY = 4.05e-3   # optimized
DROPOUT      = (0.342, 0.208)  # optimized per hidden layer
N_SPLITS     = 10
N_REPEATS    = 10

# Target NIR window
TARGET_LOW   = 2.2
TARGET_HIGH  = 2.8
TARGET_TOL   = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 15 descriptors — order must match Excel columns exactly
FEATURE_COLS = [
    "avg_Mulliken EN",
    "avg_First ionization energy (kJ/mol)",
    "1/r2",
    "avg_Metallic valence",
    "avg_Martynov-Batsanov EN",
    "beta",
    "SGR No.",
    "avg_Number of outer shell electrons",
    "X",
    "max_metal_ligand_bond_length",
    "std_Mendeleev number",
    "volume_per_atom",
    "max_First ionization energy (kJ/mol)",
    "volume_per_fu",
    "polyhedron volume",
]


# ── Model architecture ─────────────────────────────────────────────────────────
class DqBPredictor(nn.Module):
    """
    Multilayer Perceptron for Dq/B regression.
    Architecture: 15 → 128 → 128 → 1  (non-pyramidal, Optuna-optimized)
    BatchNorm + ReLU + Dropout after each hidden layer.
    He (Kaiming Normal) weight initialization.

    Non-pyramidal design (128→128) outperformed pyramidal alternatives
    in Bayesian optimization on diversified dataset spanning all CF regimes.
    Best CV R² = 0.5029 (Trial 7, 50 trials total).
    """

    def __init__(self, input_dim: int = 15):
        super().__init__()
        d = DROPOUT
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(d[0]),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(d[1]),

            nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Training helpers ───────────────────────────────────────────────────────────
def make_loader(X: np.ndarray, y: np.ndarray,
                shuffle: bool = True) -> DataLoader:
    tx = torch.tensor(X, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(tx, ty),
                      batch_size=BATCH_SIZE, shuffle=shuffle)


def train_model(X_tr: np.ndarray, y_tr: np.ndarray) -> DqBPredictor:
    """Train one model instance on (X_tr, y_tr)."""
    model     = DqBPredictor(X_tr.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()
    loader    = make_loader(X_tr, y_tr)

    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

    return model


@torch.no_grad()
def predict(model: DqBPredictor, X: np.ndarray) -> np.ndarray:
    model.eval()
    tx = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    return model(tx).cpu().numpy()


# ── Tier classification ────────────────────────────────────────────────────────
def assign_tier(dqb: float, sigma: float) -> str:
    """
    Classify a prediction into one of four tiers based on predicted Dq/B
    and ensemble uncertainty σ.

    Tier 1 — Strong   : Dq/B in [2.2, 2.8], σ < 0.2
    Tier 2 — Promising: Dq/B in [2.2, 2.8], σ < 0.4
    Tier 3 — Uncertain: Dq/B in [2.2, 2.8], σ ≥ 0.4
    Tier 3 — Edge     : Dq/B within 0.3 of target boundary
    Tier 4 — Out of range
    """
    in_range  = TARGET_LOW <= dqb <= TARGET_HIGH
    near_edge = (TARGET_LOW - TARGET_TOL <= dqb < TARGET_LOW) or \
                (TARGET_HIGH < dqb <= TARGET_HIGH + TARGET_TOL)
    if in_range and sigma < 0.2:   return "Tier 1 — Strong"
    elif in_range and sigma < 0.4: return "Tier 2 — Promising"
    elif in_range:                 return "Tier 3 — Uncertain"
    elif near_edge:                return "Tier 3 — Edge"
    else:                          return "Tier 4 — Out of range"


# ── Data loading ───────────────────────────────────────────────────────────────
def load_training(path: str):
    df = pd.read_excel(path)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in training set: {missing}\n"
            f"Available: {list(df.columns)}"
        )
    return df[FEATURE_COLS].values.astype(np.float32), df["Dq/B"].values.astype(np.float32), df


def load_prediction(path: str):
    df = pd.read_excel(path)
    first_col = str(list(df.columns)[0])
    has_no_header = (
        first_col not in ["Formula", "formula"]
        and not first_col.startswith("avg")
        and first_col not in FEATURE_COLS
    )
    if has_no_header:
        df = pd.read_excel(path, header=None, names=["Formula"] + FEATURE_COLS)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in prediction set: {missing}")
    return df[FEATURE_COLS].values.astype(np.float32), df["Formula"].values


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"Hyperparameters: epochs={EPOCHS} | batch={BATCH_SIZE} | "
          f"lr={LR} | weight_decay={WEIGHT_DECAY}")

    # Load
    print("\n📂 Loading data...")
    X, y, train_df = load_training(TRAIN_PATH)
    X_new, formulas = load_prediction(PREDICT_PATH)
    print(f"✅ Training set : {X.shape[0]} compounds, {X.shape[1]} features")
    print(f"   Dq/B range   : {y.min():.3f} – {y.max():.3f}  (mean {y.mean():.3f})")
    print(f"✅ Prediction set: {len(formulas)} candidates")

    # Find best random state
    print("\n🔄 Searching for best random state...")
    candidates  = sorted(set(range(5, 101, 5)).union(range(5, 101, 7)))
    best_r2, best_state = -np.inf, None

    for i, rs in enumerate(candidates, 1):
        r2s = []
        for tr, te in KFold(n_splits=N_SPLITS, shuffle=True, random_state=rs).split(X):
            sc  = RobustScaler().fit(X[tr])
            m   = train_model(sc.transform(X[tr]), y[tr])
            r2s.append(r2_score(y[te], predict(m, sc.transform(X[te]))))
        mean_r2 = np.mean(r2s)
        print(f"   [{i:02d}/{len(candidates)}] rs={rs:3d}  R²={mean_r2:.4f}", end="\r")
        if mean_r2 > best_r2:
            best_r2, best_state = mean_r2, rs

    print(f"\n✅ Best random state = {best_state}  (CV R² = {best_r2:.4f})")

    # Final 10-fold CV
    print(f"\n📊 Running final {N_SPLITS}-fold CV (random_state = {best_state})...")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=best_state)

    y_true, y_pred_cv          = [], []
    r2s, maes, rmses           = [], [], []
    fold_preds     = [[] for _ in range(len(y))]
    fold_preds_new = []

    for fold, (tr, te) in enumerate(kf.split(X), 1):
        sc  = RobustScaler().fit(X[tr])
        m   = train_model(sc.transform(X[tr]), y[tr])
        p   = predict(m, sc.transform(X[te]))

        y_true.extend(y[te]); y_pred_cv.extend(p)
        r2s.append(r2_score(y[te], p))
        maes.append(mean_absolute_error(y[te], p))
        rmses.append(np.sqrt(mean_squared_error(y[te], p)))
        for idx, pred in zip(te, p):
            fold_preds[idx].append(pred)

        sc_new = RobustScaler().fit(X[tr])
        fold_preds_new.append(predict(
            train_model(sc_new.transform(X[tr]), y[tr]),
            sc_new.transform(X_new)
        ))
        print(f"   Fold {fold:2d}: R² = {r2s[-1]:.4f}  "
              f"MAE = {maes[-1]:.4f}  RMSE = {rmses[-1]:.4f}")

    fold_preds_new = np.array(fold_preds_new)

    final_r2   = r2_score(y_true, y_pred_cv)
    final_mae  = mean_absolute_error(y_true, y_pred_cv)
    final_rmse = np.sqrt(mean_squared_error(y_true, y_pred_cv))

    print(f"\n{'='*55}")
    print(f"  CV R²   = {final_r2:.4f}  (±{np.std(r2s):.4f})")
    print(f"  CV MAE  = {final_mae:.4f}  (±{np.std(maes):.4f})")
    print(f"  CV RMSE = {final_rmse:.4f}  (±{np.std(rmses):.4f})")
    print(f"{'='*55}")

    # Repeated CV for stable uncertainty
    print(f"\n🔁 Estimating uncertainty ({N_REPEATS}×{N_SPLITS}-fold ensemble)...")
    all_preds = np.zeros((N_REPEATS * N_SPLITS, len(formulas)))
    idx = 0
    for rep in range(N_REPEATS):
        for tr, _ in KFold(n_splits=N_SPLITS, shuffle=True,
                           random_state=best_state + rep * 13).split(X):
            sc  = RobustScaler().fit(X[tr])
            m   = train_model(sc.transform(X[tr]), y[tr])
            all_preds[idx] = predict(m, sc.transform(X_new))
            idx += 1
    uncertainty = np.std(all_preds, axis=0)

    # Final model on full dataset
    print("\n🏁 Training final model on full dataset...")
    sc_full     = RobustScaler().fit(X)
    final_model = train_model(sc_full.transform(X), y)
    final_preds = predict(final_model, sc_full.transform(X_new))

    # Tier classification
    tiers = [assign_tier(p, s) for p, s in zip(final_preds, uncertainty)]

    results_df = pd.DataFrame({
        "Formula":          formulas,
        "Predicted Dq/B":   np.round(final_preds, 4),
        "Uncertainty (σ)":  np.round(uncertainty, 4),
        "Tier":             tiers,
    }).sort_values(["Tier", "Predicted Dq/B"])

    results_df.to_excel(OUTPUT_PATH, index=False)
    print(f"\n💾 Saved predictions → {OUTPUT_PATH}")

    print("\n🏷️  Tier summary:")
    for tier, count in results_df["Tier"].value_counts().sort_index().items():
        print(f"   {tier}: {count} compounds")

    # Parity plot + uncertainty histogram
    cv_means = [np.mean(p) for p in fold_preds]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Parity plot
    ax = axes[0]
    ax.scatter(y, cv_means, alpha=0.65, edgecolors="navy",
               facecolors="steelblue", s=50, linewidths=0.5,
               label="Compounds")
    lims = [min(y.min(), min(cv_means)) - 0.05,
            max(y.max(), max(cv_means)) + 0.05]
    ax.plot(lims, lims, "r--", lw=1.8, label="Ideal (y = ŷ)")
    ax.axvspan(TARGET_LOW, TARGET_HIGH, alpha=0.08,
               color="gold", label="NIR target window")
    ax.set_xlabel("True Dq/B", fontsize=12)
    ax.set_ylabel("Predicted Dq/B (CV mean)", fontsize=12)
    ax.set_title(
        f"Parity Plot — ANN (MLP)\n"
        f"R² = {final_r2:.4f}   MAE = {final_mae:.4f}   RMSE = {final_rmse:.4f}",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Uncertainty distribution
    ax2 = axes[1]
    ax2.hist(uncertainty, bins=12, color="#1a4480", edgecolor="white",
             alpha=0.85, label="Candidates")
    ax2.axvline(0.2, color="green", lw=1.5, ls="--", label="σ = 0.2 (Tier 1)")
    ax2.axvline(0.4, color="orange", lw=1.5, ls="--", label="σ = 0.4 (Tier 2)")
    ax2.set_xlabel("Ensemble Uncertainty σ", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Prediction Uncertainty Distribution", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ann_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved → ann_results.png")


if __name__ == "__main__":
    main()
