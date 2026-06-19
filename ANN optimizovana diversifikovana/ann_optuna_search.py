"""
Bayesian Hyperparameter Optimization for Cr³⁺ Dq/B ANN Predictor
==================================================================
Uses Optuna (TPE + MedianPruner) to optimize the 2HL MLP architecture
(15→hidden1→hidden2→1) for Dq/B crystal field parameter prediction.

Authors : Snežana Đurković, Prof. Dr. Miroslav Dramićanin
Group   : OMAS — Optical Materials and Spectroscopy
Institute: Nuclear Sciences "Vinča", University of Belgrade
ORCID   : https://orcid.org/0009-0007-6638-0682
Year    : 2026

Optimized hyperparameters
-------------------------
Architecture : hidden1 neurons, hidden2 neurons
Regularization: dropout1, dropout2, weight_decay
Optimization : learning_rate, batch_size

Fixed (not optimized)
---------------------
Epochs     : 500
Activation : ReLU
Init       : He / Kaiming Normal
Scheduler  : CosineAnnealingLR
Grad clip  : max_norm = 1.0
CV         : 10-fold (single run per trial for speed)

Usage
-----
    pip install optuna torch scikit-learn pandas openpyxl
    python ann_optuna_search.py

Output
------
    ann_optuna_results.xlsx  — all trials sorted by R²
    ann_best_params.txt      — best configuration for use in dqb_Cr3_ANN.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# ── Configuration ──────────────────────────────────────────────────────────────
TRAIN_PATH = "Cr3_dqb_training_set.xlsx"

N_TRIALS   = 50      # number of Optuna trials
N_SPLITS   = 10      # CV folds per trial
EPOCHS     = 500     # training epochs per fold
SEED       = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# ── Hyperparameter search space ────────────────────────────────────────────────
SEARCH_SPACE = {
    "hidden1":      [32, 64, 128, 256],   # neurons in first hidden layer
    "hidden2":      [16, 32, 64, 128],    # neurons in second hidden layer
    "dropout1":     (0.10, 0.40),         # dropout rate after hidden1 (continuous)
    "dropout2":     (0.05, 0.30),         # dropout rate after hidden2 (continuous)
    "lr":           (1e-4, 1e-2),         # learning rate (log scale)
    "weight_decay": (1e-4, 1e-2),         # L2 regularization (log scale)
    "batch_size":   [32, 64, 128],        # mini-batch size
}


# ── Model ──────────────────────────────────────────────────────────────────────
class DqBPredictor(nn.Module):
    """
    2HL MLP with configurable neuron counts and dropout rates.
    Architecture: 15 → hidden1 → hidden2 → 1
    """

    def __init__(self, hidden1: int, hidden2: int,
                 dropout1: float, dropout2: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, hidden1),
            nn.BatchNorm1d(hidden1), nn.ReLU(), nn.Dropout(dropout1),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2), nn.ReLU(), nn.Dropout(dropout2),

            nn.Linear(hidden2, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Training ───────────────────────────────────────────────────────────────────
def train_model(X_tr: np.ndarray, y_tr: np.ndarray,
                params: dict) -> DqBPredictor:
    model = DqBPredictor(
        params["hidden1"], params["hidden2"],
        params["dropout1"], params["dropout2"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    Xtr = torch.tensor(X_tr, dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=params["batch_size"], shuffle=True,
    )

    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

    return model


@torch.no_grad()
def predict(model: DqBPredictor, X: np.ndarray) -> np.ndarray:
    model.eval()
    tx = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    return model(tx).cpu().numpy()


# ── Optuna objective ───────────────────────────────────────────────────────────
def objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """
    Objective function for Optuna.
    Returns mean 10-fold CV R² (to be maximized).
    Pruning: reports intermediate R² after each fold;
    MedianPruner terminates unpromising trials early.
    """
    params = {
        "hidden1":      trial.suggest_categorical("hidden1",
                            SEARCH_SPACE["hidden1"]),
        "hidden2":      trial.suggest_categorical("hidden2",
                            SEARCH_SPACE["hidden2"]),
        "dropout1":     trial.suggest_float("dropout1",
                            *SEARCH_SPACE["dropout1"]),
        "dropout2":     trial.suggest_float("dropout2",
                            *SEARCH_SPACE["dropout2"]),
        "lr":           trial.suggest_float("lr",
                            *SEARCH_SPACE["lr"], log=True),
        "weight_decay": trial.suggest_float("weight_decay",
                            *SEARCH_SPACE["weight_decay"], log=True),
        "batch_size":   trial.suggest_categorical("batch_size",
                            SEARCH_SPACE["batch_size"]),
    }

    # Constraint: hidden2 must be <= hidden1 (pyramidal structure)
    if params["hidden2"] > params["hidden1"]:
        raise optuna.exceptions.TrialPruned()

    r2s = []
    kf  = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for fold, (tr, te) in enumerate(kf.split(X)):
        sc    = RobustScaler().fit(X[tr])
        model = train_model(sc.transform(X[tr]), y[tr], params)
        p     = predict(model, sc.transform(X[te]))
        r2s.append(r2_score(y[te], p))

        # Report intermediate value for pruning
        trial.report(np.mean(r2s), step=fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return float(np.mean(r2s))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"Device : {DEVICE}")
    print(f"Trials : {N_TRIALS}")
    print(f"CV     : {N_SPLITS}-fold")
    print(f"Epochs : {EPOCHS} per fold\n")

    # Load data
    print("📂 Loading data...")
    train_df = pd.read_excel(TRAIN_PATH)
    X = train_df[FEATURE_COLS].values.astype(np.float32)
    y = train_df["Dq/B"].values.astype(np.float32)
    print(f"✅ {X.shape[0]} compounds, {X.shape[1]} features\n")

    # Create study
    sampler = TPESampler(seed=SEED)
    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="dqb_ann_optimization",
    )

    # Optimize
    print("🔍 Starting Bayesian optimization (Optuna TPE)...")
    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    # Results
    print(f"\n{'='*60}")
    print(f"✅ Optimization complete — {len(study.trials)} trials")
    print(f"\n🏆 Best trial:")
    best = study.best_trial
    print(f"   R²    = {best.value:.4f}")
    print(f"   Params:")
    for k, v in best.params.items():
        print(f"     {k:<15} = {v}")

    # Save all trials
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values("value", ascending=False)
    trials_df.to_excel("ann_optuna_results.xlsx", index=False)
    print(f"\n💾 All trials saved → ann_optuna_results.xlsx")

    # Save best params
    with open("ann_best_params.txt", "w") as f:
        f.write("# Best hyperparameters from Optuna TPE search\n")
        f.write(f"# Best CV R² = {best.value:.4f}\n\n")
        for k, v in best.params.items():
            f.write(f"{k} = {v}\n")
    print(f"💾 Best params saved  → ann_best_params.txt")

    # Top 5 summary
    print(f"\n📊 Top 5 configurations:")
    print(f"  {'Trial':>6} {'R²':>8} {'hidden1':>8} {'hidden2':>8} "
          f"{'dropout1':>9} {'dropout2':>9} {'lr':>10} {'wd':>10} {'batch':>6}")
    print(f"  {'─'*75}")
    for _, row in trials_df.head(5).iterrows():
        print(f"  {int(row['number']):>6} "
              f"{row['value']:>8.4f} "
              f"{int(row['params_hidden1']):>8} "
              f"{int(row['params_hidden2']):>8} "
              f"{row['params_dropout1']:>9.3f} "
              f"{row['params_dropout2']:>9.3f} "
              f"{row['params_lr']:>10.2e} "
              f"{row['params_weight_decay']:>10.2e} "
              f"{int(row['params_batch_size']):>6}")

    print(f"\n{'='*60}")
    print(f"Copy best params into dqb_Cr3_ANN.py:")
    print(f"  HIDDEN1      = {best.params['hidden1']}")
    print(f"  HIDDEN2      = {best.params['hidden2']}")
    print(f"  DROPOUT      = ({best.params['dropout1']:.3f}, "
          f"{best.params['dropout2']:.3f})")
    print(f"  LR           = {best.params['lr']:.2e}")
    print(f"  WEIGHT_DECAY = {best.params['weight_decay']:.2e}")
    print(f"  BATCH_SIZE   = {best.params['batch_size']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
