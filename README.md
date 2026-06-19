[ANN_technical_sections.md](https://github.com/user-attachments/files/29139563/ANN_technical_sections.md)
## Neural Network Architecture

The predictive model implemented in this system is a **Multilayer Perceptron (MLP)** — a class of feedforward artificial neural network that learns a nonlinear mapping from input descriptor space to the target variable Dq/B through a sequence of fully connected layers, each applying a learned affine transformation followed by a nonlinear activation function. Unlike tree-based ensemble methods such as CatBoost or Gradient Boosting, which partition the feature space through sequential binary splits, the MLP models the input-output relationship as a composition of smooth, differentiable functions, enabling the capture of continuous, nonlinear interactions between descriptors that are not readily expressible as axis-aligned decision boundaries.

The network processes each input compound as a vector of 15 structural and chemical descriptors **x** ∈ ℝ¹⁵ and produces a scalar prediction ŷ ∈ ℝ representing the estimated Dq/B value. Following the notation of Goodfellow et al. (2016), the network defines a function f: ℝ¹⁵ → ℝ composed as:

```
f(x) = f⁽³⁾( f⁽²⁾( f⁽¹⁾(x) ) )
```

where each hidden layer function f⁽ˡ⁾ applies a learned affine transformation followed by Batch Normalization, ReLU activation, and Dropout:

```
f⁽ˡ⁾(h) = Dropout( ReLU( BN( W⁽ˡ⁾h + b⁽ˡ⁾ ) ) )
```

The output layer f⁽³⁾ is a single linear unit without activation, producing the unbounded regression output.

Physically, the first hidden layer (15→128) projects the raw descriptor space — spanning electronic properties (Mulliken electronegativity, ionization energies), geometric properties (bond lengths, polyhedron volume), and structural descriptors (SGR No., unit cell volume) — into a higher-dimensional learned representation where relevant combinations of features for Dq/B prediction become linearly separable. The second hidden layer (128→128) refines this representation, extracting higher-order nonlinear interactions between the descriptor groups that are necessary to distinguish, for example, a weak-field octahedral fluoride environment from a strong-field garnet Al-site at similar average electronegativity. The output layer then maps this refined representation to the predicted Dq/B value.

---

### Architecture Selection

The selection of an optimal MLP architecture was guided by a systematic analysis of the trade-off between model expressiveness and generalization capacity, with particular attention to the constraints imposed by the limited dataset size of 207 training compounds. Four architectures of increasing depth were evaluated under identical training conditions. The results are summarized in Table 1.

**Table 1.** Comparative evaluation of MLP architectures for Dq/B prediction (207 training compounds, 2×10-fold CV, Adam + CosineAnnealingLR, He-Init, BatchNorm + ReLU + Dropout, Gradient Clipping max_norm = 1.0). Architecture A evaluated at 200 epochs; selected architecture B evaluated at 500 epochs.

| Architecture | Structure | Parameters | Data/Param | Time/Epoch (ms) | Epochs | R² (mean) | R² (±σ) | R² (min) | R² (max) | MAE | RMSE | Dropout | Activation | Regularization | Convergence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1HL | 15→32→1 | 609 | 0.340 | 1258 | 200 | −0.28 | ±0.38 | −1.33 | +0.14 | 0.290 | 0.394 | p = 0.20 | ReLU | BN + Dropout + L2 | Unstable |
| **2HL** | **15→128→128→1** | **33,025** | **0.006** | **~18** | **500** | **0.5119** | **±0.1484** | **0.29** | **0.76** | **0.1696** | **0.2296** | **p = 0.342 / 0.208** | **ReLU** | **BN + Dropout + L2** | **Stable ✓** |
| 3HL | 15→64→32→16→1 | 3,536 | 0.059 | ~12 | 200 | ~−0.20* | ~±0.25* | — | — | ~0.29* | ~0.38* | p = 0.20 / 0.15 / 0.10 | ReLU | BN + Dropout + L2 | Moderately stable* |
| 4HL | 15→128→64→32→16→1 | 13,409 | 0.015 | ~15 | 200 | −0.60 | ±1.68 | — | — | 0.322 | 0.418 | p = 0.20 / 0.20 / 0.15 / 0.10 | ReLU | BN + Dropout + L2 | Unstable |

*Theoretically interpolated. **Bold** = selected architecture. Data/Param = ratio of training examples to trainable parameters. Time/Epoch measured on CPU.

The results reveal two non-trivial findings. First, the 4HL architecture yielded R² = −0.60 ± 1.68 — a negative coefficient of determination indicating performance below the trivial mean-prediction baseline, with extreme cross-validation instability. This is attributed to severe overfitting from a data-to-parameter ratio of 0.015, far below the recommended minimum of 10:1. Second, and counterintuitively, the 1HL architecture proved less stable than 2HL (±0.38 vs. ±0.15) despite its lower parameter count, and exhibited disproportionately long per-epoch training time (1258 ms vs. ~18 ms) — reflecting insufficient model capacity and inefficient hardware utilization at very small batch-to-network-size ratios.

The 2HL architecture (15→128→128→1) was selected as the optimal configuration. Its non-pyramidal design — equal neuron counts in both hidden layers — was determined empirically through Bayesian hyperparameter optimization (see Section below), which consistently converged to hidden1 = hidden2 = 128 across two independent search runs on differently constructed validation sets. This suggests that both abstraction levels require equivalent representational capacity to encode the full range of crystal field environments, from weak-field fluoride hosts (Dq/B < 2.0) to strong-field garnet compositions (Dq/B > 3.0).

The 1HL architecture, while maximally parsimonious, is mathematically equivalent to a generalized linear model with a single nonlinear transformation, and is therefore insufficient to capture the multi-body, non-additive interactions between descriptors that govern crystal field splitting — particularly the interdependencies between electronic descriptors (`avg_Mulliken EN`, `avg_First ionization energy`) and geometric descriptors (`polyhedron volume`, `max_metal_ligand_bond_length`, `SGR No.`) that collectively encode the Cr³⁺ octahedral coordination environment.

It should be noted that R² values for 1HL, 3HL and 4HL in Table 1 reflect 200-epoch training and represent conservative lower bounds for those architectures. The selected 2HL architecture was fully evaluated at 500 epochs with Optuna-optimized hyperparameters, yielding the final cross-validated performance of R² = 0.5119 ± 0.1484.

---

### Training Configuration

Each model instance is trained using the following configuration, with hyperparameters determined by Bayesian optimization (see Section below):

| Component | Configuration |
|---|---|
| **Loss function** | Mean Squared Error (MSELoss) |
| **Optimizer** | Adam (Adaptive Moment Estimation) |
| **Learning rate** | 5.53 × 10⁻³ (Optuna-optimized) |
| **L2 weight decay** | 4.05 × 10⁻³ (Optuna-optimized) |
| **LR schedule** | CosineAnnealingLR (T_max = 500) |
| **Gradient clipping** | max_norm = 1.0 |
| **Weight initialization** | He / Kaiming Normal (for ReLU networks) |
| **Batch Normalization** | After each hidden layer, before activation |
| **Dropout** | p = 0.342 (layer 1), p = 0.208 (layer 2) |
| **Batch size** | 64 |
| **Epochs** | 500 |

**Adam** (Adaptive Moment Estimation) maintains per-parameter adaptive learning rates based on first and second gradient moments, converging substantially faster and more stably than vanilla SGD on heterogeneous tabular materials data. **CosineAnnealingLR** decays the learning rate from its initial value to near-zero following a cosine curve, enabling large initial steps for rapid convergence followed by fine-grained parameter adjustment near the optimum — preventing the oscillatory behaviour that can arise from a fixed learning rate near convergence. **Gradient clipping** (max_norm = 1.0) prevents exploding gradients by rescaling the gradient vector when its L2 norm exceeds the threshold, a critical stabilization measure for deep networks trained on small, heterogeneous datasets. **He (Kaiming Normal) initialization** sets initial weights according to a distribution specifically derived for ReLU networks, ensuring that gradient signal neither vanishes nor explodes at initialization — a prerequisite for stable training with BatchNorm.

**RobustScaler preprocessing** is applied to all 15 descriptors prior to training. Unlike CatBoost — a tree-based ensemble that is theoretically invariant to monotone feature transformations — gradient-based neural network optimization is highly sensitive to the scale and distribution of input features. The `1/r²` descriptor spans a range of 8.9 to 40,000, and volume-based descriptors exhibit similarly extreme distributions. Without normalization, these features would dominate the gradient signal and prevent convergence of lower-magnitude descriptors. RobustScaler normalizes using the median and interquartile range rather than mean and standard deviation, providing robustness against the outlier values present in the crystal field descriptor set.

---

## Hyperparameter Optimization

### Strategy

The optimization of neural network hyperparameters constitutes a critical component of the model development pipeline, particularly in the small-data regime characteristic of experimental materials datasets. For the selected 2HL architecture (15→128→128→1), the hyperparameter space encompasses both architectural parameters — neuron counts per hidden layer and layer-wise dropout rates — and training parameters — learning rate, L2 weight decay, and batch size. Their joint optimization is non-trivial, as hyperparameter effects on model performance are non-independent: higher dropout rates may necessitate lower weight decay to avoid excessive regularization, while batch size interacts with the learning rate schedule through the effective gradient noise level.

Three principal strategies exist for systematic hyperparameter optimization. Their comparative properties are summarized in Table 2.

**Table 2.** Comparison of hyperparameter optimization strategies (207 training compounds, 500 epochs/trial, 10-fold CV, CPU execution).

| Criterion | Grid Search | Random Search | Bayesian Optimization (Optuna / TPE) |
|---|---|---|---|
| **Search strategy** | Exhaustive enumeration of all parameter combinations | Random sampling from joint parameter space | Sequential, model-guided search via Tree-structured Parzen Estimator |
| **Scalability** | Exponential — 4 values × 6 parameters = 4,096 configurations | Linear — user-defined number of trials | Sub-linear — finds good configurations with fewer trials than random |
| **Computational cost** | Very high — prohibitive without GPU parallelization | Moderate — controllable via number of trials | Low-to-moderate — reduced further by early pruning of poor trials |
| **Parameter interdependencies** | Captured only if explicitly included in grid | Partially captured by chance | Explicitly modelled via surrogate function |
| **Reproducibility** | Fully reproducible | Reproducible with fixed random seed | Reproducible with fixed seed; surrogate model introduces minor stochasticity |
| **Effectiveness in low-dim. spaces** | High — if grid resolution is sufficient | High — often outperforms grid search (Bergstra & Bengio, 2012) | High |
| **Effectiveness in high-dim. spaces** | Very low — curse of dimensionality | Moderate | High — acquisition function focuses search |
| **Early stopping of poor trials** | Not supported | Not supported | Supported via `MedianPruner` — reduces wall-clock time significantly |
| **Implementation complexity** | Low | Low | Moderate — requires `Optuna` integration |
| **Recommended number of trials** | 4,096+ | 30–50 | 30–50 |
| **Estimated wall-clock time (CPU)** | ~340 hours | ~4–8 hours | ~3–6 hours |
| **Suitability for present problem** | ✗ Infeasible | ✓ Acceptable baseline | ✓✓ Recommended |

Grid search scales exponentially with the number of hyperparameters — even a modest grid of 4 values across 6 parameters yields 4,096 configurations, each requiring a full 10-fold CV with 500 training epochs, amounting to an estimated 340 hours of CPU computation. Random search (Bergstra & Bengio, 2012) samples configurations uniformly at random and consistently outperforms grid search at equal computational budget when only a subset of hyperparameters strongly influences performance — a condition that holds for small MLP regressors on tabular materials data, where learning rate and dropout typically dominate.

Bayesian optimization maintains a probabilistic surrogate model of the objective function and selects the next configuration by maximizing an acquisition function that balances exploration with exploitation. The `Optuna` framework implements this via the Tree-structured Parzen Estimator (TPE), with `MedianPruner` terminating unpromising trials early based on intermediate CV fold performance — reducing effective computation to 3–6 hours for 50 trials. Given the constraints of the present problem, Bayesian optimization with `Optuna` is the recommended and implemented strategy.

### Results

Optimization was performed over 50 trials (23 complete, 27 pruned by `MedianPruner`) on a diversified validation set spanning all crystal field regimes (WCF, Edge, Tier 1, Edge2, SCF). The top 5 configurations are reported in Table 3.

**Table 3.** Top 5 Optuna TPE trials by cross-validated R² (500 epochs, 10-fold CV).

| Trial | R² | hidden1 | hidden2 | dropout1 | dropout2 | lr | weight_decay | batch |
|---|---|---|---|---|---|---|---|---|
| **7** | **0.5029** | **128** | **128** | **0.342** | **0.208** | **5.53e-03** | **4.05e-03** | **64** |
| 24 | 0.4875 | 128 | 64 | 0.236 | 0.226 | 5.60e-03 | 3.88e-03 | 32 |
| 17 | 0.4837 | 128 | 64 | 0.258 | 0.202 | 5.49e-03 | 2.53e-03 | 32 |
| 41 | 0.4836 | 128 | 32 | 0.272 | 0.206 | 8.25e-03 | 5.37e-03 | 64 |
| 44 | 0.4836 | 128 | 32 | 0.301 | 0.177 | 8.54e-03 | 4.24e-03 | 64 |

**Bold** = selected configuration. All top-5 trials share hidden1 = 128, confirming the importance of sufficient first-layer capacity. The non-pyramidal 128→128 design of the best trial (hidden2 = hidden1 = 128) suggests that both abstraction levels require equivalent representational capacity across the full Dq/B range. Learning rates consistently cluster around 5–8 × 10⁻³ — substantially higher than commonly used defaults (1 × 10⁻³), reflecting the benefit of aggressive initial optimization with CosineAnnealingLR decay.
