*Read this in other languages: [Español](README_es.md)*

# AI4I 2020 Predictive Maintenance - Advanced Binary Failure Classification

## Project Overview

This project develops an end-to-end Machine Learning pipeline for binary predictive maintenance classification using the AI4I 2020 Predictive Maintenance Dataset.

The core objective is to anticipate anomalous machine behavior and predict general machine failures based on operational and sensor measurements. In industrial environments, the primary business value lies in proactively alerting technicians of an impending failure rather than predicting the exact failure mode, especially when specific failure types are extremely rare.

This solution combines business-driven problem framing, physics-based feature engineering, automated benchmarking of Gradient Boosting algorithms, Optuna hyperparameter optimization and leakage-free decision threshold selection to build a robust classification system.

---

## Results (Unseen Test Set)

Final model: **XGBoost** (selected automatically by the benchmark; the Optuna search did not beat its default configuration on validation, so the defaults were kept), evaluated on a stratified test set of 1,500 machines (51 failures).

| Metric | Value |
|--------|-------|
| Macro F1 (threshold 0.50) | 0.949 |
| Failure-class Recall (threshold 0.50) | 0.90 |
| Failure-class Precision (threshold 0.50) | 0.90 |
| PR-AUC (Average Precision) | 0.937 |
| ROC-AUC | 0.988 |

Business operating points (thresholds selected on out-of-fold data, never on the test set):

| Strategy | Threshold | Failures caught | Failures missed | False alarms |
|----------|-----------|-----------------|-----------------|--------------|
| A - Default | 0.50 | 46 | 5 | 5 |
| B - F1-optimal (high precision) | ~0.87 | 46 | 5 | 1 |
| C - Recall ≥ 90% (risk-averse) | ~0.007 | 49 | 2 | 90 |

Out-of-fold, the recall at the default threshold is 0.78, so the test-set recall of 0.90 should be read as a favorable draw rather than a guaranteed level.

---

## Dataset

**AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository)

| Property | Value |
|-----------|-----------|
| Task | Binary Classification |
| Instances | 10,000 |
| Features | 6 Core Features |
| Type | Multivariate |
| Missing Values | No |
| Failure Rate | 3.39% |

### Features

- Product Type (L, M, H)
- Air Temperature [K]
- Process Temperature [K]
- Rotational Speed [rpm]
- Torque [Nm]
- Tool Wear [min]

---

## Target Definition (Architectural Shift)

Originally, the dataset contains specific failure indicators (Tool Wear Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, Random Failure). Due to the extreme class imbalance of these specific modes, predicting them individually yields poor statistical reliability.

### The Strategic Approach

All minority failure classes have been aggregated into a single macro-class, using the dataset's own `Machine failure` label:

**Machine Failure**
- 0 = Normal Operation
- 1 = Failure

### Statistical Advantage

This groups isolated failure examples into a larger, more statistically significant anomaly class, allowing advanced models to learn patterns safely and effectively.

### Data Quality Findings

The EDA shows that the failure-mode flags and the global label are not perfectly aligned (9 failures without a mode flag, 18 mode flags without a failure, 24 rows with several modes). `Machine failure` is therefore used as the single source of truth for the target.

### Data Leakage Prevention

All specific failure mode flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are strictly dropped from the feature set prior to training.

---

## Feature Engineering

Domain-specific features were engineered to capture the physical failure mechanisms documented for the dataset:

| Feature | Formula | Related failure mode |
|---------|---------|----------------------|
| `temperature_diff` | Process temperature − Air temperature [K] | Heat Dissipation Failure |
| `power_w` | Torque [Nm] × Rotational speed [rpm] × 2π / 60 [W] | Power Failure |
| `wear_torque_strain` | Tool wear [min] × Torque [Nm] | Overstrain Failure |

---

## Data Preprocessing

A modular Scikit-Learn pipeline (fitted only on training data) was implemented containing:

### Numerical Features

- Median Imputation
- Standard Scaling

Outliers are deliberately **not** clipped: extreme torque, power and strain values are the failure signal itself, so clipping them would remove exactly the information the model needs.

### Categorical Features

- Most Frequent Imputation
- Rare Category Encoding (via Custom Transformer)
- One-Hot Encoding

---

## Class Imbalance Handling

Even with a consolidated binary target, failures represent only ~3.4% of operations.

To handle this without introducing synthetic noise, balanced sample weights are computed and passed directly into the model training phase. The **same** weighting is applied consistently in the benchmark, in every Optuna trial, in the final retraining and in the cross-validation used for threshold selection.

---

## Model Benchmarking

The pipeline automatically benchmarks multiple Gradient Boosting architectures:

- Gradient Boosting Classifier (Scikit-Learn)
- HistGradientBoosting Classifier (Scikit-Learn)
- XGBoost
- LightGBM
- CatBoost

Models are ranked by **Macro F1** on the validation set, and **PR-AUC** is reported as a threshold-independent complement. Both are well suited for imbalanced datasets where false negatives (missed failures) have a high business cost.

---

## Hyperparameter Optimization

The highest-performing model from the benchmark is dynamically selected and optimized using Optuna (seeded TPE sampler, 40 trials). Tuned parameters are only kept if they beat the default configuration on validation; in the final run the best trial reached a validation Macro F1 of 0.899 versus 0.908 for the defaults, so the default XGBoost configuration was kept.

The final model is retrained on the combined training and validation sets before being evaluated once on a completely unseen, stratified test set.

---

## Decision Threshold Selection

The decision threshold is treated as a business lever. To avoid leaking information from the test set, the thresholds are selected on **out-of-fold predictions** (5-fold stratified cross-validation on train + validation), and the test set is only used to confirm each strategy.

---

## Evaluation

Model performance and robustness are assessed through:

- F1 Score (Macro and Failure class)
- PR-AUC (Average Precision)
- Precision and Recall Analysis across thresholds
- Comprehensive Classification Report
- Confusion Matrix Heatmaps

---

## Project Structure

```
├── data/
│   └── ai4i2020.csv
├── notebooks/
│   └── AI4I 2020 Predictive Maintenance.ipynb
├── requirements.txt
├── README.md
└── README_es.md
```

## How to Run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Then open `notebooks/AI4I 2020 Predictive Maintenance.ipynb` and run all cells (the dataset is loaded with a path relative to the `notebooks/` folder). A full run takes a few minutes.

---

## License

This is an educational and demonstrative Data Science project based on a publicly available industrial dataset for analytical purposes.

---

## Author

**Armando Guarnera**  
Data Scientist  
Argentina
