# 3. Modelling & Evaluation

## Overview

This document describes the modelling pipeline built to predict a Valorant player's **performance score** (an ordinal numeric value ranging from −1 to 12) from in-game statistics. Because the target is a continuous ordinal score rather than a discrete class label, all models are treated as **regressors**.

The notebook [`notebooks/3_modelling_and_evaluation.ipynb`](../notebooks/3_modelling_and_evaluation.ipynb) is split into **8 steps** that cover data loading, feature engineering, model training, evaluation, visualisation, and export. Each step is described in detail below.

---

## Step 1 Import Libraries and Load Training Data

**What we did:**  
Imported all required libraries and loaded the pre-processed training split from `data/processed/train/train.csv` into a pandas DataFrame.

**Libraries imported:**

| Library | Purpose |
|---|---|
| `pandas` | DataFrame operations and CSV loading |
| `numpy` | Array arithmetic and metric calculations |
| `matplotlib.pyplot` | Evaluation plots |
| `sklearn.preprocessing.LabelEncoder` | Encoding categorical columns |
| `sklearn.ensemble.RandomForestRegressor` | Random Forest model |
| `sklearn.ensemble.GradientBoostingRegressor` | Gradient Boosting model |
| `sklearn.neighbors.KNeighborsRegressor` | K-Nearest Neighbours model |
| `sklearn.metrics` | MAE, RMSE, R² computation |
| `xgboost.XGBRegressor` | XGBoost model |
| `scipy.stats` | Statistical utilities |

**Key variables set:**  
- `PROJECT_ROOT` — resolved workspace root via `pathlib`  
- `PROCESSED_DIR` — path to `data/processed/`  
- `train_df` — loaded training DataFrame (shape printed as a sanity check)

---

## Step 2 Feature Selection and Label Encoding

**What we did:**  
Selected the six input features and the target column, then label-encoded the two categorical columns so all models receive a fully numeric input matrix.

**Features used:**

| Column | Type | Role |
|---|---|---|
| `agent` | Categorical; label-encoded integer | Feature |
| `role` | Categorical; label-encoded integer | Feature |
| `avg_dmg_delta` | Numeric | Feature |
| `win_loss_margin` | Numeric | Feature |
| `kills` | Numeric | Feature |
| `deaths` | Numeric | Feature |
| `performance` | Numeric (-1 to 12) | **Target** |

`sklearn.preprocessing.LabelEncoder` was fitted separately on `agent` and `role`. The fitted encoder objects are stored in a `label_encoders` dictionary so that **identical mappings** can be applied to the test set and persisted with the final model , preventing category-integer mismatch at testing time.

---

## Step 3 Train Regression Models

**What we did:**  
Instantiated and trained four regression algorithms on the full training set (`X`, `y`). After fitting, each model's training-set MAE, RMSE, and R² were printed as a quick sanity check.

**Rationale for regression:**  
`performance` is an ordinal numeric score, not a class label. Treating it as a regression target allows the model to predict any value on the continuous range and enables RMSE/MAE as meaningful error measures.

**Models and hyperparameters:**

| Model | Key Hyperparameters |
|---|---|
| **Random Forest** | `n_estimators=200`, `max_depth=15`, `min_samples_leaf=4`, `min_samples_split=8`, `random_state=42` |
| **Gradient Boosting** | `n_estimators=200`, `learning_rate=0.1`, `max_depth=3`, `subsample=0.8`, `random_state=42` |
| **K-Nearest Neighbors** | `n_neighbors=15`, `metric='euclidean'` |
| **XGBoost** | `n_estimators=200`, `learning_rate=0.1`, `max_depth=4`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42` |

All four models are stored in a `results` dictionary alongside their training predictions and training metrics for easy access in later steps.

---

## Step 4 Accuracy Check: Why Accuracy Alone Falls Short

**What we did:**  
Demonstrated the limitation of using discrete accuracy as a metric for this task by computing both *exact accuracy* and *±1 tolerance accuracy* on the training and test sets for all four models.

### Step 4A Load and Encode the Test Set

`data/processed/test/test.csv` was loaded and the same `label_encoders` fitted in Step 2 were used to transform `agent` and `role`, ensuring test encoding is consistent with training encoding.

### Step 4B Compute Exact and ±1 Tolerance Accuracy

Predictions were rounded to the nearest integer and two accuracy variants were recorded:

| Metric | Definition |
|---|---|
| **Exact accuracy** | % of samples where `round(prediction) == actual` |
| **±1 tolerance** | % of samples where `|round(prediction) − actual| ≤ 1` |

**Why accuracy is misleading here:**  
The performance score is a continuous ordinal value. A prediction of 4 against an actual of 5 counts as *wrong* to the same degree as a prediction of 0 against an actual of 12, even though the former is nearly perfect. RMSE captures the magnitude of near-misses and is therefore the more honest metric which motivates Step 5.

Reference : https://www.datacamp.com/tutorial/rmse

---

## Step 5 Evaluate on Test Set (RMSE / MAE / R²)

**What we did:**  
Evaluated all four models on the held-out test split using three regression metrics: MAE, RMSE, and R².

**Metrics explained:**

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|\hat{y}_i - y_i|$$

MAE (Mean Absolute Error) means the average difference between the model’s predicted value and the actual value.

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2}$$

RMSE (Root Mean Squared Error) gives more importance to large prediction mistakes because it squares the errors before averaging them. This means that bigger errors have a stronger impact on the final value. For example, one prediction that is 4 points wrong affects the score more than four predictions that are 1 point wrong each, making RMSE very sensitive to large errors or outliers.

$$R^2 = 1 - \frac{\sum(\hat{y}_i - y_i)^2}{\sum(\bar{y} - y_i)^2}$$

R² (R-squared) shows how well the model explains the variation in the target values compared to simply predicting the average value every time. A value of 1.0 means the model predicts perfectly, while a value close to 0 means the model is not much better than predicting the average, and negative values mean the model performs worse than just using the mean prediction.

**RMSE is the primary selection criterion** because large errors (predicting an excellent performance when the player actually had a poor game) carry greater practical cost than many small near-misses. Reference : https://www.datacamp.com/tutorial/rmse

**Results:**

| Model | Train MAE | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| **Random Forest** ✓ | 0.549 | **0.891** | **1.389** | **0.651** |
| Gradient Boosting | 0.521 | 0.956 | 1.474 | 0.607 |
| K-Nearest Neighbors | 0.819 | 0.926 | 1.391 | 0.650 |
| XGBoost | 0.404 | 0.941 | 1.466 | 0.611 |

> A Test RMSE of ~1.4 on a scale of -1 to 12 means the model is typically within 1-2 score bands, practically acceptable for a performance prediction task.

---

## Step 6 Select the Best Model

**What we did:**  
Ranked all four models across three metrics (RMSE, MAE, R²) independently and summed the ranks. The model with the **lowest total rank** was selected as best, avoiding over-reliance on any single metric.

**Ranking logic:**

| Rank column | Direction | Why |
|---|---|---|
| `rank_rmse` | 1 = lowest RMSE → best | Lower error is better |
| `rank_mae` | 1 = lowest MAE → best | Lower error is better |
| `rank_r2` | 1 = highest R² → best | Higher explained variance is better |

`rank_total = rank_rmse + rank_mae + rank_r2`

**Outcome:** **Random Forest** achieved the lowest `rank_total` and was selected as the best model. Its name, colour, and result dictionary are used in all subsequent steps.

A `COLORS` dictionary maps each model name to a consistent hex colour used across all evaluation plots in Step 7.

---

## Step 7 Regression Evaluation Plots

**What we did:**  
Produced three sets of side-by-side plots comparing all four models visually. The best model's subplot is **bold-titled** and marked with `✓ best`.

### Step 7A Residual Distribution

A histogram of `actual vs predicted` residuals for each model. Two vertical lines are overlaid:
- **Dashed black line** at zero, the ideal centre for unbiased predictions.
- **Solid red line** at the mean residual, reveals systematic over- or under-prediction.

A narrow, symmetric histogram centred at zero indicates the model predicts without consistent bias. The subplot x-axis label also shows MAE and RMSE for quick reference.

### Step 7B Sorted Actual vs Predicted

Both the actual performance scores and predicted scores are **independently sorted** and overlaid as line charts. The shaded region between them (the error band) gives an intuitive picture of how closely the model tracks the full range of scores from lowest to highest. A narrow band throughout indicates well-calibrated predictions across all performance levels.

The subplot x-axis label shows R² and RMSE per model.

### Step 7C Model Comparison Bar Chart

Side-by-side bar charts for **Test MAE**, **Test RMSE**, and **Test R²** across all four models. Bar values are annotated directly. The best model's bar is outlined in bold black for instant identification. Axis subtitles clarify the direction of improvement (`↓ lower is better` / `↑ higher is better`).

---

## Step 8 Export the Best Model

**What we did:**  
Saved three artefacts to the `models/` directory so the trained model can be loaded and used for inference without re-running the notebook.

| Artefact | Path | Contents |
|---|---|---|
| Model | `models/valorant_performance_predictor.joblib` | Fitted sklearn/XGBoost estimator (binary) |
| Metadata | `models/valorant_performance_predictor_meta.json` | Model name, feature list, categorical columns, target name, and test metrics (MAE, RMSE, R²) |
| Encoders | `models/valorant_performance_predictor_encoders.joblib` | Fitted `LabelEncoder` instances for `agent` and `role` |


**Example metadata structure:**
```json
{
  "model_name": "Random Forest",
  "features": ["agent", "role", "avg_dmg_delta", "win_loss_margin", "kills", "deaths"],
  "cat_cols": ["agent", "role"],
  "target": "performance",
  "test_mae": 0.891,
  "test_rmse": 1.389,
  "test_r2": 0.651
}
```

---

## Summary of Key Findings

- **Random Forest was selected as the best model** with Test RMSE **1.389**, Test MAE **0.891**, and Test R² **0.651**,  explaining ~65 % of variance in player performance on unseen data.
- **Random Forest and KNN were almost tied on RMSE** (1.389 vs 1.391), but Random Forest also led on MAE and R², making it the clear overall winner.
- **XGBoost showed the lowest Train MAE (0.404)** but a higher Test RMSE (1.466), suggesting mild overfitting compared to Random Forest.
- **Gradient Boosting had the weakest test generalisation** (RMSE 1.474, R² 0.607) despite a competitive Train MAE (0.521).
- **KNN was surprisingly competitive** on the test set despite being the only non-ensemble, non-gradient method — though its higher Train MAE (0.819) indicates it underfits relative to tree-based models.
- Using a **multi-metric rank-sum** for model selection (Step 6) avoids the pitfall of optimising for one number at the expense of others.

---

## Research Backing

The choice of RMSE as the primary metric for a continuous ordinal target is well-established:

- **Chai & Draxler (2014)**  *Root mean square error (RMSE) or mean absolute error (MAE)?* Geoscientific Model Development, 7, 1247–1250. Argues RMSE is appropriate when large errors carry greater practical cost than small errors.
- **scikit-learn documentation  Metrics and scoring** Lists `mean_squared_error`, `mean_absolute_error`, and `r2_score` as canonical regression metrics; accuracy does not appear under the regression section.

---


