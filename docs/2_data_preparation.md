# 2. Data Preparation

**Notebook:** `notebooks/2_data_preparation.ipynb`  
**Input:** `data/interim/player_game_data.csv`  
**Output (cleaned):** `data/interim/player_game_data_cleaned.csv`  
**Output (splits):** `data/processed/train/train.csv`, `data/processed/test/test.csv`

---

## Overview

This notebook cleans the raw interim dataset, engineers two derived features, analyses feature correlations with the target variable, and produces stratified train/test splits ready for modelling.

The pipeline consists of **9 steps**:

| Step | Title |
|------|-------|
| 1 | Import libraries and load dataset |
| 2 | Load and inspect dataset |
| 3 | Handle missing / invalid values in categorical columns |
| 4 | Handle missing / invalid values in numeric columns |
| 5 | *(merged into Step 4 sub-steps)* |
| 6 | Finalise and save cleaned dataset |
| 7 | Feature engineering |
| 8 | Feature correlation analysis |
| 9 | Train / test split and save |

---

## Step 1  Import Libraries and Configure Paths

Key imports: `pandas`, `pathlib`.

Two project-level path constants are set at the start and reused throughout the notebook:

```
INTERIM_DIR   = <project_root>/data/interim/
PROCESSED_DIR = <project_root>/data/processed/
```

The custom `DataLoader` and `DataPrep` utility classes from `src/` are also imported here.

---

## Step 2  Load the Dataset

The interim dataset (`player_game_data.csv`) is loaded via `DataLoader`. It contains **1 296 rows × 20 columns**:

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | int | Unique match identifier |
| `episode` | int | Valorant episode number |
| `act` | int | Act within the episode |
| `rank` | str | Player rank at time of match |
| `date` | str | Match date |
| `agent` | str | Agent played |
| `map` | str | Map played |
| `outcome` | str | Match result (Win / Loss / Draw) |
| `round_wins` | int | Rounds won |
| `round_losses` | int | Rounds lost |
| `kills` | int | Total kills |
| `deaths` | int | Total deaths |
| `assists` | int | Total assists |
| `kdr` | float | Kill / death ratio |
| `avg_dmg_delta` | int | Average damage delta per round |
| `headshot_pct` | int | Headshot percentage |
| `avg_dmg` | int | Average damage per round |
| `acs` | int | Average combat score |
| `num_frag` | int | Number of first / multi-kills |
| `role` | str | Agent role (Duelist / Initiator / Controller / Sentinel) |

---

## Step 3 Handle Missing / Invalid Values in Categorical Columns

### 3A   `rank` column
Missing values in the rank column are filled using the mode (the most frequently occurring rank). This approach is appropriate because rank is an unordered categorical variable, and replacing missing values with the most common category helps maintain the overall distribution of the data.

### 3B    `outcome` column
Rows where the outcome value is either NaN or 'Unknown' are corrected by comparing the values of round_wins and round_losses:

    - If round_wins > round_losses, the outcome is set to 'Win'.
    - If round_wins < round_losses, the outcome is set to 'Loss'.
    - If round_wins == round_losses, the outcome is set to 'Draw'.

### 3C    `role` column
Missing values in the role column are filled using the mode of the column, ensuring that the most frequently occurring role is used to replace missing entries.

### 3D    `agent` column
Rows containing 'UnknownAgent' are replaced with the mode agent corresponding to the same role. The mode is calculated separately for each of the four roles: Duelist, Controller, Sentinel, and Initiator. This method preserves the role–agent relationship, preventing the assignment of an agent that does not logically belong to a specific role.

### 3E    Verification
After performing all data corrections, a summary table is generated again to verify the data quality. The table confirms that missing_count, missing_%, and unknown_count are 0 for all categorical columns, ensuring that the dataset is fully cleaned and ready for further analysis.

---

## Step 4    Handle Missing / Invalid Values in Numeric Columns

The numeric columns considered in this step are round_wins, round_losses, kills, and deaths.

### 4A    Identify Negative Values
A diagnostic analysis is first conducted to detect invalid negative values in the numeric columns. For each column, a report is generated containing negative_count, negative_%, min_value, and max_value prior to any data correction. This step helps quantify the extent of invalid data and guides the subsequent cleaning process.

### 4B    Fix Negative `round_wins` / `round_losses`
Negative values in round_wins and round_losses are corrected using outcome-aware imputation through the function fix_negative_by_outcome. The applied rules are summarized below:

| Situation | Fix applied |
|-----------|-------------|
| Both `round_wins` and `round_losses` negative (any outcome) | Replace both with the outcome-group median |
| `round_wins` negative, outcome = Win or Loss | Replace with the median `round_wins` for that outcome group |
| `round_losses` negative, outcome = Win or Loss | Replace with the median `round_losses` for that outcome group |
| `round_wins` negative, outcome = Draw | Copy from `round_losses` (symmetry rule) |
| `round_losses` negative, outcome = Draw | Copy from `round_wins` (symmetry rule) |

This strategy preserves the logical relationship between the match outcome and the round results.

### 4C    Verify `outcome` vs Round Wins/Losses Consistency
After correcting the round values, the outcome column is recalculated using the updated values of round_wins and round_losses. The recalculated outcome is then compared with the stored value. Any inconsistencies are resolved by replacing the stored value with the recalculated outcome. After this step, the dataset achieves 100% consistency between round results and match outcomes.

### 4D    Fix Negative `kills` / `deaths`
To correct invalid values in kills and deaths, role-level median statistics are first computed, including median_kills, median_deaths, and median_kdr. Negative values are then corrected using role-aware imputation implemented in the function fix_kills_deaths.

| Situation | Fix applied |
|-----------|-------------|
| `kills` negative, `deaths > 0` | `kills = round(median_kdr × deaths)` |
| `kills` negative, `deaths == 0` | `kills = median_kills` for that role |
| `deaths` negative, `kills > 0` | `deaths = round(kills / median_kdr)` |
| `deaths` negative, `kills == 0` | `deaths = median_deaths` for that role |
| Both negative | Both replaced with role median |
| `kills > 0`, `deaths == 0` (special case) | `kdr = kills / 1 = kills` (Valorant convention) |

Finally `kdr` is recalculated as `kills / max(deaths, 1)`. This ensures that division-by-zero errors are avoided and that the resulting metric remains meaningful.

---

## Step 6    Finalise and Save Cleaned Dataset

After completing all data cleaning and validation steps, the fully processed dataframe is saved to the interim data directory:
```
 data/interim/player_game_data_cleaned.csv
```

## Step 7    Feature Engineering

To enrich the dataset and improve its suitability for modelling, two additional derived features are created from the cleaned data.

### `win_loss_margin`
The win–loss margin represents the difference between the number of rounds won and rounds lost:
```
win_loss_margin = round_wins − round_losses
```
This feature captures the degree of match dominance or deficit. A positive value indicates that the player’s team won the match with a certain margin, while a negative value indicates a loss and quantifies the extent of that loss.

### `performance` (target variable)
A new target variable, performance, is created by combining the player's Average Combat Score (ACS) with the match outcome margin. The calculation is defined as follows:

```
combined_score  = acs + (margin_weight × win_loss_margin)
performance     = int(combined_score // bin_width) + 1
```
- `bin_width = 40`, `margin_weight = 10`

This formulation integrates individual player performance (captured through ACS) with team-level match context (represented by the win–loss margin). By combining these factors, the metric reflects both individual contribution and match outcome influence.

Instead of using raw ACS values directly, the combined score is binned into integer categories. This approach reduces the effect of ACS volatility and produces a more stable and interpretable ordinal performance grade suitable for machine learning models.

> **Feature matrix X** contains all 20 original columns plus `win_loss_margin` (21 features total). No label-encoding is applied at this stage    categorical columns remain as strings to preserve interpretability and allow the modelling notebook to choose its own encoding strategy.

---

## Step 8    Feature Correlation Analysis

Since the feature matrix `X` contains both numerical and categorical variables, the correlation analysis is performed only on numeric features. These columns are selected using:

Two visualisations are produced:
1. **Correlation table**    Pearson correlation of each numeric feature with `performance`, styled with a viridis colour gradient.
2. **Scatter + regression plots**    One subplot per numeric feature showing the scatter of values against `performance` with a linear regression line overlay.

**Key findings from the correlation analysis:**
- `acs` and `avg_dmg` show the strongest positive correlation with `performance` (expected, since `acs` is part of the target formula).
- `avg_dmg_delta` is also strongly positively correlated.
- `round_losses` and `deaths` are moderately negatively correlated.
- `win_loss_margin` shows a clear positive linear trend.
- `headshot_pct`, `assists`, and `episode`/`act` show weak or near-zero correlation.

---

## Step 9    Train / Test Split and Save

### Rare-class filtering
Performance grades appearing **fewer than 2 times** are removed before splitting to allow `stratify` to function correctly (scikit-learn requires at least 2 samples per class for stratified splitting).

### Split parameters
| Parameter | Value |
|-----------|-------|
| `test_size` | 0.2 (20 %) |
| `random_state` | 42 |
| `stratify` | `y` (performance grade) |

Stratification ensures that the distribution of performance grades is proportionally preserved in both splits, which is especially important given the class imbalance across performance grades.

### Output shapes
| Split | Rows | Columns |
|-------|------|---------|
| Train | 1 034 | 21 (features) + 1 (performance) |
| Test  | 259  | 21 (features) + 1 (performance) |

### Saved files
```
data/processed/train/train.csv
data/processed/test/test.csv
```

A bar chart comparing the performance grade distribution across train and test confirms that stratification was applied correctly.

---

## Output Column Reference

Both `train.csv` and `test.csv` contain the following 22 columns:

| # | Column | Origin |
|---|--------|--------|
| 1 | `game_id` | Original |
| 2 | `episode` | Original |
| 3 | `act` | Original |
| 4 | `rank` | Original (missing values filled) |
| 5 | `date` | Original |
| 6 | `agent` | Original (UnknownAgent replaced) |
| 7 | `map` | Original |
| 8 | `outcome` | Original (invalid values corrected) |
| 9 | `round_wins` | Original (negatives imputed) |
| 10 | `round_losses` | Original (negatives imputed) |
| 11 | `kills` | Original (negatives imputed) |
| 12 | `deaths` | Original (negatives imputed) |
| 13 | `assists` | Original |
| 14 | `kdr` | Recalculated after fixes |
| 15 | `avg_dmg_delta` | Original |
| 16 | `headshot_pct` | Original |
| 17 | `avg_dmg` | Original |
| 18 | `acs` | Original |
| 19 | `num_frag` | Original |
| 20 | `role` | Original (missing values filled) |
| 21 | `win_loss_margin` | **Derived**    `round_wins − round_losses` |
| 22 | `performance` | **Target**    binned ACS + margin score |

---

## Summary and Modelling Decision

### From Classification to Regression , The Full Story

#### Iteration 1: Predicting Agent Choice (Classification)

The original framing of this project was a **multi-class classification problem**. Given a player's historical match statistics , kills, deaths, KDR, round wins/losses, map, and rank , the goal was to predict which **Agent** the player would select in their next match. A classification model was the natural fit here: a set of K agent labels, one output class per prediction.

#### Why Classification Was Abandoned

During the **Data Understanding phase** (`notebooks/1_data_understanding.ipynb`), a correlation analysis was conducted between all numeric features and the two candidate classification targets: `agent` and `role`. The results were decisive:

- `agent` and `role` showed **near-zero correlation** with every numeric performance feature (kills, deaths, KDR, ACS, avg_dmg, round_wins, round_losses).
- The single column that had any notable correlation with `agent` was `game_id` , a unique row identifier that carries no predictive meaning and must not be used as a feature.

This revealed a fundamental problem: **a player's raw performance statistics do not reliably determine their agent choice**. Agent selection in Valorant is driven by personal preference, team composition negotiation, and meta considerations , none of which are captured in a single-player stat sheet. Training a classification model on such a weak signal would produce a model that essentially learns the marginal frequency of each agent label (Cypher ≈ 57 % of all matches in this dataset) rather than any real relationship.

#### Iteration 2: Predicting Player Performance (Regression)

Given the low-correlation finding, the prediction target was completely reframed. Instead of asking *"which agent will this player pick?"*, we now ask:

> **Given a player's context (rank, map, agent, role, round outcomes), can we predict how that player performed?**

A new target variable , **`performance`** , was engineered in this notebook (Step 7). It combines a player's Average Combat Score (`acs`) with a win/loss round margin weight to produce a single ordinal numeric score:

```
combined_score = acs + (margin_weight × win_loss_margin)
performance    = int(combined_score // bin_width) + 1
```

The cleaned dataset was then split into **train (80 %) / test (20 %)** sets using stratified sampling to preserve the performance grade distribution across both splits.

#### Why Regression and Not Another Model

| Question | Answer |
|---|---|
| **Why not classification?** | The original classification target (`agent`) lacked sufficient correlation with numeric features. The revised target (`performance`) is a continuous ordinal numeric score , naturally suited to regression. |
| **Why not keep classification for performance?** | Binned performance grades are ordinal, not nominal. Regression respects the ordering and magnitude between grades; classification treats each bin as independent and loses that ordering information. |
| **Why not a clustering or unsupervised approach?** | The dataset has clearly defined numeric labels (performance scores). Supervised regression directly optimises the prediction error against known ground truth, which is more appropriate than clustering. |
| **What makes regression viable here?** | The correlation analysis (Step 8) confirms that `acs`, `avg_dmg`, `avg_dmg_delta`, and `win_loss_margin` all show **meaningful, positive linear relationships** with `performance`. The signal is present. |

In short: the **target variable is numeric and ordinal**, the **features show measurable linear correlation** with that target, and the **classification approach was ruled out** by evidence from the data understanding phase. Regression is therefore both the statistically appropriate and data-driven choice for Iteration 2.

#### Next Steps , Modelling

With the data fully cleaned, features engineered, and train/test splits saved, the project moves into the **Modelling phase** (`notebooks/3_modelling.ipynb`). The next notebook will:

1. Encode categorical features and scale numeric features.
2. Train a regression model with `performance` as the target.
3. Analyse feature importances to validate which in-game statistics matter most.
