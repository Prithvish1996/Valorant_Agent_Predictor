# Valorant Player Performance Predictor

A machine learning project that predicts a Valorant player's **performance score** (−1 to 12) from in-game statistics using regression models. Built following the **CRISP-DM** methodology across four well-documented phases.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Setting Up on a New Machine](#3-setting-up-on-a-new-machine)
   - [3.1 Install Anaconda](#31-install-anaconda)
   - [3.2 Clone the Repository](#32-clone-the-repository)
   - [3.3 Create and Activate the Conda Environment](#33-create-and-activate-the-conda-environment)
   - [3.4 Launch JupyterLab](#34-launch-jupyterlab)
4. [How the Project is Organised](#4-how-the-project-is-organised)
5. [Running the Project — Step-by-Step](#5-running-the-project--step-by-step)
6. [Notebook Execution Order](#6-notebook-execution-order)
7. [How the Docs Folder Maps to the Notebooks](#7-how-the-docs-folder-maps-to-the-notebooks)
8. [Acknowledgements](#8-acknowledgements)

---

## 1. Project Overview

| Item | Detail |
|---|---|
| **Goal** | Predict a player's performance score from kills, deaths, damage delta, agent, role, and win/loss margin |
| **Target** | `performance` — an ordinal numeric score ranging from **−1 to 12** |
| **Approach** | Regression (Random Forest, Gradient Boosting, KNN, XGBoost) |
| **Best model** | Random Forest — Test RMSE 1.389, Test R² 0.651 |
| **Methodology** | CRISP-DM (Business Understanding → Data Understanding → Data Preparation → Modelling & Evaluation) |

---

## 2. Repository Structure

```
Valorant_Player_Performance_Predictor/
│
├── data/
│   ├── external/               # Raw source data (do not modify)
│   │   ├── valorant_games.csv          # Original dataset
│   │   ├── valorant_games_messy.csv    # Intentionally dirtied dataset used for cleaning exercises
│   │   └── agent_master_list.csv       # Reference list of all 27 agents and their roles
│   ├── interim/                # Intermediate outputs produced by notebooks
│   │   ├── player_game_data.csv        # After data understanding (merged with agent roles)
│   │   └── player_game_data_cleaned.csv # After data preparation (cleaned + engineered features)
│   └── processed/              # Final ML-ready splits
│       ├── train/train.csv
│       └── test/test.csv
│
├── docs/                       # CRISP-DM phase documentation (read alongside notebooks)
│   ├── 0_buisness_understanding.md
│   ├── 1_data_understanding.md
│   ├── 2_data_preparation.md
│   └── 3_modelling_and_evaluation.md
│
├── models/                     # Saved model artefacts (produced by notebook 3)
│   ├── valorant_performance_predictor.joblib
│   ├── valorant_performance_predictor_meta.json
│   └── valorant_performance_predictor_encoders.joblib
│
├── notebooks/                  # Executable Jupyter notebooks (run in numeric order)
│   ├── 1_data_understanding.ipynb
│   ├── 2_data_preparation.ipynb
│   └── 3_modelling_and_evaluation.ipynb
│
├── src/                        # Reusable Python source modules
│   ├── data/
│   │   ├── data_loader.py              # DataLoader utility class
│   │   └── messy/mess_my_data.py       # Script that generates the messy dataset
│   ├── features/
│   │   └── data_preparation.py         # DataPrep utility class
│   └── utils/
│       ├── anomalies.py
│       ├── data_prep.py
│       ├── data_viz_utils.py
│       ├── display_utils.py
│       └── logger.py
│
├── environment.yml             # Conda environment definition
└── README.md                   # This file
```

---

## 3. Setting Up on a New Machine

### 3.1 Install Anaconda

Anaconda provides Python, conda, and JupyterLab in a single installer.

| OS | Download Link |
|---|---|
| **Windows** | https://www.anaconda.com/download |
| **macOS** | https://www.anaconda.com/download |
| **Linux** | https://www.anaconda.com/download |

> **Tip:** During the Windows installer, tick **"Add Anaconda to my PATH"** if you plan to use the standard Command Prompt / PowerShell. Otherwise use **Anaconda Prompt** for all conda commands.

If you prefer the minimal installer (no GUI), use **Miniconda** instead:  
https://docs.conda.io/en/latest/miniconda.html

Verify the installation worked:

```bash
conda --version
# Expected: conda 24.x.x or similar
```

---

### 3.2 Extract the Project ZIP

1. Locate the ZIP file you received (e.g. `Valorant_Player_Performance_Predictor.zip`).
2. Right-click the file and select **Extract All...** (Windows) or double-click to unzip (macOS/Linux).
3. Choose a destination folder, then open a terminal and navigate into the extracted folder:

```bash
cd path\to\Valorant_Player_Performance_Predictor
```

---

### 3.3 Create and Activate the Conda Environment

The `environment.yml` file at the project root defines every required package (Python 3.10, pandas, scikit-learn, xgboost, matplotlib, jupyterlab, etc.).

```bash
# Create the environment (only needed once)
conda env create -f environment.yml

# Activate it (needed every session)
conda activate valorant_predictor
```

To verify all packages installed correctly:

```bash
conda list
```

If you ever update `environment.yml` and need to sync:

```bash
conda env update -f environment.yml --prune
```

To remove the environment entirely and start fresh:

```bash
conda deactivate
conda env remove -n valorant_predictor
```

---

### 3.4 Open the Project in VS Code (Recommended)

VS Code can run Jupyter notebooks natively and integrates directly with conda environments — no browser required.

**Install VS Code:**  
https://code.visualstudio.com/download

**Install the required extensions** (one-time setup):

1. Open VS Code.
2. Go to the **Extensions** panel (`Ctrl+Shift+X`).
3. Search for and install:
   - **Python** (by Microsoft) — `ms-python.python`
   - **Jupyter** (by Microsoft) — `ms-toolsai.jupyter`

**Open the project folder:**

1. In VS Code go to **File → Open Folder...** and select the extracted `Valorant_Player_Performance_Predictor` folder.

**Select the conda environment as the Python interpreter:**

1. Open any `.ipynb` notebook from the `notebooks/` folder.
2. Click the kernel selector in the top-right corner of the notebook (it may say "Select Kernel" or show a Python version).
3. Choose **Python Environments...** → select **`valorant_predictor`** (the conda env created in Step 3.3).

> If `valorant_predictor` does not appear, make sure you have run `conda env create -f environment.yml` first, then reload VS Code.

From this point, run any notebook cell with `Shift+Enter`, or run the entire notebook with **Run All** from the toolbar.

---

### 3.5 Launch JupyterLab (alternative to VS Code)

If you prefer a browser-based interface, with the environment activated run:

```bash
jupyter lab
```

Your browser will open at `http://localhost:8888`. Navigate to the `notebooks/` folder in the left-hand file browser to start working.

---

## 4. How the Project is Organised

This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework. Each phase has:

- A **documentation file** in `docs/` — explains the *why* and *what* behind every decision.
- A **notebook** in `notebooks/` — the executable *how*, with code and outputs.

The two always go together. Read the doc first to understand the goal of each phase, then run the notebook to execute it.

---

## 5. Running the Project — Step-by-Step

Follow these steps in order on a freshly cloned repository. Each step depends on the outputs of the previous one.

### Step 1 — Read the Business Understanding document

**File:** `docs/0_buisness_understanding.md`  
**No notebook for this phase.**

Read this document first. It explains:
- What Valorant is and why this prediction task is useful.
- The dataset source (Kaggle — 1,000 match records).
- Why a **messy version** of the dataset (`valorant_games_messy.csv`) was intentionally created to simulate real-world data quality problems.
- The Agent Master List and how it is used.

> This phase has no notebook. It is purely background context and project framing.

---

### Step 2 — Run Notebook 1: Data Understanding

**Doc:** `docs/1_data_understanding.md`  
**Notebook:** `notebooks/1_data_understanding.ipynb`  
**Input:** `data/external/valorant_games_messy.csv` + `data/external/agent_master_list.csv`  
**Output:** `data/interim/player_game_data.csv`

Open the notebook and run all cells top-to-bottom (`Run → Run All Cells`). This notebook:

1. Loads the raw messy dataset and the agent master list.
2. Merges them on agent name to add a `role` column to each match record.
3. Profiles data quality — missing values, invalid categories, impossible numeric values, inconsistent outcomes.
4. Produces early visualisations (agent distribution, role imbalance, numeric feature histograms).
5. Saves the merged (but not yet cleaned) data to `data/interim/player_game_data.csv`.

> Read `docs/1_data_understanding.md` alongside the notebook — it explains every quality issue found and why it matters for the next phase.

---

### Step 3 — Run Notebook 2: Data Preparation

**Doc:** `docs/2_data_preparation.md`  
**Notebook:** `notebooks/2_data_preparation.ipynb`  
**Input:** `data/interim/player_game_data.csv`  
**Output:** `data/interim/player_game_data_cleaned.csv`, `data/processed/train/train.csv`, `data/processed/test/test.csv`

Run all cells top-to-bottom. This notebook:

1. Loads the interim dataset produced by Notebook 1.
2. Fixes missing and invalid values in categorical columns (`agent`, `role`, `outcome`).
3. Fixes impossible numeric values (negative kills, deaths, etc.).
4. Saves the cleaned dataset to `data/interim/player_game_data_cleaned.csv`.
5. Engineers two new features: `win_loss_margin` and `performance` (the model target).
6. Analyses feature correlations with `performance`.
7. Creates stratified train/test splits and saves them to `data/processed/`.

> Read `docs/2_data_preparation.md` for a detailed explanation of every cleaning decision and how each engineered feature is derived.

---

### Step 4 — Run Notebook 3: Modelling and Evaluation

**Doc:** `docs/3_modelling_and_evaluation.md`  
**Notebook:** `notebooks/3_modelling_and_evaluation.ipynb`  
**Input:** `data/processed/train/train.csv`, `data/processed/test/test.csv`  
**Output:** `models/valorant_performance_predictor.joblib`, `models/valorant_performance_predictor_meta.json`, `models/valorant_performance_predictor_encoders.joblib`

Run all cells top-to-bottom. This notebook:

1. Loads the training split and selects six features.
2. Label-encodes categorical columns (`agent`, `role`).
3. Trains four regressors: Random Forest, Gradient Boosting, KNN, XGBoost.
4. Demonstrates why accuracy is a poor metric for this task.
5. Evaluates all models on the test split (MAE, RMSE, R²).
6. Selects the best model via a multi-metric rank-sum.
7. Produces three evaluation plots (residual distribution, sorted actual vs predicted, model comparison bar chart).
8. Saves the best model, its metadata, and the fitted label encoders to `models/`.

> Read `docs/3_modelling_and_evaluation.md` for a full explanation of metric choice, model selection logic, and what each evaluation plot shows.

---

## 6. Notebook Execution Order

```
notebooks/1_data_understanding.ipynb
        ↓
notebooks/2_data_preparation.ipynb
        ↓
notebooks/3_modelling_and_evaluation.ipynb
```

> **Important:** Each notebook reads files produced by the previous one. Running them out of order will cause file-not-found errors. Always run them in sequence on a fresh clone.

---

## 7. How the Docs Folder Maps to the Notebooks

| Doc file | Notebook | CRISP-DM phase | Purpose |
|---|---|---|---|
| `docs/0_buisness_understanding.md` | *(none)* | Business Understanding | Project background, dataset source, problem framing |
| `docs/1_data_understanding.md` | `notebooks/1_data_understanding.ipynb` | Data Understanding | Data quality audit, feature overview, early patterns |
| `docs/2_data_preparation.md` | `notebooks/2_data_preparation.ipynb` | Data Preparation | Cleaning steps, feature engineering, train/test split |
| `docs/3_modelling_and_evaluation.md` | `notebooks/3_modelling_and_evaluation.ipynb` | Modelling & Evaluation | Model training, metric rationale, evaluation plots, export |

**Recommended reading order for each phase:**

1. Read the **doc** first to understand the goal and decisions.
2. Open the **notebook** and run it cell by cell, referring back to the doc for context on each step.

The doc files follow the same step numbering as the notebook cells — e.g., "Step 3" in `docs/2_data_preparation.md` corresponds directly to the "Step 3" section in `notebooks/2_data_preparation.ipynb`.

---

## 8. Acknowledgements

| Person / Tool | Role / Contribution |
|---|---|
| **Pieter Zielstra** | Mentor and Study Coach — guidance throughout the project lifecycle |
| **Deepak Tunuguntla** | Mentor and Study Coach — feedback on methodology and implementation |
| **Saxion University of Applied Sciences** | Academic institution — project context and CRISP-DM framework |
| **GitHub Copilot** | AI coding assistant — code generation, refactoring, and documentation writing |
| **ChatGPT (OpenAI)** | Research assistant — ideas, conceptual explanations, and background research material |
| **VS Code** | Primary development environment — notebook editing, debugging, and source code management |
| **GitHub** | Version control and project hosting |
| **Anaconda** | Python distribution and package/environment management (`conda`) |
| **scikit-learn** | Machine learning library — model training and evaluation utilities |
| **XGBoost** | Gradient boosting library — XGBRegressor model |
| **Kaggle** | Dataset source — original 1,000-match Valorant game statistics dataset |
| **Riot Games / Valorant** | Game and domain — official agent roster used to build the Agent Master List |
