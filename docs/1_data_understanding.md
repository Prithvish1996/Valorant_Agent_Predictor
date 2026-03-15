# Data Understanding (CRISP-DM Phase 2)

## Overview

This document summarizes the Data Understanding phase of the Valorant Player Performance Predictor project. In this step of CRISP-DM, we answer three simple questions:

- What data do we have?
- How good is it?
- What early patterns might help us predict agent choice?

Our focus is to stay practical: describe the data clearly, highlight important issues, and extract a few key insights that will guide the next phases.

---

## 1. Data Sources

We work with two datasets that capture complementary views of Valorant gameplay:

- **Player Game Dataset** (`valorant_games_messy.csv`): 1,296 single-player match records. Each row describes one player's performance in a single match, with metrics (kills, deaths, assists, KDR, avg_dmg, ACS, round_wins, round_losses) and context (agent, map, rank, match_date, outcome). Shape before enrichment: (1296, 19).
- **Agent Master List** (`agent_master_list.csv`): reference table of 27 agents and their roles (Duelist, Initiator, Controller, Sentinel).

These are merged into a single **player_game_data** dataset (`data/interim/player_game_data.csv`) by joining on agent name. The merged dataset has 1,296 rows and 20 columns and adds a `role` feature to each player-game record.


---

## 2. Feature Overview

The merged dataset can be thought of in three groups:

- **Performance metrics (numeric):** kills, deaths, assists, kdr, avg_dmg, acs, round_wins, round_losses.
- **Contextual variables (categorical):** agent, map, rank, role, match_outcome.
- **Temporal variable:** match_date.

This simple classification helps later: numeric features will be scaled and checked for outliers; categorical features will be encoded; dates can be turned into month/season or used for time-based splits.

---

## 3. Data Exploration

### 3.1 Quality issues

We observe three main types of problems:

- **Missing values**
	- rank: 14 records (1.08%)
	- outcome: 102 records with NaN + 100 records with `'Unknown'` value (total: 202 records, ~15.6%)
	- role: 26 records (2.01%)
	- agent: some records contain `'UnknownAgent'` value

- **Impossible or suspicious numeric values**
	- Negative values in kills (38), deaths (34), round_wins (32), round_losses (38), and kdr (44).

- **Inconsistent match outcomes**
	- 289 records (≈22.3%) with Draw, `'Unknown'`, or missing (NaN) outcome values.
	- Note: `'Unknown'` is stored as a string value, not as NaN, requiring explicit replacement during data preparation.

For now, we keep all rows but mark these issues as priorities for the Data Preparation phase.

### 3.2 Categorical distributions

Agent usage is highly skewed:

- Cypher: 747 matches (~57%)
- Killjoy: 258
- Long tail of other agents (e.g., Omen 38, Breach 36, Viper 35)

Maps are more balanced (top maps: Ascent 182, Lotus 175, Bind 165, Haven 140, Split 122), providing enough variety to analyze map–agent relationships.

Rank is concentrated in higher tiers (Ascendant 1, Diamond 3, Diamond 1, Platinum 3, Gold 3), so most rows reflect a single experienced player in each match.

Roles are very imbalanced:

- Sentinel: 1,030
- Controller: 126
- Initiator: 89
- Duelist: 25

This tells us the dataset heavily reflects Sentinel-style play. It also warns us that some classes (especially Duelist) may be hard to model without careful evaluation.

### 3.3 Numeric behaviour and correlations

Box plots and summaries show:

- KDR has about 10 outliers above 3σ, plus 44 negative values (clearly invalid).
- Kills/deaths have a few extreme matches and the negative values noted above.
- Round metrics have no statistical outliers but do contain negative entries.

A correlation matrix of the main numeric features (round_wins, round_losses, kills, deaths, assists, kdr, avg_dmg, acs) shows no very strong relationships (none above |0.7|). Most correlations are weak to moderate, which suggests that each metric adds some independent information.

### 3.4 Temporal patterns

The match dates in the dataset span approximately from 2023 to 2024 and are generally well-distributed across this period. However, a small number of matches contain future dates that fall outside this expected range, extending into 2026. These anomalous entries lie outside the normal temporal distribution and could introduce noise into the analysis. To maintain data consistency and ensure meaningful temporal analysis, these out-of-range dates should either be corrected, removed, or normalized appropriately before further modeling or visualization.

Time information will be useful later if we want to account for meta changes or create time-aware train/test splits.

---

## 4. Overall Data Quality

Putting everything together:

- **Completeness:** about 98.4% of values are present, with missingness concentrated in rank, outcome, and role.
- **Accuracy:** roughly 15.5% of key numeric entries are negative and must be corrected or removed.
- **Consistency:** 289 matches have non-standard or missing outcomes that need to be normalized.

We assign a **medium** confidence rating: the data is rich enough to be useful, but it clearly needs a careful cleaning step before modeling.

---

## 5. Early Hypotheses

From this understanding, a few guiding hypotheses emerge:

1. **Role–performance alignment:** With Sentinels dominating, players may favor information/utility-focused play. We need to test whether high-frag players are truly under-represented on Duelists or if the sample is simply biased.
2. **Map–agent relationship:** Cypher is strong overall, but some agents may be more viable on specific maps; this should be visible once we control for map.
3. **Rank–agent complexity:** Concentration in Diamond/Ascendant suggests higher-ranked players might favor “complex” agents and roles; we can test how agent choice shifts across ranks.

These hypotheses will guide feature engineering and later model interpretation.

---

## 6. Data Preparation Plan

The next phase will operationalize what we learned here:

- Handle missing values in `rank` column using **mode** (most frequent value) since it's categorical.
- Handle missing/invalid `outcome` values:
  - Replace `'Unknown'` string values with NaN first.
  - Fill missing outcomes by comparing `round_wins` vs `round_losses` for each match (Win if `round_wins > round_losses`, Loss if `round_wins < round_losses`, Draw if equal).
- Handle missing values in `role` column using **mode**.
- Replace `'UnknownAgent'` values in `agent` column with **mode**.
- Fix or remove negative values in kills, deaths, round_wins, round_losses, and kdr.
- Encode categorical variables (agent, map, rank, role) before modeling.


We also plan some targeted feature engineering:

- Derived performance features (e.g., win rate, KDA, damage efficiency).


---

## 7. Summary and Next Steps

We now have a clear understanding of our dataset. It contains 1,296 single-player matches with 20 features, combining detailed performance statistics with role and context information. The main strengths of the dataset are the variety of performance metrics and the additional role information. However, there are also some challenges, such as unusual numeric values, inconsistent outcomes, and an imbalance in the number of matches for different agents and roles.

This document acts as a starting point for the next steps. It records what we currently know about the dataset before we begin cleaning or modifying it.
