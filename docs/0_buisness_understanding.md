# 0. Business Understanding

## 1. Background: Valorant

Valorant is a 5v5 tactical first-person shooter developed and published by Riot Games, released worldwide in June 2020. The game combines precise gunplay similar to Counter-Strike with a set of unique characters called Agents, each having special abilities, similar to heroes in Overwatch. Because it mixes shooting skill with the strategic use of abilities, the game quickly became very popular and now has millions of active players around the world.

#### Dataset Source

The primary dataset used in this project originates from Kaggle and is available at:  
https://www.kaggle.com/datasets/mitchellharrison/my-first-1000-valorant-games

This dataset contains statistics from the first 1000 matches played in *Valorant*. For the purposes of this project, the dataset was not used directly in its original form. Instead, it was intentionally modified to introduce several **data quality issues and anomalies**.
 These modifications were designed to simulate real-world datasets, which often contain missing values, inconsistencies, and incorrect entries. By introducing such issues, the project is able to demonstrate practical **data cleaning, validation, and preprocessing techniques**.

Two versions of the dataset are included in the project:

- **Original dataset:** `data/external/valorant_games.csv`  
- **Modified dataset used for analysis:** `data/extrernal/valorant_games_messy.csv`

The modified dataset includes intentionally introduced problems such as missing values, invalid categorical labels, inconsistent outcomes, and negative numeric values. These anomalies provide a realistic environment to implement and evaluate the data preparation pipeline developed in this project.

In addition to the main dataset, an **Agent Master List** was created as a supporting reference dataset. This list was compiled manually based on the current pool of playable agents in Valorant, available on the official Valorant website:

https://playvalorant.com/en-us/agents/

The Agent Master List is used during data cleaning and validation to ensure that agent names are valid and correctly associated with their corresponding roles within the game.

**Note:** To perform the same transformations or intentionally introduce anomalies into the dataset, you can use the script provided in the project directory:  

`src/data/messy/mess_my_data.py`  

This script contains all the logic used to generate the `valorant_games_messy.csv` file from the original dataset, including the addition of missing values, invalid entries, and other controlled inconsistencies for data cleaning practice.



### 1.1 Dataset Features

The dataset contains one row per player per game. The table below describes every column.

#### Raw columns (from source dataset)

| Feature | Type | Description |
|---|---|---|
| `game_id` | Integer | Unique identifier for each match — row index only, not a modelling feature |
| `episode` | Integer | Valorant competitive season episode number |
| `act` | Integer | Act within the episode (each episode has 3 acts) |
| `rank` | Categorical | Player's competitive rank at the time of the match (e.g., `Gold 1`, `Platinum 3`) |
| `date` | Date | Date the match was played |
| `agent` | Categorical | Name of the agent the player selected for the match (e.g., `Cypher`, `Jett`) |
| `map` | Categorical | Name of the map the match was played on (e.g., `Ascent`, `Bind`) |
| `outcome` | Categorical | Match result from the player's perspective — `Win`, `Loss`, or `Draw` |
| `round_wins` | Integer | Number of rounds the player's team won |
| `round_losses` | Integer | Number of rounds the player's team lost |
| `kills` | Integer | Total kills secured by the player in the match |
| `deaths` | Integer | Total times the player died in the match |
| `assists` | Integer | Total assists (contributed to a kill without landing the final shot) |
| `kdr` | Float | Kill/Death Ratio — `kills ÷ deaths`; values above 1.0 indicate more kills than deaths |
| `avg_dmg_delta` | Float | Average damage dealt minus average damage received per round; positive = net contributor |
| `headshot_pct` | Float | Percentage of kills secured via headshots; higher = more precise aim |
| `avg_dmg` | Float | Average damage dealt to enemies per round |
| `acs` | Integer | Average Combat Score — Riot's official per-round contribution metric combining damage, kills, and multi-kills |
| `num_frag` | Integer | Number of rounds in which the player secured the first kill (opening frag) |
| `role` | Categorical | Agent role grouping — `Duelist`, `Initiator`, `Controller`, or `Sentinel` |



#### Supporting reference dataset: Agent Master List (`data/external/agent_master_list.csv`)

A two-column lookup table compiled manually from the official Valorant agent roster at [playvalorant.com/en-us/agents](https://playvalorant.com/en-us/agents/). It is used during data cleaning to validate agent names and fill missing `role` values by joining on `Agent`.

| Column | Type | Description |
|---|---|---|
| `Agent` | Categorical | Official agent name as it appears in the game (e.g., `Jett`, `KAY/O`) |
| `Role` | Categorical | Role the agent belongs to — `Duelist`, `Initiator`, `Controller`, or `Sentinel` |

---

### 1.2 How a Match Works

| Concept | Description |
|---|---|
| **Teams** | Two teams of 5 players each |
| **Rounds** | A standard match plays up to 25 rounds (first to 13 wins) |
| **Roles** | Attackers attempt to plant a bomb (Spike); Defenders try to stop them |
| **Maps** | Each match is played on one of several symmetrically designed maps |
| **Economy** | Players earn credits per round to buy weapons and abilities |

### 1.3 Agents and Roles

Every player selects one Agent before a match begins. Agents are grouped into four **roles**:

| Role | Description | Example Agents |
|---|---|---|
| **Duelist** | Aggressive fraggers designed to create space and secure kills | Jett, Reyna, Neon |
| **Initiator** | Set up plays by gathering information and disrupting enemies | Sova, Breach, Fade |
| **Controller** | Control sightlines and area denial using smoke/zone abilities | Omen, Brimstone, Astra |
| **Sentinel** | Defensive anchor; locks down flanks and supports the team | Sage, Cypher, Killjoy |

Agent selection is a important strategic decision team compositions that balance roles tend to outperform one-dimensional lineups, making agent choice a meaningful predictor of match outcomes and individual performance.

### 1.4 Rank System

Valorant uses a **competitive rank ladder** to matchmake and evaluate players. Ranks progress from lowest to highest:

`Iron → Bronze → Silver → Gold → Platinum → Diamond → Ascendant → Immortal → Radiant`

Each tier (except Radiant) has three sub-divisions (e.g., Silver 1, Silver 2, Silver 3). A player's rank reflects their overall skill level and directly influences the quality of opponents they face, performance expectations, and statistical benchmarks.

---

## 2. Business Problem

### 2.1 Itteration - 01

#### 2.1.1 Problem Statement

Given a player's in-game performance history  including kills, deaths, KDR, round wins/losses, map, and current rank  **can we predict the Agent a player is most likely to select in their next match?**

Agent prediction has practical value for:

- **Players**: Understanding which agents suit their playstyle and performance patterns.
- **Coaches / analysts**: Tailoring team compositions and practice regimens.
- **Game developers**: Informing balance decisions by revealing which agents are picked by high/low-performing players.

#### 2.1.1 Project Goal

Build a supervised multi-class classification model that predicts the **Agent** a player will choose, using features derived from their historical match data.

### 2.2 Iteration 02

#### 2.2.1 Findings from Iteration 01

After completing the data understanding phase (see `notebooks/1_data_understanding.ipynb`), the correlation matrix revealed a critical limitation of the original problem framing:

- **`agent`** and **`role`**  the target variables from Iteration 01 show **very low correlation** with all numeric features (kills, deaths, KDR, round wins/losses).
- The only column exhibiting a notable correlation with `agent`/`role` is `game_id`, which is a **unique row identifier**, not a meaningful feature, and must not be used for model training.

This strongly suggests that a player's agent choice is not reliably predictable from their raw performance statistics alone. The dataset does not carry enough signal to train a performant agent classification model.

#### 2.2.2 Problem Statement

Given the findings above, the prediction target is revised. Instead of predicting which agent a player selects, we now ask:

> **Given a subset of a player's in-game features (rank, map, agent, role, round outcomes), can we predict how that player performed **

This reframing shifts the problem from agent recommendation to **player performance prediction**, which is better supported by the available data.

#### 2.2.3 Project Goal

Build a supervised regression or classification model that predicts a player's **performance** (e.g., KDR bracket or match outcome) from contextual and historical match features.

#### 2.2.4 Success Criteria

| Criterion | Target |
|---|---|
| Regression (KDR)  RMSE | As low as possible; baseline = predicting mean KDR |
| Classification (outcome)  Accuracy | > 80% on held-out test set |
| Feature importance | Interpretable feature contributions available |





