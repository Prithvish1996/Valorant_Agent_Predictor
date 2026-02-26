import pandas as pd
import numpy as np
import random
from pathlib import Path
from datetime import datetime, timedelta

from src.data.data_loader import DataLoader

# -----------------------------
# 1️⃣ Set project and data paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # Adjust based on script location
DATA_DIR = PROJECT_ROOT / "data" / "external"

# Initialize data loader
data_loader = DataLoader(DATA_DIR)

# Load the clean Kaggle data
df = data_loader.load_data("valorant_games.csv")  # <- Make sure you have this file

# Output path for messy CSV
messy_file = DATA_DIR / "valorant_games_messy.csv"

# -----------------------------
# 2️⃣ Prepare lists for random generation
# -----------------------------
agents = df['agent'].unique().tolist()
maps = df['map'].unique().tolist()
ranks = df['rank'].unique().tolist()

# -----------------------------
# 3️⃣ Function to create messy row
# -----------------------------
def create_light_messy_row():
    row = {
        'game_id': df['game_id'].max() + random.randint(1, 1000),
        'episode': random.choice(df['episode'].unique()),
        'act': random.choice(df['act'].unique()),
        'rank': random.choice(ranks + [None])
    }

    # Date can be future sometimes
    if random.random() < 0.1:
        row['date'] = (datetime.now() + timedelta(days=random.randint(1,30))).strftime("%m/%d/%Y")
    else:
        row['date'] = random.choice(df['date'])

    # Agent, with some unknown
    row['agent'] = random.choice(agents + ["UnknownAgent"])

    # Map
    row['map'] = random.choice(maps)

    # Outcome with some missing
    row['outcome'] = random.choice([None,"Draw","Unknown"])

    # Random numeric stats, allow negative values sometimes
    row['round_wins'] = random.randint(-2,15)
    row['round_losses'] = random.randint(-2,15)
    row['kills'] = random.randint(-3,20)
    row['deaths'] = random.randint(-3,20)
    row['assists'] = random.randint(0,10)
    row['kdr'] = round(random.uniform(-0.5, 2.5), 1)
    row['avg_dmg_delta'] = random.randint(-50,50)
    row['headshot_pct'] = random.randint(0,100)
    row['avg_dmg'] = random.randint(0,250)
    row['acs'] = random.randint(0,300)
    row['num_frag'] = random.randint(0,10)
    return row

# -----------------------------
# 4️⃣ Generate 200 messy rows
# -----------------------------
messy_rows = [create_light_messy_row() for _ in range(296)]
df_messy = pd.concat([df, pd.DataFrame(messy_rows)], ignore_index=True)

# -----------------------------
# 5️⃣ Save to CSV
# -----------------------------
df_messy.to_csv(messy_file, index=False)
print(f"Messy dataset saved at: {messy_file}")
