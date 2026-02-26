# src/utils/data_viz_utils.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib

sns.set_style("whitegrid")
sns.set_palette("Set2")

def plot_numeric_box(df: pd.DataFrame, col: str):
    """Box plot for numeric column"""
    plt.figure(figsize=(10,4))
    sns.boxplot(x=df[col], color='skyblue')
    plt.title(f"{col} Distribution (Box Plot)", fontsize=14)
    plt.xlabel(col)
    plt.show()

def plot_date_scatter(df: pd.DataFrame, col: str):
    """Scatter plot for date column to show matches over time"""
    df_copy = df.copy()
    df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
    df_valid = df_copy.dropna(subset=[col])

    # Count matches per date
    date_counts = df_valid[col].dt.date.value_counts().sort_index().reset_index()
    date_counts.columns = ['date', 'match_count']
    date_counts['date'] = pd.to_datetime(date_counts['date'])

    plt.figure(figsize=(14, 5))
    plt.scatter(date_counts['date'], date_counts['match_count'],
                alpha=0.6, c='steelblue', edgecolors='navy', s=50)

    plt.title(f"Matches Over Time - {col} (Scatter Plot)", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Number of Matches")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_categorical_bottom(df: pd.DataFrame, col: str, bottom_n=10):
    """Bar plot for bottom N rare categories"""
    if 'date' in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col]):
        # Date column → violin plot
        df[col] = pd.to_datetime(df[col], errors='coerce')
        date_numeric = df[col].dropna().map(lambda x: x.toordinal())

        plt.figure(figsize=(12,4))
        sns.violinplot(x=date_numeric, color='mediumseagreen')

        xticks = plt.xticks()[0]
        xtick_labels = [pd.Timestamp.fromordinal(int(x)).strftime('%Y-%m-%d') for x in xticks]
        plt.xticks(xticks, xtick_labels, rotation=45)
        plt.title(f"{col} Distribution Over Time (Violin Plot)", fontsize=14)
        plt.xlabel(col)
        plt.show()
    else:
        plt.figure(figsize=(12,5))
        bottom_vals = df[col].value_counts().nsmallest(bottom_n)
        cmap = matplotlib.colormaps["Reds_r"]
        colors = [cmap(i / len(bottom_vals)) for i in range(len(bottom_vals))]

        for i, (category, value) in enumerate(bottom_vals.items()):
            plt.barh(category, value, color=colors[i])

        plt.title(f"Bottom {bottom_n} {col} Counts (Rare/Unknown Categories)", fontsize=14)
        plt.xlabel("Count")
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()
