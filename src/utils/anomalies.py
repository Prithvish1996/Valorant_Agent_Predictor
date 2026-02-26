import pandas as pd

def find_inconsistent_outcomes(df):
    return df[
        (df['round_wins'] + df['round_losses'] > 0) &
        (df['outcome'].isna() | (~df['outcome'].isin(['Win', 'Loss'])))
        ]

def find_rare_categories(df, col, n=5):
    return df[col].value_counts().nsmallest(n)

def find_future_dates(df, date_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    return df[df[date_col] > pd.Timestamp.today()]

def numeric_anomaly_summary(df, numeric_cols):
    return pd.DataFrame({
        "Feature": numeric_cols,
        "Negative Values": [(df[col] < 0).sum() for col in numeric_cols],
        "Zero Values": [(df[col] == 0).sum() for col in numeric_cols],
        "Outliers (>3σ)": [
            ((df[col] - df[col].mean()).abs() > 3*df[col].std()).sum()
            for col in numeric_cols
        ]
    })

def categorical_anomaly_summary(df, categorical_cols, unknown_label="UnknownAgent"):
    return pd.DataFrame({
        "Feature": categorical_cols,
        "Unknown / Rare": [(df[col] == unknown_label).sum() for col in categorical_cols]
    })

def date_anomaly_summary(df, date_cols):
    return pd.DataFrame({
        "Feature": date_cols,
        "Future Dates": [(df[col] > pd.Timestamp.today()).sum() for col in date_cols]
    })
