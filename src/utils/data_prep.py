"""
Data Preparation Utilities

A utility class containing common DataFrame operations for data cleaning and preparation.
"""

import pandas as pd


class DataPrep:
    """
    Utility class for common data preparation operations like filling missing values,
    replacing invalid values, and data validation.
    """

    @staticmethod
    def fill_with_mode(df: pd.DataFrame, column: str, verbose: bool = True) -> pd.DataFrame:
        """
        Fill missing values in a column with the mode (most frequent value).
        """
        missing_before = df[column].isnull().sum()
        mode_value = df[column].mode()[0]
        df[column] = df[column].fillna(mode_value)
        missing_after = df[column].isnull().sum()

        if verbose:
            print(f"[{column}] Missing before: {missing_before} | Mode: {mode_value} | Missing after: {missing_after}")

        return df

    @staticmethod
    def determine_outcome(row):
        """
        Determine the outcome of a match based on round wins and losses.
        """
        if row['round_wins'] > row['round_losses']:
            return 'Win'
        elif row['round_wins'] < row['round_losses']:
            return 'Loss'
        else:
            return 'Draw'

    @staticmethod
    def fill_invalid_with_func(df: pd.DataFrame, column: str, valid_values: list, func, verbose: bool = True) -> pd.DataFrame:
        """
        Fill invalid values in a column using a custom function.

        Args:
            df: DataFrame to modify
            column: Column name to fill
            valid_values: List of valid values for the column
            func: Function to apply to rows with invalid values
            verbose: Whether to print progress
        """
        invalid_before = (~df[column].isin(valid_values)).sum()

        # Apply function to rows with invalid values
        mask = (~df[column].isin(valid_values)) | df[column].isnull()
        df.loc[mask, column] = df[mask].apply(func, axis=1)

        invalid_after = (~df[column].isin(valid_values)).sum()

        if verbose:
            print(f"[{column}] Invalid before: {invalid_before} | Invalid after: {invalid_after}")

        return df





