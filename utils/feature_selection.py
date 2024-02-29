"""
Handles feature selection utility functions.
"""

import pandas as pd
import pathlib


def get_X_y(data_path: pathlib.PosixPath, target_variable: str):
    """get_X_y returns the target and predictive variables."""

    df = pd.read_csv(data_path)
    X = df.drop([target_variable],axis=1).values
    y = df[target_variable].values

    return X, y