# src/detect_drift.py

import pandas as pd
import numpy as np
import sys
from scipy.spatial.distance import jensenshannon

def detect_data_drift(base_data_path, new_data_path, column, threshold=0.1):
    """
    Detects data drift between a base dataset and new data for a specific column.

    Exits with code 1 if drift is detected, otherwise exits with code 0.
    """
    # Load the datasets
    base_df = pd.read_csv(base_data_path)
    new_df = pd.read_csv(new_data_path)

    # Create histograms (probability distributions) for the specified column
    # We use a common set of bins to ensure the distributions are comparable.
    min_val = min(base_df[column].min(), new_df[column].min())
    max_val = max(base_df[column].max(), new_df[column].max())
    bins = np.linspace(min_val, max_val, 50)

    base_hist = np.histogram(base_df[column].dropna(), bins=bins)[0]
    new_hist = np.histogram(new_df[column].dropna(), bins=bins)[0]

    # Normalize histograms to get probability distributions
    base_dist = base_hist / base_hist.sum()
    new_dist = new_hist / new_hist.sum()

    # Calculate Jensen-Shannon divergence
    # It's a value between 0 (identical distributions) and 1 (maximally different).
    js_divergence = jensenshannon(base_dist, new_dist)

    print(f"Analyzing drift for column: '{column}'")
    print(f"Jensen-Shannon divergence: {js_divergence:.4f}")

    if js_divergence > threshold:
        print(f"ALERT: Data drift detected! Divergence ({js_divergence:.4f}) is above the threshold ({threshold}).")
        sys.exit(1) # Exit with code 1 to indicate drift
    else:
        print("No significant data drift detected.")
        sys.exit(0) # Exit with code 0 to indicate no drift

if __name__ == "__main__":
    # We will use sys.argv to get file paths from the command line
    # sys.argv[1] will be the base data path, sys.argv[2] the new data path
    if len(sys.argv) != 3:
        print("Usage: python detect_drift.py <base_data_path> <new_data_path>")
        sys.exit(2)

    # We'll check the 'Age' column for drift as an example
    detect_data_drift(sys.argv[1], sys.argv[2], column='Age')