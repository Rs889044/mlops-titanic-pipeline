#!/bin/bash

# Make sure we are in the virtual environment
source venv/bin/activate

echo "--- Step 1: Checking for data drift ---"

# Run the drift detection script. It will exit with 1 if drift is detected.
python src/detect_drift.py data/raw/train.csv data/raw/new_data.csv

# Check the exit code of the last command ($?)
if [ $? -eq 1 ]; then
    echo "--- Step 2: Drift detected! Proceeding with retraining pipeline. ---"

    # Simulate integrating new data into the main training set
    echo "--- Merging new data into training set... ---"
    tail -n +2 data/raw/new_data.csv >> data/raw/train.csv

    # Update DVC to track the new version of our training data
    echo "--- Versioning new data with DVC... ---"
    dvc add data/raw/train.csv

    # Run the full DVC pipeline to preprocess data and retrain the model
    echo "--- Running DVC pipeline to retrain the model... ---"
    dvc repro

    echo "--- Retraining pipeline complete. A new model version has been registered in MLflow. ---"
else
    echo "--- Step 2: No data drift detected. Pipeline will not be re-run. ---"
fi