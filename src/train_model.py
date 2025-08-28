# src/train_model.py

import argparse
import mlflow
import mlflow.spark
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# --- MLflow setup ---
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Titanic Survival Prediction")

def create_spark_session():
    """Creates and returns a Spark session."""
    return SparkSession.builder \
        .appName("Titanic Model Training") \
        .master("local[*]") \
        .getOrCreate()

def train_model(spark, input_path_train, model_output_path):
    """
    Loads raw data, defines a full preprocessing and training pipeline,
    tunes it, and logs the best version to MLflow.
    """
    # Load raw data
    train_df = spark.read.csv(input_path_train, header=True, inferSchema=True)

    with mlflow.start_run() as run:
        print(f"Starting MLflow Run ID: {run.info.run_id}")

        # --- Define All Pipeline Stages ---

        # 1. Imputer for missing 'Age' and 'Fare' values
        imputer = Imputer(
            inputCols=["Age", "Fare"],
            outputCols=["Age_imputed", "Fare_imputed"]
        ).setStrategy("mean")

        # 2. StringIndexer for categorical columns
        categorical_cols = ["Sex", "Embarked", "Pclass"]
        string_indexers = [
            StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid="keep") 
            for col in categorical_cols
        ]

        # 3. VectorAssembler to create the 'features' vector
        feature_cols = ["Age_imputed", "Fare_imputed", "SibSp", "Parch"] + [c + "_index" for c in categorical_cols]
        vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

        # 4. The Logistic Regression model
        lr = LogisticRegression(featuresCol="features", labelCol="Survived")
        
        # --- Create the Full Pipeline ---
        pipeline = Pipeline(stages=[imputer] + string_indexers + [vector_assembler, lr])

        # Define Hyperparameter Grid for Cross-Validation
        paramGrid = ParamGridBuilder() \
            .addGrid(lr.regParam, [0.01, 0.1]) \
            .addGrid(lr.maxIter, [10, 20]) \
            .build()
        
        # Define Evaluator
        evaluator = BinaryClassificationEvaluator(labelCol="Survived", metricName="areaUnderROC")

        # Set up Cross-Validator
        crossval = CrossValidator(estimator=pipeline,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=evaluator,
                                  numFolds=3)
        
        print("Starting full pipeline training with cross-validation...")
        cvModel = crossval.fit(train_df)
        print("Model training complete.")
        
        # The best model is the entire fitted pipeline
        bestPipelineModel = cvModel.bestModel

        # Log metrics (optional, but good practice)
        predictions = bestPipelineModel.transform(train_df)
        auc = evaluator.evaluate(predictions)
        mlflow.log_metric("auc", auc)
        print(f"AUC on training data: {auc}")

        # Log the entire fitted pipeline model
        mlflow.spark.log_model(
            spark_model=bestPipelineModel,
            artifact_path="spark-model",
            registered_model_name="TitanicSurvivalModel"
        )
        print("Full pipeline model logged and registered in MLflow Model Registry.")

        # Create dummy output for DVC
        import os
        if not os.path.exists(model_output_path):
            os.makedirs(model_output_path)
        with open(os.path.join(model_output_path, "train_status.txt"), "w") as f:
            f.write(f"Training completed for run_id: {run.info.run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path_train", help="Path to the raw training data CSV")
    parser.add_argument("--output_path", help="Path to save a dummy status file")
    args = parser.parse_args()

    spark_session = create_spark_session()
    train_model(spark_session, args.input_path_train, args.output_path)
    spark_session.stop()