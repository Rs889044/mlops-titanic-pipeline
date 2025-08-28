# src/train_model.py

import argparse
import mlflow
import mlflow.spark
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col

# --- MLflow setup ---
# Set the tracking URI to the local server we just started
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Set the experiment name
mlflow.set_experiment("Titanic Survival Prediction")

def create_spark_session():
    """Creates and returns a Spark session."""
    return SparkSession.builder \
        .appName("Titanic Model Training") \
        .master("local[*]") \
        .getOrCreate()

def train_model(spark, input_path, model_output_path):
    """
    Loads processed data, trains a model with hyperparameter tuning,
    and logs the entire process with MLflow.
    """
    train_df = spark.read.parquet(f"{input_path}/train")

    # Start an MLflow run. All logging will be recorded under this run.
    with mlflow.start_run() as run:
        print(f"Starting MLflow Run ID: {run.info.run_id}")

        # 1. Define the Model and Hyperparameter Grid
        lr = LogisticRegression(featuresCol="features", labelCol="Survived")
        paramGrid = ParamGridBuilder() \
            .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
            .addGrid(lr.maxIter, [10, 20, 30]) \
            .build()
        
        # Log the grid of hyperparameters we are searching over
        mlflow.log_param("regParam_grid", [0.01, 0.1, 0.5])
        mlflow.log_param("maxIter_grid", [10, 20, 30])

        # 2. Define Evaluators
        # AUC for overall performance
        auc_evaluator = BinaryClassificationEvaluator(labelCol="Survived")
        # Other metrics like accuracy, precision, etc.
        f1_evaluator = MulticlassClassificationEvaluator(labelCol="Survived", metricName="f1")
        accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="Survived", metricName="accuracy")

        # 3. Set up and run Cross-Validation
        crossval = CrossValidator(estimator=lr,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=auc_evaluator,
                                  numFolds=3)
        
        print("Starting model training with cross-validation...")
        cvModel = crossval.fit(train_df)
        print("Model training complete.")
        
        bestModel = cvModel.bestModel

        # 4. Log Best Parameters
        best_reg_param = bestModel.getOrDefault('regParam')
        best_max_iter = bestModel.getOrDefault('maxIter')
        mlflow.log_param("best_regParam", best_reg_param)
        mlflow.log_param("best_maxIter", best_max_iter)

        # 5. Evaluate and Log Metrics
        predictions = bestModel.transform(train_df)
        
        auc = auc_evaluator.evaluate(predictions)
        f1_score = f1_evaluator.evaluate(predictions)
        accuracy = accuracy_evaluator.evaluate(predictions)

        print(f"AUC: {auc}")
        print(f"F1 Score: {f1_score}")
        print(f"Accuracy: {accuracy}")

        mlflow.log_metric("auc", auc)
        mlflow.log_metric("f1_score", f1_score)
        mlflow.log_metric("accuracy", accuracy)

        # 6. Log Artifacts (e.g., a confusion matrix)
        preds_and_labels = predictions.select("prediction", "Survived").toPandas()
        confusion_matrix = pd.crosstab(preds_and_labels['Survived'], preds_and_labels['prediction'])
        
        plt.figure(figsize=(6, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig("confusion_matrix.png")
        
        mlflow.log_artifact("confusion_matrix.png")

        # 7. Log the model to MLflow and register it in the Model Registry
        # The 'registered_model_name' argument is what triggers the registration
        mlflow.spark.log_model(
            spark_model=bestModel,
            artifact_path="spark-model",
            registered_model_name="TitanicSurvivalModel"
        )
        print("Model logged and registered in MLflow Model Registry.")
    # Inside the train_model function, at the very end
    import os
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    with open(os.path.join(model_output_path, "train_status.txt"), "w") as f:
        f.write(f"Training completed for run_id: {run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Path to the processed data directory")
    # We keep output_path for DVC's sake, but it's not used by the model itself
    parser.add_argument("--output_path", help="Path to save a dummy status file")
    args = parser.parse_args()

    spark_session = create_spark_session()
    train_model(spark_session, args.input_path, args.output_path)
    spark_session.stop()