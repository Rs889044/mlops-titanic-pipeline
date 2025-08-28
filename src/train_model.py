# src/train_model.py

import argparse
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def create_spark_session():
    """Creates and returns a Spark session."""
    return SparkSession.builder \
        .appName("Titanic Model Training") \
        .master("local[*]") \
        .getOrCreate()

def train_model(spark, input_path, model_output_path):
    """
    Loads processed data and trains a model with hyperparameter tuning.
    """
    # 1. Load the processed data from Task 1
    print("Loading processed data...")
    train_df = spark.read.parquet(f"{input_path}/train")

    # 2. Define the Model
    lr = LogisticRegression(featuresCol="features", labelCol="Survived")

    # 3. Define the Hyperparameter Grid for Tuning
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
        .addGrid(lr.maxIter, [10, 20, 30]) \
        .build()

    # 4. Define the Evaluator
    evaluator = BinaryClassificationEvaluator(labelCol="Survived", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

    # 5. Set up Cross-Validation
    crossval = CrossValidator(estimator=lr,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=3)

    # 6. Train the model
    print("Starting model training with cross-validation...")
    cvModel = crossval.fit(train_df)
    print("Model training complete.")

    # 7. Save the best model
    bestModel = cvModel.bestModel
    bestModel.write().overwrite().save(model_output_path)

    print(f"Best model saved to: {model_output_path}")

    # Optional: Print the best parameters
    best_reg_param = bestModel.getOrDefault('regParam')
    best_max_iter = bestModel.getOrDefault('maxIter')
    print(f"Best Parameters: regParam = {best_reg_param}, maxIter = {best_max_iter}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Path to the processed data directory")
    parser.add_argument("--output_path", help="Path to save the trained model")
    args = parser.parse_args()

    spark_session = create_spark_session()
    train_model(spark_session, args.input_path, args.output_path)
    spark_session.stop()