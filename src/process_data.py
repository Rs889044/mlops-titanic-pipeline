# src/process_data.py

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, when, count, udf
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer

def create_spark_session():
    """Creates and returns a Spark session."""
    return SparkSession.builder \
        .appName("Titanic Data Processing") \
        .master("local[*]") \
        .getOrCreate()

def process_data(spark, input_path_train, input_path_test, output_path):
    """
    Loads, processes, and feature engineers the Titanic dataset using Spark.
    """
    train_df = spark.read.csv(input_path_train, header=True, inferSchema=True)
    test_df = spark.read.csv(input_path_test, header=True, inferSchema=True)

    # Impute 'Age' and 'Fare' with the mean. # <<< CHANGED (Added 'Fare')
    imputer = Imputer(
        inputCols=["Age", "Fare"], 
        outputCols=["Age_imputed", "Fare_imputed"]
    ).setStrategy("mean")

    embarked_mode = train_df.groupBy("Embarked").count().orderBy("count", ascending=False).first()[0]

    train_df = train_df.fillna({"Embarked": embarked_mode, "Cabin": "Unknown"})
    test_df = test_df.fillna({"Embarked": embarked_mode, "Cabin": "Unknown"})

    train_df = train_df.withColumn("IsAlone", when((col("SibSp") + col("Parch")) == 0, 1).otherwise(0))
    test_df = test_df.withColumn("IsAlone", when((col("SibSp") + col("Parch")) == 0, 1).otherwise(0))

    categorical_cols = ["Sex", "Embarked", "Pclass"]
    string_indexers = [StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid="keep") for col in categorical_cols]

    # Assemble all feature columns into a single vector # <<< CHANGED (Use imputed Fare)
    feature_cols = ["Age_imputed", "SibSp", "Parch", "Fare_imputed", "IsAlone"] + [c + "_index" for c in categorical_cols]
    vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    preprocessing_pipeline = Pipeline(stages=[imputer] + string_indexers + [vector_assembler])

    pipeline_model = preprocessing_pipeline.fit(train_df)

    processed_train_df = pipeline_model.transform(train_df)
    processed_test_df = pipeline_model.transform(test_df)

    final_train_df = processed_train_df.select("Survived", "features")
    final_test_df = processed_test_df.select("PassengerId", "features")

    final_train_df.write.mode("overwrite").parquet(f"{output_path}/train")
    final_test_df.write.mode("overwrite").parquet(f"{output_path}/test")

    print("Data processing complete. Processed files saved to:", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path_train", help="Path to the raw training data CSV")
    parser.add_argument("--input_path_test", help="Path to the raw test data CSV")
    parser.add_argument("--output_path", help="Path to save the processed data")
    args = parser.parse_args()

    spark_session = create_spark_session()
    process_data(spark_session, args.input_path_train, args.input_path_test, args.output_path)
    spark_session.stop()