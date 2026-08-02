"""
Feature engineering utilities for the Bus Service Reliability pipeline.

Mirrors the target/feature preparation steps in 04_model_training.ipynb, refactored
into reusable functions. Ensures training and evaluation notebooks build the exact
same feature vector, avoiding the size-mismatch issues encountered during development.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, countDistinct, stddev
from pyspark.ml.feature import StringIndexer, VectorAssembler


def add_target_variable(df: DataFrame) -> DataFrame:
    """
    Define the binary classification target: 1 if route_size == 'Long', else 0.

    Note: this is a structural proxy for reliability/efficiency, justified in
    04_model_training.ipynb, since the dataset lacks real-time GPS/delay data.
    """
    return df.withColumn(
        "target",
        when(col("route_size") == "Long", 1).otherwise(0),
    )


def add_service_code_index(df: DataFrame) -> DataFrame:
    """Encode service_code as a numeric index for use as a model feature."""
    indexer = StringIndexer(inputCol="service_code", outputCol="service_code_index")
    return indexer.fit(df).transform(df)


def add_service_level_features(df: DataFrame) -> DataFrame:
    """
    Derive per-service_code aggregate features:
      - num_journeys: distinct journey count per service
      - stop_seq_std: standard deviation of stop_sequence per service
    Missing stop_seq_std (services with a single journey) is filled with 0.
    """
    service_features = df.groupBy("service_code").agg(
        countDistinct("journey_id").alias("num_journeys"),
        stddev("stop_sequence").alias("stop_seq_std"),
    )
    df = df.join(service_features, on="service_code", how="left")
    df = df.fillna(0, subset=["stop_seq_std"])
    return df


def build_feature_vector(df: DataFrame) -> DataFrame:
    """Assemble the final 'features' vector column used by all models."""
    assembler = VectorAssembler(
        inputCols=["service_code_index", "num_journeys", "stop_seq_std"],
        outputCol="features",
    )
    return assembler.transform(df)


def build_dataset(df: DataFrame) -> DataFrame:
    """
    Full feature engineering pipeline: target + service_code_index +
    service-level aggregates + assembled feature vector.

    Use this SAME function in both training and evaluation notebooks to guarantee
    a consistent feature schema between train_data and test_data.
    """
    df = add_target_variable(df)
    df = add_service_code_index(df)
    df = add_service_level_features(df)
    df = build_feature_vector(df)
    return df