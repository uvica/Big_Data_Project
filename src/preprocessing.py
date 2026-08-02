"""
Preprocessing utilities for the Bus Service Reliability pipeline.

Mirrors the cleaning steps performed in 02_preprocessing.ipynb, refactored into
reusable functions so they can be imported by other notebooks/scripts.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, when, when as spark_when


def check_missing_values(df: DataFrame) -> DataFrame:
    """Return a single-row DataFrame with the null count for every column."""
    return df.select([
        count(when(col(c).isNull(), c)).alias(c)
        for c in df.columns
    ])


def check_duplicates(df: DataFrame) -> dict:
    """Return total row count, unique row count, and duplicate count."""
    total_rows = df.count()
    unique_rows = df.dropDuplicates().count()
    return {
        "total_rows": total_rows,
        "unique_rows": unique_rows,
        "duplicate_rows": total_rows - unique_rows,
    }


def convert_stop_sequence_type(df: DataFrame) -> DataFrame:
    """Cast stop_sequence to integer type."""
    return df.withColumn("stop_sequence", col("stop_sequence").cast("integer"))


def add_route_size_feature(df: DataFrame) -> DataFrame:
    """
    Engineer the route_size feature from stop_sequence:
      < 20 stops  -> Short
      < 50 stops  -> Medium
      otherwise   -> Long
    """
    return df.withColumn(
        "route_size",
        spark_when(col("stop_sequence") < 20, "Short")
        .when(col("stop_sequence") < 50, "Medium")
        .otherwise("Long"),
    )


def preprocess_pipeline(df: DataFrame) -> DataFrame:
    """Run the full preprocessing pipeline: type conversion + feature engineering."""
    df = convert_stop_sequence_type(df)
    df = add_route_size_feature(df)
    return df