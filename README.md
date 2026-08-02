# Bus Service Reliability Prediction Using BODS Timetable Data

A PySpark-based big data pipeline that ingests, cleans, explores, and models UK bus
timetable data from the Bus Open Data Service (BODS), predicting route length category
as a structural proxy for service complexity.

## Project overview

This project processes 271,552 timetable records extracted from Stagecoach TransXChange
XML files, performing exploratory data analysis and comparing three classification models
(Logistic Regression, Random Forest, Decision Tree) to predict whether a route is
long-distance based on service structure features.

## Pipeline structure

Notebooks are numbered and must be run in order — each depends on outputs from the
previous stage.

| Notebook | Purpose |
|---|---|
| `01_data_ingestion.ipynb` | Extracts and parses BODS TransXChange XML files into a PySpark DataFrame; demonstrates repartitioning (12 → 4 partitions) and caching; saves to Parquet |
| `02_preprocessing.ipynb` | Null/duplicate checks, data type conversion, engineers `route_size` feature (Short/Medium/Long) |
| `03_eda.ipynb` | Statistical profiling (mean, median, std, skewness, kurtosis), data quality assessment (nulls, cardinality, outliers via IQR), visualizations, correlation analysis |
| `04_model_training.ipynb` | Trains and compares 3 classification models; hyperparameter tuning via `CrossValidator`; computes Model Efficiency (F1-score per second of training) |
| `05_evaluation.ipynb` | Evaluates all 3 models on held-out test data using Accuracy, Precision, Recall, F1-score, and ROC-AUC |

## Data storage strategy

This project uses **Apache Parquet** with PySpark DataFrames rather than a relational
database. This is justified by Parquet's columnar compression, native Spark integration,
and embedded schema — well suited to this project's aggregation-heavy, single-table
workload at 271K+ record scale. See the "Data Storage Strategy" section in
`01_data_ingestion.ipynb` for full justification.

## Target variable

The dataset lacks real-time GPS, delay, or disruption data required for the assignment
brief's defined reliability metrics (Service Reliability, Headway Regularity, Travel Time
Variability). As a justified alternative, this project uses **route length classification**
(`route_size`: Short/Medium/Long → binary target: Long vs. not-Long) as an operational
proxy — longer routes are structurally more exposed to delay accumulation and headway
drift. Full justification is documented in `04_model_training.ipynb`.

## Results summary

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Training time (s) | Efficiency (F1/s) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.8576 | 0.8224 | 0.8576 | 0.8303 | **0.8464** | 3.45 | **0.2403** |
| Random Forest | 0.8624 | 0.7438 | 0.8624 | 0.7987 | 0.5000 | 5.81 | 0.1374 |
| Decision Tree | 0.8511 | 0.7438 | 0.8511 | 0.7932 | 0.4937 | 3.89 | 0.2052 |

**Logistic Regression** is the recommended model — it is the only one with genuine
discriminative power (ROC-AUC well above chance), while also being the fastest and most
efficient to train. Random Forest and Decision Tree both collapsed to majority-class
prediction (ROC-AUC ≈ 0.50), a known risk given the ~86%/14% class imbalance in the target.

## Setup

### Requirements
- Python 3.12
- Java 11+ (required by PySpark)
- See `requirements.txt` for Python package versions

### Installation
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Running the pipeline
1. Place the BODS timetable ZIP file in `data/timetable/timetable.zip`
2. Run notebooks in order: `01_data_ingestion.ipynb` through `05_evaluation.ipynb`
3. Outputs (Parquet files, trained models) are saved to `outputs/` and `models/`

### Monitoring Spark jobs
While any notebook cell is running, the Spark UI is available at
`http://localhost:4040` (or `4041` if the default port is in use) to inspect job
partitioning, caching, and task execution.

## Project structure