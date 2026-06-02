# Building Production ML Pipelines on AWS

> A hands-on guide for Cloud Architects with EKS, Serverless, Event-Driven, and CloudFormation expertise.

---

## Table of Contents

1. [ML Pipeline Architecture Overview](#1-ml-pipeline-architecture-overview)
2. [Data Ingestion & Preparation](#2-data-ingestion--preparation)
3. [SageMaker Feature Store](#3-sagemaker-feature-store)
4. [Model Training](#4-model-training)
5. [MLOps Pipeline](#5-mlops-pipeline)
6. [Model Deployment](#6-model-deployment)
7. [Monitoring & Governance](#7-monitoring--governance)
8. [Cost Optimization](#8-cost-optimization)
9. [CloudFormation Snippets](#9-cloudformation-snippets)
10. [End-to-End Project: Customer Churn Prediction](#10-end-to-end-project-customer-churn-prediction)

---

## 1. ML Pipeline Architecture Overview

### Training Pipeline vs Inference Pipeline

These are fundamentally different workloads with different SLAs, scaling characteristics, and failure modes.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE (Batch)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  S3 Data Lake → Processing Job → Feature Store → Training Job       │
│       │              │                │               │              │
│       │              ▼                ▼               ▼              │
│       │         Validated         Features       Model Artifact      │
│       │          Dataset          (offline)        → S3             │
│       │                                              │              │
│       │                                              ▼              │
│       │                                     Model Registry          │
│       │                                     (Approval Gate)         │
│                                                                     │
│  Trigger: Schedule (EventBridge) | Data Arrival | Drift Detection   │
│  SLA: Hours | Optimize for: Throughput, Cost                        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE (Real-time)                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  API Gateway → Lambda/EKS → Feature Store → SageMaker Endpoint     │
│                 (enrichment)   (online)      (or KServe on EKS)     │
│                                                      │              │
│                                                      ▼              │
│                                              Prediction Response     │
│                                                      │              │
│                                                      ▼              │
│                                              Capture → S3           │
│                                              (for monitoring)       │
│                                                                     │
│  Trigger: API Request (synchronous) | Kinesis/SQS (async)          │
│  SLA: p99 < 100ms | Optimize for: Latency, Availability            │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Decisions

| Decision | Training Pipeline | Inference Pipeline |
|----------|------------------|--------------------|
| Compute | Ephemeral (Spot OK) | Persistent (On-Demand) |
| Scaling | Vertical (bigger instance) | Horizontal (more instances) |
| State | Checkpoints to S3 | Stateless (features from store) |
| Failure Mode | Retry entire job | Circuit breaker + fallback |
| Orchestration | Step Functions / SageMaker Pipelines | API Gateway + Load Balancer |
| Cost Model | Pay-per-job-hour | Pay-per-endpoint-hour |

### Where EKS Fits

If you already run EKS at scale, you have two viable paths:

1. **SageMaker-native**: Use SageMaker for training + endpoints. Best when your ML team is separate from platform team.
2. **EKS-native (Kubeflow/Ray)**: Run everything on EKS with Karpenter autoscaling GPU nodes. Best when you want unified platform and already have GPU node pools.
3. **Hybrid**: Train on SageMaker (managed Spot, distributed training), serve on EKS (unified observability, service mesh integration).

---

## 2. Data Ingestion & Preparation

### 2.1 S3 Data Lake Patterns for ML

#### Layered Architecture

```
s3://company-data-lake/
├── raw/                          # Landing zone (immutable)
│   ├── streaming/                # Kinesis Firehose output
│   │   └── year=2024/month=06/hour=12/
│   └── batch/                    # Daily dumps
│       └── source=crm/dt=2024-06-15/
├── curated/                      # Cleaned, validated, partitioned
│   ├── customers/                # Hive-style partitions
│   │   └── dt=2024-06-15/
│   └── events/
│       └── event_type=login/dt=2024-06-15/
├── features/                     # Feature Store offline backing
│   └── feature_group=customer_features/
├── training/                     # Training datasets (versioned)
│   └── experiment=churn-v3/
│       ├── train/
│       ├── validation/
│       └── test/
└── models/                       # Model artifacts
    └── churn-model/
        └── version=12/model.tar.gz
```

#### Key Patterns

```python
# Partition strategy for time-series ML data
# Use EventBridge + Lambda to trigger processing on data arrival

# S3 Event → EventBridge Rule → Step Functions
{
    "source": ["aws.s3"],
    "detail-type": ["Object Created"],
    "detail": {
        "bucket": {"name": ["company-data-lake"]},
        "object": {"key": [{"prefix": "raw/streaming/"}]}
    }
}
```

#### Data Versioning with S3

```python
# Option 1: S3 versioning (simple but expensive at scale)
# Option 2: Manifest-based versioning (production pattern)

# manifest.json - tracks exact dataset composition
{
    "dataset_id": "churn-training-v3",
    "created_at": "2024-06-15T10:00:00Z",
    "files": [
        {"key": "curated/customers/dt=2024-06-14/part-0000.parquet", "etag": "abc123", "size": 1048576},
        {"key": "curated/events/event_type=login/dt=2024-06-14/part-0000.parquet", "etag": "def456", "size": 2097152}
    ],
    "statistics": {
        "total_records": 1500000,
        "feature_count": 47,
        "label_distribution": {"churn": 0.12, "retain": 0.88}
    }
}
```

### 2.2 SageMaker Processing Jobs

Processing jobs are ephemeral compute for data transformation. Think of them as "Lambda on steroids" — they spin up a container, run your code, and terminate.

#### Spark Processing Job

```python
from sagemaker.spark.processing import PySparkProcessor

spark_processor = PySparkProcessor(
    base_job_name="churn-feature-engineering",
    framework_version="3.3",
    role=sagemaker_role,
    instance_count=5,
    instance_type="ml.m5.4xlarge",
    max_runtime_in_seconds=7200,
    tags=[
        {"Key": "Project", "Value": "churn-prediction"},
        {"Key": "Environment", "Value": "production"}
    ]
)

spark_processor.run(
    submit_app="s3://ml-pipeline-code/processing/feature_engineering.py",
    submit_py_files=["s3://ml-pipeline-code/processing/utils.py"],
    arguments=[
        "--input-path", "s3://company-data-lake/curated/",
        "--output-path", "s3://company-data-lake/features/",
        "--execution-date", "2024-06-15",
        "--feature-group", "customer_churn_features"
    ],
    spark_event_logs_s3_uri="s3://ml-pipeline-logs/spark/",
    logs=True
)
```

#### Sklearn Processing Job (for smaller datasets < 50GB)

```python
from sagemaker.processing import ScriptProcessor

sklearn_processor = ScriptProcessor(
    command=["python3"],
    image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    role=sagemaker_role,
    instance_count=1,
    instance_type="ml.m5.2xlarge",
    base_job_name="churn-data-validation",
    env={
        "VALIDATION_THRESHOLD": "0.95",
        "ALERT_SNS_TOPIC": "arn:aws:sns:us-east-1:123456789012:ml-alerts"
    }
)

sklearn_processor.run(
    code="s3://ml-pipeline-code/processing/validate_data.py",
    inputs=[
        ProcessingInput(
            source="s3://company-data-lake/curated/customers/",
            destination="/opt/ml/processing/input/customers",
            s3_data_distribution_type="ShardedByS3Key"  # Parallel processing
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output/validation_report",
            destination="s3://ml-pipeline-artifacts/validation/",
            output_name="validation_report"
        )
    ]
)
```

### 2.3 Data Quality Checks & Validation

#### Production Validation Script (`validate_data.py`)

```python
"""
Data validation for ML pipeline.
Runs as SageMaker Processing Job.
Outputs: validation report + pass/fail signal for pipeline orchestration.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
import great_expectations as ge
import boto3

INPUT_DIR = Path("/opt/ml/processing/input/customers")
OUTPUT_DIR = Path("/opt/ml/processing/output/validation_report")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load all parquet files from input directory."""
    files = list(INPUT_DIR.glob("**/*.parquet"))
    if not files:
        raise ValueError(f"No parquet files found in {INPUT_DIR}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def schema_validation(df):
    """Validate schema hasn't drifted from expected."""
    expected_schema = {
        "customer_id": "object",
        "tenure_months": "int64",
        "monthly_charges": "float64",
        "total_charges": "float64",
        "contract_type": "object",
        "churn": "int64"
    }
    
    issues = []
    for col, dtype in expected_schema.items():
        if col not in df.columns:
            issues.append(f"Missing column: {col}")
        elif str(df[col].dtype) != dtype:
            issues.append(f"Column {col}: expected {dtype}, got {df[col].dtype}")
    
    return {"passed": len(issues) == 0, "issues": issues}

def statistical_validation(df):
    """Check for statistical anomalies that indicate data pipeline issues."""
    checks = {}
    
    # Null rate check
    null_rates = df.isnull().mean()
    high_null_cols = null_rates[null_rates > 0.05].to_dict()
    checks["null_rate"] = {
        "passed": len(high_null_cols) == 0,
        "details": high_null_cols
    }
    
    # Duplicate check
    dup_rate = df.duplicated(subset=["customer_id"]).mean()
    checks["duplicates"] = {
        "passed": dup_rate < 0.001,
        "duplicate_rate": dup_rate
    }
    
    # Distribution drift (compare to baseline statistics)
    # In production, load baseline from S3
    checks["label_distribution"] = {
        "passed": 0.05 < df["churn"].mean() < 0.30,
        "churn_rate": df["churn"].mean(),
        "note": "Churn rate outside [5%, 30%] indicates data issue"
    }
    
    # Freshness check
    if "event_timestamp" in df.columns:
        max_ts = pd.to_datetime(df["event_timestamp"]).max()
        staleness_hours = (pd.Timestamp.now() - max_ts).total_seconds() / 3600
        checks["freshness"] = {
            "passed": staleness_hours < 48,
            "staleness_hours": staleness_hours
        }
    
    return checks

def volume_validation(df, min_records=10000):
    """Ensure sufficient data volume for training."""
    return {
        "passed": len(df) >= min_records,
        "record_count": len(df),
        "minimum_required": min_records
    }

def main():
    df = load_data()
    
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "record_count": len(df),
        "schema_validation": schema_validation(df),
        "statistical_validation": statistical_validation(df),
        "volume_validation": volume_validation(df)
    }
    
    # Determine overall pass/fail
    all_passed = (
        report["schema_validation"]["passed"]
        and all(v["passed"] for v in report["statistical_validation"].values())
        and report["volume_validation"]["passed"]
    )
    report["overall_passed"] = all_passed
    
    # Write report
    with open(OUTPUT_DIR / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    # Signal pass/fail for pipeline orchestration
    # SageMaker Pipelines reads this as a property file
    with open(OUTPUT_DIR / "validation_status.json", "w") as f:
        json.dump({"passed": all_passed}, f)
    
    if not all_passed:
        # Send alert but don't fail the job — let the pipeline decide
        sns = boto3.client("sns")
        sns.publish(
            TopicArn=os.environ["ALERT_SNS_TOPIC"],
            Subject="[ML Pipeline] Data Validation Failed",
            Message=json.dumps(report, indent=2, default=str)
        )
    
    print(f"Validation {'PASSED' if all_passed else 'FAILED'}")
    print(json.dumps(report, indent=2, default=str))

if __name__ == "__main__":
    main()
```

### 2.4 Feature Engineering Patterns

#### Pattern 1: Time-windowed aggregations (PySpark)

```python
"""feature_engineering.py - Runs as SageMaker Spark Processing Job"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("ChurnFeatureEngineering").getOrCreate()

# Load curated data
customers = spark.read.parquet("s3://company-data-lake/curated/customers/")
events = spark.read.parquet("s3://company-data-lake/curated/events/")

# Time-windowed behavioral features
windows = [7, 14, 30, 60, 90]

for days in windows:
    window_events = events.filter(
        F.col("event_timestamp") >= F.date_sub(F.current_date(), days)
    )
    
    agg_features = window_events.groupBy("customer_id").agg(
        F.count("*").alias(f"event_count_{days}d"),
        F.countDistinct("event_type").alias(f"distinct_events_{days}d"),
        F.sum(F.when(F.col("event_type") == "support_ticket", 1).otherwise(0))
            .alias(f"support_tickets_{days}d"),
        F.avg("session_duration_sec").alias(f"avg_session_duration_{days}d"),
        F.max("event_timestamp").alias(f"last_activity_{days}d")
    )
    
    customers = customers.join(agg_features, "customer_id", "left")

# Derived features
customers = customers.withColumn(
    "engagement_trend",
    (F.col("event_count_7d") * 4) / F.greatest(F.col("event_count_30d"), F.lit(1))
).withColumn(
    "days_since_last_activity",
    F.datediff(F.current_date(), F.col("last_activity_7d"))
).withColumn(
    "support_intensity",
    F.col("support_tickets_30d") / F.greatest(F.col("tenure_months"), F.lit(1))
)

# Write to Feature Store offline format
customers.write.mode("overwrite").parquet(
    "s3://company-data-lake/features/feature_group=customer_churn_features/"
)
```

#### Pattern 2: Real-time feature computation (Lambda + DynamoDB)

```python
"""
Lambda function for real-time feature computation.
Triggered by Kinesis stream of user events.
Updates DynamoDB (or Feature Store online) with latest features.
"""
import boto3
import json
from decimal import Decimal

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("customer-realtime-features")

def lambda_handler(event, context):
    for record in event["Records"]:
        payload = json.loads(record["kinesis"]["data"])
        customer_id = payload["customer_id"]
        
        # Atomic counter updates for sliding window features
        table.update_item(
            Key={"customer_id": customer_id},
            UpdateExpression="""
                SET last_event_timestamp = :ts,
                    last_event_type = :et,
                    #ttl = :ttl
                ADD event_count_24h :one,
                    total_session_duration :dur
            """,
            ExpressionAttributeNames={"#ttl": "ttl"},
            ExpressionAttributeValues={
                ":ts": payload["timestamp"],
                ":et": payload["event_type"],
                ":one": 1,
                ":dur": Decimal(str(payload.get("session_duration", 0))),
                ":ttl": int(payload["timestamp"]) + 86400  # 24h TTL
            }
        )
    
    return {"statusCode": 200, "processed": len(event["Records"])}
```

---

## 3. SageMaker Feature Store

### 3.1 Online vs Offline Store

| Aspect | Online Store | Offline Store |
|--------|-------------|---------------|
| Latency | Single-digit ms | Seconds (Athena query) |
| Storage | DynamoDB (managed) | S3 (Parquet, auto-partitioned) |
| Use Case | Real-time inference | Training dataset creation |
| Retention | Latest value only | Full history (time-travel) |
| Cost Model | Per-read/write unit | S3 storage + Athena queries |
| Scale | Millions of reads/sec | Petabyte-scale |

### 3.2 Feature Group Definition & Ingestion

```python
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.session import Session

sagemaker_session = Session()
region = sagemaker_session.boto_region_name

# Define feature group
customer_feature_group = FeatureGroup(
    name="customer-churn-features",
    sagemaker_session=sagemaker_session
)

# Define schema
feature_definitions = [
    {"FeatureName": "customer_id", "FeatureType": "String"},
    {"FeatureName": "event_time", "FeatureType": "Fractional"},  # Required: Unix timestamp
    {"FeatureName": "tenure_months", "FeatureType": "Integral"},
    {"FeatureName": "monthly_charges", "FeatureType": "Fractional"},
    {"FeatureName": "total_charges", "FeatureType": "Fractional"},
    {"FeatureName": "contract_type", "FeatureType": "String"},
    {"FeatureName": "event_count_30d", "FeatureType": "Integral"},
    {"FeatureName": "support_tickets_30d", "FeatureType": "Integral"},
    {"FeatureName": "engagement_trend", "FeatureType": "Fractional"},
    {"FeatureName": "days_since_last_activity", "FeatureType": "Integral"},
]

customer_feature_group.load_feature_definitions(
    data_frame=feature_df  # Or define programmatically
)

# Create feature group with both online and offline store
customer_feature_group.create(
    s3_uri=f"s3://company-data-lake/features/",
    record_identifier_name="customer_id",
    event_time_feature_name="event_time",
    role_arn=sagemaker_role,
    enable_online_store=True,
    online_store_kms_key_id="alias/ml-encryption-key",
    offline_store_kms_key_id="alias/ml-encryption-key",
    tags=[{"Key": "Project", "Value": "churn-prediction"}]
)

# Batch ingestion (from DataFrame)
customer_feature_group.ingest(
    data_frame=features_df,
    max_workers=5,
    max_processes=4,
    wait=True
)
```

#### Real-time Feature Retrieval (for inference)

```python
import boto3

featurestore_runtime = boto3.client("sagemaker-featurestore-runtime")

def get_customer_features(customer_id: str) -> dict:
    """Retrieve features for real-time inference. ~5ms latency."""
    response = featurestore_runtime.get_record(
        FeatureGroupName="customer-churn-features",
        RecordIdentifierValueAsString=customer_id,
        FeatureNames=[
            "tenure_months", "monthly_charges", "event_count_30d",
            "support_tickets_30d", "engagement_trend", "days_since_last_activity"
        ]
    )
    
    # Convert to dict
    features = {
        item["FeatureName"]: item["ValueAsString"]
        for item in response["Record"]
    }
    return features

# Batch retrieval (for multiple customers)
def get_batch_features(customer_ids: list) -> list:
    """Batch get — up to 100 records per call."""
    response = featurestore_runtime.batch_get_record(
        Identifiers=[
            {
                "FeatureGroupName": "customer-churn-features",
                "RecordIdentifiersValueAsString": [cid],
                "FeatureNames": ["tenure_months", "monthly_charges", "engagement_trend"]
            }
            for cid in customer_ids
        ]
    )
    return response["Records"]
```

#### Training Dataset from Offline Store (Athena)

```python
from sagemaker.feature_store.dataset_builder import DatasetBuilder

# Point-in-time correct join (prevents data leakage!)
query = customer_feature_group.athena_query()

query_string = """
SELECT customer_id, tenure_months, monthly_charges, total_charges,
       event_count_30d, support_tickets_30d, engagement_trend,
       days_since_last_activity
FROM "sagemaker_featurestore"."customer-churn-features"
WHERE event_time BETWEEN 
    CAST(to_unixtime(timestamp '2024-01-01') AS double) 
    AND CAST(to_unixtime(timestamp '2024-06-01') AS double)
"""

query.run(query_string=query_string, output_location="s3://ml-pipeline-artifacts/athena-results/")
query.wait()
dataset = query.as_dataframe()
```

### 3.3 When to Use Feature Store vs DynamoDB/S3

| Scenario | Recommendation | Rationale |
|----------|---------------|-----------|
| < 10 features, simple lookup | DynamoDB | Overkill for Feature Store |
| Need point-in-time correctness | **Feature Store** | Built-in time-travel, prevents training-serving skew |
| Multiple models share features | **Feature Store** | Feature discovery, reuse, governance |
| Features change schema frequently | DynamoDB + S3 | Feature Store schema changes are painful |
| Ultra-low latency (< 1ms) | DynamoDB DAX | Feature Store online is ~5ms |
| Already have feature pipelines writing to S3 | S3 + Glue Catalog | Don't migrate unless you need online serving |
| Need feature lineage & monitoring | **Feature Store** | Built-in lineage tracking |

**Rule of thumb**: If you have > 3 ML models in production sharing features, Feature Store pays for itself in consistency and governance. For a single model, DynamoDB + S3 is simpler.

---

## 4. Model Training

### 4.1 SageMaker Training Jobs

#### Built-in Algorithm (XGBoost)

```python
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

xgb_estimator = Estimator(
    image_uri=sagemaker.image_uris.retrieve("xgboost", region, version="1.7-1"),
    role=sagemaker_role,
    instance_count=1,
    instance_type="ml.m5.4xlarge",
    output_path="s3://ml-pipeline-artifacts/models/",
    base_job_name="churn-xgboost",
    max_run=3600,
    tags=[{"Key": "Project", "Value": "churn-prediction"}],
    
    # Spot training (save 60-90%)
    use_spot_instances=True,
    max_wait=7200,  # Max time including spot interruptions
    checkpoint_s3_uri="s3://ml-pipeline-artifacts/checkpoints/churn-xgboost/",
    
    # Encryption
    output_kms_key="alias/ml-encryption-key",
    volume_kms_key="alias/ml-encryption-key",
    
    # Network isolation for compliance
    enable_network_isolation=False,  # Set True if model can't reach internet
    subnets=["subnet-abc123", "subnet-def456"],
    security_group_ids=["sg-ml-training"]
)

# Hyperparameters
xgb_estimator.set_hyperparameters(
    objective="binary:logistic",
    num_round=500,
    max_depth=6,
    eta=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    scale_pos_weight=7.5,  # Handle class imbalance (1/churn_rate)
    early_stopping_rounds=20
)

# Training inputs
train_input = TrainingInput(
    s3_data="s3://company-data-lake/training/experiment=churn-v3/train/",
    content_type="text/csv",
    distribution="FullyReplicated"
)

validation_input = TrainingInput(
    s3_data="s3://company-data-lake/training/experiment=churn-v3/validation/",
    content_type="text/csv",
    distribution="FullyReplicated"
)

xgb_estimator.fit(
    inputs={"train": train_input, "validation": validation_input},
    wait=False  # Async — poll or use EventBridge for completion
)
```

#### Custom Container Training

```dockerfile
# Dockerfile for custom training container
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1-gpu-py310-cu121-ubuntu20.04-sagemaker

# Install additional dependencies
COPY requirements.txt /opt/ml/code/requirements.txt
RUN pip install -r /opt/ml/code/requirements.txt

# Copy training code
COPY src/ /opt/ml/code/

# SageMaker entry point
ENV SAGEMAKER_PROGRAM train.py
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
```

```python
# train.py - Custom training script
"""
SageMaker contract:
- Input data: /opt/ml/input/data/{channel_name}/
- Model output: /opt/ml/model/
- Hyperparameters: /opt/ml/input/config/hyperparameters.json
- Checkpoints: /opt/ml/checkpoints/ (synced to S3)
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # SageMaker passes hyperparameters as command-line args
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--checkpoint-path", type=str, default="/opt/ml/checkpoints")
    return parser.parse_args()

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_dataset = load_dataset(args.train)
    val_dataset = load_dataset(args.validation)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Model
    model = ChurnModel(input_dim=47, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([7.5]).to(device))
    
    # Resume from checkpoint (for Spot interruptions)
    start_epoch = 0
    checkpoint_file = os.path.join(args.checkpoint_path, "checkpoint.pth")
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from checkpoint at epoch {start_epoch}")
    
    # Training loop
    best_auc = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for batch in train_loader:
            # ... training step ...
            pass
        
        # Validation
        val_auc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: val_auc={val_auc:.4f}")
        
        # Checkpoint every epoch (critical for Spot training)
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_auc": val_auc
        }, checkpoint_file)
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))
    
    # Save final model artifacts
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))
    # Save model metadata for inference
    with open(os.path.join(args.model_dir, "model_config.json"), "w") as f:
        json.dump({"input_dim": 47, "hidden_dim": 128, "version": "v3"}, f)

if __name__ == "__main__":
    args = parse_args()
    train(args)
```

### 4.2 Distributed Training

#### Data Parallel (for large datasets, model fits in single GPU)

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train.py",
    source_dir="src/",
    role=sagemaker_role,
    instance_count=4,
    instance_type="ml.p4d.24xlarge",  # 8x A100 per instance = 32 GPUs total
    framework_version="2.1",
    py_version="py310",
    
    # SageMaker Distributed Data Parallel
    distribution={
        "smdistributed": {
            "dataparallel": {
                "enabled": True
            }
        }
    },
    
    # Or PyTorch DDP (simpler, good enough for most cases)
    # distribution={"pytorchddp": {"enabled": True}},
    
    hyperparameters={
        "epochs": 50,
        "batch-size": 256,  # Per-GPU batch size
        "learning-rate": 0.001
    }
)
```

#### Model Parallel (for models that don't fit in single GPU — LLM fine-tuning)

```python
estimator = PyTorch(
    entry_point="train_llm.py",
    source_dir="src/",
    role=sagemaker_role,
    instance_count=2,
    instance_type="ml.p4d.24xlarge",
    framework_version="2.1",
    py_version="py310",
    
    distribution={
        "smdistributed": {
            "modelparallel": {
                "enabled": True,
                "parameters": {
                    "partitions": 4,                    # Split model across 4 GPUs
                    "microbatches": 8,                  # Pipeline parallelism
                    "pipeline_parallel_degree": 2,
                    "tensor_parallel_degree": 4,
                    "ddp": True,                        # Combine with data parallel
                    "optimize": "speed",                # or "memory"
                    "auto_partition": True               # Let SM figure out split points
                }
            }
        }
    }
)
```

### 4.3 Hyperparameter Tuning

```python
from sagemaker.tuner import (
    HyperparameterTuner,
    ContinuousParameter,
    IntegerParameter,
    CategoricalParameter
)

# Define search space
hyperparameter_ranges = {
    "eta": ContinuousParameter(0.01, 0.3, scaling_type="Logarithmic"),
    "max_depth": IntegerParameter(3, 10),
    "subsample": ContinuousParameter(0.5, 1.0),
    "colsample_bytree": ContinuousParameter(0.5, 1.0),
    "num_round": IntegerParameter(100, 1000),
    "min_child_weight": IntegerParameter(1, 10),
    "gamma": ContinuousParameter(0, 5),
    "alpha": ContinuousParameter(0, 2),  # L1 regularization
    "lambda": ContinuousParameter(0, 2)  # L2 regularization
}

tuner = HyperparameterTuner(
    estimator=xgb_estimator,
    objective_metric_name="validation:auc",
    objective_type="Maximize",
    hyperparameter_ranges=hyperparameter_ranges,
    
    # Bayesian optimization (default, most efficient)
    strategy="Bayesian",
    max_jobs=50,
    max_parallel_jobs=5,  # Balance exploration vs cost
    
    # Early stopping
    early_stopping_type="Auto",
    
    # Warm start from previous tuning job
    warm_start_config=WarmStartConfig(
        warm_start_type=WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM,
        parents=["previous-tuning-job-name"]
    ),
    
    tags=[{"Key": "Project", "Value": "churn-prediction"}]
)

tuner.fit(
    inputs={"train": train_input, "validation": validation_input},
    wait=False
)

# Get best model
best_training_job = tuner.best_training_job()
```

### 4.4 Spot Training with Checkpointing

```python
# Key settings for reliable Spot training:
estimator = Estimator(
    # ... other params ...
    
    # Spot configuration
    use_spot_instances=True,
    max_wait=14400,          # 4 hours max (including interruptions)
    max_run=7200,            # 2 hours actual training time
    
    # Checkpoint configuration — CRITICAL for Spot
    checkpoint_s3_uri="s3://ml-pipeline-artifacts/checkpoints/job-name/",
    checkpoint_local_path="/opt/ml/checkpoints",
    
    # Retry policy
    max_retry_attempts=2     # Auto-retry on spot interruption
)

# Your training script MUST:
# 1. Check for existing checkpoint at start
# 2. Save checkpoint after every epoch (or every N steps for long epochs)
# 3. The checkpoint_s3_uri is automatically synced by SageMaker
```

**Cost savings calculation:**
- `ml.p4d.24xlarge` On-Demand: ~$32.77/hr
- Spot price: ~$9.83/hr (70% savings)
- For a 4-hour training job: $131 → $39 savings per run
- Over 100 training jobs/month: $9,200 saved

### 4.5 Training on EKS with Kubeflow/Ray

If you're already running EKS with GPU node pools (Karpenter provisioning P4d/G5 nodes), training on EKS gives you unified infrastructure.

#### Kubeflow Training Operator

```yaml
# pytorchjob.yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: churn-training-v3
  namespace: ml-training
  labels:
    project: churn-prediction
    experiment: v3
spec:
  elasticPolicy:
    rdzvBackend: etcd
    minReplicas: 2
    maxReplicas: 8  # Auto-scale training workers
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          nodeSelector:
            karpenter.sh/provisioner-name: gpu-training
          tolerations:
            - key: nvidia.com/gpu
              operator: Exists
              effect: NoSchedule
          containers:
            - name: pytorch
              image: 123456789012.dkr.ecr.us-east-1.amazonaws.com/churn-training:v3
              resources:
                limits:
                  nvidia.com/gpu: 4
                  memory: "64Gi"
                requests:
                  nvidia.com/gpu: 4
                  memory: "48Gi"
              env:
                - name: S3_CHECKPOINT_URI
                  value: "s3://ml-pipeline-artifacts/checkpoints/churn-v3/"
              volumeMounts:
                - name: shm
                  mountPath: /dev/shm
          volumes:
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: "16Gi"
    Worker:
      replicas: 3
      restartPolicy: OnFailure
      template:
        spec:
          nodeSelector:
            karpenter.sh/provisioner-name: gpu-training-spot
          tolerations:
            - key: nvidia.com/gpu
              operator: Exists
              effect: NoSchedule
            - key: karpenter.sh/disruption
              operator: Exists
              effect: NoSchedule
          containers:
            - name: pytorch
              image: 123456789012.dkr.ecr.us-east-1.amazonaws.com/churn-training:v3
              resources:
                limits:
                  nvidia.com/gpu: 4
                  memory: "64Gi"
```

#### Karpenter Provisioner for GPU Training

```yaml
apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: gpu-training-spot
spec:
  template:
    spec:
      requirements:
        - key: karpenter.k8s.aws/instance-category
          operator: In
          values: ["g", "p"]
        - key: karpenter.k8s.aws/instance-family
          operator: In
          values: ["g5", "p4d"]
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["spot"]
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]
      nodeClassRef:
        name: gpu-training
  disruption:
    consolidationPolicy: WhenEmpty
    consolidateAfter: 5m
  limits:
    resources:
      nvidia.com/gpu: "64"  # Max 64 GPUs in cluster
---
apiVersion: karpenter.k8s.aws/v1beta1
kind: EC2NodeClass
metadata:
  name: gpu-training
spec:
  amiFamily: AL2
  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: ml-cluster
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: ml-cluster
  blockDeviceMappings:
    - deviceName: /dev/xvda
      ebs:
        volumeSize: 500Gi  # Large for dataset caching
        volumeType: gp3
        iops: 10000
        throughput: 500
  instanceProfile: "KarpenterNodeInstanceProfile-ml-cluster"
```

#### Ray Train (simpler than Kubeflow, better for experimentation)

```python
# ray_training.py - Submit to Ray on EKS
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig

ray.init(address="ray://ray-head.ml-training.svc.cluster.local:10001")

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=ScalingConfig(
        num_workers=4,
        use_gpu=True,
        resources_per_worker={"GPU": 1, "CPU": 8}
    ),
    run_config=RunConfig(
        name="churn-training-v3",
        storage_path="s3://ml-pipeline-artifacts/ray-results/",
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="val_auc",
            checkpoint_score_order="max"
        )
    ),
    datasets={
        "train": ray.data.read_parquet("s3://company-data-lake/training/train/"),
        "val": ray.data.read_parquet("s3://company-data-lake/training/validation/")
    }
)

result = trainer.fit()
print(f"Best model AUC: {result.metrics['val_auc']}")
```

---

## 5. MLOps Pipeline

### 5.1 SageMaker Pipelines (Step-by-Step)

```python
"""
Complete SageMaker Pipeline for Churn Prediction.
This is the orchestration layer — equivalent to Step Functions for ML.
"""
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    ProcessingStep, TrainingStep, CreateModelStep,
    RegisterModel, TransformStep
)
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet

# Pipeline parameters (configurable at runtime)
input_data_uri = ParameterString(
    name="InputDataUri",
    default_value="s3://company-data-lake/curated/customers/"
)
model_approval_status = ParameterString(
    name="ModelApprovalStatus",
    default_value="PendingManualApproval"
)
auc_threshold = ParameterFloat(
    name="AucThreshold",
    default_value=0.85
)

# ============ Step 1: Data Validation ============
validation_processor = ScriptProcessor(
    command=["python3"],
    image_uri=sklearn_image_uri,
    role=sagemaker_role,
    instance_count=1,
    instance_type="ml.m5.xlarge"
)

# Property file to read validation results
validation_report = PropertyFile(
    name="ValidationReport",
    output_name="validation_output",
    path="validation_status.json"
)

step_validate = ProcessingStep(
    name="ValidateData",
    processor=validation_processor,
    code="s3://ml-pipeline-code/processing/validate_data.py",
    inputs=[
        ProcessingInput(source=input_data_uri, destination="/opt/ml/processing/input/")
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output/",
            destination="s3://ml-pipeline-artifacts/validation/",
            output_name="validation_output"
        )
    ],
    property_files=[validation_report]
)

# ============ Step 2: Feature Engineering ============
step_features = ProcessingStep(
    name="FeatureEngineering",
    processor=spark_processor,
    code="s3://ml-pipeline-code/processing/feature_engineering.py",
    inputs=[
        ProcessingInput(
            source=input_data_uri,
            destination="/opt/ml/processing/input/"
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output/train",
            destination="s3://ml-pipeline-artifacts/datasets/train/",
            output_name="train"
        ),
        ProcessingOutput(
            source="/opt/ml/processing/output/validation",
            destination="s3://ml-pipeline-artifacts/datasets/validation/",
            output_name="validation"
        ),
        ProcessingOutput(
            source="/opt/ml/processing/output/test",
            destination="s3://ml-pipeline-artifacts/datasets/test/",
            output_name="test"
        )
    ]
)

# ============ Step 3: Model Training ============
step_train = TrainingStep(
    name="TrainModel",
    estimator=xgb_estimator,  # Defined earlier with Spot + checkpointing
    inputs={
        "train": TrainingInput(
            s3_data=step_features.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="text/csv"
        ),
        "validation": TrainingInput(
            s3_data=step_features.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
            content_type="text/csv"
        )
    }
)

# ============ Step 4: Model Evaluation ============
evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)

step_evaluate = ProcessingStep(
    name="EvaluateModel",
    processor=sklearn_processor,
    code="s3://ml-pipeline-code/evaluation/evaluate_model.py",
    inputs=[
        ProcessingInput(
            source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=step_features.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test"
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/evaluation",
            destination="s3://ml-pipeline-artifacts/evaluation/",
            output_name="evaluation"
        )
    ],
    property_files=[evaluation_report]
)

# ============ Step 5: Conditional Registration ============
# Only register model if AUC >= threshold
step_register = RegisterModel(
    name="RegisterModel",
    estimator=xgb_estimator,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.m5.xlarge", "ml.c5.xlarge"],
    transform_instances=["ml.m5.4xlarge"],
    model_package_group_name="churn-prediction-models",
    approval_status=model_approval_status,
    model_metrics={
        "ModelQuality": {
            "Statistics": {
                "ContentType": "application/json",
                "S3Uri": f"s3://ml-pipeline-artifacts/evaluation/evaluation.json"
            }
        }
    }
)

# Condition: Register only if model meets quality bar
cond_register = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name=step_evaluate.name,
        property_file=evaluation_report,
        json_path="metrics.auc"
    ),
    right=auc_threshold
)

step_condition = ConditionStep(
    name="CheckModelQuality",
    conditions=[cond_register],
    if_steps=[step_register],
    else_steps=[]  # Pipeline stops without registration
)

# ============ Assemble Pipeline ============
pipeline = Pipeline(
    name="churn-prediction-pipeline",
    parameters=[input_data_uri, model_approval_status, auc_threshold],
    steps=[step_validate, step_features, step_train, step_evaluate, step_condition],
    sagemaker_session=sagemaker_session
)

# Create/update pipeline
pipeline.upsert(role_arn=sagemaker_role)

# Execute pipeline
execution = pipeline.start(
    parameters={
        "InputDataUri": "s3://company-data-lake/curated/customers/dt=2024-06-15/",
        "AucThreshold": 0.87
    }
)
```

### 5.2 Model Registry

```python
import boto3

sm_client = boto3.client("sagemaker")

# Create model package group (one-time)
sm_client.create_model_package_group(
    ModelPackageGroupName="churn-prediction-models",
    ModelPackageGroupDescription="Production churn prediction models",
    Tags=[
        {"Key": "Project", "Value": "churn-prediction"},
        {"Key": "Team", "Value": "data-science"}
    ]
)

# Approve a model version (manual gate or automated)
def approve_model(model_package_arn: str, approver: str):
    """Called by ML Engineer after reviewing model metrics."""
    sm_client.update_model_package(
        ModelPackageArn=model_package_arn,
        ModelApprovalStatus="Approved",
        ApprovalDescription=f"Approved by {approver} - AUC meets production threshold",
        CustomerMetadataProperties={
            "approved_by": approver,
            "approved_at": datetime.now().isoformat(),
            "deployment_target": "production"
        }
    )

# List model versions
def get_latest_approved_model(group_name: str) -> str:
    """Get the latest approved model for deployment."""
    response = sm_client.list_model_packages(
        ModelPackageGroupName=group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1
    )
    if response["ModelPackageSummaryList"]:
        return response["ModelPackageSummaryList"][0]["ModelPackageArn"]
    raise ValueError(f"No approved models found in {group_name}")
```

### 5.3 CI/CD for ML

#### EventBridge Rule — Trigger Deployment on Model Approval

```json
{
    "source": ["aws.sagemaker"],
    "detail-type": ["SageMaker Model Package State Change"],
    "detail": {
        "ModelPackageGroupName": ["churn-prediction-models"],
        "ModelApprovalStatus": ["Approved"]
    }
}
```

#### CodePipeline Integration

```yaml
# buildspec.yml for CodeBuild - Model Deployment Stage
version: 0.2
env:
  variables:
    MODEL_PACKAGE_GROUP: "churn-prediction-models"
    ENDPOINT_NAME: "churn-prediction-prod"
    
phases:
  install:
    commands:
      - pip install sagemaker boto3
      
  pre_build:
    commands:
      - echo "Getting latest approved model..."
      - |
        MODEL_ARN=$(python -c "
        import boto3
        client = boto3.client('sagemaker')
        response = client.list_model_packages(
            ModelPackageGroupName='${MODEL_PACKAGE_GROUP}',
            ModelApprovalStatus='Approved',
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )
        print(response['ModelPackageSummaryList'][0]['ModelPackageArn'])
        ")
      - echo "Model ARN: ${MODEL_ARN}"
      
  build:
    commands:
      - echo "Deploying model with canary traffic shifting..."
      - python deploy/deploy_endpoint.py --model-arn "${MODEL_ARN}" --endpoint "${ENDPOINT_NAME}"
      
  post_build:
    commands:
      - echo "Running smoke tests..."
      - python tests/smoke_test.py --endpoint "${ENDPOINT_NAME}"
      - echo "Deployment complete"
```

#### Step Functions for ML Deployment (more flexible than CodePipeline)

```json
{
  "Comment": "ML Model Deployment with Canary + Automated Rollback",
  "StartAt": "GetApprovedModel",
  "States": {
    "GetApprovedModel": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123456789012:function:get-latest-model",
      "Next": "CreateEndpointConfig"
    },
    "CreateEndpointConfig": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createEndpointConfig",
      "Parameters": {
        "EndpointConfigName.$": "States.Format('churn-prod-{}', $.model_version)",
        "ProductionVariants": [{
          "VariantName": "new-model",
          "ModelName.$": "$.model_name",
          "InstanceType": "ml.m5.xlarge",
          "InitialInstanceCount": 2,
          "InitialVariantWeight": 0
        }]
      },
      "Next": "UpdateEndpointCanary"
    },
    "UpdateEndpointCanary": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:updateEndpoint",
      "Parameters": {
        "EndpointName": "churn-prediction-prod",
        "EndpointConfigName.$": "$.endpoint_config_name",
        "DeploymentConfig": {
          "BlueGreenUpdatePolicy": {
            "TrafficRoutingConfiguration": {
              "Type": "CANARY",
              "CanarySize": {
                "Type": "INSTANCE_COUNT",
                "Value": 1
              },
              "WaitIntervalInSeconds": 600
            },
            "TerminationWaitInSeconds": 300,
            "MaximumExecutionTimeoutInSeconds": 3600
          },
          "AutoRollbackConfiguration": {
            "Alarms": [
              {"AlarmName": "churn-endpoint-5xx-rate"},
              {"AlarmName": "churn-endpoint-latency-p99"},
              {"AlarmName": "churn-model-quality-drift"}
            ]
          }
        }
      },
      "Next": "MonitorCanary"
    },
    "MonitorCanary": {
      "Type": "Wait",
      "Seconds": 900,
      "Next": "CheckCanaryMetrics"
    },
    "CheckCanaryMetrics": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123456789012:function:check-canary-health",
      "Next": "CanaryDecision"
    },
    "CanaryDecision": {
      "Type": "Choice",
      "Choices": [
        {
          "Variable": "$.canary_healthy",
          "BooleanEquals": true,
          "Next": "PromoteToFull"
        }
      ],
      "Default": "RollbackDeployment"
    },
    "PromoteToFull": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123456789012:function:promote-canary",
      "Next": "NotifySuccess"
    },
    "RollbackDeployment": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123456789012:function:rollback-endpoint",
      "Next": "NotifyFailure"
    },
    "NotifySuccess": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sns:publish",
      "Parameters": {
        "TopicArn": "arn:aws:sns:us-east-1:123456789012:ml-deployments",
        "Message.$": "States.Format('Model {} deployed successfully to production', $.model_version)"
      },
      "End": true
    },
    "NotifyFailure": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sns:publish",
      "Parameters": {
        "TopicArn": "arn:aws:sns:us-east-1:123456789012:ml-alerts",
        "Message.$": "States.Format('ROLLBACK: Model {} failed canary checks', $.model_version)"
      },
      "End": true
    }
  }
}
```

### 5.4 A/B Testing & Shadow Deployments

#### A/B Testing (SageMaker Production Variants)

```python
from sagemaker.model import Model

# Deploy two model versions with traffic split
model_v2 = Model(
    model_data="s3://ml-pipeline-artifacts/models/churn-v2/model.tar.gz",
    image_uri=inference_image_uri,
    role=sagemaker_role
)
model_v3 = Model(
    model_data="s3://ml-pipeline-artifacts/models/churn-v3/model.tar.gz",
    image_uri=inference_image_uri,
    role=sagemaker_role
)

# A/B test: 90% traffic to v2 (champion), 10% to v3 (challenger)
endpoint = model_v2.deploy(
    initial_instance_count=2,
    instance_type="ml.m5.xlarge",
    endpoint_name="churn-prediction-ab-test",
    data_capture_config=DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,
        capture_options=["Input", "Output"],
        destination_s3_uri="s3://ml-pipeline-artifacts/data-capture/"
    )
)

# Add challenger variant
sm_client.create_endpoint_config(
    EndpointConfigName="churn-ab-test-config",
    ProductionVariants=[
        {
            "VariantName": "champion-v2",
            "ModelName": "churn-model-v2",
            "InstanceType": "ml.m5.xlarge",
            "InitialInstanceCount": 2,
            "InitialVariantWeight": 9  # 90%
        },
        {
            "VariantName": "challenger-v3",
            "ModelName": "churn-model-v3",
            "InstanceType": "ml.m5.xlarge",
            "InitialInstanceCount": 1,
            "InitialVariantWeight": 1  # 10%
        }
    ]
)
```

#### Shadow Deployment (capture predictions without serving)

```python
"""
Shadow deployment pattern using SageMaker inference pipeline.
The shadow model receives the same requests but responses are logged, not returned.
"""
import boto3
import json
from concurrent.futures import ThreadPoolExecutor

sagemaker_runtime = boto3.client("sagemaker-runtime")
s3 = boto3.client("s3")

def invoke_with_shadow(endpoint_name: str, shadow_endpoint: str, payload: str):
    """Invoke production endpoint + shadow, return only production response."""
    
    # Production call (blocking)
    prod_response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="text/csv",
        Body=payload
    )
    
    # Shadow call (fire-and-forget via thread pool)
    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(_shadow_invoke, shadow_endpoint, payload, prod_response)
    
    return prod_response

def _shadow_invoke(shadow_endpoint, payload, prod_response):
    """Log shadow prediction for offline comparison."""
    try:
        shadow_response = sagemaker_runtime.invoke_endpoint(
            EndpointName=shadow_endpoint,
            ContentType="text/csv",
            Body=payload
        )
        
        # Log both predictions for later analysis
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "production_prediction": json.loads(prod_response["Body"].read()),
            "shadow_prediction": json.loads(shadow_response["Body"].read()),
            "input_hash": hashlib.md5(payload.encode()).hexdigest()
        }
        
        # Write to Kinesis Firehose → S3 for batch analysis
        firehose = boto3.client("firehose")
        firehose.put_record(
            DeliveryStreamName="shadow-comparison-stream",
            Record={"Data": json.dumps(comparison) + "\n"}
        )
    except Exception as e:
        # Shadow failures must never impact production
        logger.warning(f"Shadow invocation failed: {e}")
```

---

## 6. Model Deployment

### 6.1 Real-time Endpoints

#### Autoscaling Configuration

```python
import boto3

aas_client = boto3.client("application-autoscaling")

# Register scalable target
aas_client.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId=f"endpoint/{endpoint_name}/variant/{variant_name}",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=2,
    MaxCapacity=20
)

# Target tracking scaling (recommended)
aas_client.put_scaling_policy(
    PolicyName="churn-endpoint-scaling",
    ServiceNamespace="sagemaker",
    ResourceId=f"endpoint/{endpoint_name}/variant/{variant_name}",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 750.0,  # Target invocations per instance per minute
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
        },
        "ScaleInCooldown": 300,   # 5 min cool-down before scale-in
        "ScaleOutCooldown": 60    # 1 min before scale-out (be responsive)
    }
)

# Step scaling for bursty workloads (use with CloudWatch alarm)
aas_client.put_scaling_policy(
    PolicyName="churn-endpoint-step-scaling",
    ServiceNamespace="sagemaker",
    ResourceId=f"endpoint/{endpoint_name}/variant/{variant_name}",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="StepScaling",
    StepScalingPolicyConfiguration={
        "AdjustmentType": "ChangeInCapacity",
        "StepAdjustments": [
            {"MetricIntervalLowerBound": 0, "MetricIntervalUpperBound": 1000, "ScalingAdjustment": 2},
            {"MetricIntervalLowerBound": 1000, "ScalingAdjustment": 5}
        ],
        "Cooldown": 60
    }
)
```

#### Multi-Model Endpoint (cost optimization for many models)

```python
from sagemaker.multidatamodel import MultiDataModel

# Deploy 100+ models on a single endpoint
multi_model = MultiDataModel(
    name="customer-models",
    model_data_prefix="s3://ml-pipeline-artifacts/models/per-customer/",
    model=base_model,  # Base container that can load any model
    sagemaker_session=sagemaker_session
)

predictor = multi_model.deploy(
    initial_instance_count=3,
    instance_type="ml.m5.4xlarge",
    endpoint_name="customer-models-endpoint"
)

# Invoke specific model
response = sagemaker_runtime.invoke_endpoint(
    EndpointName="customer-models-endpoint",
    ContentType="text/csv",
    TargetModel="customer-segment-a/model.tar.gz",  # Dynamically loaded
    Body=payload
)
```

### 6.2 Serverless Inference

```python
from sagemaker.serverless import ServerlessInferenceConfig

# Best for: intermittent traffic, < 60 second cold start OK, < 6MB payload
serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=4096,          # 1024, 2048, 3072, 4096, 5120, or 6144
    max_concurrency=50,              # Max concurrent invocations
    provisioned_concurrency=5        # Keep 5 instances warm (costs $)
)

model.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name="churn-prediction-serverless"
)
```

**When to use Serverless Inference vs Real-time Endpoints:**

| Aspect | Serverless | Real-time Endpoint |
|--------|-----------|-------------------|
| Traffic pattern | Intermittent, unpredictable | Steady, high-volume |
| Cold start | 1-5 seconds (first request) | None (always warm) |
| Max payload | 6MB | 6MB (100MB with async) |
| Max response time | 60 seconds | No hard limit |
| GPU support | ❌ | ✅ |
| Cost at 0 TPS | Near $0 | Full instance cost |
| Cost at 1000 TPS | Expensive | Cheaper per-request |
| Autoscaling | Automatic | Manual configuration |

### 6.3 Batch Transform

```python
transformer = xgb_estimator.transformer(
    instance_count=5,
    instance_type="ml.m5.4xlarge",
    output_path="s3://ml-pipeline-artifacts/batch-predictions/",
    strategy="MultiRecord",           # Process multiple records per request
    max_concurrent_transforms=10,     # Parallel requests to container
    max_payload=6,                    # MB per request
    assemble_with="Line",
    accept="text/csv"
)

transformer.transform(
    data="s3://company-data-lake/curated/customers/dt=2024-06-15/",
    content_type="text/csv",
    split_type="Line",
    join_source="Input",              # Include input in output for joining
    wait=False
)
```

### 6.4 EKS-Based Serving

#### KServe (formerly KFServing) — Production-grade

```yaml
# inferenceservice.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: churn-prediction
  namespace: ml-serving
  annotations:
    serving.kserve.io/autoscalerClass: hpa
    serving.kserve.io/targetUtilizationPercentage: "70"
spec:
  predictor:
    minReplicas: 2
    maxReplicas: 20
    scaleTarget: 10       # Target concurrent requests per pod
    scaleMetric: concurrency
    model:
      modelFormat:
        name: xgboost
      storageUri: "s3://ml-pipeline-artifacts/models/churn-v3/"
      resources:
        limits:
          cpu: "4"
          memory: "8Gi"
        requests:
          cpu: "2"
          memory: "4Gi"
    containerConcurrency: 10
    timeout: 30
  transformer:
    containers:
      - name: feature-enrichment
        image: 123456789012.dkr.ecr.us-east-1.amazonaws.com/churn-transformer:v3
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
        env:
          - name: FEATURE_STORE_ENDPOINT
            value: "https://featurestore-runtime.us-east-1.amazonaws.com"
---
# Canary rollout with Knative
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: churn-prediction
  annotations:
    serving.kserve.io/canaryTrafficPercent: "10"
spec:
  predictor:
    canaryTrafficPercent: 10
    model:
      modelFormat:
        name: xgboost
      storageUri: "s3://ml-pipeline-artifacts/models/churn-v4/"  # New version
```

#### Ray Serve on EKS

```python
# ray_serve_deployment.py
import ray
from ray import serve
from ray.serve import Application
import xgboost as xgb
import numpy as np
import boto3

@serve.deployment(
    num_replicas=3,
    ray_actor_options={"num_cpus": 2, "memory": 4 * 1024**3},
    autoscaling_config={
        "min_replicas": 2,
        "max_replicas": 20,
        "target_ongoing_requests": 10,
        "upscale_delay_s": 10,
        "downscale_delay_s": 300
    },
    health_check_period_s=10,
    health_check_timeout_s=30
)
class ChurnPredictor:
    def __init__(self):
        # Load model from S3
        s3 = boto3.client("s3")
        s3.download_file("ml-pipeline-artifacts", "models/churn-v3/model.xgb", "/tmp/model.xgb")
        self.model = xgb.Booster()
        self.model.load_model("/tmp/model.xgb")
        
        # Feature store client
        self.featurestore = boto3.client("sagemaker-featurestore-runtime")
    
    async def __call__(self, request) -> dict:
        data = await request.json()
        customer_id = data["customer_id"]
        
        # Get features from Feature Store
        features = self._get_features(customer_id)
        
        # Predict
        dmatrix = xgb.DMatrix(np.array([features]))
        prediction = self.model.predict(dmatrix)[0]
        
        return {
            "customer_id": customer_id,
            "churn_probability": float(prediction),
            "risk_level": "high" if prediction > 0.7 else "medium" if prediction > 0.4 else "low"
        }
    
    def _get_features(self, customer_id: str) -> list:
        response = self.featurestore.get_record(
            FeatureGroupName="customer-churn-features",
            RecordIdentifierValueAsString=customer_id
        )
        return [float(r["ValueAsString"]) for r in response["Record"]]

app = ChurnPredictor.bind()
```

```yaml
# RayService on EKS
apiVersion: ray.io/v1alpha1
kind: RayService
metadata:
  name: churn-prediction-service
  namespace: ml-serving
spec:
  serveConfig:
    importPath: ray_serve_deployment:app
    runtimeEnv:
      pip:
        - xgboost==2.0.0
        - boto3
  rayClusterConfig:
    headGroupSpec:
      rayStartParams:
        dashboard-host: "0.0.0.0"
      template:
        spec:
          containers:
            - name: ray-head
              image: rayproject/ray-ml:2.9.0-py310
              resources:
                limits:
                  cpu: "4"
                  memory: "8Gi"
    workerGroupSpecs:
      - replicas: 3
        minReplicas: 2
        maxReplicas: 20
        groupName: serve-workers
        rayStartParams: {}
        template:
          spec:
            containers:
              - name: ray-worker
                image: rayproject/ray-ml:2.9.0-py310
                resources:
                  limits:
                    cpu: "4"
                    memory: "8Gi"
```

---

## 7. Monitoring & Governance

### 7.1 Model Monitor

#### Data Quality Monitoring

```python
from sagemaker.model_monitor import (
    DefaultModelMonitor,
    DataCaptureConfig,
    CronExpressionGenerator
)

# Step 1: Enable data capture on endpoint
data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,
    capture_options=["Input", "Output"],
    destination_s3_uri="s3://ml-pipeline-artifacts/data-capture/churn-endpoint/",
    kms_key_id="alias/ml-encryption-key"
)

# Step 2: Create baseline from training data
monitor = DefaultModelMonitor(
    role=sagemaker_role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size_in_gb=100,
    max_runtime_in_seconds=3600
)

monitor.suggest_baseline(
    baseline_dataset="s3://company-data-lake/training/experiment=churn-v3/train/train.csv",
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri="s3://ml-pipeline-artifacts/monitoring/baselines/data-quality/",
    wait=True
)

# Step 3: Schedule monitoring job
monitor.create_monitoring_schedule(
    monitor_schedule_name="churn-data-quality-monitor",
    endpoint_input=endpoint_name,
    output_s3_uri="s3://ml-pipeline-artifacts/monitoring/reports/data-quality/",
    statistics=monitor.baseline_statistics(),
    constraints=monitor.suggested_constraints(),
    schedule_cron_expression=CronExpressionGenerator.hourly(),
    enable_cloudwatch_metrics=True
)
```

#### Model Quality Monitoring (Prediction accuracy over time)

```python
from sagemaker.model_monitor import ModelQualityMonitor

model_quality_monitor = ModelQualityMonitor(
    role=sagemaker_role,
    instance_count=1,
    instance_type="ml.m5.xlarge"
)

# Baseline from test set predictions
model_quality_monitor.suggest_baseline(
    problem_type="BinaryClassification",
    baseline_dataset="s3://ml-pipeline-artifacts/evaluation/test_predictions.csv",
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri="s3://ml-pipeline-artifacts/monitoring/baselines/model-quality/",
    ground_truth_attribute="actual_churn",
    inference_attribute="predicted_churn",
    probability_attribute="churn_probability",
    probability_threshold_attribute=0.5
)

# Schedule with ground truth merge
# Ground truth arrives with delay (e.g., churn happens weeks later)
model_quality_monitor.create_monitoring_schedule(
    monitor_schedule_name="churn-model-quality-monitor",
    endpoint_input=endpoint_name,
    output_s3_uri="s3://ml-pipeline-artifacts/monitoring/reports/model-quality/",
    problem_type="BinaryClassification",
    ground_truth_input="s3://company-data-lake/ground-truth/churn/",
    constraints=model_quality_monitor.suggested_constraints(),
    schedule_cron_expression=CronExpressionGenerator.daily()
)
```

#### Bias Monitoring (SageMaker Clarify)

```python
from sagemaker.clarify import (
    BiasConfig, DataConfig, ModelConfig, ModelPredictedLabelConfig
)

bias_config = BiasConfig(
    label_values_or_threshold=[1],
    facet_name="contract_type",  # Monitor bias across contract types
    facet_values_or_threshold=["Month-to-month"],
    group_name="age_group"
)

# Schedule Clarify bias monitoring
clarify_monitor.create_monitoring_schedule(
    monitor_schedule_name="churn-bias-monitor",
    endpoint_input=endpoint_name,
    output_s3_uri="s3://ml-pipeline-artifacts/monitoring/reports/bias/",
    schedule_cron_expression=CronExpressionGenerator.daily(),
    analysis_config={
        "bias_config": bias_config,
        "headers": feature_names,
        "label": "churn"
    }
)
```

### 7.2 CloudWatch Metrics for ML

```python
# Custom CloudWatch metrics for ML-specific monitoring
import boto3

cloudwatch = boto3.client("cloudwatch")

def publish_ml_metrics(predictions: list, actuals: list = None):
    """Publish custom ML metrics to CloudWatch."""
    metrics = []
    
    # Prediction distribution
    metrics.append({
        "MetricName": "PredictionMean",
        "Dimensions": [{"Name": "Endpoint", "Value": "churn-prediction-prod"}],
        "Value": np.mean(predictions),
        "Unit": "None"
    })
    metrics.append({
        "MetricName": "HighRiskPredictionRate",
        "Dimensions": [{"Name": "Endpoint", "Value": "churn-prediction-prod"}],
        "Value": sum(1 for p in predictions if p > 0.7) / len(predictions),
        "Unit": "None"
    })
    
    # Feature drift proxy (prediction distribution shift)
    metrics.append({
        "MetricName": "PredictionStdDev",
        "Dimensions": [{"Name": "Endpoint", "Value": "churn-prediction-prod"}],
        "Value": np.std(predictions),
        "Unit": "None"
    })
    
    # If ground truth available (delayed)
    if actuals:
        from sklearn.metrics import roc_auc_score, precision_score, recall_score
        metrics.extend([
            {
                "MetricName": "LiveAUC",
                "Dimensions": [{"Name": "Endpoint", "Value": "churn-prediction-prod"}],
                "Value": roc_auc_score(actuals, predictions),
                "Unit": "None"
            },
            {
                "MetricName": "LivePrecision",
                "Dimensions": [{"Name": "Endpoint", "Value": "churn-prediction-prod"}],
                "Value": precision_score(actuals, [1 if p > 0.5 else 0 for p in predictions]),
                "Unit": "None"
            }
        ])
    
    cloudwatch.put_metric_data(Namespace="ML/ChurnPrediction", MetricData=metrics)
```

#### CloudWatch Alarms for ML

```yaml
# CloudFormation for ML-specific alarms
ChurnEndpointLatencyAlarm:
  Type: AWS::CloudWatch::Alarm
  Properties:
    AlarmName: churn-endpoint-latency-p99
    MetricName: ModelLatency
    Namespace: AWS/SageMaker
    Statistic: p99
    Period: 300
    EvaluationPeriods: 3
    Threshold: 200000  # 200ms in microseconds
    ComparisonOperator: GreaterThanThreshold
    Dimensions:
      - Name: EndpointName
        Value: churn-prediction-prod
      - Name: VariantName
        Value: AllTraffic
    AlarmActions:
      - !Ref MLAlertsTopic

ChurnPredictionDriftAlarm:
  Type: AWS::CloudWatch::Alarm
  Properties:
    AlarmName: churn-prediction-distribution-drift
    MetricName: PredictionMean
    Namespace: ML/ChurnPrediction
    Statistic: Average
    Period: 3600
    EvaluationPeriods: 6  # 6 hours of drift
    Threshold: 0.25       # Mean prediction > 25% indicates drift
    ComparisonOperator: GreaterThanThreshold
    AlarmActions:
      - !Ref MLAlertsTopic
      - !Ref RetrainingTriggerTopic  # Auto-trigger retraining
```

### 7.3 Automated Retraining Triggers

```python
"""
EventBridge rules for automated retraining.
Triggers: data drift detected, model quality degradation, scheduled.
"""

# EventBridge Rule: Trigger retraining on drift detection
eventbridge_rule = {
    "Name": "churn-model-retrain-on-drift",
    "Description": "Trigger retraining when Model Monitor detects drift",
    "EventPattern": json.dumps({
        "source": ["aws.sagemaker"],
        "detail-type": ["SageMaker Model Monitor Status Change"],
        "detail": {
            "MonitoringScheduleName": ["churn-data-quality-monitor"],
            "MonitoringScheduleStatus": ["Failed"]  # "Failed" = violations detected
        }
    }),
    "Targets": [{
        "Id": "retrain-step-function",
        "Arn": "arn:aws:states:us-east-1:123456789012:stateMachine:churn-retrain-pipeline",
        "Input": json.dumps({
            "trigger": "drift_detected",
            "source": "model_monitor"
        })
    }]
}
```

#### Complete Retraining Logic (Lambda)

```python
"""
Lambda: Evaluate whether retraining is needed and trigger pipeline.
Invoked by EventBridge on drift alarm or CloudWatch alarm.
"""
import boto3
import json
from datetime import datetime, timedelta

sm_client = boto3.client("sagemaker")
sfn_client = boto3.client("stepfunctions")

def lambda_handler(event, context):
    trigger_source = event.get("trigger", "unknown")
    
    # Guard: Don't retrain too frequently
    last_training = get_last_training_time()
    if last_training and (datetime.now() - last_training) < timedelta(hours=6):
        print(f"Skipping retrain — last training was {last_training}")
        return {"action": "skipped", "reason": "cooldown_period"}
    
    # Guard: Check if there's sufficient new data
    new_data_volume = check_new_data_volume()
    if new_data_volume < 10000:  # Minimum records for meaningful retrain
        print(f"Skipping retrain — only {new_data_volume} new records")
        return {"action": "skipped", "reason": "insufficient_data"}
    
    # Trigger SageMaker Pipeline
    response = sm_client.start_pipeline_execution(
        PipelineName="churn-prediction-pipeline",
        PipelineExecutionDisplayName=f"retrain-{trigger_source}-{datetime.now().strftime('%Y%m%d-%H%M')}",
        PipelineParameters=[
            {"Name": "InputDataUri", "Value": "s3://company-data-lake/curated/customers/"},
            {"Name": "ModelApprovalStatus", "Value": "PendingManualApproval"},
            {"Name": "AucThreshold", "Value": "0.87"}
        ]
    )
    
    return {
        "action": "retrain_triggered",
        "pipeline_execution_arn": response["PipelineExecutionArn"],
        "trigger": trigger_source
    }
```

### 7.4 Model Cards & Lineage Tracking

```python
from sagemaker.model_card import (
    ModelCard, ModelOverview, TrainingDetails, 
    EvaluationDetails, IntendedUses, AdditionalInformation
)

model_card = ModelCard(
    name="churn-prediction-v3",
    status="Draft",  # Draft → PendingReview → Approved → Archived
    model_overview=ModelOverview(
        model_description="XGBoost binary classifier predicting 30-day customer churn",
        model_creator="Data Science Team",
        model_owner="ML Platform Team",
        algorithm_type="XGBoost",
        problem_type="Binary Classification"
    ),
    intended_uses=IntendedUses(
        purpose_of_model="Identify customers at risk of churning to enable proactive retention",
        intended_uses="Real-time scoring in customer engagement platform",
        factors_affecting_model_efficiency="Model performs best on customers with >30 days tenure",
        risk_rating="Medium",
        explanations_for_risk_rating="False positives may waste retention budget; false negatives miss at-risk customers"
    ),
    training_details=TrainingDetails(
        objective_function={"function": "binary:logistic", "notes": "With class weight 7.5x"},
        training_observations="Model shows slight bias toward month-to-month contracts",
        training_job_details={
            "training_arn": training_job_arn,
            "training_datasets": ["s3://company-data-lake/training/experiment=churn-v3/"],
            "training_environment": {
                "container_image": xgb_image_uri,
                "instance_type": "ml.m5.4xlarge"
            },
            "hyperparameters": best_hyperparameters
        }
    ),
    evaluation_details=[
        EvaluationDetails(
            name="Test Set Evaluation",
            evaluation_observation="AUC: 0.89, Precision@0.5: 0.78, Recall@0.5: 0.72",
            datasets=["s3://company-data-lake/training/experiment=churn-v3/test/"],
            metric_groups=[
                {"name": "Classification Metrics", "value": [
                    {"name": "AUC", "type": "number", "value": 0.89},
                    {"name": "Precision", "type": "number", "value": 0.78},
                    {"name": "Recall", "type": "number", "value": 0.72},
                    {"name": "F1", "type": "number", "value": 0.75}
                ]}
            ]
        )
    ]
)

model_card.create()
```

#### Lineage Tracking (Automatic with SageMaker)

```python
from sagemaker.lineage.context import Context
from sagemaker.lineage.artifact import Artifact
from sagemaker.lineage.association import Association

# SageMaker automatically tracks lineage for:
# - Training jobs → Model artifacts
# - Processing jobs → Datasets
# - Endpoints → Model packages

# Query lineage: "What data was this model trained on?"
from sagemaker.lineage.query import LineageQuery

query = LineageQuery(sagemaker_session)
results = query.query(
    start_arns=[model_artifact_arn],
    direction="Ascendants",  # Go backward to find inputs
    include_edges=True
)

for vertex in results.vertices:
    print(f"{vertex.arn} ({vertex.lineage_type})")
```

---

## 8. Cost Optimization

### 8.1 Instance Selection Guide

| Instance | GPU | Use Case | $/hr (On-Demand) | When to Use |
|----------|-----|----------|-------------------|-------------|
| **ml.m5.xlarge** | None | XGBoost, sklearn, small DL | $0.23 | Tabular data, < 10GB |
| **ml.c5.4xlarge** | None | CPU-intensive preprocessing | $0.68 | Feature engineering |
| **ml.g5.xlarge** | 1x A10G (24GB) | Single-GPU training/inference | $1.41 | Fine-tuning, small models |
| **ml.g5.12xlarge** | 4x A10G (96GB) | Multi-GPU DL training | $7.09 | Medium DL models |
| **ml.p4d.24xlarge** | 8x A100 (320GB) | Large model training | $32.77 | LLM fine-tuning, distributed |
| **ml.p5.48xlarge** | 8x H100 (640GB) | Largest models | $98.32 | Foundation model training |
| **ml.trn1.32xlarge** | 16x Trainium | Training (AWS custom) | $21.50 | 30-50% cheaper than P4d for supported ops |
| **ml.inf2.xlarge** | 1x Inferentia2 | Inference only | $0.76 | 4x better $/perf than GPU for inference |
| **ml.inf2.48xlarge** | 12x Inferentia2 | Large model inference | $12.98 | LLM serving at scale |

### Decision Matrix

```
Training Decision:
├── Tabular data (XGBoost, LightGBM, sklearn)?
│   └── ml.m5.xlarge → ml.m5.4xlarge (NO GPU needed)
├── Deep Learning, model fits in 24GB?
│   └── ml.g5.xlarge (Spot: ~$0.42/hr)
├── Deep Learning, needs multi-GPU?
│   └── ml.g5.12xlarge or ml.p4d.24xlarge (Spot: ~$9.83/hr)
├── LLM fine-tuning (7B-70B parameters)?
│   └── ml.p4d.24xlarge × 1-4 instances
└── LLM pre-training / 70B+ models?
    └── ml.p5.48xlarge × 4-32 instances or ml.trn1.32xlarge

Inference Decision:
├── < 100 TPS, latency not critical (> 100ms OK)?
│   └── Serverless Inference (pay per request)
├── 100-1000 TPS, p99 < 50ms?
│   └── ml.c5.xlarge or ml.m5.xlarge (CPU is fine for XGBoost)
├── GPU-accelerated model (PyTorch/TF)?
│   └── ml.g5.xlarge or ml.inf2.xlarge (Inferentia 4x cheaper)
├── LLM serving (< 13B)?
│   └── ml.inf2.xlarge - ml.inf2.8xlarge
└── LLM serving (> 13B)?
    └── ml.inf2.48xlarge or ml.p4d.24xlarge
```

### 8.2 Spot Instances for Training

```python
# Spot savings by instance type (typical):
# ml.m5.xlarge:    ~60% savings
# ml.g5.xlarge:    ~70% savings  
# ml.p4d.24xlarge: ~70% savings
# ml.p5.48xlarge:  ~60% savings (less available)

# Best practices:
# 1. Always enable checkpointing
# 2. Set max_wait = 2 × max_run (allows for 1 interruption)
# 3. Use max_retry_attempts=2
# 4. Monitor spot interruption rate by instance type

estimator = Estimator(
    # ...
    use_spot_instances=True,
    max_run=7200,              # 2 hour job
    max_wait=14400,            # 4 hour max (including spot waits)
    max_retry_attempts=2,      # Retry on interruption
    checkpoint_s3_uri=f"s3://ml-pipeline-artifacts/checkpoints/{job_name}/"
)
```

### 8.3 Savings Plans for Inference

```python
# SageMaker Savings Plans:
# - 1-year commitment: ~20% discount
# - 3-year commitment: ~45% discount
# - Applies to: Inference (endpoints), Notebooks, Training
# - Measured in: $/hr commitment

# Example calculation:
# Production endpoint: 3x ml.m5.xlarge running 24/7
# On-Demand: 3 × $0.23 × 24 × 30 = $496/month
# 1-year SP:  3 × $0.184 × 24 × 30 = $397/month (20% savings)
# 3-year SP:  3 × $0.127 × 24 × 30 = $274/month (45% savings)

# Recommendation: 
# Commit to your baseline (minimum always-on capacity)
# Let autoscaling handle peaks with On-Demand
```

### 8.4 Right-sizing Inference Endpoints

```python
"""
Endpoint right-sizing script.
Run this monthly to identify over-provisioned endpoints.
"""
import boto3
from datetime import datetime, timedelta

cloudwatch = boto3.client("cloudwatch")
sm_client = boto3.client("sagemaker")

def analyze_endpoint_utilization(endpoint_name: str, days: int = 7):
    """Analyze endpoint utilization and recommend sizing."""
    
    metrics = {}
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # Get invocation metrics
    for metric_name in ["Invocations", "ModelLatency", "OverheadLatency", "CPUUtilization", "MemoryUtilization"]:
        response = cloudwatch.get_metric_statistics(
            Namespace="AWS/SageMaker" if metric_name in ["Invocations", "ModelLatency", "OverheadLatency"] else "/aws/sagemaker/Endpoints",
            MetricName=metric_name,
            Dimensions=[
                {"Name": "EndpointName", "Value": endpoint_name},
                {"Name": "VariantName", "Value": "AllTraffic"}
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,  # Hourly
            Statistics=["Average", "Maximum", "Sum"]
        )
        metrics[metric_name] = response["Datapoints"]
    
    # Analysis
    avg_cpu = np.mean([d["Average"] for d in metrics.get("CPUUtilization", [])])
    max_cpu = max([d["Maximum"] for d in metrics.get("CPUUtilization", [])], default=0)
    avg_invocations = np.mean([d["Sum"] for d in metrics.get("Invocations", [])])
    
    recommendations = []
    
    if avg_cpu < 20 and max_cpu < 50:
        recommendations.append("DOWNSIZE: CPU utilization consistently low. Consider smaller instance type.")
    if avg_cpu > 70:
        recommendations.append("UPSIZE: CPU utilization high. Consider larger instance or more replicas.")
    if avg_invocations < 10:  # < 10 requests/hour
        recommendations.append("SERVERLESS: Very low traffic. Switch to Serverless Inference.")
    
    return {
        "endpoint": endpoint_name,
        "avg_cpu_utilization": avg_cpu,
        "max_cpu_utilization": max_cpu,
        "avg_hourly_invocations": avg_invocations,
        "recommendations": recommendations
    }
```

---

## 9. CloudFormation Snippets

### 9.1 SageMaker Execution Role

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'ML Pipeline Infrastructure'

Parameters:
  Environment:
    Type: String
    AllowedValues: [dev, staging, prod]
  ProjectName:
    Type: String
    Default: churn-prediction

Resources:
  # ============ IAM ============
  SageMakerExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: !Sub '${ProjectName}-sagemaker-role-${Environment}'
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service:
                - sagemaker.amazonaws.com
                - states.amazonaws.com
                - events.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
      Policies:
        - PolicyName: MLPipelinePolicy
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - s3:GetObject
                  - s3:PutObject
                  - s3:ListBucket
                  - s3:DeleteObject
                Resource:
                  - !Sub 'arn:aws:s3:::${DataLakeBucket}'
                  - !Sub 'arn:aws:s3:::${DataLakeBucket}/*'
                  - !Sub 'arn:aws:s3:::${ArtifactsBucket}'
                  - !Sub 'arn:aws:s3:::${ArtifactsBucket}/*'
              - Effect: Allow
                Action:
                  - kms:Decrypt
                  - kms:Encrypt
                  - kms:GenerateDataKey
                Resource: !GetAtt MLEncryptionKey.Arn
              - Effect: Allow
                Action:
                  - logs:CreateLogGroup
                  - logs:CreateLogStream
                  - logs:PutLogEvents
                Resource: '*'
              - Effect: Allow
                Action:
                  - ecr:GetAuthorizationToken
                  - ecr:BatchCheckLayerAvailability
                  - ecr:GetDownloadUrlForLayer
                  - ecr:BatchGetImage
                Resource: '*'
              - Effect: Allow
                Action:
                  - cloudwatch:PutMetricData
                Resource: '*'
                Condition:
                  StringEquals:
                    cloudwatch:namespace:
                      - 'ML/ChurnPrediction'
                      - '/aws/sagemaker/Endpoints'
```

### 9.2 S3 Buckets with Lifecycle

```yaml
  DataLakeBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub '${ProjectName}-data-lake-${AWS::AccountId}-${Environment}'
      BucketEncryption:
        ServerSideEncryptionConfiguration:
          - ServerSideEncryptionByDefault:
              SSEAlgorithm: aws:kms
              KMSMasterKeyID: !Ref MLEncryptionKey
      VersioningConfiguration:
        Status: Enabled
      LifecycleConfiguration:
        Rules:
          - Id: TransitionRawToIA
            Prefix: raw/
            Status: Enabled
            Transitions:
              - TransitionInDays: 30
                StorageClass: STANDARD_IA
              - TransitionInDays: 90
                StorageClass: GLACIER
          - Id: CleanupTempData
            Prefix: temp/
            Status: Enabled
            ExpirationInDays: 7
          - Id: ExpireOldVersions
            Status: Enabled
            NoncurrentVersionExpirationInDays: 30
      NotificationConfiguration:
        EventBridgeConfiguration:
          EventBridgeEnabled: true
      Tags:
        - Key: Project
          Value: !Ref ProjectName
        - Key: Environment
          Value: !Ref Environment

  ArtifactsBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub '${ProjectName}-artifacts-${AWS::AccountId}-${Environment}'
      BucketEncryption:
        ServerSideEncryptionConfiguration:
          - ServerSideEncryptionByDefault:
              SSEAlgorithm: aws:kms
              KMSMasterKeyID: !Ref MLEncryptionKey
      LifecycleConfiguration:
        Rules:
          - Id: CleanupCheckpoints
            Prefix: checkpoints/
            Status: Enabled
            ExpirationInDays: 14
          - Id: TransitionOldModels
            Prefix: models/
            Status: Enabled
            Transitions:
              - TransitionInDays: 90
                StorageClass: STANDARD_IA
```

### 9.3 KMS Key for ML

```yaml
  MLEncryptionKey:
    Type: AWS::KMS::Key
    Properties:
      Description: !Sub 'Encryption key for ${ProjectName} ML pipeline'
      KeyPolicy:
        Version: '2012-10-17'
        Statement:
          - Sid: EnableIAMUserPermissions
            Effect: Allow
            Principal:
              AWS: !Sub 'arn:aws:iam::${AWS::AccountId}:root'
            Action: 'kms:*'
            Resource: '*'
          - Sid: AllowSageMakerUse
            Effect: Allow
            Principal:
              Service: sagemaker.amazonaws.com
            Action:
              - kms:Decrypt
              - kms:Encrypt
              - kms:GenerateDataKey
              - kms:ReEncryptFrom
              - kms:ReEncryptTo
              - kms:DescribeKey
            Resource: '*'
      Tags:
        - Key: Project
          Value: !Ref ProjectName

  MLEncryptionKeyAlias:
    Type: AWS::KMS::Alias
    Properties:
      AliasName: !Sub 'alias/${ProjectName}-ml-key-${Environment}'
      TargetKeyId: !Ref MLEncryptionKey
```

### 9.4 SageMaker Endpoint with Autoscaling

```yaml
  ChurnModel:
    Type: AWS::SageMaker::Model
    Properties:
      ModelName: !Sub '${ProjectName}-model-${Environment}'
      ExecutionRoleArn: !GetAtt SageMakerExecutionRole.Arn
      PrimaryContainer:
        Image: !Sub '683313688378.dkr.ecr.${AWS::Region}.amazonaws.com/sagemaker-xgboost:1.7-1'
        ModelDataUrl: !Sub 's3://${ArtifactsBucket}/models/latest/model.tar.gz'
        Environment:
          SAGEMAKER_PROGRAM: inference.py
      Tags:
        - Key: Project
          Value: !Ref ProjectName

  ChurnEndpointConfig:
    Type: AWS::SageMaker::EndpointConfig
    Properties:
      EndpointConfigName: !Sub '${ProjectName}-endpoint-config-${Environment}'
      ProductionVariants:
        - VariantName: AllTraffic
          ModelName: !GetAtt ChurnModel.ModelName
          InstanceType: ml.m5.xlarge
          InitialInstanceCount: 2
          InitialVariantWeight: 1.0
      DataCaptureConfig:
        EnableCapture: true
        InitialSamplingPercentage: 100
        DestinationS3Uri: !Sub 's3://${ArtifactsBucket}/data-capture/'
        CaptureOptions:
          - CaptureMode: Input
          - CaptureMode: Output
        CaptureContentTypeHeader:
          CsvContentTypes:
            - text/csv
      KmsKeyId: !GetAtt MLEncryptionKey.Arn
      Tags:
        - Key: Project
          Value: !Ref ProjectName

  ChurnEndpoint:
    Type: AWS::SageMaker::Endpoint
    Properties:
      EndpointName: !Sub '${ProjectName}-endpoint-${Environment}'
      EndpointConfigName: !GetAtt ChurnEndpointConfig.EndpointConfigName
      Tags:
        - Key: Project
          Value: !Ref ProjectName

  # Autoscaling
  EndpointScalingTarget:
    Type: AWS::ApplicationAutoScaling::ScalableTarget
    Properties:
      MaxCapacity: 20
      MinCapacity: 2
      ResourceId: !Sub 'endpoint/${ChurnEndpoint.EndpointName}/variant/AllTraffic'
      ScalableDimension: sagemaker:variant:DesiredInstanceCount
      ServiceNamespace: sagemaker
      RoleARN: !GetAtt AutoScalingRole.Arn

  EndpointScalingPolicy:
    Type: AWS::ApplicationAutoScaling::ScalingPolicy
    Properties:
      PolicyName: !Sub '${ProjectName}-endpoint-scaling'
      PolicyType: TargetTrackingScaling
      ScalingTargetId: !Ref EndpointScalingTarget
      TargetTrackingScalingPolicyConfiguration:
        TargetValue: 750
        PredefinedMetricSpecification:
          PredefinedMetricType: SageMakerVariantInvocationsPerInstance
        ScaleInCooldown: 300
        ScaleOutCooldown: 60
```

### 9.5 EventBridge Rules for ML Pipeline

```yaml
  # Trigger retraining on data arrival
  DataArrivalRule:
    Type: AWS::Events::Rule
    Properties:
      Name: !Sub '${ProjectName}-data-arrival-${Environment}'
      Description: 'Trigger ML pipeline when new data arrives'
      EventPattern:
        source:
          - aws.s3
        detail-type:
          - Object Created
        detail:
          bucket:
            name:
              - !Ref DataLakeBucket
          object:
            key:
              - prefix: 'curated/customers/'
      Targets:
        - Id: TriggerPipeline
          Arn: !GetAtt RetrainingStateMachine.Arn
          RoleArn: !GetAtt EventBridgeRole.Arn
          InputTransformer:
            InputPathsMap:
              bucket: '$.detail.bucket.name'
              key: '$.detail.object.key'
            InputTemplate: |
              {
                "trigger": "data_arrival",
                "source_bucket": "<bucket>",
                "source_key": "<key>"
              }

  # Trigger retraining on model drift
  ModelDriftRule:
    Type: AWS::Events::Rule
    Properties:
      Name: !Sub '${ProjectName}-model-drift-${Environment}'
      Description: 'Trigger retraining when Model Monitor detects drift'
      EventPattern:
        source:
          - aws.sagemaker
        detail-type:
          - SageMaker Model Monitor Status Change
        detail:
          MonitoringScheduleName:
            - prefix: !Sub '${ProjectName}'
      Targets:
        - Id: TriggerRetraining
          Arn: !GetAtt RetrainingLambda.Arn

  # Schedule weekly retraining
  WeeklyRetrainingRule:
    Type: AWS::Events::Rule
    Properties:
      Name: !Sub '${ProjectName}-weekly-retrain-${Environment}'
      Description: 'Weekly scheduled retraining'
      ScheduleExpression: 'cron(0 6 ? * MON *)'  # Monday 6 AM UTC
      Targets:
        - Id: WeeklyRetrain
          Arn: !GetAtt RetrainingStateMachine.Arn
          RoleArn: !GetAtt EventBridgeRole.Arn
          Input: '{"trigger": "scheduled", "schedule": "weekly"}'

  # Model approval → deploy
  ModelApprovalRule:
    Type: AWS::Events::Rule
    Properties:
      Name: !Sub '${ProjectName}-model-approved-${Environment}'
      Description: 'Deploy model when approved in Model Registry'
      EventPattern:
        source:
          - aws.sagemaker
        detail-type:
          - SageMaker Model Package State Change
        detail:
          ModelPackageGroupName:
            - !Sub '${ProjectName}-models'
          ModelApprovalStatus:
            - Approved
      Targets:
        - Id: DeployModel
          Arn: !GetAtt DeploymentStateMachine.Arn
          RoleArn: !GetAtt EventBridgeRole.Arn
```

### 9.6 Model Monitoring Schedule

```yaml
  DataQualityMonitoringSchedule:
    Type: AWS::SageMaker::MonitoringSchedule
    Properties:
      MonitoringScheduleName: !Sub '${ProjectName}-data-quality-${Environment}'
      MonitoringScheduleConfig:
        MonitoringJobDefinition:
          MonitoringAppSpecification:
            ImageUri: !Sub '156813124566.dkr.ecr.${AWS::Region}.amazonaws.com/sagemaker-model-monitor-analyzer'
          MonitoringInputs:
            - EndpointInput:
                EndpointName: !GetAtt ChurnEndpoint.EndpointName
                LocalPath: /opt/ml/processing/input/endpoint
          MonitoringOutputConfig:
            MonitoringOutputs:
              - S3Output:
                  S3Uri: !Sub 's3://${ArtifactsBucket}/monitoring/data-quality/'
                  LocalPath: /opt/ml/processing/output
          MonitoringResources:
            ClusterConfig:
              InstanceCount: 1
              InstanceType: ml.m5.xlarge
              VolumeSizeInGB: 50
          RoleArn: !GetAtt SageMakerExecutionRole.Arn
          BaselineConfig:
            ConstraintsResource:
              S3Uri: !Sub 's3://${ArtifactsBucket}/monitoring/baselines/constraints.json'
            StatisticsResource:
              S3Uri: !Sub 's3://${ArtifactsBucket}/monitoring/baselines/statistics.json'
        ScheduleConfig:
          ScheduleExpression: 'cron(0 * * * ? *)'  # Hourly
      EndpointName: !GetAtt ChurnEndpoint.EndpointName
      Tags:
        - Key: Project
          Value: !Ref ProjectName
```

### 9.7 Feature Store Feature Group

```yaml
  CustomerChurnFeatureGroup:
    Type: AWS::SageMaker::FeatureGroup
    Properties:
      FeatureGroupName: !Sub '${ProjectName}-features-${Environment}'
      RecordIdentifierFeatureName: customer_id
      EventTimeFeatureName: event_time
      OnlineStoreConfig:
        EnableOnlineStore: true
        SecurityConfig:
          KmsKeyId: !GetAtt MLEncryptionKey.Arn
      OfflineStoreConfig:
        S3StorageConfig:
          S3Uri: !Sub 's3://${DataLakeBucket}/features/'
          KmsKeyId: !GetAtt MLEncryptionKey.Arn
        DisableGlueTableCreation: false
      FeatureDefinitions:
        - FeatureName: customer_id
          FeatureType: String
        - FeatureName: event_time
          FeatureType: Fractional
        - FeatureName: tenure_months
          FeatureType: Integral
        - FeatureName: monthly_charges
          FeatureType: Fractional
        - FeatureName: total_charges
          FeatureType: Fractional
        - FeatureName: event_count_30d
          FeatureType: Integral
        - FeatureName: support_tickets_30d
          FeatureType: Integral
        - FeatureName: engagement_trend
          FeatureType: Fractional
        - FeatureName: days_since_last_activity
          FeatureType: Integral
      RoleArn: !GetAtt SageMakerExecutionRole.Arn
      Tags:
        - Key: Project
          Value: !Ref ProjectName
```

### 9.8 Complete Outputs

```yaml
Outputs:
  SageMakerRoleArn:
    Description: SageMaker Execution Role ARN
    Value: !GetAtt SageMakerExecutionRole.Arn
    Export:
      Name: !Sub '${ProjectName}-${Environment}-SageMakerRoleArn'
  
  EndpointName:
    Description: SageMaker Endpoint Name
    Value: !GetAtt ChurnEndpoint.EndpointName
    Export:
      Name: !Sub '${ProjectName}-${Environment}-EndpointName'
  
  DataLakeBucketName:
    Description: Data Lake S3 Bucket
    Value: !Ref DataLakeBucket
    Export:
      Name: !Sub '${ProjectName}-${Environment}-DataLakeBucket'
  
  FeatureGroupName:
    Description: Feature Store Feature Group
    Value: !Sub '${ProjectName}-features-${Environment}'
    Export:
      Name: !Sub '${ProjectName}-${Environment}-FeatureGroup'
  
  MonitoringSchedule:
    Description: Model Monitoring Schedule
    Value: !Sub '${ProjectName}-data-quality-${Environment}'
```

---

## 10. End-to-End Project: Customer Churn Prediction

### Project Overview

Build a production customer churn prediction system that:
1. Ingests data from multiple sources (CRM, events, billing)
2. Engineers features with time-windowed aggregations
3. Trains an XGBoost model with hyperparameter tuning
4. Deploys with canary rollout and auto-rollback
5. Monitors for drift and auto-retrains
6. Serves predictions at < 50ms p99

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CHURN PREDICTION SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐   ┌──────────────┐   ┌───────────────┐                   │
│  │ CRM System  │──▶│ Kinesis      │──▶│ Firehose      │──▶ S3 raw/       │
│  │ (events)    │   │ Data Stream  │   │ (buffered)    │                    │
│  └─────────────┘   └──────────────┘   └───────────────┘                   │
│                                                                             │
│  ┌─────────────┐   ┌──────────────┐                                       │
│  │ Billing DB  │──▶│ DMS/Glue ETL │──────────────────────▶ S3 curated/   │
│  │ (daily)     │   └──────────────┘                                       │
│  └─────────────┘                                                           │
│                                                                             │
│  ──── TRAINING PIPELINE (Weekly + On-Drift) ────────────────────────────   │
│                                                                             │
│  EventBridge ──▶ SageMaker Pipeline:                                       │
│    ┌────────────┐   ┌────────────┐   ┌──────────┐   ┌────────────┐       │
│    │ Validate   │──▶│ Engineer   │──▶│  Train   │──▶│  Evaluate  │       │
│    │ Data       │   │ Features   │   │  (Spot)  │   │  Model     │       │
│    └────────────┘   └────────────┘   └──────────┘   └────────────┘       │
│                          │                                  │              │
│                          ▼                                  ▼              │
│                    Feature Store              Model Registry (Approval)    │
│                                                             │              │
│  ──── INFERENCE PIPELINE (Real-time) ───────────────────────│────────────  │
│                                                             ▼              │
│  API GW ──▶ Lambda ──▶ Feature Store ──▶ SageMaker Endpoint            │
│             (enrich)    (online, 5ms)    (autoscaled, 2-20 instances)     │
│                                                  │                        │
│                                                  ▼                        │
│                                          Data Capture ──▶ S3             │
│                                                             │             │
│  ──── MONITORING ────────────────────────────────────────────────────────  │
│                                                             │             │
│  Model Monitor (hourly) ──▶ Drift Detected? ──▶ EventBridge ──▶ Retrain │
│  CloudWatch Alarms ──▶ Auto-rollback on latency/error spike             │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Step 1: Project Setup

```bash
# Project structure
churn-prediction/
├── infrastructure/
│   ├── main.yaml              # CloudFormation (from Section 9)
│   ├── parameters/
│   │   ├── dev.json
│   │   ├── staging.json
│   │   └── prod.json
│   └── scripts/
│       └── deploy.sh
├── src/
│   ├── processing/
│   │   ├── validate_data.py
│   │   ├── feature_engineering.py
│   │   └── prepare_training_data.py
│   ├── training/
│   │   ├── train.py
│   │   └── hyperparameters.json
│   ├── evaluation/
│   │   └── evaluate_model.py
│   ├── inference/
│   │   ├── inference.py
│   │   └── Dockerfile
│   └── monitoring/
│       ├── check_drift.py
│       └── publish_metrics.py
├── pipelines/
│   ├── training_pipeline.py    # SageMaker Pipeline definition
│   └── deployment_pipeline.json # Step Functions definition
├── tests/
│   ├── unit/
│   ├── integration/
│   └── smoke_test.py
└── notebooks/
    ├── 01_exploration.ipynb
    ├── 02_feature_engineering.ipynb
    └── 03_model_development.ipynb
```

### Step 2: Data Pipeline

```python
# src/processing/prepare_training_data.py
"""
Prepares training dataset with proper train/val/test split.
Ensures no data leakage (temporal split, not random).
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import os

def temporal_split(df, date_col="event_time", train_ratio=0.7, val_ratio=0.15):
    """Split by time to prevent look-ahead bias."""
    df = df.sort_values(date_col)
    n = len(df)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    return train, val, test

def prepare_features(df):
    """Final feature preparation for training."""
    # Drop ID columns and raw timestamps
    feature_cols = [c for c in df.columns if c not in [
        "customer_id", "event_time", "churn"
    ]]
    
    # Handle categoricals
    categorical_cols = df[feature_cols].select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Handle nulls (tree models handle NaN, but explicit is better)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    
    # Load features from Feature Store offline
    input_dir = args.input_path
    df = pd.read_parquet(input_dir)
    
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    print(f"Churn rate: {df['churn'].mean():.3f}")
    
    # Prepare features
    df = prepare_features(df)
    
    # Temporal split
    train, val, test = temporal_split(df)
    
    print(f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Save as CSV (XGBoost format: label as first column, no header)
    output_dir = args.output_path
    for name, data in [("train", train), ("validation", val), ("test", test)]:
        # XGBoost expects: label, feature1, feature2, ...
        cols = ["churn"] + [c for c in data.columns if c != "churn"]
        path = os.path.join(output_dir, name, "data.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data[cols].to_csv(path, index=False, header=False)
        print(f"Saved {name}: {path} ({len(data)} records)")

if __name__ == "__main__":
    main()
```

### Step 3: Model Evaluation

```python
# src/evaluation/evaluate_model.py
"""
Comprehensive model evaluation for pipeline gate decision.
Outputs metrics as JSON property file for SageMaker Pipeline condition step.
"""
import json
import os
import tarfile
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, f1_score,
    confusion_matrix, classification_report, 
    average_precision_score, brier_score_loss
)

MODEL_DIR = "/opt/ml/processing/model"
TEST_DIR = "/opt/ml/processing/test"
OUTPUT_DIR = "/opt/ml/processing/evaluation"

def load_model():
    """Extract and load model from tar.gz artifact."""
    model_path = os.path.join(MODEL_DIR, "model.tar.gz")
    with tarfile.open(model_path, "r:gz") as tar:
        tar.extractall(path=MODEL_DIR)
    
    model = xgb.Booster()
    model.load_model(os.path.join(MODEL_DIR, "xgboost-model"))
    return model

def load_test_data():
    """Load test dataset."""
    test_file = os.path.join(TEST_DIR, "data.csv")
    df = pd.read_csv(test_file, header=None)
    y_test = df.iloc[:, 0].values
    X_test = df.iloc[:, 1:].values
    return X_test, y_test

def evaluate(model, X_test, y_test):
    """Comprehensive evaluation metrics."""
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Core metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Precision-Recall at various thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Find optimal threshold (maximize F1)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    # Business metrics
    # Assume: cost of false negative (missed churn) = $500
    # Cost of false positive (unnecessary retention offer) = $50
    cost_fn = 500
    cost_fp = 50
    total_cost = fn * cost_fn + fp * cost_fp
    baseline_cost = sum(y_test) * cost_fn  # Cost if we predict no churn
    cost_savings = baseline_cost - total_cost
    
    report = {
        "metrics": {
            "auc": float(auc),
            "average_precision": float(avg_precision),
            "brier_score": float(brier),
            "f1": float(f1),
            "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
            "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0
        },
        "confusion_matrix": {
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        },
        "optimal_threshold": float(optimal_threshold),
        "business_metrics": {
            "cost_savings_vs_baseline": float(cost_savings),
            "total_cost_at_threshold_0.5": float(total_cost),
            "baseline_cost_no_model": float(baseline_cost),
            "roi_percentage": float(cost_savings / baseline_cost * 100) if baseline_cost > 0 else 0
        },
        "dataset_stats": {
            "test_samples": int(len(y_test)),
            "positive_rate": float(y_test.mean()),
            "prediction_mean": float(y_pred_proba.mean()),
            "prediction_std": float(y_pred_proba.std())
        }
    }
    
    return report

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model = load_model()
    X_test, y_test = load_test_data()
    
    report = evaluate(model, X_test, y_test)
    
    # Save full report
    with open(os.path.join(OUTPUT_DIR, "evaluation.json"), "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"AUC:              {report['metrics']['auc']:.4f}")
    print(f"F1:               {report['metrics']['f1']:.4f}")
    print(f"Precision:        {report['metrics']['precision']:.4f}")
    print(f"Recall:           {report['metrics']['recall']:.4f}")
    print(f"Optimal Threshold:{report['optimal_threshold']:.3f}")
    print(f"Cost Savings:     ${report['business_metrics']['cost_savings_vs_baseline']:,.0f}")
    print(f"ROI:              {report['business_metrics']['roi_percentage']:.1f}%")
    print("=" * 60)
    
    # Gate decision
    passed = report["metrics"]["auc"] >= 0.85 and report["metrics"]["recall"] >= 0.65
    print(f"\nQuality Gate: {'PASSED ✓' if passed else 'FAILED ✗'}")

if __name__ == "__main__":
    main()
```

### Step 4: Inference Container

```python
# src/inference/inference.py
"""
SageMaker inference script (entry point for serving container).
Handles: model loading, input deserialization, prediction, output serialization.
"""
import os
import json
import xgboost as xgb
import numpy as np
import logging

logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """Load model from the model directory."""
    model_path = os.path.join(model_dir, "xgboost-model")
    model = xgb.Booster()
    model.load_model(model_path)
    
    # Load model config for feature names
    config_path = os.path.join(model_dir, "model_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            model.config = json.load(f)
    
    logger.info(f"Model loaded from {model_path}")
    return model

def input_fn(request_body, request_content_type):
    """Deserialize input data."""
    if request_content_type == "text/csv":
        # CSV format: feature1,feature2,...
        lines = request_body.strip().split("\n")
        data = np.array([[float(x) for x in line.split(",")] for line in lines])
        return xgb.DMatrix(data)
    elif request_content_type == "application/json":
        # JSON format: {"instances": [[f1, f2, ...], [f1, f2, ...]]}
        payload = json.loads(request_body)
        if "instances" in payload:
            data = np.array(payload["instances"])
        else:
            data = np.array([payload["features"]])
        return xgb.DMatrix(data)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Generate predictions."""
    predictions = model.predict(input_data)
    return predictions

def output_fn(prediction, accept):
    """Serialize predictions."""
    if accept == "application/json":
        result = {
            "predictions": [
                {
                    "churn_probability": float(p),
                    "risk_level": "high" if p > 0.7 else "medium" if p > 0.4 else "low",
                    "recommended_action": get_recommendation(p)
                }
                for p in prediction
            ]
        }
        return json.dumps(result), "application/json"
    elif accept == "text/csv":
        return "\n".join([str(p) for p in prediction]), "text/csv"
    else:
        raise ValueError(f"Unsupported accept type: {accept}")

def get_recommendation(probability):
    """Business logic: map prediction to action."""
    if probability > 0.7:
        return "immediate_outreach"
    elif probability > 0.4:
        return "retention_offer"
    elif probability > 0.2:
        return "engagement_campaign"
    else:
        return "standard_service"
```

### Step 5: Complete Pipeline Definition

```python
# pipelines/training_pipeline.py
"""
Complete SageMaker Pipeline definition.
Deploy with: python training_pipeline.py --action create|update|run
"""
import argparse
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.parameters import ParameterString, ParameterFloat, ParameterInteger
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.spark.processing import PySparkProcessor
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter

def create_pipeline(region, role, default_bucket):
    """Create the complete training pipeline."""
    
    session = sagemaker.Session()
    
    # ─── Parameters ───
    input_data_uri = ParameterString(name="InputDataUri", default_value=f"s3://{default_bucket}/curated/customers/")
    instance_type_processing = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.4xlarge")
    instance_type_training = ParameterString(name="TrainingInstanceType", default_value="ml.m5.4xlarge")
    instance_count_training = ParameterInteger(name="TrainingInstanceCount", default_value=1)
    auc_threshold = ParameterFloat(name="AucThreshold", default_value=0.85)
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
    
    # ─── Step 1: Validate ───
    sklearn_image = sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1")
    
    validate_processor = ScriptProcessor(
        command=["python3"],
        image_uri=sklearn_image,
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        base_job_name="churn-validate"
    )
    
    validation_report = PropertyFile(name="ValidationReport", output_name="output", path="validation_status.json")
    
    step_validate = ProcessingStep(
        name="ValidateData",
        processor=validate_processor,
        code="src/processing/validate_data.py",
        inputs=[ProcessingInput(source=input_data_uri, destination="/opt/ml/processing/input/")],
        outputs=[ProcessingOutput(source="/opt/ml/processing/output/", output_name="output")],
        property_files=[validation_report]
    )
    
    # ─── Step 2: Feature Engineering (Spark) ───
    spark_processor = PySparkProcessor(
        base_job_name="churn-features",
        framework_version="3.3",
        role=role,
        instance_count=3,
        instance_type=instance_type_processing,
        max_runtime_in_seconds=7200
    )
    
    step_features = ProcessingStep(
        name="EngineerFeatures",
        processor=spark_processor,
        code="src/processing/feature_engineering.py",
        inputs=[ProcessingInput(source=input_data_uri, destination="/opt/ml/processing/input/")],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/output/train", output_name="train"),
            ProcessingOutput(source="/opt/ml/processing/output/validation", output_name="validation"),
            ProcessingOutput(source="/opt/ml/processing/output/test", output_name="test")
        ]
    )
    step_features.add_depends_on([step_validate])
    
    # ─── Step 3: Training (XGBoost with Spot) ───
    xgb_image = sagemaker.image_uris.retrieve("xgboost", region, version="1.7-1")
    
    estimator = Estimator(
        image_uri=xgb_image,
        role=role,
        instance_count=instance_count_training,
        instance_type=instance_type_training,
        output_path=f"s3://{default_bucket}/models/",
        base_job_name="churn-train",
        use_spot_instances=True,
        max_wait=14400,
        max_run=7200,
        checkpoint_s3_uri=f"s3://{default_bucket}/checkpoints/"
    )
    
    estimator.set_hyperparameters(
        objective="binary:logistic",
        num_round=500,
        max_depth=6,
        eta=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        scale_pos_weight=7.5,
        early_stopping_rounds=20
    )
    
    step_train = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_features.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv"
            ),
            "validation": TrainingInput(
                s3_data=step_features.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv"
            )
        }
    )
    
    # ─── Step 4: Evaluate ───
    eval_processor = ScriptProcessor(
        command=["python3"],
        image_uri=sklearn_image,
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        base_job_name="churn-evaluate"
    )
    
    evaluation_report = PropertyFile(name="EvalReport", output_name="evaluation", path="evaluation.json")
    
    step_evaluate = ProcessingStep(
        name="EvaluateModel",
        processor=eval_processor,
        code="src/evaluation/evaluate_model.py",
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(
                source=step_features.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test"
            )
        ],
        outputs=[ProcessingOutput(source="/opt/ml/processing/evaluation", output_name="evaluation")],
        property_files=[evaluation_report]
    )
    
    # ─── Step 5: Conditional Registration ───
    step_register = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv", "application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.xlarge", "ml.c5.xlarge", "ml.m5.2xlarge"],
        transform_instances=["ml.m5.4xlarge"],
        model_package_group_name="churn-prediction-models",
        approval_status=model_approval_status
    )
    
    cond_quality_check = ConditionGreaterThanOrEqualTo(
        left=JsonGet(step_name=step_evaluate.name, property_file=evaluation_report, json_path="metrics.auc"),
        right=auc_threshold
    )
    
    step_condition = ConditionStep(
        name="QualityGate",
        conditions=[cond_quality_check],
        if_steps=[step_register],
        else_steps=[]
    )
    
    # ─── Assemble ───
    pipeline = Pipeline(
        name="churn-prediction-pipeline",
        parameters=[input_data_uri, instance_type_processing, instance_type_training,
                   instance_count_training, auc_threshold, model_approval_status],
        steps=[step_validate, step_features, step_train, step_evaluate, step_condition],
        sagemaker_session=session
    )
    
    return pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["create", "update", "run"], required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--role", required=True)
    parser.add_argument("--bucket", required=True)
    args = parser.parse_args()
    
    pipeline = create_pipeline(args.region, args.role, args.bucket)
    
    if args.action in ["create", "update"]:
        pipeline.upsert(role_arn=args.role)
        print(f"Pipeline {'created' if args.action == 'create' else 'updated'}")
    elif args.action == "run":
        execution = pipeline.start()
        print(f"Pipeline execution started: {execution.arn}")

if __name__ == "__main__":
    main()
```

### Step 6: Deployment Script

```python
# deploy/deploy_endpoint.py
"""
Deploy or update SageMaker endpoint with blue/green canary strategy.
Called by CI/CD pipeline after model approval.
"""
import boto3
import argparse
import time
import json
from datetime import datetime

sm_client = boto3.client("sagemaker")
cw_client = boto3.client("cloudwatch")

def deploy_model(model_package_arn: str, endpoint_name: str, instance_type: str = "ml.m5.xlarge"):
    """Deploy approved model with canary rollout."""
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"churn-model-{timestamp}"
    config_name = f"churn-config-{timestamp}"
    
    # Create model from package
    sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=SAGEMAKER_ROLE,
        Containers=[{
            "ModelPackageName": model_package_arn
        }]
    )
    
    # Create endpoint config
    sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InstanceType": instance_type,
            "InitialInstanceCount": 2,
            "InitialVariantWeight": 1.0
        }],
        DataCaptureConfig={
            "EnableCapture": True,
            "InitialSamplingPercentage": 100,
            "DestinationS3Uri": f"s3://{ARTIFACTS_BUCKET}/data-capture/{endpoint_name}/",
            "CaptureOptions": [
                {"CaptureMode": "Input"},
                {"CaptureMode": "Output"}
            ]
        }
    )
    
    # Check if endpoint exists
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_exists = True
    except sm_client.exceptions.ClientError:
        endpoint_exists = False
    
    if endpoint_exists:
        # Update with blue/green deployment
        sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
            DeploymentConfig={
                "BlueGreenUpdatePolicy": {
                    "TrafficRoutingConfiguration": {
                        "Type": "CANARY",
                        "CanarySize": {
                            "Type": "INSTANCE_COUNT",
                            "Value": 1
                        },
                        "WaitIntervalInSeconds": 600  # 10 min canary bake
                    },
                    "TerminationWaitInSeconds": 300,
                    "MaximumExecutionTimeoutInSeconds": 3600
                },
                "AutoRollbackConfiguration": {
                    "Alarms": [
                        {"AlarmName": f"{endpoint_name}-5xx-rate"},
                        {"AlarmName": f"{endpoint_name}-latency-p99"}
                    ]
                }
            }
        )
        print(f"Endpoint update initiated with canary deployment")
    else:
        # Create new endpoint
        sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        print(f"New endpoint creation initiated")
    
    # Wait for endpoint to be in service
    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=endpoint_name,
        WaiterConfig={"Delay": 30, "MaxAttempts": 120}
    )
    
    print(f"Endpoint {endpoint_name} is InService")
    return endpoint_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-arn", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    args = parser.parse_args()
    
    deploy_model(args.model_arn, args.endpoint, args.instance_type)
```

### Step 7: Smoke Test

```python
# tests/smoke_test.py
"""
Post-deployment smoke test.
Validates endpoint is responding correctly with expected latency.
"""
import boto3
import json
import time
import numpy as np
import sys

sagemaker_runtime = boto3.client("sagemaker-runtime")

def smoke_test(endpoint_name: str, n_requests: int = 100):
    """Run smoke test against deployed endpoint."""
    
    # Test payload (representative of production traffic)
    test_payloads = [
        "0.5,45.0,1200.0,50,3,0.8,2",   # Medium risk
        "0.1,10.0,100.0,5,0,1.5,0",      # Low risk
        "0.9,80.0,5000.0,200,10,0.2,15", # High risk
    ]
    
    latencies = []
    errors = []
    predictions = []
    
    for i in range(n_requests):
        payload = test_payloads[i % len(test_payloads)]
        
        start = time.time()
        try:
            response = sagemaker_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="text/csv",
                Accept="application/json",
                Body=payload
            )
            
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
            
            result = json.loads(response["Body"].read())
            predictions.append(result)
            
        except Exception as e:
            errors.append(str(e))
    
    # Validate results
    print(f"\n{'='*60}")
    print(f"SMOKE TEST RESULTS - {endpoint_name}")
    print(f"{'='*60}")
    print(f"Requests sent:     {n_requests}")
    print(f"Successful:        {len(latencies)}")
    print(f"Errors:            {len(errors)}")
    print(f"Error rate:        {len(errors)/n_requests*100:.1f}%")
    print(f"Latency p50:       {np.percentile(latencies, 50):.1f}ms")
    print(f"Latency p95:       {np.percentile(latencies, 95):.1f}ms")
    print(f"Latency p99:       {np.percentile(latencies, 99):.1f}ms")
    print(f"{'='*60}")
    
    # Assertions
    assert len(errors) == 0, f"Got {len(errors)} errors: {errors[:3]}"
    assert np.percentile(latencies, 99) < 200, f"p99 latency too high: {np.percentile(latencies, 99)}ms"
    
    # Validate prediction format
    sample = predictions[0]
    assert "predictions" in sample, "Missing 'predictions' key in response"
    assert "churn_probability" in sample["predictions"][0], "Missing churn_probability"
    assert 0 <= sample["predictions"][0]["churn_probability"] <= 1, "Probability out of range"
    
    print("\n✓ All smoke tests PASSED")
    return True

if __name__ == "__main__":
    endpoint_name = sys.argv[1] if len(sys.argv) > 1 else "churn-prediction-prod"
    success = smoke_test(endpoint_name)
    sys.exit(0 if success else 1)
```

### Step 8: Operational Runbook

```markdown
## Operational Runbook

### Daily Operations
- [ ] Check Model Monitor reports (CloudWatch Dashboard)
- [ ] Review prediction distribution (alert if mean shifts > 10%)
- [ ] Monitor endpoint latency and error rates

### Weekly Operations  
- [ ] Review A/B test results (if running)
- [ ] Check Feature Store ingestion health
- [ ] Review cost reports (Cost Explorer tag: Project=churn-prediction)
- [ ] Validate ground truth data pipeline

### On Drift Detection
1. Verify drift is real (not a monitoring false positive)
2. Check upstream data sources for schema/distribution changes
3. If real drift: approve automated retraining pipeline execution
4. If false positive: update baseline statistics

### On Deployment Failure
1. Check CloudWatch Alarms that triggered rollback
2. Inspect canary metrics in SageMaker endpoint events
3. Compare new model predictions vs champion on same test set
4. Investigate if training data quality has degraded
5. If model is good but latency is bad: check instance type/model size

### Emergency Rollback
```bash
# Immediate rollback to previous endpoint config
aws sagemaker update-endpoint \
  --endpoint-name churn-prediction-prod \
  --endpoint-config-name churn-config-PREVIOUS_TIMESTAMP

# Or via Step Functions
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:123456789012:stateMachine:churn-rollback \
  --input '{"reason": "manual_emergency_rollback"}'
```
```

---

## Quick Reference: When to Use What

| Need | SageMaker Native | EKS Native | Hybrid |
|------|-----------------|------------|--------|
| Training | SageMaker Training Jobs + Spot | Kubeflow PyTorchJob + Karpenter Spot | Train on SM, dev on EKS |
| Feature Store | SageMaker Feature Store | Redis + S3 | Feature Store (managed) |
| Serving < 100ms | SageMaker Endpoints | KServe / Ray Serve | Serve on EKS (unified mesh) |
| Serving > 1s batch | Batch Transform | Spark on EKS | Batch Transform |
| Orchestration | SageMaker Pipelines | Argo Workflows | Step Functions |
| Monitoring | Model Monitor | Prometheus + Grafana | Model Monitor + CloudWatch |
| Experiment Tracking | SageMaker Experiments | MLflow on EKS | MLflow (flexible) |
| Cost | Higher (managed tax) | Lower (DIY operations) | Balanced |

---

## Summary

This guide covers the production patterns for building ML pipelines on AWS. The key principles:

1. **Separate training from inference** — different SLAs, different scaling, different cost models.
2. **Automate everything** — EventBridge triggers, SageMaker Pipelines, Step Functions for deployment.
3. **Monitor aggressively** — Data drift, model quality, business metrics. Auto-retrain on drift.
4. **Optimize costs** — Spot for training (70% savings), right-size inference, Savings Plans for baseline.
5. **Ship with confidence** — Canary deployments, auto-rollback on CloudWatch alarms, shadow testing.
6. **Leverage what you know** — EKS + Karpenter for GPU workloads, EventBridge for orchestration, CloudFormation for IaC.
