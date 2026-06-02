# End-to-End MLOps: AWS SageMaker vs MLflow/Open-Source

> **Production-Grade Reference Guide for Cloud Architects**  
> Last Updated: June 2026 | Audience: Cloud/ML Platform Engineers with AWS & EKS expertise

---

## Table of Contents

1. [Overview & Philosophy](#1-overview--philosophy)
2. [Data Sources & Ingestion](#2-data-sources--ingestion)
3. [Data Cleaning & Labeling](#3-data-cleaning--labeling)
4. [Feature Engineering & Feature Store](#4-feature-engineering--feature-store)
5. [Model Training & Fine-Tuning](#5-model-training--fine-tuning)
6. [Model Registry & Versioning](#6-model-registry--versioning)
7. [Model Deployment](#7-model-deployment)
8. [Canary Testing & Progressive Rollout](#8-canary-testing--progressive-rollout)
9. [Monitoring & Observability](#9-monitoring--observability)
10. [Automated Retraining](#10-automated-retraining)
11. [Cost Comparison](#11-cost-comparison)
12. [Complete Pipeline Diagrams](#12-complete-pipeline-diagrams)
13. [Code Examples](#13-code-examples)
14. [Decision Matrix](#14-decision-matrix)

---

## 1. Overview & Philosophy

### SageMaker: Fully Managed, Integrated AWS Ecosystem

| Aspect | Details |
|--------|---------|
| **Philosophy** | "One service to rule them all" — tightly integrated, managed infrastructure |
| **Target** | Teams wanting fast time-to-production with minimal infra management |
| **Lock-in** | High — deeply coupled to S3, IAM, CloudWatch, ECR, VPC |
| **Pricing** | Pay-per-use (instance-hours, storage, inference requests) |
| **Strengths** | Zero infra management, built-in governance, enterprise compliance |
| **Weaknesses** | Vendor lock-in, limited customization, opaque debugging |

### MLflow + OSS: Open-Source, Portable, Cloud-Agnostic

| Aspect | Details |
|--------|---------|
| **Philosophy** | "Best-of-breed tooling" — composable, extensible, community-driven |
| **Target** | Teams with strong platform engineering, multi-cloud needs |
| **Lock-in** | Low — runs anywhere (EKS, GKE, bare metal, local) |
| **Pricing** | Infrastructure cost only (compute, storage, networking) |
| **Strengths** | Full control, portability, no vendor dependency, customizable |
| **Weaknesses** | Higher operational burden, integration complexity, more moving parts |

### Decision Quick-Reference

```
Choose SageMaker when:
  ✓ All-in on AWS (no multi-cloud requirement)
  ✓ Small ML team (< 5 ML engineers)
  ✓ Need governance/compliance out-of-box (SOC2, HIPAA, FedRAMP)
  ✓ Want to minimize platform engineering effort
  ✓ Budget allows managed service premiums

Choose MLflow/OSS when:
  ✓ Multi-cloud or hybrid strategy
  ✓ Large platform team (can absorb operational overhead)
  ✓ Need deep customization of training/serving infrastructure
  ✓ Cost-sensitive at scale (10,000+ training jobs/month)
  ✓ Existing Kubernetes (EKS) investment
  ✓ Want to avoid vendor lock-in for strategic reasons
```

---

## 2. Data Sources & Ingestion

### SageMaker Approach

```
┌─────────────────────────────────────────────────────────────────┐
│                    SageMaker Data Ingestion                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  S3 Data Lake ─────┐                                             │
│  (Parquet/CSV/JSON) │                                            │
│                     ├──→ SageMaker Data Wrangler ──→ S3 (clean)  │
│  Redshift ──────────┤    (visual data prep,                      │
│                     │     300+ transforms)                        │
│  Athena ────────────┤                                            │
│                     │                                             │
│  RDS (Aurora/PG) ───┘                                            │
│                                                                   │
│  Kinesis Data Streams ──→ Kinesis Firehose ──→ S3 (streaming)    │
│                                                                   │
│  SageMaker Feature Store ←── Offline store (S3/Glue Catalog)     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Key Services:**
- **S3**: Central data lake — all SageMaker jobs read from/write to S3
- **Athena**: Serverless SQL over S3 (Parquet/ORC preferred for cost)
- **Redshift Spectrum**: Federated queries from Redshift to S3
- **Data Wrangler**: Visual ETL with 300+ built-in transforms, export to Processing jobs
- **Kinesis**: Real-time streaming → S3 for near-real-time ML pipelines
- **Glue Catalog**: Unified metadata catalog across all sources

**CloudFormation — S3 Data Lake with Lifecycle:**

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: ML Data Lake - S3 buckets for SageMaker MLOps

Resources:
  MLDataLakeBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub '${AWS::AccountId}-ml-data-lake-${AWS::Region}'
      VersioningConfiguration:
        Status: Enabled
      LifecycleConfiguration:
        Rules:
          - Id: TransitionToIA
            Status: Enabled
            Transitions:
              - StorageClass: STANDARD_IA
                TransitionInDays: 90
              - StorageClass: GLACIER
                TransitionInDays: 365
          - Id: CleanupIncompleteUploads
            Status: Enabled
            AbortIncompleteMultipartUpload:
              DaysAfterInitiation: 7
      BucketEncryption:
        ServerSideEncryptionConfiguration:
          - ServerSideEncryptionByDefault:
              SSEAlgorithm: aws:kms
              KMSMasterKeyID: !Ref MLKMSKey
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true
      Tags:
        - Key: Project
          Value: MLOps
        - Key: Environment
          Value: Production

  MLKMSKey:
    Type: AWS::KMS::Key
    Properties:
      Description: KMS key for ML data encryption
      KeyPolicy:
        Version: '2012-10-17'
        Statement:
          - Sid: AllowRootAccount
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
              - kms:GenerateDataKey
            Resource: '*'

  MLArtifactsBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub '${AWS::AccountId}-ml-artifacts-${AWS::Region}'
      VersioningConfiguration:
        Status: Enabled
      BucketEncryption:
        ServerSideEncryptionConfiguration:
          - ServerSideEncryptionByDefault:
              SSEAlgorithm: aws:kms
              KMSMasterKeyID: !Ref MLKMSKey

Outputs:
  DataLakeBucket:
    Value: !Ref MLDataLakeBucket
    Export:
      Name: MLDataLakeBucket
  ArtifactsBucket:
    Value: !Ref MLArtifactsBucket
    Export:
      Name: MLArtifactsBucket
```

### MLflow/OSS Approach

```
┌─────────────────────────────────────────────────────────────────┐
│                    MLflow/OSS Data Ingestion                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  S3/GCS/ADLS ──────┐                                            │
│  (any cloud store)  │                                            │
│                     ├──→ Spark / Polars / Pandas ──→ Delta Lake   │
│  Databases ─────────┤    (programmatic ETL)          (versioned)  │
│  (PG, MySQL, etc.)  │                                            │
│                     │                                             │
│  Kafka/Flink ───────┘──→ Apache Iceberg ──→ Feature pipelines    │
│                                                                   │
│  DVC (Data Version Control) ──→ Track datasets in Git            │
│                                                                   │
│  Great Expectations ──→ Data contracts & validation              │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Key Tools:**
- **DVC (Data Version Control)**: Git-like versioning for datasets and models
- **Delta Lake / Apache Iceberg**: ACID transactions, time-travel, schema evolution on data lakes
- **Great Expectations**: Declarative data validation ("data contracts")
- **Spark/Polars/Pandas**: Flexible compute engines for any source
- **Kafka + Flink**: Streaming ingestion for real-time features

**DVC Setup Example:**

```bash
# Initialize DVC in your ML project
dvc init
dvc remote add -d s3store s3://my-ml-data-lake/dvc-cache

# Track a large dataset
dvc add data/training_dataset.parquet
git add data/training_dataset.parquet.dvc data/.gitignore
git commit -m "Add training dataset v1.0"
dvc push

# Version a new dataset
dvc add data/training_dataset.parquet  # after updating
git add data/training_dataset.parquet.dvc
git commit -m "Update training dataset v1.1 - added Q2 data"
dvc push
```

---

## 3. Data Cleaning & Labeling

### SageMaker Approach

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **Processing Jobs** | Scalable data cleaning | Spark, scikit-learn, custom containers; auto-scales EC2 |
| **Ground Truth** | Human labeling at scale | Built-in labeling UIs, active learning, workforce management |
| **Data Wrangler** | Visual data prep | 300+ transforms, bias detection, feature importance |
| **Clarify** | Bias detection | Pre-training bias metrics (CI, DPL, CDDL), post-training metrics |

**SageMaker Processing Job (CloudFormation):**

```yaml
  DataCleaningProcessingJob:
    Type: AWS::SageMaker::Pipeline
    Properties:
      PipelineName: data-cleaning-pipeline
      PipelineDefinition:
        PipelineDefinitionBody: !Sub |
          {
            "Version": "2020-12-01",
            "Steps": [
              {
                "Name": "DataCleaning",
                "Type": "Processing",
                "Arguments": {
                  "ProcessingResources": {
                    "ClusterConfig": {
                      "InstanceCount": 2,
                      "InstanceType": "ml.m5.4xlarge",
                      "VolumeSizeInGB": 100
                    }
                  },
                  "AppSpecification": {
                    "ImageUri": "${AWS::AccountId}.dkr.ecr.${AWS::Region}.amazonaws.com/ml-processing:latest",
                    "ContainerEntrypoint": ["python3", "/opt/ml/processing/code/clean.py"]
                  },
                  "ProcessingInputs": [
                    {
                      "InputName": "raw-data",
                      "S3Input": {
                        "S3Uri": "s3://${MLDataLakeBucket}/raw/",
                        "LocalPath": "/opt/ml/processing/input",
                        "S3DataType": "S3Prefix"
                      }
                    }
                  ],
                  "ProcessingOutputConfig": {
                    "Outputs": [
                      {
                        "OutputName": "cleaned-data",
                        "S3Output": {
                          "S3Uri": "s3://${MLDataLakeBucket}/cleaned/",
                          "LocalPath": "/opt/ml/processing/output",
                          "S3UploadMode": "EndOfJob"
                        }
                      }
                    ]
                  }
                }
              }
            ]
          }
      RoleArn: !GetAtt SageMakerExecutionRole.Arn
```

### MLflow/OSS Approach

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **Pandas/Polars/Spark** | Programmatic cleaning | Full flexibility, any transform, any scale |
| **Label Studio** | Open-source labeling | Multi-modal (text, image, audio), ML-assisted, self-hosted |
| **Snorkel** | Programmatic labeling | Labeling functions, weak supervision, no manual labels needed |
| **Great Expectations** | Data validation | Expectation suites, data docs, pipeline integration |
| **dbt** | SQL transformations | Modular SQL, testing, documentation, lineage |

**Great Expectations Data Contract Example:**

```python
import great_expectations as gx

context = gx.get_context()

# Define data source
datasource = context.data_sources.add_pandas_filesystem(
    name="ml_training_data",
    base_directory="s3://my-ml-data-lake/cleaned/"
)

# Create expectation suite (data contract)
suite = context.suites.add(gx.ExpectationSuite(name="training_data_contract"))

suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(column="target")
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="feature_age", min_value=0, max_value=150
    )
)
suite.add_expectation(
    gx.expectations.ExpectColumnDistinctValuesToBeInSet(
        column="category", value_set=["A", "B", "C", "D"]
    )
)
suite.add_expectation(
    gx.expectations.ExpectTableRowCountToBeBetween(
        min_value=10000, max_value=10000000
    )
)

# Run validation
checkpoint = context.checkpoints.add(
    gx.Checkpoint(
        name="training_data_validation",
        validations=[
            gx.CheckpointValidation(
                datasource=datasource,
                asset_name="training_features.parquet",
                expectation_suite=suite,
            )
        ]
    )
)
result = checkpoint.run()
assert result.success, f"Data contract violated: {result.to_json_dict()}"
```

---

## 4. Feature Engineering & Feature Store

### SageMaker Feature Store

```
┌───────────────────────────────────────────────────────────┐
│            SageMaker Feature Store Architecture             │
├───────────────────────────────────────────────────────────┤
│                                                             │
│  Feature Ingestion ──→ Feature Group ──→ Online Store       │
│  (PutRecord API)       (schema, TTL)     (< 10ms latency)  │
│                              │                              │
│                              └──→ Offline Store             │
│                                   (S3 + Glue Catalog)       │
│                                   (time-travel queries)     │
│                                                             │
│  Training:  OfflineStore → Point-in-time joins → Dataset    │
│  Inference: OnlineStore → GetRecord → Feature vector        │
│                                                             │
└───────────────────────────────────────────────────────────┘
```

**CloudFormation — SageMaker Feature Group:**

```yaml
  CustomerFeatureGroup:
    Type: AWS::SageMaker::FeatureGroup
    Properties:
      FeatureGroupName: customer-features-v1
      RecordIdentifierFeatureName: customer_id
      EventTimeFeatureName: event_time
      OnlineStoreConfig:
        EnableOnlineStore: true
        SecurityConfig:
          KmsKeyId: !Ref MLKMSKey
      OfflineStoreConfig:
        S3StorageConfig:
          S3Uri: !Sub 's3://${MLArtifactsBucket}/feature-store/'
          KmsKeyId: !Ref MLKMSKey
        DataCatalogConfig:
          TableName: customer_features
          Catalog: AwsDataCatalog
          Database: ml_feature_store
      FeatureDefinitions:
        - FeatureName: customer_id
          FeatureType: String
        - FeatureName: event_time
          FeatureType: String
        - FeatureName: total_spend_30d
          FeatureType: Fractional
        - FeatureName: login_count_7d
          FeatureType: Integral
        - FeatureName: churn_risk_score
          FeatureType: Fractional
        - FeatureName: segment
          FeatureType: String
      RoleArn: !GetAtt SageMakerExecutionRole.Arn
      Tags:
        - Key: Team
          Value: ml-platform
```

### Feast (Open-Source Feature Store)

```
┌───────────────────────────────────────────────────────────┐
│              Feast Feature Store Architecture               │
├───────────────────────────────────────────────────────────┤
│                                                             │
│  Feature Repo (Git) ──→ feast apply ──→ Registry (S3/PG)   │
│  (definitions,                                              │
│   transformations)                                          │
│                                                             │
│  Offline Store: S3/Redshift/BigQuery (batch features)       │
│  Online Store: Redis/DynamoDB/PostgreSQL (low-latency)      │
│                                                             │
│  Training:  feast get_historical_features() → DataFrame     │
│  Inference: feast get_online_features() → Feature vector    │
│                                                             │
│  Materialization: feast materialize (offline → online)       │
│                                                             │
└───────────────────────────────────────────────────────────┘
```

**Feast Feature Definition (feature_repo/features.py):**

```python
from datetime import timedelta
from feast import Entity, Feature, FeatureView, FileSource, ValueType
from feast.types import Float32, Int64, String

# Entity definition
customer = Entity(
    name="customer_id",
    value_type=ValueType.STRING,
    description="Unique customer identifier",
)

# Offline source (S3 Parquet)
customer_features_source = FileSource(
    path="s3://my-ml-data-lake/features/customer_features.parquet",
    timestamp_field="event_time",
    created_timestamp_column="created_at",
)

# Feature View
customer_features_view = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=7),
    schema=[
        Feature(name="total_spend_30d", dtype=Float32),
        Feature(name="login_count_7d", dtype=Int64),
        Feature(name="churn_risk_score", dtype=Float32),
        Feature(name="segment", dtype=String),
    ],
    source=customer_features_source,
    online=True,
    tags={"team": "ml-platform", "version": "v1"},
)
```

**Feast on EKS (Helm values):**

```yaml
# feast-values.yaml for Helm deployment on EKS
feast:
  registry:
    registry_type: s3
    path: s3://my-ml-artifacts/feast-registry/registry.pb
  online_store:
    type: redis
    connection_string: "redis-cluster.internal:6379"
  offline_store:
    type: redshift
    cluster_id: ml-redshift-cluster
    region: us-east-1
    database: ml_features
    user: feast_service
    iam_role: arn:aws:iam::123456789012:role/FeastRedshiftRole

  feature_server:
    enabled: true
    replicas: 3
    resources:
      requests:
        cpu: "500m"
        memory: "1Gi"
      limits:
        cpu: "2"
        memory: "4Gi"
    serviceAccount:
      annotations:
        eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/FeastServiceRole
```

---

## 5. Model Training & Fine-Tuning

### SageMaker Training

| Feature | Details |
|---------|---------|
| **Built-in Algorithms** | XGBoost, Linear Learner, K-Means, DeepAR, BlazingText, etc. |
| **Custom Containers** | Any framework (PyTorch, TF, JAX) in custom Docker images |
| **Distributed Training** | Data parallel (Horovod, SMDataParallel), model parallel (SMModelParallel) |
| **HPO** | Bayesian optimization, random search, grid search, Hyperband |
| **Spot Training** | Up to 90% cost savings with automatic checkpointing |
| **JumpStart** | Pre-trained foundation models (Llama, Mistral, Falcon) |
| **LoRA/QLoRA** | Fine-tune LLMs on single/multi-GPU with PEFT |

**CloudFormation — SageMaker Training Job (via Pipeline):**

```yaml
  ModelTrainingPipeline:
    Type: AWS::SageMaker::Pipeline
    Properties:
      PipelineName: model-training-pipeline-v1
      PipelineDefinition:
        PipelineDefinitionBody: !Sub |
          {
            "Version": "2020-12-01",
            "Parameters": [
              {"Name": "TrainingInstanceType", "Type": "String", "DefaultValue": "ml.g5.2xlarge"},
              {"Name": "TrainingInstanceCount", "Type": "Integer", "DefaultValue": 1},
              {"Name": "Epochs", "Type": "Integer", "DefaultValue": 10},
              {"Name": "LearningRate", "Type": "Float", "DefaultValue": 0.001}
            ],
            "Steps": [
              {
                "Name": "TrainModel",
                "Type": "Training",
                "Arguments": {
                  "AlgorithmSpecification": {
                    "TrainingImage": "${AWS::AccountId}.dkr.ecr.${AWS::Region}.amazonaws.com/custom-training:latest",
                    "TrainingInputMode": "FastFile"
                  },
                  "ResourceConfig": {
                    "InstanceType": {"Get": "Parameters.TrainingInstanceType"},
                    "InstanceCount": {"Get": "Parameters.TrainingInstanceCount"},
                    "VolumeSizeInGB": 200
                  },
                  "StoppingCondition": {
                    "MaxRuntimeInSeconds": 86400
                  },
                  "HyperParameters": {
                    "epochs": {"Get": "Parameters.Epochs"},
                    "learning_rate": {"Get": "Parameters.LearningRate"},
                    "batch_size": "64",
                    "model_name": "custom-classifier-v2"
                  },
                  "InputDataConfig": [
                    {
                      "ChannelName": "train",
                      "DataSource": {
                        "S3DataSource": {
                          "S3Uri": "s3://${MLDataLakeBucket}/train/",
                          "S3DataType": "S3Prefix"
                        }
                      }
                    },
                    {
                      "ChannelName": "validation",
                      "DataSource": {
                        "S3DataSource": {
                          "S3Uri": "s3://${MLDataLakeBucket}/validation/",
                          "S3DataType": "S3Prefix"
                        }
                      }
                    }
                  ],
                  "OutputDataConfig": {
                    "S3OutputPath": "s3://${MLArtifactsBucket}/models/"
                  },
                  "EnableManagedSpotTraining": true,
                  "CheckpointConfig": {
                    "S3Uri": "s3://${MLArtifactsBucket}/checkpoints/"
                  },
                  "RoleArn": "${SageMakerExecutionRole.Arn}"
                }
              }
            ]
          }
      RoleArn: !GetAtt SageMakerExecutionRole.Arn
```

**SageMaker HPO (Python SDK):**

```python
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter

tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name="validation:auc",
    objective_type="Maximize",
    hyperparameter_ranges={
        "learning_rate": ContinuousParameter(0.0001, 0.1, scaling_type="Logarithmic"),
        "num_layers": IntegerParameter(2, 8),
        "dropout": ContinuousParameter(0.0, 0.5),
        "batch_size": IntegerParameter(16, 128),
    },
    max_jobs=50,
    max_parallel_jobs=5,
    strategy="Bayesian",
    early_stopping_type="Auto",
)

tuner.fit({"train": train_s3, "validation": val_s3}, wait=False)
```

### MLflow/OSS Training

| Feature | Details |
|---------|---------|
| **MLflow Tracking** | Log params, metrics, artifacts, tags; compare runs in UI |
| **Ray Train** | Distributed training (data parallel, FSDP) on EKS/Ray clusters |
| **Optuna** | State-of-art HPO (TPE, CMA-ES, pruning) |
| **Hugging Face + PEFT** | LoRA/QLoRA fine-tuning with full HF ecosystem |
| **DeepSpeed/FSDP** | Large model training (ZeRO-1/2/3, activation checkpointing) |
| **W&B** | Advanced experiment tracking with sweeps |
| **Kubeflow/Ray on EKS** | Managed training infrastructure on Kubernetes |

**MLflow Training with Ray on EKS:**

```python
import mlflow
import ray
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
import torch
import torch.nn as nn

# Connect to MLflow tracking server (deployed on EKS)
mlflow.set_tracking_uri("http://mlflow.ml-platform.svc.cluster.local:5000")
mlflow.set_experiment("customer-churn-model")

def train_func(config):
    """Distributed training function executed on each worker."""
    import torch
    from torch.utils.data import DataLoader
    
    # Get distributed context
    model = build_model(config["num_layers"], config["hidden_dim"])
    model = train.torch.prepare_model(model)
    
    dataset = load_dataset_from_s3(config["data_path"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    dataloader = train.torch.prepare_data_loader(dataloader)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        # Report metrics to Ray + MLflow
        train.report({"loss": avg_loss, "epoch": epoch})
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
    
    # Save final model
    return model.state_dict()

# Launch distributed training on Ray cluster (EKS)
with mlflow.start_run(run_name="churn-model-distributed-v2") as run:
    mlflow.log_params({
        "num_workers": 4,
        "instance_type": "g5.2xlarge",
        "framework": "pytorch",
        "distributed_strategy": "DDP",
    })
    
    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            "lr": 0.001,
            "epochs": 20,
            "batch_size": 64,
            "num_layers": 4,
            "hidden_dim": 256,
            "data_path": "s3://my-ml-data-lake/train/",
        },
        scaling_config=ScalingConfig(
            num_workers=4,
            use_gpu=True,
            resources_per_worker={"GPU": 1, "CPU": 8},
        ),
    )
    
    result = trainer.fit()
    
    # Log model artifact
    mlflow.pytorch.log_model(
        pytorch_model=result.checkpoint,
        artifact_path="model",
        registered_model_name="customer-churn-classifier",
    )
    
    mlflow.log_metrics({
        "final_loss": result.metrics["loss"],
        "training_time_sec": result.metrics.get("time_total_s", 0),
    })
```

**Optuna HPO with MLflow:**

```python
import optuna
import mlflow

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    num_layers = trial.suggest_int("num_layers", 2, 8)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    
    with mlflow.start_run(nested=True, run_name=f"trial-{trial.number}"):
        mlflow.log_params(trial.params)
        
        model = train_model(lr=lr, num_layers=num_layers, 
                          dropout=dropout, batch_size=batch_size)
        auc = evaluate_model(model)
        
        mlflow.log_metric("auc", auc)
        return auc

with mlflow.start_run(run_name="hpo-sweep"):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner(),
    )
    study.optimize(objective, n_trials=50, n_jobs=5)
    
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_auc", study.best_value)
```

---

## 6. Model Registry & Versioning

### SageMaker Model Registry

```
┌─────────────────────────────────────────────────────────────┐
│              SageMaker Model Registry Flow                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Training Job ──→ Model Artifact (S3) ──→ Create Model Pkg   │
│                                                ↓              │
│                                       Model Package Group     │
│                                       (customer-churn-model)  │
│                                                ↓              │
│                                       Version 1 [PendingApproval]
│                                       Version 2 [Approved] ←─ CI/CD
│                                       Version 3 [Rejected]    │
│                                                               │
│  Model Card: accuracy, bias metrics, intended use, limits     │
│  Lineage: data → processing → training → model → endpoint    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Approval Workflow States:**
- `PendingManualApproval` → Human review required
- `Approved` → Ready for deployment
- `Rejected` → Does not meet quality bar

**CloudFormation — Model Package Group:**

```yaml
  ChurnModelPackageGroup:
    Type: AWS::SageMaker::ModelPackageGroup
    Properties:
      ModelPackageGroupName: customer-churn-model
      ModelPackageGroupDescription: "Production churn prediction models"
      Tags:
        - Key: Team
          Value: ml-platform
        - Key: UseCase
          Value: customer-retention

  # EventBridge rule to trigger deployment on model approval
  ModelApprovalRule:
    Type: AWS::Events::Rule
    Properties:
      Description: Trigger deployment when model is approved
      EventPattern:
        source:
          - aws.sagemaker
        detail-type:
          - SageMaker Model Package State Change
        detail:
          ModelPackageGroupName:
            - customer-churn-model
          ModelApprovalStatus:
            - Approved
      Targets:
        - Id: TriggerDeploymentPipeline
          Arn: !GetAtt DeploymentStateMachine.Arn
          RoleArn: !GetAtt EventBridgeRole.Arn
```

### MLflow Model Registry

```
┌─────────────────────────────────────────────────────────────┐
│                MLflow Model Registry Flow                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Training Run ──→ mlflow.log_model() ──→ Register Model      │
│                                               ↓               │
│                                    Registered Model            │
│                                    (customer-churn-classifier) │
│                                               ↓               │
│                                    Version 1 [Staging]        │
│                                    Version 2 [Production] ←─ Promote
│                                    Version 3 [Archived]       │
│                                                               │
│  Annotations: description, tags, aliases ("champion", "challenger")
│  Artifacts: model files, requirements.txt, conda.yaml        │
│  Lineage: run_id → experiment → parameters → metrics         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**MLflow Model Registration:**

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model from a training run
model_uri = f"runs:/{run_id}/model"
mv = mlflow.register_model(model_uri, "customer-churn-classifier")

# Add metadata
client.update_model_version(
    name="customer-churn-classifier",
    version=mv.version,
    description="XGBoost v2 - trained on Q1-Q2 2026 data, AUC=0.94"
)

# Set alias for deployment reference
client.set_registered_model_alias(
    name="customer-churn-classifier",
    alias="champion",
    version=mv.version
)

# Transition stages (if using classic stage-based workflow)
client.transition_model_version_stage(
    name="customer-churn-classifier",
    version=mv.version,
    stage="Production",
    archive_existing_versions=True,
)

# Tag with deployment metadata
client.set_model_version_tag(
    name="customer-churn-classifier",
    version=mv.version,
    key="deployment_approved_by",
    value="ml-team-lead"
)
client.set_model_version_tag(
    name="customer-churn-classifier",
    version=mv.version,
    key="validation_auc",
    value="0.94"
)
```

---

## 7. Model Deployment

### SageMaker Deployment Options

| Mode | Latency | Use Case | Scaling |
|------|---------|----------|---------|
| **Real-time Endpoint** | < 100ms | Online inference, APIs | Auto-scaling (1-N instances) |
| **Multi-Model Endpoint** | < 200ms | 100s of models on shared infra | Model loading on-demand |
| **Serverless Inference** | Cold start ~1s | Sporadic traffic | Scales to zero |
| **Batch Transform** | N/A | Bulk scoring | Parallel instances |
| **Async Inference** | Seconds-minutes | Large payloads, queued | Queue-based scaling |

**CloudFormation — Real-time Endpoint with Auto-scaling:**

```yaml
  ChurnModel:
    Type: AWS::SageMaker::Model
    Properties:
      ModelName: !Sub 'customer-churn-${Environment}'
      ExecutionRoleArn: !GetAtt SageMakerExecutionRole.Arn
      PrimaryContainer:
        Image: !Sub '${AWS::AccountId}.dkr.ecr.${AWS::Region}.amazonaws.com/inference:latest'
        ModelDataUrl: !Sub 's3://${MLArtifactsBucket}/models/churn-model/model.tar.gz'
        Environment:
          SAGEMAKER_PROGRAM: inference.py
          MODEL_VERSION: "2.1.0"

  ChurnEndpointConfig:
    Type: AWS::SageMaker::EndpointConfig
    Properties:
      EndpointConfigName: !Sub 'churn-endpoint-config-${Environment}'
      ProductionVariants:
        - VariantName: primary
          ModelName: !Ref ChurnModel
          InstanceType: ml.g5.xlarge
          InitialInstanceCount: 2
          InitialVariantWeight: 1.0
          ContainerStartupHealthCheckTimeoutInSeconds: 300
      DataCaptureConfig:
        EnableCapture: true
        InitialSamplingPercentage: 100
        DestinationS3Uri: !Sub 's3://${MLArtifactsBucket}/data-capture/'
        CaptureOptions:
          - CaptureMode: Input
          - CaptureMode: Output
        CaptureContentTypeHeader:
          CsvContentTypes:
            - 'text/csv'
          JsonContentTypes:
            - 'application/json'

  ChurnEndpoint:
    Type: AWS::SageMaker::Endpoint
    Properties:
      EndpointName: !Sub 'churn-endpoint-${Environment}'
      EndpointConfigName: !Ref ChurnEndpointConfig
      Tags:
        - Key: Project
          Value: customer-retention
        - Key: Environment
          Value: !Ref Environment

  # Auto-scaling
  EndpointScalingTarget:
    Type: AWS::ApplicationAutoScaling::ScalableTarget
    Properties:
      MaxCapacity: 10
      MinCapacity: 2
      ResourceId: !Sub 'endpoint/${ChurnEndpoint}/variant/primary'
      RoleARN: !GetAtt AutoScalingRole.Arn
      ScalableDimension: sagemaker:variant:DesiredInstanceCount
      ServiceNamespace: sagemaker

  EndpointScalingPolicy:
    Type: AWS::ApplicationAutoScaling::ScalingPolicy
    Properties:
      PolicyName: churn-endpoint-scaling
      PolicyType: TargetTrackingScaling
      ScalableTargetId: !Ref EndpointScalingTarget
      TargetTrackingScalingPolicyConfiguration:
        TargetValue: 750  # invocations per instance per minute
        ScaleInCooldown: 300
        ScaleOutCooldown: 60
        PredefinedMetricSpecification:
          PredefinedMetricType: SageMakerVariantInvocationsPerInstance
```

### MLflow/OSS Deployment Options

| Tool | Latency | Use Case | Platform |
|------|---------|----------|----------|
| **KServe** | < 50ms | Production inference on K8s | EKS/GKE |
| **Seldon Core** | < 50ms | Advanced inference graphs | EKS/GKE |
| **Ray Serve** | < 20ms | High-throughput, batching | EKS/Ray |
| **BentoML** | < 50ms | Easy packaging & serving | Any |
| **vLLM** | Varies | LLM serving (PagedAttention) | GPU nodes |
| **TorchServe** | < 100ms | PyTorch models | Any |

**KServe on EKS (InferenceService):**

```yaml
# kserve-inference-service.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: customer-churn-classifier
  namespace: ml-serving
  annotations:
    serving.kserve.io/autoscalerClass: hpa
    serving.kserve.io/targetUtilizationPercentage: "70"
spec:
  predictor:
    minReplicas: 2
    maxReplicas: 20
    scaleTarget: 10  # concurrent requests per pod
    model:
      modelFormat:
        name: mlflow
      storageUri: "s3://my-ml-artifacts/mlflow-models/customer-churn-classifier/2"
      resources:
        requests:
          cpu: "2"
          memory: "4Gi"
          nvidia.com/gpu: "1"
        limits:
          cpu: "4"
          memory: "8Gi"
          nvidia.com/gpu: "1"
    serviceAccountName: kserve-sa  # IRSA for S3 access
  transformer:
    containers:
      - name: feature-transformer
        image: 123456789012.dkr.ecr.us-east-1.amazonaws.com/feature-transformer:v1
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
```

**Ray Serve Deployment:**

```python
import ray
from ray import serve
import mlflow

@serve.deployment(
    num_replicas=3,
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={
        "min_replicas": 2,
        "max_replicas": 20,
        "target_ongoing_requests": 10,
    },
)
class ChurnModelDeployment:
    def __init__(self):
        # Load model from MLflow registry
        self.model = mlflow.pyfunc.load_model(
            "models:/customer-churn-classifier@champion"
        )
    
    async def __call__(self, request):
        data = await request.json()
        prediction = self.model.predict(data["features"])
        return {"prediction": prediction.tolist(), "model_version": "champion"}

# Deploy
app = ChurnModelDeployment.bind()
serve.run(app, route_prefix="/predict", host="0.0.0.0", port=8000)
```

---

## 8. Canary Testing & Progressive Rollout

### SageMaker Canary Deployments

```
┌─────────────────────────────────────────────────────────────────┐
│           SageMaker Progressive Deployment                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Phase 1: Shadow Testing                                         │
│  ┌───────┐     ┌──────────────┐                                 │
│  │Traffic│────→│ Primary (v1) │──→ Response to user              │
│  └───────┘     └──────────────┘                                 │
│       │        ┌──────────────┐                                 │
│       └──copy─→│ Shadow (v2)  │──→ Log only (compare metrics)   │
│                └──────────────┘                                 │
│                                                                   │
│  Phase 2: Canary (10% → 50% → 100%)                             │
│  ┌───────┐     ┌──────────────┐                                 │
│  │Traffic│─90%→│ Primary (v1) │                                 │
│  └───────┘     └──────────────┘                                 │
│       │        ┌──────────────┐                                 │
│       └──10%─→ │ Canary (v2)  │ ← Monitor latency, errors       │
│                └──────────────┘                                 │
│                                                                   │
│  Phase 3: Blue/Green with Guardrails                             │
│  Auto-rollback if: latency_p99 > 500ms OR error_rate > 1%       │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**SageMaker Canary Deployment (Python SDK):**

```python
import boto3
from sagemaker.model import Model
from sagemaker.session import Session

sm_client = boto3.client("sagemaker")

# Create new endpoint config with canary variant
sm_client.create_endpoint_config(
    EndpointConfigName="churn-canary-config-v2",
    ProductionVariants=[
        {
            "VariantName": "primary",
            "ModelName": "churn-model-v1",
            "InstanceType": "ml.g5.xlarge",
            "InitialInstanceCount": 2,
            "InitialVariantWeight": 0.9,  # 90% traffic
        },
        {
            "VariantName": "canary",
            "ModelName": "churn-model-v2",
            "InstanceType": "ml.g5.xlarge",
            "InitialInstanceCount": 1,
            "InitialVariantWeight": 0.1,  # 10% traffic
        },
    ],
)

# Update endpoint with blue/green deployment guardrails
sm_client.update_endpoint(
    EndpointName="churn-endpoint-prod",
    EndpointConfigName="churn-canary-config-v2",
    DeploymentConfig={
        "BlueGreenUpdatePolicy": {
            "TrafficRoutingConfiguration": {
                "Type": "CANARY",
                "CanarySize": {
                    "Type": "INSTANCE_COUNT",
                    "Value": 1,
                },
                "WaitIntervalInSeconds": 600,  # 10 min bake time
            },
            "TerminationWaitInSeconds": 300,
            "MaximumExecutionTimeoutInSeconds": 3600,
        },
        "AutoRollbackConfiguration": {
            "Alarms": [
                {"AlarmName": "churn-endpoint-high-latency"},
                {"AlarmName": "churn-endpoint-high-error-rate"},
            ]
        },
    },
)
```

**CloudFormation — CloudWatch Alarms for Auto-Rollback:**

```yaml
  EndpointHighLatencyAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: churn-endpoint-high-latency
      AlarmDescription: P99 latency exceeds 500ms - trigger rollback
      MetricName: ModelLatency
      Namespace: AWS/SageMaker
      Statistic: p99
      Period: 60
      EvaluationPeriods: 3
      Threshold: 500000  # microseconds
      ComparisonOperator: GreaterThanThreshold
      Dimensions:
        - Name: EndpointName
          Value: !Ref ChurnEndpoint
        - Name: VariantName
          Value: canary
      TreatMissingData: breaching

  EndpointHighErrorAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: churn-endpoint-high-error-rate
      AlarmDescription: Error rate exceeds 1% - trigger rollback
      MetricName: Invocation5XXErrors
      Namespace: AWS/SageMaker
      Statistic: Average
      Period: 60
      EvaluationPeriods: 2
      Threshold: 0.01
      ComparisonOperator: GreaterThanThreshold
      Dimensions:
        - Name: EndpointName
          Value: !Ref ChurnEndpoint
        - Name: VariantName
          Value: canary
```

### MLflow/OSS Canary Deployments

**Argo Rollouts (Canary on EKS):**

```yaml
# argo-rollout-canary.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: churn-model-rollout
  namespace: ml-serving
spec:
  replicas: 5
  revisionHistoryLimit: 3
  selector:
    matchLabels:
      app: churn-model
  strategy:
    canary:
      canaryService: churn-model-canary
      stableService: churn-model-stable
      trafficRouting:
        istio:
          virtualServices:
            - name: churn-model-vsvc
              routes:
                - primary
      steps:
        - setWeight: 10
        - pause: {duration: 10m}
        - analysis:
            templates:
              - templateName: churn-model-analysis
            args:
              - name: service-name
                value: churn-model-canary
        - setWeight: 30
        - pause: {duration: 10m}
        - analysis:
            templates:
              - templateName: churn-model-analysis
        - setWeight: 60
        - pause: {duration: 10m}
        - setWeight: 100
      rollbackWindow:
        revisions: 2
  template:
    metadata:
      labels:
        app: churn-model
    spec:
      containers:
        - name: model-server
          image: 123456789012.dkr.ecr.us-east-1.amazonaws.com/churn-model:v2
          ports:
            - containerPort: 8080
          resources:
            requests:
              cpu: "2"
              memory: "4Gi"
              nvidia.com/gpu: "1"
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 30
---
# Analysis Template - automated canary validation
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: churn-model-analysis
  namespace: ml-serving
spec:
  metrics:
    - name: latency-p99
      provider:
        prometheus:
          address: http://prometheus.monitoring:9090
          query: |
            histogram_quantile(0.99,
              sum(rate(http_request_duration_seconds_bucket{
                service="{{args.service-name}}"
              }[5m])) by (le)
            )
      successCondition: result[0] < 0.5  # < 500ms
      interval: 60s
      count: 5
      failureLimit: 2
    - name: error-rate
      provider:
        prometheus:
          address: http://prometheus.monitoring:9090
          query: |
            sum(rate(http_requests_total{
              service="{{args.service-name}}",
              status=~"5.."
            }[5m])) /
            sum(rate(http_requests_total{
              service="{{args.service-name}}"
            }[5m]))
      successCondition: result[0] < 0.01  # < 1%
      interval: 60s
      count: 5
      failureLimit: 1
    - name: prediction-quality
      provider:
        prometheus:
          address: http://prometheus.monitoring:9090
          query: |
            avg(model_prediction_confidence{
              service="{{args.service-name}}"
            })
      successCondition: result[0] > 0.7
      interval: 120s
      count: 3
```

**Istio VirtualService for Traffic Splitting:**

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: churn-model-vsvc
  namespace: ml-serving
spec:
  hosts:
    - churn-model.ml-serving.svc.cluster.local
  http:
    - name: primary
      route:
        - destination:
            host: churn-model-stable
            port:
              number: 8080
          weight: 90
        - destination:
            host: churn-model-canary
            port:
              number: 8080
          weight: 10
      retries:
        attempts: 3
        perTryTimeout: 2s
```

---

## 9. Monitoring & Observability

### SageMaker Model Monitor

| Monitor Type | What It Detects | Frequency |
|-------------|----------------|-----------|
| **Data Quality** | Schema drift, missing values, outliers | Hourly/Daily |
| **Model Quality** | Accuracy degradation, precision/recall drop | When ground truth available |
| **Bias Drift** | Fairness metric changes (DPL, DPPL) | Scheduled |
| **Feature Attribution** | SHAP value drift, feature importance changes | Scheduled |

**CloudFormation — Model Monitor:**

```yaml
  DataQualityMonitoringSchedule:
    Type: AWS::SageMaker::MonitoringSchedule
    Properties:
      MonitoringScheduleName: churn-data-quality-monitor
      MonitoringScheduleConfig:
        MonitoringJobDefinition:
          MonitoringAppSpecification:
            ImageUri: !Sub '${AWS::AccountId}.dkr.ecr.${AWS::Region}.amazonaws.com/sagemaker-model-monitor-analyzer'
          MonitoringInputs:
            - EndpointInput:
                EndpointName: !Ref ChurnEndpoint
                LocalPath: /opt/ml/processing/input
                S3DataDistributionType: FullyReplicated
                S3InputMode: File
          MonitoringOutputConfig:
            MonitoringOutputs:
              - S3Output:
                  LocalPath: /opt/ml/processing/output
                  S3Uri: !Sub 's3://${MLArtifactsBucket}/monitoring/data-quality/'
          MonitoringResources:
            ClusterConfig:
              InstanceCount: 1
              InstanceType: ml.m5.xlarge
              VolumeSizeInGB: 50
          BaselineConfig:
            ConstraintsResource:
              S3Uri: !Sub 's3://${MLArtifactsBucket}/baselines/constraints.json'
            StatisticsResource:
              S3Uri: !Sub 's3://${MLArtifactsBucket}/baselines/statistics.json'
          RoleArn: !GetAtt SageMakerExecutionRole.Arn
        ScheduleConfig:
          ScheduleExpression: cron(0 * * * ? *)  # hourly
      EndpointName: !Ref ChurnEndpoint

  # CloudWatch alarm on drift detection
  DriftDetectionAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: churn-model-drift-detected
      MetricName: data_quality_violations
      Namespace: aws/sagemaker/Endpoints/data-metrics
      Statistic: Maximum
      Period: 3600
      EvaluationPeriods: 1
      Threshold: 0
      ComparisonOperator: GreaterThanThreshold
      AlarmActions:
        - !Ref DriftNotificationTopic
```

### MLflow/OSS Monitoring Stack

```
┌─────────────────────────────────────────────────────────────────┐
│              OSS Monitoring Architecture (EKS)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Model Server ──→ OpenTelemetry Collector ──→ Prometheus          │
│  (metrics,         (traces, metrics)          (time-series DB)   │
│   predictions)                                     ↓              │
│                                               Grafana             │
│                                               (dashboards)        │
│                                                                   │
│  Inference Logs ──→ Evidently AI ──→ Drift Reports               │
│  (predictions,      (statistical tests,   (HTML reports,          │
│   features)          drift detection)      Prometheus metrics)    │
│                                                                   │
│  Model Quality ──→ NannyML ──→ Performance Estimation            │
│  (no ground truth    (CBPE, DLE algorithms)                       │
│   needed!)                                                        │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Evidently AI Drift Detection:**

```python
import evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    ColumnDriftMetric,
)
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift, TestShareOfDriftedColumns
import pandas as pd

# Load reference (training) and current (production) data
reference_data = pd.read_parquet("s3://my-ml-data-lake/reference/features.parquet")
current_data = pd.read_parquet("s3://my-ml-data-lake/production/today_features.parquet")

# Drift Report
drift_report = Report(metrics=[
    DatasetDriftMetric(),
    DataDriftTable(),
    ColumnDriftMetric(column_name="total_spend_30d"),
    ColumnDriftMetric(column_name="login_count_7d"),
])

drift_report.run(reference_data=reference_data, current_data=current_data)
drift_report.save_html("reports/drift_report.html")

# Get drift results programmatically
results = drift_report.as_dict()
dataset_drift = results["metrics"][0]["result"]["dataset_drift"]
drift_share = results["metrics"][0]["result"]["drift_share"]

# Automated test suite (for CI/CD integration)
test_suite = TestSuite(tests=[
    TestShareOfDriftedColumns(lt=0.3),  # < 30% features drifted
    TestColumnDrift(column_name="churn_risk_score"),
])

test_suite.run(reference_data=reference_data, current_data=current_data)
test_results = test_suite.as_dict()

if not test_results["summary"]["all_passed"]:
    # Trigger retraining pipeline
    trigger_retraining(reason="drift_detected", details=test_results)
```

**Prometheus + Grafana Monitoring (Kubernetes manifests):**

```yaml
# prometheus-servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ml-model-metrics
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: churn-model
  endpoints:
    - port: metrics
      interval: 15s
      path: /metrics
  namespaceSelector:
    matchNames:
      - ml-serving
---
# grafana-dashboard-configmap.yaml (key panels)
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-model-dashboard
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
data:
  ml-model.json: |
    {
      "title": "ML Model Performance",
      "panels": [
        {
          "title": "Inference Latency P99",
          "targets": [{"expr": "histogram_quantile(0.99, rate(inference_duration_seconds_bucket[5m]))"}]
        },
        {
          "title": "Prediction Distribution",
          "targets": [{"expr": "histogram_quantile(0.5, rate(prediction_score_bucket[5m]))"}]
        },
        {
          "title": "Feature Drift Score",
          "targets": [{"expr": "evidently_column_drift_score{feature=~\".*\"}"}]
        },
        {
          "title": "Requests per Second",
          "targets": [{"expr": "sum(rate(inference_requests_total[1m]))"}]
        }
      ]
    }
```

---

## 10. Automated Retraining

### SageMaker Automated Retraining

```
┌─────────────────────────────────────────────────────────────────┐
│         SageMaker Automated Retraining Architecture              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Model Monitor ──→ CloudWatch Alarm ──→ EventBridge Rule         │
│  (drift detected)   (threshold breach)    │                      │
│                                           ↓                      │
│                                    SageMaker Pipeline             │
│                                    (or Step Functions)            │
│                                           │                      │
│                           ┌───────────────┼───────────────┐      │
│                           ↓               ↓               ↓      │
│                    Process Data     Train Model      Evaluate     │
│                           │               │               │      │
│                           └───────────────┼───────────────┘      │
│                                           ↓                      │
│                                   Conditional Step               │
│                                   (AUC > 0.90?)                  │
│                                    ↓ Yes     ↓ No                │
│                             Register Model   Alert Team          │
│                             (Auto-approve)                        │
│                                    ↓                             │
│                             Deploy Canary                         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**CloudFormation — EventBridge + Step Functions for Retraining:**

```yaml
  DriftRetrainingRule:
    Type: AWS::Events::Rule
    Properties:
      Description: Trigger retraining on drift detection
      EventPattern:
        source:
          - aws.cloudwatch
        detail-type:
          - CloudWatch Alarm State Change
        detail:
          alarmName:
            - churn-model-drift-detected
          state:
            value:
              - ALARM
      Targets:
        - Id: StartRetraining
          Arn: !GetAtt RetrainingStateMachine.Arn
          RoleArn: !GetAtt EventBridgeRole.Arn
          Input: !Sub |
            {
              "trigger": "drift_detection",
              "pipeline_name": "model-training-pipeline-v1",
              "timestamp": "$.time"
            }

  # Scheduled retraining (weekly)
  ScheduledRetrainingRule:
    Type: AWS::Events::Rule
    Properties:
      Description: Weekly scheduled retraining
      ScheduleExpression: cron(0 2 ? * SUN *)  # Every Sunday 2 AM UTC
      Targets:
        - Id: WeeklyRetrain
          Arn: !GetAtt RetrainingStateMachine.Arn
          RoleArn: !GetAtt EventBridgeRole.Arn
          Input: '{"trigger": "scheduled_weekly"}'

  RetrainingStateMachine:
    Type: AWS::StepFunctions::StateMachine
    Properties:
      StateMachineName: ml-retraining-orchestrator
      RoleArn: !GetAtt StepFunctionsRole.Arn
      DefinitionString: !Sub |
        {
          "StartAt": "StartPipeline",
          "States": {
            "StartPipeline": {
              "Type": "Task",
              "Resource": "arn:aws:states:::sagemaker:startPipelineExecution.sync",
              "Parameters": {
                "PipelineName": "model-training-pipeline-v1",
                "PipelineExecutionDisplayName.$": "States.Format('retrain-{}', $.trigger)"
              },
              "Next": "EvaluateModel"
            },
            "EvaluateModel": {
              "Type": "Task",
              "Resource": "arn:aws:states:::lambda:invoke",
              "Parameters": {
                "FunctionName": "${EvaluationLambda}",
                "Payload.$": "$"
              },
              "Next": "QualityGate"
            },
            "QualityGate": {
              "Type": "Choice",
              "Choices": [
                {
                  "Variable": "$.Payload.auc",
                  "NumericGreaterThanEquals": 0.90,
                  "Next": "RegisterAndDeploy"
                }
              ],
              "Default": "AlertTeam"
            },
            "RegisterAndDeploy": {
              "Type": "Task",
              "Resource": "arn:aws:states:::sagemaker:createModelPackage",
              "Parameters": {
                "ModelPackageGroupName": "customer-churn-model",
                "ModelApprovalStatus": "Approved"
              },
              "Next": "Success"
            },
            "AlertTeam": {
              "Type": "Task",
              "Resource": "arn:aws:states:::sns:publish",
              "Parameters": {
                "TopicArn": "${AlertTopic}",
                "Message": "Retrained model did not meet quality bar"
              },
              "Next": "Failed"
            },
            "Success": {"Type": "Succeed"},
            "Failed": {"Type": "Fail"}
          }
        }
```

### MLflow/OSS Automated Retraining

**Dagster Pipeline (Production-grade):**

```python
from dagster import (
    asset, op, job, schedule, sensor,
    AssetExecutionContext, RunRequest, SensorResult
)
import mlflow
import pandas as pd

@asset(group_name="ml_pipeline")
def check_drift(context: AssetExecutionContext) -> dict:
    """Check for data/model drift using Evidently."""
    from evidently.test_suite import TestSuite
    from evidently.tests import TestShareOfDriftedColumns
    
    reference = pd.read_parquet("s3://my-data/reference/features.parquet")
    current = pd.read_parquet("s3://my-data/production/latest_features.parquet")
    
    suite = TestSuite(tests=[TestShareOfDriftedColumns(lt=0.3)])
    suite.run(reference_data=reference, current_data=current)
    
    results = suite.as_dict()
    drift_detected = not results["summary"]["all_passed"]
    
    context.log.info(f"Drift detected: {drift_detected}")
    return {"drift_detected": drift_detected, "drift_share": results["summary"]}


@asset(group_name="ml_pipeline", deps=[check_drift])
def retrain_model(context: AssetExecutionContext, check_drift: dict) -> dict:
    """Retrain model if drift detected."""
    if not check_drift["drift_detected"]:
        context.log.info("No drift - skipping retraining")
        return {"action": "skipped"}
    
    mlflow.set_tracking_uri("http://mlflow.ml-platform:5000")
    mlflow.set_experiment("customer-churn-auto-retrain")
    
    with mlflow.start_run(run_name="auto-retrain") as run:
        # Load fresh training data
        train_data = pd.read_parquet("s3://my-data/train/latest/")
        
        # Train model
        model = train_xgboost(train_data)
        
        # Evaluate
        metrics = evaluate_model(model, test_data)
        mlflow.log_metrics(metrics)
        
        if metrics["auc"] >= 0.90:
            # Register and promote
            mlflow.sklearn.log_model(model, "model",
                registered_model_name="customer-churn-classifier")
            
            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions(
                "customer-churn-classifier")[0].version
            client.set_registered_model_alias(
                "customer-churn-classifier", "challenger", latest_version)
            
            return {"action": "retrained", "version": latest_version, 
                    "metrics": metrics}
        else:
            return {"action": "below_threshold", "metrics": metrics}


@sensor(minimum_interval_seconds=3600)
def drift_sensor(context):
    """Sensor that triggers retraining on drift detection."""
    # Check Prometheus for drift metrics
    import requests
    resp = requests.get(
        "http://prometheus:9090/api/v1/query",
        params={"query": 'evidently_dataset_drift{job="churn-model"} == 1'}
    )
    
    if resp.json()["data"]["result"]:
        yield RunRequest(
            run_key=f"drift-retrain-{context.cursor}",
            tags={"trigger": "drift_sensor"}
        )


@schedule(cron_schedule="0 3 * * 0", job=retrain_job)  # Weekly Sunday 3AM
def weekly_retrain_schedule():
    return RunRequest(tags={"trigger": "weekly_schedule"})
```

**Argo Workflows (Kubernetes-native):**

```yaml
apiVersion: argoproj.io/v1alpha1
kind: CronWorkflow
metadata:
  name: ml-retraining-weekly
  namespace: ml-pipelines
spec:
  schedule: "0 3 * * 0"  # Weekly Sunday 3AM
  workflowSpec:
    entrypoint: retrain-pipeline
    serviceAccountName: ml-pipeline-sa
    templates:
      - name: retrain-pipeline
        dag:
          tasks:
            - name: check-drift
              template: drift-check
            - name: prepare-data
              template: data-prep
              dependencies: [check-drift]
              when: "{{tasks.check-drift.outputs.parameters.drift_detected}} == true"
            - name: train-model
              template: training
              dependencies: [prepare-data]
            - name: evaluate
              template: evaluation
              dependencies: [train-model]
            - name: deploy
              template: canary-deploy
              dependencies: [evaluate]
              when: "{{tasks.evaluate.outputs.parameters.quality_passed}} == true"

      - name: drift-check
        container:
          image: 123456789012.dkr.ecr.us-east-1.amazonaws.com/drift-checker:v1
          command: [python, check_drift.py]
        outputs:
          parameters:
            - name: drift_detected
              valueFrom:
                path: /tmp/drift_result.txt

      - name: training
        container:
          image: 123456789012.dkr.ecr.us-east-1.amazonaws.com/model-trainer:v1
          command: [python, train.py]
          resources:
            requests:
              nvidia.com/gpu: 1
              memory: "16Gi"
            limits:
              nvidia.com/gpu: 1
              memory: "32Gi"
        nodeSelector:
          node.kubernetes.io/instance-type: g5.2xlarge
```

---

## 11. Cost Comparison

### Medium-Scale Project Estimates (Monthly)

**Assumptions:**
- 10 training runs/month (4 hours each, g5.2xlarge equivalent)
- 2 real-time endpoints (24/7, ml.g5.xlarge equivalent)
- 1TB data stored, 10TB processed
- 1M inference requests/month
- 3 ML engineers

| Component | SageMaker | MLflow/OSS (EKS) | Notes |
|-----------|-----------|-------------------|-------|
| **Training Compute** | $3,200 | $2,100 | SM: on-demand; OSS: spot on EKS |
| **Spot Savings** | -$1,600 (50% off) | -$1,300 (60% off) | SM spot less discount than EC2 |
| **Inference Compute** | $2,400 | $1,800 | SM: endpoint hours; OSS: EKS pods |
| **Storage (S3)** | $23 | $23 | Same S3 costs |
| **Feature Store** | $500 | $200 | SM: managed; OSS: Redis + S3 |
| **Data Processing** | $800 | $400 | SM Processing vs Spark on EKS |
| **Monitoring** | $150 | $100 | SM Monitor vs Prometheus/Grafana |
| **MLflow Server** | N/A | $200 | EKS pod + RDS backend |
| **Networking** | $100 | $150 | VPC endpoints vs service mesh |
| **EKS Cluster** | N/A | $73 | Control plane cost |
| **Platform Engineering** | $0 | $2,000 | Engineer time maintaining infra |
| | | | |
| **Total (compute+storage)** | **~$5,573** | **~$3,743** | Excluding people cost |
| **Total (with people)** | **~$5,573** | **~$5,743** | Similar when including ops |

### Cost Optimization Tips

**SageMaker:**
- Use Spot Training (save 50-90% on training)
- Serverless Inference for < 100 req/min workloads
- Multi-model endpoints for many small models
- Reserved capacity (SageMaker Savings Plans) for steady-state inference
- Use `ml.inf2` Inferentia instances for inference (up to 4x cheaper than GPU)

**MLflow/OSS:**
- Karpenter for intelligent spot instance selection on EKS
- Scale-to-zero with KEDA for batch workloads
- Use Graviton (ARM) instances for non-GPU workloads (20% cheaper)
- Shared GPU scheduling with MIG or time-slicing
- Preemptible training with checkpointing

---

## 12. Complete Pipeline Diagrams

### SageMaker End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        AWS SageMaker MLOps Pipeline (Production)                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                           │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐                  │
│  │ S3 Data │───→│ Data Wrangler│───→│ Processing  │───→│ Feature Store│                  │
│  │ Lake    │    │ (visual prep)│    │ Job (Spark) │    │ (online+off) │                  │
│  └─────────┘    └──────────────┘    └─────────────┘    └──────┬───────┘                  │
│       │                                                        │                          │
│       │         ┌──────────────┐                               │                          │
│       └────────→│ Ground Truth │ (labeling)                    │                          │
│                 └──────────────┘                               │                          │
│                                                                ↓                          │
│                                                    ┌──────────────────┐                   │
│                                                    │ Training Job     │                   │
│                                                    │ (HPO, Spot,      │                   │
│                                                    │  Distributed)    │                   │
│                                                    └────────┬─────────┘                   │
│                                                             │                             │
│                                                             ↓                             │
│                                              ┌─────────────────────────┐                  │
│                                              │ Model Registry          │                  │
│                                              │ (Pending → Approved)    │                  │
│                                              └────────────┬────────────┘                  │
│                                                           │                               │
│                                                           ↓                               │
│  ┌──────────────┐    ┌─────────────────┐    ┌─────────────────────────┐                  │
│  │ CloudWatch   │←───│ Model Monitor   │←───│ Endpoint (Canary/B-G)   │                  │
│  │ Alarms       │    │ (Drift, Bias)   │    │ Auto-scaling            │                  │
│  └──────┬───────┘    └─────────────────┘    └─────────────────────────┘                  │
│         │                                                                                 │
│         ↓                                                                                 │
│  ┌──────────────┐    ┌─────────────────┐                                                 │
│  │ EventBridge  │───→│ Step Functions  │───→ (Retrain Loop)                              │
│  │ (trigger)    │    │ (orchestrate)   │                                                 │
│  └──────────────┘    └─────────────────┘                                                 │
│                                                                                           │
│  Infrastructure: CloudFormation │ CI/CD: CodePipeline/CodeBuild │ IAM: Least Privilege    │
│                                                                                           │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### MLflow/OSS End-to-End Pipeline (on EKS)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    MLflow/OSS MLOps Pipeline on EKS (Production)                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                           │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐                  │
│  │ S3/Delta│───→│ Spark/Polars │───→│ Great Expct │───→│ Feast Feature│                  │
│  │ Lake    │    │ (ETL)        │    │ (validate)  │    │ Store        │                  │
│  └─────────┘    └──────────────┘    └─────────────┘    └──────┬───────┘                  │
│       │                                                        │                          │
│       │         ┌──────────────┐                               │                          │
│       └────────→│ Label Studio │ (labeling)                    │                          │
│                 └──────────────┘                               │                          │
│                                                                ↓                          │
│                                                    ┌──────────────────┐                   │
│                                                    │ Ray Train / K8s  │                   │
│                                                    │ (Distributed,    │                   │
│                                                    │  Optuna HPO)     │                   │
│                                                    └────────┬─────────┘                   │
│                                                             │                             │
│       ┌───────────────────────────────────────┐             │                             │
│       │ MLflow Tracking (params, metrics, art)│←────────────┘                             │
│       └───────────────────┬───────────────────┘                                          │
│                           ↓                                                               │
│                ┌─────────────────────────┐                                                │
│                │ MLflow Model Registry   │                                                │
│                │ (Staging → Production)  │                                                │
│                └────────────┬────────────┘                                                │
│                             │                                                             │
│                             ↓                                                             │
│  ┌──────────────┐    ┌─────────────────┐    ┌─────────────────────────┐                  │
│  │ Prometheus + │←───│ Evidently AI    │←───│ KServe / Ray Serve      │                  │
│  │ Grafana      │    │ (Drift Monitor) │    │ (Argo Rollouts Canary)  │                  │
│  └──────┬───────┘    └─────────────────┘    └─────────────────────────┘                  │
│         │                                                                                 │
│         ↓                                                                                 │
│  ┌──────────────┐    ┌─────────────────┐                                                 │
│  │ Alert Manager│───→│ Dagster/Argo WF │───→ (Retrain Loop)                              │
│  │ (trigger)    │    │ (orchestrate)   │                                                 │
│  └──────────────┘    └─────────────────┘                                                 │
│                                                                                           │
│  Infrastructure: CloudFormation (EKS) │ CI/CD: ArgoCD/GitHub Actions │ GitOps: FluxCD    │
│                                                                                           │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### CloudFormation — EKS Cluster for MLOps (Foundation)

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: EKS Cluster for MLOps - MLflow/OSS Stack

Parameters:
  ClusterName:
    Type: String
    Default: ml-platform-eks
  KubernetesVersion:
    Type: String
    Default: '1.30'
  Environment:
    Type: String
    AllowedValues: [dev, staging, prod]
    Default: prod

Resources:
  EKSCluster:
    Type: AWS::EKS::Cluster
    Properties:
      Name: !Ref ClusterName
      Version: !Ref KubernetesVersion
      RoleArn: !GetAtt EKSClusterRole.Arn
      ResourcesVpcConfig:
        SubnetIds:
          - !Ref PrivateSubnet1
          - !Ref PrivateSubnet2
          - !Ref PrivateSubnet3
        SecurityGroupIds:
          - !Ref ClusterSecurityGroup
        EndpointPublicAccess: false
        EndpointPrivateAccess: true
      Logging:
        ClusterLogging:
          EnabledTypes:
            - Type: api
            - Type: audit
            - Type: controllerManager
      EncryptionConfig:
        - Provider:
            KeyArn: !GetAtt EKSEncryptionKey.Arn
          Resources:
            - secrets

  # GPU Node Group for Training
  GPUTrainingNodeGroup:
    Type: AWS::EKS::Nodegroup
    Properties:
      ClusterName: !Ref EKSCluster
      NodegroupName: gpu-training
      NodeRole: !GetAtt NodeInstanceRole.Arn
      Subnets:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2
      InstanceTypes:
        - g5.2xlarge
        - g5.4xlarge
      CapacityType: SPOT
      ScalingConfig:
        DesiredSize: 0
        MinSize: 0
        MaxSize: 10
      Labels:
        workload-type: training
        gpu: "true"
      Taints:
        - Key: nvidia.com/gpu
          Value: "true"
          Effect: NO_SCHEDULE
      Tags:
        Project: MLOps
        Environment: !Ref Environment

  # GPU Node Group for Inference (On-Demand for stability)
  GPUInferenceNodeGroup:
    Type: AWS::EKS::Nodegroup
    Properties:
      ClusterName: !Ref EKSCluster
      NodegroupName: gpu-inference
      NodeRole: !GetAtt NodeInstanceRole.Arn
      Subnets:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2
      InstanceTypes:
        - g5.xlarge
      CapacityType: ON_DEMAND
      ScalingConfig:
        DesiredSize: 2
        MinSize: 2
        MaxSize: 20
      Labels:
        workload-type: inference
        gpu: "true"
      Taints:
        - Key: nvidia.com/gpu
          Value: "true"
          Effect: NO_SCHEDULE

  # CPU Node Group for MLflow, monitoring, etc.
  CPUNodeGroup:
    Type: AWS::EKS::Nodegroup
    Properties:
      ClusterName: !Ref EKSCluster
      NodegroupName: cpu-general
      NodeRole: !GetAtt NodeInstanceRole.Arn
      Subnets:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2
        - !Ref PrivateSubnet3
      InstanceTypes:
        - m6g.2xlarge  # Graviton for cost savings
        - m6g.xlarge
      CapacityType: ON_DEMAND
      ScalingConfig:
        DesiredSize: 3
        MinSize: 2
        MaxSize: 10
      Labels:
        workload-type: platform

  # IRSA for MLflow (S3, RDS access)
  MLflowServiceAccount:
    Type: AWS::IAM::Role
    Properties:
      RoleName: mlflow-service-role
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Federated: !Sub 'arn:aws:iam::${AWS::AccountId}:oidc-provider/${OIDCProvider}'
            Action: sts:AssumeRoleWithWebIdentity
            Condition:
              StringEquals:
                !Sub '${OIDCProvider}:sub': 'system:serviceaccount:ml-platform:mlflow'
      Policies:
        - PolicyName: mlflow-s3-access
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
                  - !Sub 'arn:aws:s3:::${MLArtifactsBucket}'
                  - !Sub 'arn:aws:s3:::${MLArtifactsBucket}/*'
```

---

## 13. Code Examples

### 13.1 Training Job Launch

**SageMaker:**

```python
import sagemaker
from sagemaker.pytorch import PyTorch

session = sagemaker.Session()
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

estimator = PyTorch(
    entry_point="train.py",
    source_dir="./src",
    role=role,
    instance_count=2,
    instance_type="ml.g5.2xlarge",
    framework_version="2.2.0",
    py_version="py310",
    hyperparameters={
        "epochs": 20,
        "batch-size": 64,
        "learning-rate": 0.001,
    },
    distribution={"torch_distributed": {"enabled": True}},
    use_spot_instances=True,
    max_wait=7200,
    max_run=3600,
    checkpoint_s3_uri="s3://ml-artifacts/checkpoints/",
    output_path="s3://ml-artifacts/models/",
    tags=[
        {"Key": "Project", "Value": "ChurnModel"},
        {"Key": "Team", "Value": "ML-Platform"},
    ],
)

estimator.fit(
    inputs={
        "train": "s3://ml-data-lake/train/",
        "validation": "s3://ml-data-lake/validation/",
    },
    wait=False,
    job_name="churn-model-train-20260602",
)
```

**MLflow/OSS (Ray on EKS):**

```python
import mlflow
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig

ray.init(address="ray://ray-head.ml-platform:10001")
mlflow.set_tracking_uri("http://mlflow.ml-platform:5000")

with mlflow.start_run(experiment_id="churn-model", run_name="train-v2"):
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "epochs": 20,
            "batch_size": 64,
            "lr": 0.001,
            "data_path": "s3://ml-data-lake/train/",
        },
        scaling_config=ScalingConfig(
            num_workers=2,
            use_gpu=True,
            resources_per_worker={"GPU": 1, "CPU": 8},
        ),
        run_config=RunConfig(
            checkpoint_config=CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min",
            ),
            storage_path="s3://ml-artifacts/ray-checkpoints/",
        ),
    )
    
    result = trainer.fit()
    mlflow.log_metrics({"final_loss": result.metrics["loss"]})
    mlflow.pytorch.log_model(result.checkpoint, "model",
                            registered_model_name="churn-classifier")
```

### 13.2 Model Registration

**SageMaker:**

```python
import boto3

sm_client = boto3.client("sagemaker")

# Create model package (register in registry)
response = sm_client.create_model_package(
    ModelPackageGroupName="customer-churn-model",
    ModelPackageDescription="Churn classifier v2.1 - AUC 0.94",
    InferenceSpecification={
        "Containers": [
            {
                "Image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/inference:v2",
                "ModelDataUrl": "s3://ml-artifacts/models/churn-v2/model.tar.gz",
                "Framework": "PYTORCH",
                "FrameworkVersion": "2.2.0",
            }
        ],
        "SupportedContentTypes": ["application/json"],
        "SupportedResponseMIMETypes": ["application/json"],
        "SupportedRealtimeInferenceInstanceTypes": [
            "ml.g5.xlarge", "ml.g5.2xlarge"
        ],
    },
    ModelApprovalStatus="PendingManualApproval",
    ModelMetrics={
        "ModelQuality": {
            "Statistics": {
                "ContentType": "application/json",
                "S3Uri": "s3://ml-artifacts/evaluation/metrics.json",
            }
        },
        "Bias": {
            "Report": {
                "ContentType": "application/json",
                "S3Uri": "s3://ml-artifacts/evaluation/bias_report.json",
            }
        },
    },
    CustomerMetadataProperties={
        "TrainingJobName": "churn-model-train-20260602",
        "DatasetVersion": "v1.3",
        "GitCommit": "abc123def",
    },
)

model_package_arn = response["ModelPackageArn"]
print(f"Registered: {model_package_arn}")

# Approve model (after human review)
sm_client.update_model_package(
    ModelPackageArn=model_package_arn,
    ModelApprovalStatus="Approved",
    ApprovalDescription="Approved by ML lead - meets all quality criteria",
)
```

**MLflow:**

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register from run artifact
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="customer-churn-classifier",
    tags={
        "framework": "pytorch",
        "dataset_version": "v1.3",
        "git_commit": "abc123def",
    },
)

# Add evaluation metadata
client.update_model_version(
    name="customer-churn-classifier",
    version=model_version.version,
    description="Churn classifier v2.1 - AUC 0.94, trained on Q1-Q2 2026 data",
)

# Set aliases for deployment
client.set_registered_model_alias(
    name="customer-churn-classifier",
    alias="challenger",
    version=model_version.version,
)

# Promote to champion after validation
client.set_registered_model_alias(
    name="customer-churn-classifier",
    alias="champion",
    version=model_version.version,
)

# Delete old alias
client.delete_registered_model_alias(
    name="customer-churn-classifier",
    alias="challenger",
)
```

### 13.3 Endpoint Deployment

**SageMaker:**

```python
from sagemaker.model import Model
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

model = Model(
    image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/inference:v2",
    model_data="s3://ml-artifacts/models/churn-v2/model.tar.gz",
    role=role,
    sagemaker_session=session,
)

predictor = model.deploy(
    initial_instance_count=2,
    instance_type="ml.g5.xlarge",
    endpoint_name="churn-endpoint-prod",
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
    data_capture_config=sagemaker.model_monitor.DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,
        destination_s3_uri="s3://ml-artifacts/data-capture/",
    ),
    wait=True,
)

# Test inference
result = predictor.predict({
    "features": [0.5, 23, 1200.50, 3, 0.8]
})
print(f"Prediction: {result}")
```

**KServe (kubectl apply):**

```python
import subprocess
import yaml

inference_service = {
    "apiVersion": "serving.kserve.io/v1beta1",
    "kind": "InferenceService",
    "metadata": {
        "name": "customer-churn-classifier",
        "namespace": "ml-serving",
    },
    "spec": {
        "predictor": {
            "minReplicas": 2,
            "maxReplicas": 20,
            "model": {
                "modelFormat": {"name": "mlflow"},
                "storageUri": "s3://ml-artifacts/mlflow-models/churn-classifier/champion",
                "resources": {
                    "requests": {"cpu": "2", "memory": "4Gi", "nvidia.com/gpu": "1"},
                    "limits": {"cpu": "4", "memory": "8Gi", "nvidia.com/gpu": "1"},
                },
            },
        }
    },
}

# Apply via kubectl
with open("/tmp/isvc.yaml", "w") as f:
    yaml.dump(inference_service, f)

subprocess.run(["kubectl", "apply", "-f", "/tmp/isvc.yaml"], check=True)
```

### 13.4 Canary Traffic Split

**SageMaker:**

```python
import boto3

sm_client = boto3.client("sagemaker")

# Update variant weights for canary
sm_client.update_endpoint_weights_and_capacities(
    EndpointName="churn-endpoint-prod",
    DesiredWeightsAndCapacities=[
        {"VariantName": "primary", "DesiredWeight": 0.9},
        {"VariantName": "canary", "DesiredWeight": 0.1},
    ],
)

# Monitor for 30 minutes, then shift more traffic
import time
time.sleep(1800)

# Check canary metrics
cw_client = boto3.client("cloudwatch")
response = cw_client.get_metric_statistics(
    Namespace="AWS/SageMaker",
    MetricName="Invocation5XXErrors",
    Dimensions=[
        {"Name": "EndpointName", "Value": "churn-endpoint-prod"},
        {"Name": "VariantName", "Value": "canary"},
    ],
    StartTime=time.time() - 1800,
    EndTime=time.time(),
    Period=300,
    Statistics=["Sum"],
)

error_count = sum(dp["Sum"] for dp in response["Datapoints"])

if error_count == 0:
    # Promote canary to 100%
    sm_client.update_endpoint_weights_and_capacities(
        EndpointName="churn-endpoint-prod",
        DesiredWeightsAndCapacities=[
            {"VariantName": "primary", "DesiredWeight": 0.0},
            {"VariantName": "canary", "DesiredWeight": 1.0},
        ],
    )
    print("Canary promoted to 100%")
else:
    # Rollback
    sm_client.update_endpoint_weights_and_capacities(
        EndpointName="churn-endpoint-prod",
        DesiredWeightsAndCapacities=[
            {"VariantName": "primary", "DesiredWeight": 1.0},
            {"VariantName": "canary", "DesiredWeight": 0.0},
        ],
    )
    print(f"Rollback: {error_count} errors detected")
```

**Argo Rollouts (kubectl patch):**

```python
import subprocess
import json

# Promote canary to next step
subprocess.run([
    "kubectl", "argo", "rollouts", "promote",
    "churn-model-rollout", "-n", "ml-serving"
], check=True)

# Check rollout status
result = subprocess.run(
    ["kubectl", "argo", "rollouts", "status", 
     "churn-model-rollout", "-n", "ml-serving", "-o", "json"],
    capture_output=True, text=True
)
status = json.loads(result.stdout)
print(f"Phase: {status['status']['phase']}")
print(f"Canary weight: {status['status'].get('canary', {}).get('weight', 0)}%")

# Manual abort if needed
if status["status"]["phase"] == "Degraded":
    subprocess.run([
        "kubectl", "argo", "rollouts", "abort",
        "churn-model-rollout", "-n", "ml-serving"
    ], check=True)
```

### 13.5 Drift Detection Setup

**SageMaker:**

```python
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

# Create baseline from training data
monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size_in_gb=50,
    max_runtime_in_seconds=3600,
)

monitor.suggest_baseline(
    baseline_dataset="s3://ml-data-lake/baseline/features.csv",
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri="s3://ml-artifacts/baselines/",
    wait=True,
)

# Schedule continuous monitoring
monitor.create_monitoring_schedule(
    monitor_schedule_name="churn-drift-monitor",
    endpoint_input="churn-endpoint-prod",
    output_s3_uri="s3://ml-artifacts/monitoring/",
    statistics=monitor.baseline_statistics(),
    constraints=monitor.suggested_constraints(),
    schedule_cron_expression="cron(0 * ? * * *)",  # hourly
)
```

**Evidently AI (OSS):**

```python
from evidently.ui.workspace import Workspace
from evidently.ui.dashboards import DashboardConfig, PanelConfig, ReportConfig
from evidently.metrics import DataDriftTable, DatasetDriftMetric
from evidently.report import Report
import schedule
import time

# Create Evidently workspace (self-hosted dashboard)
workspace = Workspace.create("s3://ml-artifacts/evidently-workspace/")

project = workspace.create_project("Churn Model Monitoring")
project.dashboard = DashboardConfig(
    name="Drift Dashboard",
    panels=[
        PanelConfig(title="Dataset Drift", metrics=[DatasetDriftMetric]),
        PanelConfig(title="Feature Drift Detail", metrics=[DataDriftTable]),
    ],
)

def run_drift_check():
    """Scheduled drift check - runs every hour."""
    import pandas as pd
    import requests
    
    reference = pd.read_parquet("s3://ml-data-lake/reference/features.parquet")
    current = pd.read_parquet("s3://ml-data-lake/production/last_hour.parquet")
    
    report = Report(metrics=[DatasetDriftMetric(), DataDriftTable()])
    report.run(reference_data=reference, current_data=current)
    
    # Save to workspace for dashboard
    workspace.add_report(project.id, report)
    
    # Export metrics to Prometheus
    results = report.as_dict()
    drift_detected = results["metrics"][0]["result"]["dataset_drift"]
    drift_share = results["metrics"][0]["result"]["drift_share"]
    
    # Push to Prometheus Pushgateway
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
    
    registry = CollectorRegistry()
    drift_gauge = Gauge("model_dataset_drift", "Dataset drift detected",
                       registry=registry)
    drift_share_gauge = Gauge("model_drift_share", "Share of drifted features",
                            registry=registry)
    
    drift_gauge.set(1 if drift_detected else 0)
    drift_share_gauge.set(drift_share)
    
    push_to_gateway("prometheus-pushgateway:9091", job="drift_monitor",
                   registry=registry)
    
    if drift_detected and drift_share > 0.3:
        # Trigger retraining via Argo Events
        requests.post(
            "http://argo-events-webhook.ml-pipelines:12000/retrain",
            json={"reason": "drift", "drift_share": drift_share}
        )

# Schedule hourly
schedule.every(1).hours.do(run_drift_check)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## 14. Decision Matrix

### When to Use SageMaker vs MLflow/OSS

| Factor | SageMaker ✓ | MLflow/OSS ✓ | Notes |
|--------|-------------|--------------|-------|
| **Team Size** | 1-10 ML engineers | 5+ platform + ML engineers | OSS needs dedicated platform team |
| **Budget** | Medium-High (managed premium) | Low-Medium (DIY savings) | At scale, OSS can be 30-40% cheaper |
| **Cloud Strategy** | AWS-only | Multi-cloud / Hybrid | MLflow runs anywhere |
| **Time to Market** | Fast (weeks) | Slower (months for platform) | SageMaker: faster first model |
| **Compliance** | Strong (built-in) | Custom (must build) | SageMaker: SOC2, HIPAA, FedRAMP OOTB |
| **Customization** | Limited | Unlimited | OSS: full control over every component |
| **Existing K8s** | N/A | Leverage existing EKS investment | If you have EKS, OSS is natural fit |
| **Model Scale** | 1-50 models | 50-1000+ models | Multi-model endpoints help, but OSS scales better |
| **LLM Serving** | JumpStart/Bedrock + SageMaker | vLLM/TGI on EKS | OSS has more flexibility for LLM serving |
| **Experiment Velocity** | Medium (job submission overhead) | High (local + cluster) | OSS: faster iteration cycles |
| **Governance** | Built-in (Model Cards, Lineage) | Must build (MLflow + custom) | SageMaker wins on governance OOTB |
| **Vendor Relationship** | AWS enterprise agreement | N/A | SageMaker: negotiate discounts at scale |

### Recommended Architecture Patterns

#### Pattern 1: "SageMaker First" (Recommended for most teams)

```
Best for: Teams < 10, all-in AWS, need fast time-to-value
Stack: SageMaker (everything) + CloudFormation + CodePipeline
Cost: $$$ but lower operational burden
```

#### Pattern 2: "Hybrid" (Best of both worlds)

```
Best for: Teams with EKS, want flexibility + some managed services
Stack: MLflow (tracking/registry) + SageMaker (training) + KServe (serving)
Cost: $$ balanced approach
```

#### Pattern 3: "Full OSS on EKS" (Maximum control)

```
Best for: Large platform teams, multi-cloud, cost-sensitive at scale
Stack: MLflow + Ray + KServe + Argo + Prometheus + Feast (all on EKS)
Cost: $ at scale, but $$$ in platform engineering time
```

#### Pattern 4: "SageMaker + EKS Inference" (Common at DISH-scale)

```
Best for: AWS shops with EKS investment wanting flexible serving
Stack: SageMaker (training/registry) + vLLM/Ray Serve on EKS (inference)
Cost: $$ leverages existing EKS, SageMaker for heavy lifting
Advantage: Use SageMaker's managed training + spot, serve on your terms
```

### Final Recommendation for DISH Cloud Architecture

Given your deep AWS/EKS expertise and CloudFormation preference:

1. **Training & Experimentation**: SageMaker Training Jobs (managed spot, distributed) + MLflow Tracking (for flexibility)
2. **Feature Store**: SageMaker Feature Store (tight S3/Glue integration)
3. **Model Registry**: MLflow Model Registry on EKS (cloud-agnostic, better developer UX)
4. **Serving**: KServe or Ray Serve on existing EKS clusters (vLLM for LLMs)
5. **Canary/Rollout**: Argo Rollouts + Istio on EKS (already have the infrastructure)
6. **Monitoring**: Evidently AI + Prometheus/Grafana on EKS (unified observability)
7. **Orchestration**: Dagster or Argo Workflows on EKS
8. **IaC**: CloudFormation for AWS resources, ArgoCD for K8s manifests

This hybrid approach gives you:
- Managed compute for expensive training (no GPU node management)
- Full control over inference (cost, customization, multi-model)
- Cloud-agnostic model artifacts (portable if needed)
- Unified Kubernetes observability stack

---

## Appendix A: IAM Role for SageMaker

```yaml
  SageMakerExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: !Sub 'SageMakerExecRole-${Environment}'
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service:
                - sagemaker.amazonaws.com
                - events.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
      Policies:
        - PolicyName: S3Access
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - s3:GetObject
                  - s3:PutObject
                  - s3:DeleteObject
                  - s3:ListBucket
                Resource:
                  - !Sub 'arn:aws:s3:::${MLDataLakeBucket}'
                  - !Sub 'arn:aws:s3:::${MLDataLakeBucket}/*'
                  - !Sub 'arn:aws:s3:::${MLArtifactsBucket}'
                  - !Sub 'arn:aws:s3:::${MLArtifactsBucket}/*'
        - PolicyName: KMSAccess
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - kms:Decrypt
                  - kms:GenerateDataKey
                  - kms:DescribeKey
                Resource:
                  - !GetAtt MLKMSKey.Arn
        - PolicyName: ECRAccess
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - ecr:GetDownloadUrlForLayer
                  - ecr:BatchGetImage
                  - ecr:GetAuthorizationToken
                Resource: '*'
```

## Appendix B: Quick Reference Commands

```bash
# === SageMaker CLI ===
# List training jobs
aws sagemaker list-training-jobs --status-equals Completed --max-results 10

# Describe endpoint
aws sagemaker describe-endpoint --endpoint-name churn-endpoint-prod

# Update endpoint traffic
aws sagemaker update-endpoint-weights-and-capacities \
  --endpoint-name churn-endpoint-prod \
  --desired-weights-and-capacities '[{"VariantName":"primary","DesiredWeight":0.5},{"VariantName":"canary","DesiredWeight":0.5}]'

# List model packages
aws sagemaker list-model-packages --model-package-group-name customer-churn-model

# === MLflow CLI ===
# Serve model locally
mlflow models serve -m "models:/customer-churn-classifier@champion" -p 5001

# List experiments
mlflow experiments search --filter "name LIKE '%churn%'"

# === Kubernetes (OSS stack) ===
# Check KServe inference services
kubectl get inferenceservices -n ml-serving

# Argo Rollouts status
kubectl argo rollouts get rollout churn-model-rollout -n ml-serving

# Feast materialize features
feast materialize $(date -d "yesterday" +%Y-%m-%dT00:00:00) $(date +%Y-%m-%dT00:00:00)

# Check drift metrics in Prometheus
curl -s "http://prometheus:9090/api/v1/query?query=model_dataset_drift" | jq .
```

---

*Guide authored for production MLOps at enterprise scale. All code examples are production-ready with error handling omitted for clarity. Adapt CloudFormation parameters, instance types, and scaling configs to your specific workload characteristics.*
