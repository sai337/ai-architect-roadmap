# MLOps vs AIOps vs DevOps: Comprehensive Comparison

> **Context**: This guide is tailored for Cloud Architects operating at enterprise scale (400+ AWS accounts, EKS clusters, Control Tower governance) — the kind of environment where understanding these three disciplines and their intersections is critical.

---

## 1. Definitions & Core Focus

### DevOps — Software Delivery Lifecycle Automation

**Definition**: A set of practices, tools, and cultural philosophies that automate and integrate the processes of software development (Dev) and IT operations (Ops) to shorten the systems development lifecycle while delivering features, fixes, and updates frequently and reliably.

**Core Focus**: *How do we ship code faster, safer, and more reliably?*

```
Developer writes code → Automated build → Automated test → Automated deploy → Monitor → Repeat
```

**Key Principle**: Break down silos between development and operations teams. Everything is code (infrastructure, configuration, policy). Automate everything that can be automated.

---

### MLOps — Machine Learning Model Lifecycle Automation

**Definition**: A set of practices that combines Machine Learning, DevOps, and Data Engineering to deploy and maintain ML systems in production reliably and efficiently. It's the discipline of operationalizing ML models — not just training them in notebooks, but running them as production services.

**Core Focus**: *How do we get ML models from experimentation to production and keep them performing well over time?*

```
Data → Feature Engineering → Training → Evaluation → Registry → Deploy → Monitor → Retrain
```

**Key Principle**: Models degrade. Data drifts. A model that was 95% accurate last month might be 70% accurate today. MLOps treats models as living artifacts that require continuous monitoring, retraining, and governance — not one-time deployments.

---

### AIOps — AI-Powered IT Operations

**Definition**: The application of artificial intelligence (machine learning, NLP, anomaly detection) TO IT operations — using AI as a tool to manage, monitor, and automate infrastructure and application operations at scale that would be impossible for humans alone.

**Core Focus**: *How do we use AI to operate our infrastructure better than humans can alone?*

```
Telemetry floods in → AI correlates & clusters → Anomalies detected → Root cause identified → Auto-remediation triggered
```

**Key Principle**: At scale (400+ accounts, thousands of services, millions of metrics), humans cannot process the signal-to-noise ratio. AIOps uses ML models to find patterns, predict failures, correlate events, and take action — turning operational chaos into actionable intelligence.

---

### The Critical Distinction

| | DevOps | MLOps | AIOps |
|---|---|---|---|
| **Relationship to AI** | Doesn't use AI (uses automation) | Manages AI (lifecycle of models) | Uses AI (AI as the operator) |
| **One-liner** | "Automate software delivery" | "Operationalize ML models" | "AI operates your infrastructure" |

---

## 2. Key Differences Table

| Dimension | DevOps | MLOps | AIOps |
|-----------|--------|-------|-------|
| **Goal/Purpose** | Ship software faster and more reliably | Get ML models to production and keep them healthy | Use AI to detect, diagnose, and resolve operational issues |
| **Primary Users** | Software Engineers, SREs, Platform Engineers | Data Scientists, ML Engineers, Data Engineers | SREs, Operations Teams, NOC Engineers |
| **What's Being Managed** | Application code, infrastructure config | Models + Data + Code + Features + Experiments | IT infrastructure, telemetry, incidents |
| **Pipeline Types** | CI/CD (build → test → deploy) | ML Pipeline (data → train → evaluate → deploy) | Observability pipeline (ingest → correlate → detect → act) |
| **Testing Methods** | Unit tests, integration tests, E2E tests, load tests | Model validation, A/B tests, shadow testing, data quality checks, bias detection | Anomaly detection accuracy, false positive rates, remediation success rates |
| **Monitoring Focus** | Uptime, latency, error rates, throughput (SLIs/SLOs) | Model accuracy, data drift, prediction latency, feature distribution shifts | Infrastructure health, event correlation, noise reduction, MTTR |
| **Deployment Patterns** | Blue/green, canary, rolling update | Shadow deployment, champion/challenger, multi-armed bandit | Progressive rollout of detection rules, playbook automation |
| **Versioning** | Code (Git) | Code + Data + Model + Hyperparams + Environment + Features | Detection rules, runbooks, correlation patterns |
| **Feedback Loops** | User feedback → backlog → sprint → deploy | Prediction outcomes → drift detection → retrain trigger | Incident outcomes → model refinement → better detection |
| **Key Metrics** | Deployment frequency, lead time, MTTR, change failure rate (DORA) | Model accuracy, F1 score, inference latency, retraining frequency | Alert noise reduction, MTTD, MTTR, false positive rate |
| **Tools Ecosystem** | Jenkins, GitHub Actions, ArgoCD, Terraform, Helm | SageMaker, MLflow, Kubeflow, Feature Stores, Model Registries | Dynatrace, Splunk ITSI, PagerDuty AIOps, AWS DevOps Guru |
| **Data Dependency** | Low (code is deterministic) | Very High (model quality = data quality) | High (quality of detection depends on telemetry quality) |
| **Reproducibility Challenge** | Low (same code = same build) | High (same code ≠ same model without same data + environment) | Medium (same data may produce different anomaly thresholds over time) |

---

## 3. Real-Time Examples

### DevOps Examples

#### Example 1: CI/CD Pipeline for a Microservice (EKS Deployment)

**Scenario**: A DISH platform team deploys a billing microservice across 3 EKS clusters.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DevOps CI/CD Pipeline — Billing Service                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Developer          CodePipeline          CodeBuild           EKS           │
│  ─────────          ────────────          ─────────           ───           │
│                                                                             │
│  git push ──────►  Source Stage  ──────►  Build Stage                       │
│  (feature/          (webhook           (docker build,                       │
│   billing-fix)       trigger)            unit tests,                        │
│                                          SAST scan)                         │
│                                              │                              │
│                                              ▼                              │
│                                         Test Stage                          │
│                                       (integration tests                    │
│                                        against staging DB)                  │
│                                              │                              │
│                                              ▼                              │
│                                        Deploy to Dev                        │
│                                       (ArgoCD sync to                       │
│                                        dev EKS cluster)                     │
│                                              │                              │
│                                              ▼                              │
│                                     Approval Gate (auto                     │
│                                     for dev, manual for                     │
│                                     prod)                                   │
│                                              │                              │
│                                              ▼                              │
│                                       Deploy to Prod                        │
│                                      (canary → 10% ──►                      │
│                                       50% ──► 100%)                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**What happens**:
1. Developer pushes to `main` branch
2. CodePipeline triggers automatically
3. CodeBuild runs `docker build`, unit tests, Trivy container scan
4. Helm chart is rendered and validated
5. ArgoCD deploys to dev cluster, runs smoke tests
6. On success → canary deployment to production EKS cluster
7. CloudWatch monitors error rate; auto-rollback if errors > 1%

---

#### Example 2: Blue/Green Deployment of a Web Application

**Scenario**: DISH customer portal needs zero-downtime deployments with instant rollback capability.

```
                    ┌──────────────────┐
                    │   Route 53 DNS   │
                    │  (weighted 100%) │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │       ALB        │
                    │  (Target Group   │
                    │   switching)     │
                    └───┬─────────┬────┘
                        │         │
            ┌───────────▼──┐  ┌──▼───────────┐
            │  BLUE (v2.1) │  │ GREEN (v2.2) │
            │  (current)   │  │  (new)       │
            │              │  │              │
            │  EKS Pods    │  │  EKS Pods    │
            │  ██████████  │  │  ██████████  │
            │              │  │              │
            └──────────────┘  └──────────────┘
                 100% ◄──────────── 0%
                                    │
            After validation:       │
                 0%  ────────────► 100%
```

**Process**:
1. Green environment deployed with v2.2 (no traffic yet)
2. Synthetic tests run against green (health checks, smoke tests)
3. CodeDeploy shifts ALB target group: Blue → Green
4. If errors spike → instant rollback by switching back to Blue
5. Blue environment becomes the next "green" for future deploy

---

#### Example 3: Infrastructure as Code with Terraform + Control Tower

**Scenario**: Provisioning a new AWS account in the DISH landing zone for a new business unit.

```yaml
# Account Vending Machine — Terraform
module "new_account" {
  source = "git::https://github.dish.internal/platform/account-factory.git"

  account_name    = "dish-streaming-analytics-prod"
  ou_path         = "Production/Streaming"
  account_email   = "aws+streaming-analytics-prod@dish.com"

  # Standard guardrails applied automatically via Control Tower
  enable_guardduty    = true
  enable_config       = true
  enable_cloudtrail   = true
  vpc_cidr            = "10.45.0.0/16"
  
  # EKS baseline
  deploy_eks          = true
  eks_version         = "1.28"
  node_groups = {
    general = { instance_types = ["m6i.xlarge"], min = 3, max = 10 }
    gpu     = { instance_types = ["g5.xlarge"], min = 0, max = 5 }
  }

  tags = {
    CostCenter  = "streaming-analytics"
    Environment = "production"
    ManagedBy   = "terraform"
  }
}
```

**What happens automatically**:
1. Terraform plan runs in CodeBuild (PR shows diff)
2. On merge → Terraform apply via CodePipeline
3. Control Tower enrolls account, applies SCPs
4. Baseline CloudFormation StackSets deploy (VPC, security, logging)
5. EKS cluster bootstrapped with ArgoCD, Dynatrace agent, Falco
6. Account appears in DISH's central monitoring within minutes

---

### MLOps Examples

#### Example 1: End-to-End Training Pipeline

**Scenario**: DISH builds a churn prediction model using customer usage data across 400+ accounts.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              MLOps Training Pipeline — Churn Prediction Model                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────────┐     │
│  │  Data    │    │   Feature    │    │  Model    │    │  Model       │     │
│  │  Ingest  │───►│  Engineering │───►│  Training │───►│  Evaluation  │     │
│  │          │    │              │    │           │    │              │     │
│  │ S3 Data  │    │ SageMaker    │    │ XGBoost + │    │ Accuracy:    │     │
│  │ Lake     │    │ Feature      │    │ Neural    │    │ 94.2%        │     │
│  │ (cross-  │    │ Store        │    │ Network   │    │ AUC: 0.97    │     │
│  │ account) │    │              │    │ ensemble  │    │ Bias check:  │     │
│  └──────────┘    └──────────────┘    └───────────┘    │ PASS         │     │
│                                                       └──────┬───────┘     │
│                                                              │              │
│       ┌─────────────────────────────────────────────────────┘              │
│       │                                                                     │
│       ▼                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────┐             │
│  │   Model      │    │   Shadow     │    │   Production      │             │
│  │   Registry   │───►│   Deploy     │───►│   Endpoint        │             │
│  │              │    │              │    │                   │             │
│  │  v3.2.1     │    │  Compare vs  │    │  SageMaker        │             │
│  │  Approved   │    │  v3.1.0      │    │  Real-time        │             │
│  │  by ML Lead │    │  (no impact) │    │  Inference        │             │
│  └──────────────┘    └──────────────┘    └───────────────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Pipeline (SageMaker Pipelines)**:
1. **Data Ingestion**: Pull 6 months of customer data from centralized S3 data lake
2. **Feature Engineering**: Compute 47 features (usage patterns, billing history, support tickets), store in Feature Store
3. **Training**: Train 3 model variants, hyperparameter tuning with Bayesian optimization
4. **Evaluation**: Check accuracy > 93%, bias across demographic groups, latency < 50ms
5. **Registry**: Register model with metadata, lineage, approval status
6. **Shadow Deploy**: New model receives live traffic but predictions aren't used (compared silently)
7. **Production**: After 7 days of shadow validation, promote to production endpoint

---

#### Example 2: Model Retraining Triggered by Data Drift

**Scenario**: The churn model's accuracy degrades because customer behavior changed post-product launch.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  Data Drift Detection → Auto-Retrain Flow                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Production          SageMaker              EventBridge        SageMaker    │
│  Endpoint            Model Monitor                             Pipeline     │
│  ─────────           ─────────────          ───────────        ─────────    │
│                                                                             │
│  Predictions ──────► Capture inference                                      │
│  served              data (input +                                          │
│  normally            predictions)                                           │
│       │                    │                                                │
│       │                    ▼                                                │
│       │              Weekly drift                                           │
│       │              analysis:                                              │
│       │              ┌─────────────────┐                                    │
│       │              │ Feature: avg_    │                                    │
│       │              │ stream_hours     │                                    │
│       │              │ Baseline: μ=4.2  │                                    │
│       │              │ Current:  μ=7.8  │                                    │
│       │              │ Drift: DETECTED  │──────► Rule triggers              │
│       │              │ (PSI > 0.25)     │        retraining                 │
│       │              └─────────────────┘        pipeline                    │
│       │                                              │                      │
│       │                                              ▼                      │
│       │                                         New model                   │
│       │                                         trained on                  │
│       │                                         recent 6 months             │
│       │                                              │                      │
│       │                                              ▼                      │
│       │◄──────────────────────────────────── New model deployed             │
│       │         (after evaluation passes)    (v3.3.0)                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**What happens**:
1. SageMaker Model Monitor runs weekly data quality checks
2. Detects that `avg_stream_hours` distribution shifted significantly (PSI > 0.25)
3. CloudWatch alarm fires → EventBridge rule triggers SageMaker Pipeline
4. Pipeline retrains on latest 6 months of data
5. If new model accuracy > current model + bias checks pass → auto-deploy
6. If not → alert ML team for manual investigation

---

#### Example 3: A/B Testing Between Model Versions

**Scenario**: Testing whether a new recommendation algorithm improves engagement for DISH streaming users.

```
                      ┌──────────────────────┐
                      │    API Gateway       │
                      │  /recommendations    │
                      └──────────┬───────────┘
                                 │
                      ┌──────────▼───────────┐
                      │  SageMaker Endpoint  │
                      │  (Production Variant)│
                      └──────────┬───────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
              ┌─────▼─────┐     │     ┌──────▼─────┐
              │ Variant A  │     │     │ Variant B  │
              │ (Control)  │     │     │(Treatment) │
              │            │     │     │            │
              │ CF-based   │     │     │ Deep       │
              │ Filtering  │     │     │ Learning   │
              │ v2.1       │     │     │ v3.0-beta  │
              │            │     │     │            │
              │ Traffic:70%│     │     │ Traffic:30%│
              └────────────┘     │     └────────────┘
                                 │
                      ┌──────────▼───────────┐
                      │  Metrics Collection  │
                      │  ─────────────────── │
                      │  Click-through rate  │
                      │  Watch completion    │
                      │  User satisfaction   │
                      │  Latency p99         │
                      └──────────────────────┘
```

**Outcome tracking**:
- After 2 weeks: Variant B shows +12% click-through, +8% watch completion
- Statistical significance reached (p < 0.05)
- Automated promotion: Variant B becomes 100% traffic
- Variant A archived in Model Registry

---

#### Example 4: Feature Store Powering Real-Time Recommendations

**Scenario**: Real-time personalized content recommendations using pre-computed and real-time features.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Feature Store Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OFFLINE (Batch Features)              ONLINE (Real-Time Features)          │
│  ─────────────────────────             ───────────────────────────          │
│                                                                             │
│  ┌──────────────┐                     ┌──────────────────┐                  │
│  │ Glue ETL Job │                     │ Kinesis Stream   │                  │
│  │ (nightly)    │                     │ (user clicks,    │                  │
│  │              │                     │  searches, views)│                  │
│  └──────┬───────┘                     └────────┬─────────┘                  │
│         │                                      │                            │
│         ▼                                      ▼                            │
│  ┌──────────────────────────────────────────────────────┐                   │
│  │           SageMaker Feature Store                    │                   │
│  │                                                      │                   │
│  │  Feature Group: "user_preferences"                   │                   │
│  │  ├── genre_affinity_vector (128-dim) [offline]       │                   │
│  │  ├── avg_watch_duration [offline]                    │                   │
│  │  ├── last_5_watched_ids [online]                     │                   │
│  │  ├── session_duration_current [online]               │                   │
│  │  └── time_of_day_bucket [online]                     │                   │
│  │                                                      │                   │
│  └──────────────────────────┬───────────────────────────┘                   │
│                             │                                               │
│                             ▼                                               │
│                  ┌─────────────────────┐                                    │
│                  │  Recommendation     │                                    │
│                  │  Model Endpoint     │                                    │
│                  │  (< 20ms latency)   │                                    │
│                  └─────────────────────┘                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### AIOps Examples

#### Example 1: Anomaly Detection on CloudWatch Metrics

**Scenario**: Detecting unusual CPU patterns across 400+ DISH AWS accounts before thresholds are breached.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│            AIOps Anomaly Detection — Multi-Account Monitoring                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  400+ AWS Accounts                                                          │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                                  │
│  │Acct1│ │Acct2│ │Acct3│ │ ... │ │Ac400│                                   │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘                                  │
│     │       │       │       │       │                                       │
│     └───────┴───────┴───────┴───────┘                                       │
│                     │                                                        │
│                     ▼                                                        │
│     ┌───────────────────────────────────┐                                   │
│     │  CloudWatch Cross-Account         │                                   │
│     │  Observability (metrics stream)   │                                   │
│     └───────────────────┬───────────────┘                                   │
│                         │                                                    │
│                         ▼                                                    │
│     ┌───────────────────────────────────┐                                   │
│     │  CloudWatch Anomaly Detection     │                                   │
│     │  ─────────────────────────────    │                                   │
│     │  ML model learns "normal"         │                                   │
│     │  patterns per metric:             │                                   │
│     │                                   │                                   │
│     │  Normal: CPU follows weekly       │                                   │
│     │  cycle, peaks Mon-Fri 9am-5pm     │                                   │
│     │                                   │                                   │
│     │  ⚠️  ANOMALY DETECTED:            │                                   │
│     │  EKS node group "payments"        │                                   │
│     │  CPU at 78% on Sunday 2am         │                                   │
│     │  (expected band: 15-25%)          │                                   │
│     │  Confidence: 99.2%                │                                   │
│     └───────────────────┬───────────────┘                                   │
│                         │                                                    │
│                         ▼                                                    │
│     ┌───────────────────────────────────┐                                   │
│     │  Alert BEFORE threshold breach    │                                   │
│     │  (CPU would hit 100% in ~40min    │                                   │
│     │   based on trend extrapolation)   │                                   │
│     └───────────────────────────────────┘                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why this matters**: A static alert (CPU > 80%) would fire too late. The AI detected that 78% CPU on Sunday 2am is anomalous because the *pattern* is wrong, not just the *threshold*. This gives the team 40 minutes of lead time.

---

#### Example 2: Log Clustering/Correlation Across 400+ Accounts

**Scenario**: A deployment in one account causes cascading failures detected by AI correlation.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              AIOps Log Correlation — Cascading Failure Detection             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  11:42:03  [acct-payments]   ERROR: Connection timeout to auth-service      │
│  11:42:04  [acct-payments]   ERROR: Connection timeout to auth-service      │
│  11:42:04  [acct-streaming]  ERROR: 503 from payments-api                   │
│  11:42:05  [acct-auth]       WARN:  Memory pressure, GC taking 4.2s        │
│  11:42:05  [acct-payments]   ERROR: Connection pool exhausted               │
│  11:42:06  [acct-streaming]  ERROR: Retry failed for billing check          │
│  11:42:06  [acct-mobile-bff] ERROR: Upstream timeout (payments)             │
│  11:42:07  [acct-auth]       ERROR: OOMKilled pod auth-service-7b4f2        │
│  ... (347 more error lines in 30 seconds)                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  AI CORRELATION ENGINE                                      │            │
│  │  ───────────────────────                                    │            │
│  │                                                             │            │
│  │  Input: 400+ raw log events across 12 accounts              │            │
│  │                                                             │            │
│  │  Analysis:                                                  │            │
│  │  1. Temporal clustering: all events within 4-second window  │            │
│  │  2. Dependency graph: auth → payments → streaming → mobile  │            │
│  │  3. Root cause identification: auth-service OOM             │            │
│  │                                                             │            │
│  │  OUTPUT:                                                    │            │
│  │  ┌─────────────────────────────────────────────────┐        │            │
│  │  │ INCIDENT #4521                                  │        │            │
│  │  │ Root Cause: auth-service OOM in acct-auth       │        │            │
│  │  │ Impact: 4 downstream services affected          │        │            │
│  │  │ Accounts: payments, streaming, mobile-bff, web  │        │            │
│  │  │ Blast Radius: ~45,000 users                     │        │            │
│  │  │ Suggested Fix: Scale auth-service, investigate  │        │            │
│  │  │   memory leak in recent deploy (v2.4.1)         │        │            │
│  │  └─────────────────────────────────────────────────┘        │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Without AIOps**: On-call gets 400+ alerts, spends 30 minutes correlating manually.
**With AIOps**: One incident, root cause identified, blast radius known — in seconds.

---

#### Example 3: Auto-Remediation Flow

**Scenario**: AI detects a known issue pattern and auto-resolves without human intervention.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AIOps Auto-Remediation Pipeline                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐     ┌──────────────┐     ┌─────────────┐     ┌───────────┐  │
│  │  DETECT  │────►│   CLASSIFY   │────►│   DECIDE    │────►│  EXECUTE  │  │
│  └──────────┘     └──────────────┘     └─────────────┘     └───────────┘  │
│                                                                             │
│  CloudWatch       Pattern matching:     Confidence > 95%?   SSM Runbook:   │
│  detects disk     "EBS volume full      Historical success  1. Identify    │
│  usage > 90%      on EKS node,          rate for this       old logs       │
│  on EKS worker    caused by             pattern: 99.1%      2. Archive to  │
│  node             unrotated logs"       (resolved 847/854   S3             │
│                                         times automatically)3. Clean /var  │
│                   Known pattern:                             4. Verify node │
│                   ID #127                Auto-approve ✓      health         │
│                   Confidence: 98.7%                          5. Close       │
│                                                             incident       │
│                                                                             │
│  Timeline:                                                                  │
│  11:42:00 — Anomaly detected                                                │
│  11:42:03 — Pattern classified                                              │
│  11:42:05 — Auto-remediation approved                                       │
│  11:42:12 — Runbook execution started                                       │
│  11:42:45 — Disk freed (from 92% → 34%)                                    │
│  11:42:47 — Incident auto-closed                                            │
│  11:42:48 — Team notified (FYI, already resolved)                           │
│                                                                             │
│  Total time: 48 seconds (vs. 23 minutes average manual MTTR)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### Example 4: Predictive Capacity Planning

**Scenario**: AI predicts that DISH's streaming EKS cluster will run out of capacity during a major sporting event.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              AIOps Predictive Capacity Planning                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Historical Pattern Analysis:                                               │
│                                                                             │
│  Capacity                                                                   │
│  Usage (%)                                                                  │
│  100│                                           ╱ ← PREDICTED              │
│     │                                         ╱    (Super Bowl              │
│   80│                                  ●    ╱       Sunday)                 │
│     │                                ●    ╱                                 │
│   60│              ●               ●    ╱                                   │
│     │            ●   ●           ●                                          │
│   40│     ●    ●       ●       ●                                            │
│     │   ●   ●           ●   ●                                              │
│   20│ ●                    ●                                                │
│     │                                                                       │
│    0└──────────────────────────────────────────── Time                      │
│     Oct   Nov   Dec   Jan   Feb                                             │
│                                    ▲                                         │
│                                    │                                         │
│                          AI Prediction (21 days ahead):                      │
│                          "Current capacity will be exhausted                 │
│                           by Feb 12 (Super Bowl). Need +340                  │
│                           additional pods / 45 new nodes."                   │
│                                                                             │
│  Action taken automatically:                                                │
│  1. Karpenter provisioner limits increased                                  │
│  2. Node warm pool pre-scaled (spot + on-demand mix)                        │
│  3. Reserved capacity reserved via Capacity Reservations                    │
│  4. CDN cache warming scheduled for T-2 hours                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### Example 5: Noise Reduction in Alerting

**Scenario**: A network blip causes 100+ alerts across accounts — AI reduces to 3 actionable incidents.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AIOps Alert Noise Reduction                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RAW ALERTS (5-minute window):                                              │
│  ═══════════════════════════════                                             │
│  🔴 [acct-012] EKS pod CrashLoopBackoff (x23 pods)                         │
│  🔴 [acct-012] ALB 5xx spike > 10%                                         │
│  🔴 [acct-015] DynamoDB throttling                                          │
│  🔴 [acct-015] Lambda timeout errors                                        │
│  🔴 [acct-018] RDS connection failures                                      │
│  🟡 [acct-012] Memory utilization > 85%                                     │
│  🟡 [acct-022] API Gateway 429 errors                                       │
│  🔴 [acct-018] ECS task failures                                            │
│  ... (+94 more alerts)                                                      │
│                                                                             │
│  Total: 102 alerts across 14 accounts                                       │
│                                                                             │
│         │                                                                   │
│         │  AI Processing:                                                   │
│         │  1. Temporal correlation (all within 90s window)                   │
│         │  2. Topology-aware grouping (shared VPC peering)                   │
│         │  3. Causal analysis (network = upstream of all)                    │
│         │  4. Deduplication (23 pod alerts = 1 event)                        │
│         │  5. Severity ranking by blast radius                               │
│         ▼                                                                   │
│                                                                             │
│  ACTIONABLE INCIDENTS:                                                      │
│  ═════════════════════                                                       │
│  🔴 INCIDENT 1 (P1): Transit Gateway throughput exceeded                    │
│     Root cause: TGW in shared-network account at capacity                   │
│     Impact: 14 accounts, all services using cross-account routing           │
│     Action: Scale TGW bandwidth / failover to secondary                     │
│                                                                             │
│  🟡 INCIDENT 2 (P3): Unrelated — DynamoDB hot partition                     │
│     Root cause: Partition key skew in acct-015                              │
│     Impact: 1 account, 2 services                                           │
│     Action: Review partition key design                                      │
│                                                                             │
│  🟡 INCIDENT 3 (P4): Pre-existing — Memory leak in acct-012                │
│     Root cause: Known issue, tracked in JIRA-4521                           │
│     Impact: 1 account, contained                                            │
│     Action: Already scheduled for next sprint                               │
│                                                                             │
│  Noise reduction: 102 alerts → 3 incidents (97% reduction)                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. How They Overlap & Complement Each Other

### The Relationship Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                    THE VIRTUOUS CYCLE                                        │
│                                                                             │
│         ┌──────────────────────────────────────────────┐                    │
│         │                                              │                    │
│         ▼                                              │                    │
│  ┌─────────────┐         ┌─────────────┐         ┌────┴────────┐          │
│  │             │         │             │         │             │          │
│  │   DevOps   │────────►│   MLOps     │────────►│   AIOps     │          │
│  │             │         │             │         │             │          │
│  │ Builds the │         │ Trains the  │         │ Monitors    │          │
│  │ platform   │         │ models      │         │ everything  │          │
│  │ & pipelines│         │             │         │             │          │
│  └─────────────┘         └─────────────┘         └─────────────┘          │
│        ▲                                               │                    │
│        │                                               │                    │
│        └───────────────────────────────────────────────┘                    │
│              AIOps insights improve the platform                             │
│              (auto-scaling, self-healing, optimization)                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Specific Overlaps

#### DevOps Enables MLOps
```
┌────────────────────────────────────────────────────────────┐
│  DevOps Provides:              MLOps Uses:                 │
│  ─────────────────             ──────────────              │
│  CI/CD pipelines        →      Model deployment pipelines  │
│  Container orchestration →      Training job scheduling    │
│  IaC (Terraform)        →      Infrastructure for GPUs    │
│  GitOps (ArgoCD)        →      Model config management    │
│  Monitoring (CW)        →      Model monitoring baseline  │
│  Secrets management     →      API keys, data credentials │
└────────────────────────────────────────────────────────────┘
```

#### MLOps Feeds AIOps
```
┌────────────────────────────────────────────────────────────┐
│  MLOps Produces:               AIOps Consumes:             │
│  ───────────────               ──────────────              │
│  Anomaly detection models →    Detect infra anomalies     │
│  NLP models              →    Log parsing & clustering    │
│  Time series models      →    Capacity prediction        │
│  Classification models   →    Alert categorization       │
│  Recommendation models   →    Suggested remediation      │
└────────────────────────────────────────────────────────────┘
```

#### AIOps Improves DevOps
```
┌────────────────────────────────────────────────────────────┐
│  AIOps Provides:               DevOps Benefits:            │
│  ───────────────               ────────────────            │
│  Deployment risk scoring →     Safer releases             │
│  Failure prediction      →     Proactive scaling          │
│  Root cause analysis     →     Faster incident response   │
│  Change correlation      →     Identify bad deploys       │
│  Capacity forecasting    →     Better resource planning   │
└────────────────────────────────────────────────────────────┘
```

### Real DISH Example of the Full Loop

```
1. DevOps team deploys EKS monitoring stack via ArgoCD (DevOps)
       │
       ▼
2. ML team trains anomaly detection model on 6 months of 
   CloudWatch metrics from 400+ accounts (MLOps)
       │
       ▼
3. Model deployed to SageMaker endpoint, integrated with 
   CloudWatch via Lambda (MLOps → AIOps)
       │
       ▼
4. AIOps detects that new ArgoCD deployment caused memory 
   regression in 3 accounts (AIOps)
       │
       ▼
5. AIOps auto-rolls back the deployment via ArgoCD API (AIOps → DevOps)
       │
       ▼
6. Rollback data feeds back into ML training set, improving 
   future deployment risk predictions (AIOps → MLOps)
       │
       ▼
7. ML team retrains "deployment risk scorer" model with new 
   failure patterns (MLOps)
       │
       ▼
8. Updated model now prevents similar deployments from 
   progressing past staging (MLOps → DevOps)
```

---

## 5. Maturity Model

### Organizational Evolution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MATURITY MODEL — 5 STAGES                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VALUE                                                                      │
│  DELIVERED                                                                  │
│    ▲                                                                        │
│    │                                                    ╭──── Stage 5       │
│    │                                                ╭───╯    FULL LOOP      │
│    │                                            ╭───╯                       │
│    │                                    ╭───────╯ Stage 4                   │
│    │                              ╭─────╯        AIOps IN                   │
│    │                        ╭─────╯              PRODUCTION                 │
│    │                  ╭─────╯  Stage 3                                      │
│    │            ╭─────╯       MLOps                                         │
│    │      ╭─────╯            FORMALIZED                                     │
│    │╭─────╯  Stage 2                                                        │
│    ││       EARLY ML                                                        │
│    ││                                                                       │
│    │╯ Stage 1                                                               │
│    │  DevOps ONLY                                                           │
│    └──────────────────────────────────────────────────────► TIME            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Stage 1: DevOps Only
**Characteristics**:
- CI/CD pipelines for application code
- Infrastructure as Code (Terraform/CloudFormation)
- Container orchestration (EKS)
- Static threshold alerting
- Manual incident response

**DISH Context**: Basic account vending, EKS deployments automated, but monitoring is threshold-based ("alert if CPU > 80%"). All 400+ accounts managed via Control Tower + IaC.

**Typical Tools**: CodePipeline, ArgoCD, Terraform, CloudWatch (basic), PagerDuty (basic)

---

### Stage 2: DevOps + Early ML Experiments
**Characteristics**:
- Data scientists working in notebooks (SageMaker Studio)
- Models trained ad-hoc, deployed manually
- No model versioning or monitoring
- "Throw it over the wall" to ops team
- ML experiments not reproducible

**DISH Context**: A data team experiments with churn prediction or network anomaly detection in notebooks. Models are exported as pickle files and deployed manually to EC2 instances. No drift monitoring.

**Gap**: Models degrade silently. No one knows when a model stops being accurate.

---

### Stage 3: MLOps Formalized
**Characteristics**:
- Automated training pipelines (SageMaker Pipelines)
- Model Registry with versioning and approval workflows
- Feature Store for consistent feature computation
- Model monitoring (drift detection, accuracy tracking)
- A/B testing infrastructure for models
- Data lineage and experiment tracking

**DISH Context**: ML models are treated as first-class production artifacts. Training pipelines run on schedule or trigger on data changes. Feature Store ensures online/offline consistency. Model performance dashboards alongside application dashboards.

**Key Milestone**: First automated model retrain without human intervention.

---

### Stage 4: AIOps Using ML Models in Production
**Characteristics**:
- ML models actively monitoring infrastructure
- Anomaly detection replaces static thresholds
- Log clustering reduces alert noise by 90%+
- Auto-remediation for known patterns
- Predictive scaling and capacity planning
- Change correlation (linking deploys to issues)

**DISH Context**: Across 400+ accounts, AI correlates events, predicts failures, and auto-remediates known issues. On-call load drops significantly. MTTR decreases from hours to minutes for automated cases.

**Key Milestone**: First incident resolved end-to-end without human intervention.

---

### Stage 5: Full Loop (AIOps Insights Feed MLOps Retraining)
**Characteristics**:
- AIOps generates training data for improved models
- Incident outcomes feed model improvement
- Self-optimizing systems (models improve from operations data)
- Deployment risk scoring prevents bad releases
- Autonomous operations for commodity tasks
- Humans focus on novel problems only

**DISH Context**: The system learns from every incident. When auto-remediation fails, the failure data trains better classification models. Deployment risk scores prevent risky changes from reaching production. The platform becomes increasingly self-managing.

**Key Milestone**: Month-over-month improvement in MTTR/noise reduction without human model tuning.

---

### Maturity Assessment Checklist

| Capability | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 |
|-----------|---------|---------|---------|---------|---------|
| CI/CD for apps | ✅ | ✅ | ✅ | ✅ | ✅ |
| IaC | ✅ | ✅ | ✅ | ✅ | ✅ |
| ML experiments | ❌ | ✅ | ✅ | ✅ | ✅ |
| Automated training pipelines | ❌ | ❌ | ✅ | ✅ | ✅ |
| Model monitoring & drift detection | ❌ | ❌ | ✅ | ✅ | ✅ |
| Feature Store | ❌ | ❌ | ✅ | ✅ | ✅ |
| AI-powered anomaly detection | ❌ | ❌ | ❌ | ✅ | ✅ |
| Auto-remediation | ❌ | ❌ | ❌ | ✅ | ✅ |
| Predictive operations | ❌ | ❌ | ❌ | ✅ | ✅ |
| Self-improving models | ❌ | ❌ | ❌ | ❌ | ✅ |
| Autonomous incident resolution | ❌ | ❌ | ❌ | ❌ | ✅ |

---

## 6. AWS Services Mapped to Each Discipline

### DevOps Services

| Service | Purpose | DISH Use Case |
|---------|---------|---------------|
| **CodePipeline** | CI/CD orchestration | Orchestrate build/test/deploy for 50+ microservices |
| **CodeBuild** | Build & test execution | Docker builds, unit tests, SAST scanning |
| **CodeDeploy** | Deployment automation | Blue/green and canary deploys to EKS |
| **CloudFormation** | Infrastructure as Code | Account baselines, StackSets across 400+ accounts |
| **EKS** | Container orchestration | Primary compute platform for microservices |
| **ECR** | Container registry | Store and scan Docker images |
| **Systems Manager** | Operations management | Patch management, parameter store, runbooks |
| **Control Tower** | Multi-account governance | Landing zone, guardrails, account vending |
| **Service Catalog** | Self-service provisioning | Teams provision approved architectures |
| **CloudWatch** | Monitoring & logging | Metrics, logs, dashboards, alarms |

### MLOps Services

| Service | Purpose | DISH Use Case |
|---------|---------|---------------|
| **SageMaker Studio** | ML IDE | Data scientists develop and experiment |
| **SageMaker Pipelines** | ML workflow orchestration | Automated training → evaluation → deployment |
| **SageMaker Model Registry** | Model versioning & governance | Track all model versions, approval workflows |
| **SageMaker Feature Store** | Feature management | Online/offline feature serving for real-time inference |
| **SageMaker Model Monitor** | Model quality monitoring | Detect data drift, model quality degradation |
| **SageMaker Endpoints** | Model hosting | Real-time and batch inference endpoints |
| **SageMaker Ground Truth** | Data labeling | Label training data for supervised models |
| **S3** | Data lake | Store training data, model artifacts, features |
| **Glue** | ETL & data catalog | Feature computation, data preparation |
| **Step Functions** | Workflow orchestration | Complex multi-step ML workflows |
| **EventBridge** | Event-driven triggers | Trigger retraining on drift detection |

### AIOps Services

| Service | Purpose | DISH Use Case |
|---------|---------|---------------|
| **DevOps Guru** | AI-powered operations insights | Anomaly detection across 400+ accounts |
| **CloudWatch Anomaly Detection** | ML-based metric monitoring | Learn normal patterns, alert on deviations |
| **CloudWatch Logs Insights** | Log analysis | Query and correlate logs across accounts |
| **EventBridge** | Event routing & automation | Trigger remediation from detected anomalies |
| **Systems Manager Automation** | Runbook execution | Auto-remediation playbooks |
| **Lambda** | Serverless compute | Custom anomaly detection, remediation logic |
| **GuardDuty** | AI-powered threat detection | Security anomaly detection |
| **Health Dashboard** | Service health | Correlate AWS service issues with app issues |
| **Trusted Advisor** | Best practice recommendations | Proactive optimization suggestions |
| **Compute Optimizer** | Resource right-sizing | AI-recommended instance types |
| **Cost Anomaly Detection** | Cost monitoring | Detect unexpected cost spikes |

### Service Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     AWS Services — How They Connect                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DEVOPS LAYER                                                               │
│  ┌────────────────────────────────────────────────────────┐                 │
│  │ CodePipeline → CodeBuild → ECR → CodeDeploy → EKS     │                 │
│  │       ↑                                      ↓         │                 │
│  │ CloudFormation / Terraform          CloudWatch Metrics  │                 │
│  │ Control Tower (governance)          CloudWatch Logs     │                 │
│  └────────────────────────────────────────────┬───────────┘                 │
│                                               │                             │
│                                               │ (provides infrastructure)   │
│                                               ▼                             │
│  MLOPS LAYER                                                                │
│  ┌────────────────────────────────────────────────────────┐                 │
│  │ S3 (data) → Glue (ETL) → Feature Store → SageMaker    │                 │
│  │                              Pipelines → Model Registry │                 │
│  │                                    → Model Monitor      │                 │
│  │                                    → Endpoints          │                 │
│  └────────────────────────────────────────────┬───────────┘                 │
│                                               │                             │
│                                               │ (produces ML models)        │
│                                               ▼                             │
│  AIOPS LAYER                                                                │
│  ┌────────────────────────────────────────────────────────┐                 │
│  │ DevOps Guru ← CloudWatch Anomaly Detection             │                 │
│  │      ↓              ↓                                  │                 │
│  │ EventBridge → SSM Automation → Auto-Remediation        │                 │
│  │      ↓                                                 │                 │
│  │ Cost Anomaly Detection + Compute Optimizer             │                 │
│  │      ↓                                                 │                 │
│  │ Insights feed back to DevOps (scaling, rollbacks)      │                 │
│  └────────────────────────────────────────────────────────┘                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Architecture Diagrams

### DevOps Architecture — Multi-Account EKS Deployment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 DevOps Architecture — DISH Multi-Account                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────┐                        │
│  │            SHARED SERVICES ACCOUNT               │                        │
│  │                                                 │                        │
│  │  ┌───────────┐   ┌──────────┐   ┌───────────┐  │                        │
│  │  │  GitHub   │──►│CodePipeline──►│ CodeBuild │  │                        │
│  │  │  (source) │   │          │   │ (build +  │  │                        │
│  │  └───────────┘   └──────────┘   │  test)    │  │                        │
│  │                                  └─────┬─────┘  │                        │
│  │                                        │        │                        │
│  │  ┌───────────┐                         ▼        │                        │
│  │  │    ECR    │◄──── Docker image pushed          │                        │
│  │  │ (registry)│                                  │                        │
│  │  └─────┬─────┘   ┌──────────────────────┐      │                        │
│  │        │         │  ArgoCD (GitOps)     │      │                        │
│  │        │         │  - Watches Helm repo │      │                        │
│  │        │         │  - Syncs to clusters │      │                        │
│  │        │         └──────────┬───────────┘      │                        │
│  └────────┼────────────────────┼──────────────────┘                        │
│           │                    │                                             │
│           │    ┌───────────────┼───────────────────┐                        │
│           │    │               │                   │                        │
│           ▼    ▼               ▼                   ▼                        │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                    │
│  │   DEV EKS    │   │  STAGING EKS │   │   PROD EKS   │                    │
│  │   (acct-dev) │   │ (acct-stage) │   │  (acct-prod) │                    │
│  │              │   │              │   │              │                    │
│  │  ┌────────┐  │   │  ┌────────┐  │   │  ┌────────┐  │                    │
│  │  │Service │  │   │  │Service │  │   │  │Service │  │                    │
│  │  │ Pods   │  │   │  │ Pods   │  │   │  │ Pods   │  │                    │
│  │  └────────┘  │   │  └────────┘  │   │  └────────┘  │                    │
│  │  Auto-deploy │   │  Auto-deploy │   │  Manual gate │                    │
│  │  on PR merge │   │  on dev pass │   │  + canary    │                    │
│  └──────────────┘   └──────────────┘   └──────────────┘                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### MLOps Architecture — Model Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               MLOps Architecture — End-to-End Model Lifecycle                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DATA LAYER                                                                 │
│  ══════════                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐              │
│  │  S3 Data Lake│    │  Glue Catalog│    │  Glue ETL Jobs   │              │
│  │  (raw data   │───►│  (metadata,  │───►│  (transform,     │              │
│  │   from 400+  │    │   schema)    │    │   clean, join)   │              │
│  │   accounts)  │    └──────────────┘    └────────┬─────────┘              │
│  └──────────────┘                                 │                         │
│                                                   ▼                         │
│  FEATURE LAYER                          ┌──────────────────┐               │
│  ═════════════                          │  SageMaker       │               │
│                                         │  Feature Store   │               │
│                                         │  ┌────────────┐  │               │
│                                         │  │ Offline    │  │               │
│                                         │  │ (S3/Glue)  │  │◄── Training  │
│                                         │  ├────────────┤  │               │
│                                         │  │ Online     │  │◄── Inference │
│                                         │  │ (DynamoDB) │  │               │
│                                         │  └────────────┘  │               │
│                                         └────────┬─────────┘               │
│                                                  │                          │
│  TRAINING LAYER                                  ▼                          │
│  ══════════════                       ┌────────────────────┐                │
│                                       │ SageMaker Pipeline │                │
│  ┌──────────┐  ┌───────────┐         │ ┌──────────────┐   │                │
│  │Experiment│  │Hyperparamr│         │ │  Processing  │   │                │
│  │ Tracking │  │  Tuning   │         │ │  (preprocess)│   │                │
│  │ (MLflow) │  │ (Bayesian)│         │ ├──────────────┤   │                │
│  └──────────┘  └───────────┘         │ │  Training    │   │                │
│                                       │ │  (GPU/dist)  │   │                │
│                                       │ ├──────────────┤   │                │
│                                       │ │  Evaluation  │   │                │
│                                       │ │  (metrics)   │   │                │
│                                       │ ├──────────────┤   │                │
│                                       │ │  Condition   │   │                │
│                                       │ │  (accuracy   │   │                │
│                                       │ │   > 93%?)    │   │                │
│                                       │ └──────┬───────┘   │                │
│                                       └────────┼───────────┘                │
│                                                │                            │
│  DEPLOYMENT LAYER                              ▼                            │
│  ════════════════               ┌──────────────────────┐                    │
│                                 │  Model Registry      │                    │
│                                 │  v3.2.1 → Approved   │                    │
│                                 │  v3.1.0 → Production │                    │
│                                 │  v3.0.2 → Archived   │                    │
│                                 └──────────┬───────────┘                    │
│                                            │                                │
│                              ┌─────────────┼─────────────┐                  │
│                              ▼             ▼             ▼                  │
│                    ┌──────────────┐ ┌────────────┐ ┌──────────┐            │
│                    │   Shadow     │ │  Canary    │ │   Batch  │            │
│                    │   (compare)  │ │  (10%→100%)│ │Transform │            │
│                    └──────────────┘ └────────────┘ └──────────┘            │
│                                                                             │
│  MONITORING LAYER                                                           │
│  ════════════════                                                           │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │  SageMaker Model Monitor                                 │              │
│  │  ├── Data Quality (schema violations, missing values)    │              │
│  │  ├── Model Quality (accuracy, precision, recall)         │              │
│  │  ├── Bias Drift (demographic fairness metrics)           │              │
│  │  └── Feature Attribution Drift (SHAP value changes)      │              │
│  │                                                          │              │
│  │  Drift Detected? → EventBridge → Trigger Retrain Pipeline│              │
│  └──────────────────────────────────────────────────────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### AIOps Architecture — Intelligent Operations Platform

```
┌─────────────────────────────────────────────────────────────────────────────┐
│             AIOps Architecture — Multi-Account Intelligent Ops               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TELEMETRY COLLECTION (400+ Accounts)                                       │
│  ═════════════════════════════════════                                       │
│  ┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐               │
│  │Acct1││Acct2││Acct3││Acct4││ ... ││Ac398││Ac399││Ac400│               │
│  └──┬──┘└──┬──┘└──┬──┘└──┬──┘└──┬──┘└──┬──┘└──┬──┘└──┬──┘               │
│     │      │      │      │      │      │      │      │                    │
│     └──────┴──────┴──────┴──────┴──────┴──────┴──────┘                    │
│                          │                                                  │
│          ┌───────────────┼───────────────┐                                  │
│          ▼               ▼               ▼                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                        │
│  │  CloudWatch  │ │  CloudTrail  │ │  VPC Flow    │                        │
│  │  Metrics +   │ │  (API calls) │ │  Logs        │                        │
│  │  Logs        │ │              │ │              │                        │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘                        │
│         │                │                │                                 │
│         └────────────────┼────────────────┘                                 │
│                          │                                                  │
│  INTELLIGENCE LAYER      ▼                                                  │
│  ══════════════════════════                                                 │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │                                                          │              │
│  │  ┌─────────────────┐  ┌──────────────┐  ┌────────────┐  │              │
│  │  │ Anomaly         │  │ Event        │  │ Pattern    │  │              │
│  │  │ Detection       │  │ Correlation  │  │ Recognition│  │              │
│  │  │                 │  │              │  │            │  │              │
│  │  │ • CloudWatch AD │  │ • Temporal   │  │ • Known    │  │              │
│  │  │ • DevOps Guru   │  │ • Topological│  │   failure  │  │              │
│  │  │ • Custom Models │  │ • Causal     │  │   modes    │  │              │
│  │  │   (SageMaker)   │  │              │  │ • Runbook  │  │              │
│  │  └────────┬────────┘  └──────┬───────┘  │   matching │  │              │
│  │           │                  │          └─────┬──────┘  │              │
│  │           └──────────────────┼────────────────┘         │              │
│  │                              │                          │              │
│  └──────────────────────────────┼──────────────────────────┘              │
│                                 │                                           │
│  DECISION LAYER                 ▼                                           │
│  ══════════════    ┌─────────────────────────────┐                          │
│                    │     Decision Engine          │                          │
│                    │                             │                          │
│                    │  Confidence > 95%?          │                          │
│                    │  ├── YES → Auto-Remediate   │                          │
│                    │  ├── 80-95% → Suggest +     │                          │
│                    │  │           Human Approve   │                          │
│                    │  └── < 80% → Alert Only     │                          │
│                    └──────────────┬──────────────┘                          │
│                                  │                                           │
│  ACTION LAYER                    ▼                                           │
│  ════════════     ┌──────────────────────────────┐                          │
│                   │                              │                          │
│      ┌────────────┼──────────────┬───────────────┤                          │
│      ▼            ▼              ▼               ▼                          │
│  ┌────────┐  ┌─────────┐  ┌──────────┐  ┌───────────┐                      │
│  │  SSM   │  │EventBrdg│  │  Lambda  │  │ PagerDuty │                      │
│  │Runbooks│  │  Rules  │  │  Custom  │  │ /Slack    │                      │
│  │(auto-  │  │(scaling,│  │  Actions │  │ (notify)  │                      │
│  │ fix)   │  │ routing)│  │          │  │           │                      │
│  └────────┘  └─────────┘  └──────────┘  └───────────┘                      │
│                                                                             │
│  LEARNING LAYER                                                             │
│  ══════════════                                                             │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │  Outcome Tracking:                                       │              │
│  │  • Was auto-remediation successful?                      │              │
│  │  • Did the suggested action resolve the issue?           │              │
│  │  • False positive? → Feed back to model training         │              │
│  │  • New pattern? → Add to knowledge base                  │              │
│  │                                                          │              │
│  │  ──────► Feeds back to MLOps retraining pipeline ──────► │              │
│  └──────────────────────────────────────────────────────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Combined Architecture — The Full Loop at DISH

```
┌─────────────────────────────────────────────────────────────────────────────┐
│          COMBINED ARCHITECTURE — DevOps + MLOps + AIOps at Scale            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────── CONTROL TOWER ───────────────────────────────┐    │
│  │  Landing Zone │ 400+ Accounts │ SCPs │ Guardrails │ Account Factory │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  DEVOPS PLATFORM (Shared Services Account)                          │    │
│  │                                                                     │    │
│  │  CodePipeline ──► CodeBuild ──► ECR ──► ArgoCD ──► EKS Clusters    │    │
│  │       │                                               │             │    │
│  │  Terraform ──► CloudFormation StackSets               │             │    │
│  │                                                       │             │    │
│  └───────────────────────────────────────────────────────┼─────────────┘    │
│                                                          │                  │
│         ┌────────────────────────────────────────────────┘                  │
│         │  Deploys & manages infrastructure for:                            │
│         │                                                                   │
│         ├──────────────────────────────────────┐                            │
│         ▼                                      ▼                            │
│  ┌─────────────────────────────┐   ┌─────────────────────────────────┐     │
│  │  MLOPS PLATFORM             │   │  AIOPS PLATFORM                  │     │
│  │  (ML Account)               │   │  (Observability Account)         │     │
│  │                             │   │                                  │     │
│  │  SageMaker Studio           │   │  Cross-account CloudWatch        │     │
│  │       │                     │   │       │                          │     │
│  │       ▼                     │   │       ▼                          │     │
│  │  Feature Store              │   │  DevOps Guru (multi-account)     │     │
│  │       │                     │   │       │                          │     │
│  │       ▼                     │   │       ▼                          │     │
│  │  Training Pipeline          │   │  Anomaly Detection Models        │     │
│  │       │                     │   │  (from MLOps) ◄──────────────────│─┐   │
│  │       ▼                     │   │       │                          │ │   │
│  │  Model Registry ────────────│───│──►    │                          │ │   │
│  │       │                     │   │       ▼                          │ │   │
│  │       ▼                     │   │  Decision Engine                 │ │   │
│  │  Model Endpoints            │   │       │                          │ │   │
│  │       │                     │   │       ├── Auto-Remediate         │ │   │
│  │       │                     │   │       ├── Scale (Karpenter)      │ │   │
│  │       │                     │   │       ├── Rollback (ArgoCD) ─────│─│──►│
│  │       ▼                     │   │       └── Alert (PagerDuty)     │ │   │
│  │  Model Monitor              │   │                                  │ │   │
│  │       │                     │   │  Outcome Data ───────────────────│─┘   │
│  │       │ drift detected      │   │  (feeds retraining)             │     │
│  │       └──► Retrain Pipeline │   │                                  │     │
│  │                             │   │                                  │     │
│  └─────────────────────────────┘   └─────────────────────────────────┘     │
│                                                                             │
│  ┌─────────────────────── WORKLOAD ACCOUNTS ───────────────────────────┐    │
│  │                                                                     │    │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐              │    │
│  │   │Streaming│  │ Payments│  │  Auth   │  │  Mobile │   ...×400    │    │
│  │   │  EKS    │  │  EKS    │  │  EKS    │  │  EKS    │              │    │
│  │   └─────────┘  └─────────┘  └─────────┘  └─────────┘              │    │
│  │        │            │            │            │                     │    │
│  │        └────────────┴────────────┴────────────┘                     │    │
│  │                          │                                          │    │
│  │              Telemetry flows up to AIOps                            │    │
│  │              Deployments flow down from DevOps                      │    │
│  │              Predictions served from MLOps                          │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary: When to Use What

| If you need to... | Use... |
|---|---|
| Ship application code faster | **DevOps** |
| Manage infrastructure at scale | **DevOps** |
| Train and deploy ML models | **MLOps** |
| Monitor model accuracy over time | **MLOps** |
| Detect infrastructure anomalies automatically | **AIOps** |
| Reduce alert noise | **AIOps** |
| Auto-remediate known issues | **AIOps** |
| Deploy models with CI/CD | **DevOps + MLOps** |
| Use ML to improve operations | **MLOps + AIOps** |
| Build a self-healing platform | **All three together** |

---

## Key Takeaway

> **DevOps** is the *foundation* — without it, you can't reliably deploy anything.
> **MLOps** is the *specialization* — it extends DevOps principles to the unique challenges of ML (data versioning, drift, experimentation).
> **AIOps** is the *evolution* — it uses the ML models (built by MLOps, deployed by DevOps) to make operations intelligent and self-managing.

At DISH's scale (400+ accounts, thousands of services, millions of metrics), you need all three working together. The goal is a platform that:
1. **Deploys reliably** (DevOps)
2. **Learns continuously** (MLOps)
3. **Operates autonomously** (AIOps)

---

*Document created: 2026-06-02 | For Cloud Architecture reference at enterprise scale*
