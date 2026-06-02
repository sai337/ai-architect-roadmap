# Enterprise AI Platform Reference Architecture

## A Blueprint for Production-Grade Agentic AI at Scale

---

**Author:** Sai Potharaju | Cloud & AI Architect  
**Version:** 1.0 | June 2026  
**Scope:** Enterprise AI Platform — Multi-Account AWS, Hybrid Cloud, Agentic AI  

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Principles](#architecture-principles)
3. [Full-Stack Architecture Overview](#full-stack-architecture-overview)
4. [Layer 0: Cloud Foundation](#layer-0-cloud-foundation)
5. [Layer 1: AI Infrastructure](#layer-1-ai-infrastructure)
6. [Layer 2: Model Serving & Training](#layer-2-model-serving--training)
7. [Layer 3: AI Orchestration](#layer-3-ai-orchestration)
8. [Layer 4: AI Applications](#layer-4-ai-applications)
9. [Cross-Cutting Concerns](#cross-cutting-concerns)
10. [Deployment Patterns](#deployment-patterns)
11. [Maturity Model](#maturity-model)
12. [Reference Implementations](#reference-implementations)
13. [Conclusion](#conclusion)

---

## Executive Summary

The enterprise AI landscape has shifted from experimentation to production at scale. Organizations no longer ask "should we adopt AI?" — they ask "how do we operationalize AI across hundreds of teams, thousands of workloads, and millions of users while maintaining security, governance, and cost control?"

This reference architecture presents a **production-proven, five-layer enterprise AI platform** that enables:

- **Multi-agent orchestration** across 400+ AWS accounts with centralized governance
- **Self-service AI agent deployment** supporting LangGraph, LangChain, and AWS Strands frameworks
- **Hybrid model serving** spanning cloud GPU clusters and on-premises infrastructure
- **Protocol-native integration** via MCP (Model Context Protocol) and A2A (Agent-to-Agent) standards
- **Enterprise-grade security** with zero-trust access control, data classification, and audit trails

This is not a theoretical framework. Every component described here has been implemented, tested, and operates in production across a Fortune 200 telecommunications enterprise managing complex multi-cloud infrastructure.

### Who This Document Is For

| Audience | Value |
|----------|-------|
| **Platform Engineers** | Detailed component selection rationale and integration patterns |
| **Cloud Architects** | Multi-account topology and hybrid deployment strategies |
| **AI/ML Engineers** | Model serving, training pipelines, and orchestration patterns |
| **Engineering Leadership** | Maturity roadmap and investment prioritization |
| **Security & Governance** | Access control, audit, and compliance architecture |

### Key Differentiators

```
┌─────────────────────────────────────────────────────────────────────┐
│  What Makes This Architecture Different                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ✦ Protocol-First: MCP + A2A as foundational integration layer     │
│  ✦ Multi-Account Native: Built for 400+ account scale from day 1   │
│  ✦ Framework-Agnostic: LangGraph, Strands, LangChain — all work    │
│  ✦ Hybrid-Ready: On-prem ↔ Cloud with LoRA adapter management     │
│  ✦ Self-Service: Developers deploy agents without platform tickets │
│  ✦ Production-Proven: Not a PoC — running at enterprise scale      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Principles

These principles guided every architectural decision:

| # | Principle | Implication |
|---|-----------|-------------|
| 1 | **Open Standards Over Proprietary Lock-in** | MCP, A2A/JSON-RPC 2.0, OpenTelemetry, OCI |
| 2 | **Self-Service with Guardrails** | Developers move fast; platform ensures safety |
| 3 | **Multi-Account by Default** | Blast radius isolation, per-team governance |
| 4 | **Hybrid-First Architecture** | On-prem + cloud as first-class deployment targets |
| 5 | **Cost-Aware from Day Zero** | GPU FinOps, chargeback, right-sizing built in |
| 6 | **Security as Code** | IAM policies, network rules, guardrails — all GitOps |
| 7 | **Observable Everything** | Every LLM call, agent step, and tool invocation traced |
| 8 | **Framework Agnostic** | Platform serves developers, not the other way around |

---

## Full-Stack Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ENTERPRISE AI PLATFORM — FULL STACK                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │ LAYER 4: AI APPLICATIONS                                                  │  │
│  │ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │  │
│  │ │ Copilots │ │ Agents   │ │ RAG Apps │ │ Automati-│ │ Observability    │ │  │
│  │ │ (Chat)   │ │ (Auton.) │ │ (Search) │ │  ons     │ │ Agents           │ │  │
│  │ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                        │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │ LAYER 3: AI ORCHESTRATION                                                 │  │
│  │ ┌──────────────┐ ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌────────────┐  │  │
│  │ │ AgentCore    │ │ A2A /   │ │ MCP      │ │ LangGraph │ │ Guardrails │  │  │
│  │ │ Runtime &    │ │ JSON-RPC│ │ Server   │ │ Strands   │ │ & Policy   │  │  │
│  │ │ Gateway      │ │ 2.0     │ │ Fleet    │ │ LangChain │ │ Engine     │  │  │
│  │ └──────────────┘ └─────────┘ └──────────┘ └───────────┘ └────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                        │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │ LAYER 2: MODEL SERVING & TRAINING                                         │  │
│  │ ┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌───────────┐ ┌───────────┐  │  │
│  │ │ AI       │ │ vLLM /   │ │ SageMaker    │ │ Trainium/ │ │ LoRA      │  │  │
│  │ │ Gateway  │ │ TGI      │ │ Endpoints    │ │ Inferentia│ │ Fine-Tune │  │  │
│  │ │ (Solo+TF)│ │ Serving  │ │              │ │           │ │ Pipeline  │  │  │
│  │ └──────────┘ └──────────┘ └──────────────┘ └───────────┘ └───────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                        │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │ LAYER 1: AI INFRASTRUCTURE                                                │  │
│  │ ┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌───────────┐ ┌───────────┐  │  │
│  │ │ GPU      │ │ AI-Opt   │ │ High-Perf    │ │ EKS/ECS   │ │ On-Prem   │  │  │
│  │ │ Clusters │ │ Storage  │ │ Networking   │ │ Compute   │ │ GPU Nodes │  │  │
│  │ │ (p5/p4d) │ │ (FSx/S3) │ │ (EFA/RDMA)  │ │           │ │           │  │  │
│  │ └──────────┘ └──────────┘ └──────────────┘ └───────────┘ └───────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                        │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │ LAYER 0: CLOUD FOUNDATION                                                 │  │
│  │ ┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌───────────┐ ┌───────────┐  │  │
│  │ │ Control  │ │ Multi-   │ │ Networking   │ │ IAM &     │ │ Governance│  │  │
│  │ │ Tower    │ │ Account  │ │ (TGW/VPC)    │ │ Identity  │ │ & Guardr. │  │  │
│  │ └──────────┘ └──────────┘ └──────────────┘ └───────────┘ └───────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │ CROSS-CUTTING: Security │ Observability │ Cost Mgmt │ CI/CD & MLOps      │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer 0: Cloud Foundation

### Purpose

Layer 0 provides the multi-account cloud foundation upon which all AI workloads are deployed. It establishes identity, networking, governance, and account topology — the bedrock that makes everything above possible at scale.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LAYER 0: CLOUD FOUNDATION                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    AWS ORGANIZATION (400+ Accounts)                   │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                       │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │   │
│  │  │ Management  │    │ Security    │    │ Shared Services         │  │   │
│  │  │ Account     │    │ Account     │    │ (Networking, Logging)   │  │   │
│  │  │             │    │             │    │                         │  │   │
│  │  │ • Ctrl Tower│    │ • GuardDuty │    │ • Transit Gateway       │  │   │
│  │  │ • SCPs      │    │ • Sec Hub   │    │ • Route 53              │  │   │
│  │  │ • Config    │    │ • Inspector │    │ • Direct Connect        │  │   │
│  │  │ • CloudTrail│    │ • Macie     │    │ • Central Logging       │  │   │
│  │  └─────────────┘    └─────────────┘    └─────────────────────────┘  │   │
│  │                                                                       │   │
│  │  ┌───────────────────────────────────────────────────────────────┐   │   │
│  │  │              AI WORKLOAD OUs (Organizational Units)            │   │   │
│  │  ├───────────────────────────────────────────────────────────────┤   │   │
│  │  │                                                               │   │   │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐             │   │   │
│  │  │  │ AI-Platform│  │ AI-Dev     │  │ AI-Prod    │             │   │   │
│  │  │  │ (Shared)   │  │ (Non-Prod) │  │ (Workloads)│             │   │   │
│  │  │  │            │  │            │  │            │             │   │   │
│  │  │  │ • Model    │  │ • Sandbox  │  │ • Serving  │             │   │   │
│  │  │  │   Registry │  │ • Training │  │ • Agents   │             │   │   │
│  │  │  │ • AI GW    │  │ • Experim. │  │ • RAG      │             │   │   │
│  │  │  │ • MCP Hub  │  │ • Fine-tune│  │ • Customer │             │   │   │
│  │  │  └────────────┘  └────────────┘  └────────────┘             │   │   │
│  │  │                                                               │   │   │
│  │  └───────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    NETWORK TOPOLOGY                                   │   │
│  │                                                                       │   │
│  │  On-Prem DC ←─── Direct Connect ───→ Transit Gateway ←──→ VPCs     │   │
│  │       │                                      │                        │   │
│  │       └──── VPN Backup ─────────────────────┘                        │   │
│  │                                                                       │   │
│  │  PrivateLink: AI Gateway ←→ MCP Servers ←→ Bedrock/SageMaker       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | AWS Service | Purpose |
|-----------|------------|---------|
| **Account Factory** | Control Tower + AFT | Automated account provisioning with AI-ready baselines |
| **Governance** | SCPs, Config Rules, CloudTrail | Preventive and detective controls for AI workloads |
| **Networking** | Transit Gateway, PrivateLink, Direct Connect | Hub-spoke connectivity, private AI service access |
| **Identity** | IAM Identity Center (SSO), Cognito | Federated identity for humans and machine-to-machine |
| **DNS** | Route 53 Private Hosted Zones | Service discovery for MCP servers and AI endpoints |
| **Logging** | CloudWatch, S3, OpenSearch | Centralized audit and operational logs |

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Dedicated AI OUs** | Separate SCPs for GPU quotas, model access, data residency |
| **PrivateLink for AI traffic** | LLM inference traffic never traverses public internet |
| **Shared AI Platform account** | Central model registry, AI Gateway, MCP hub — consumed by all |
| **Account-per-team for workloads** | Blast radius isolation, per-team cost attribution |
| **Transit Gateway over VPC Peering** | Scalable hub-spoke; 400+ accounts can't use full-mesh peering |

### Account Topology for AI

```
Organization Root
├── Management OU
│   └── Management Account (Control Tower, Billing)
├── Security OU
│   ├── Security Tooling Account (GuardDuty, Security Hub)
│   └── Log Archive Account (CloudTrail, Config)
├── Infrastructure OU
│   ├── Network Hub Account (TGW, Direct Connect, DNS)
│   └── Shared Services Account (CI/CD, Artifact Repos)
├── AI Platform OU                          ← NEW FOR AI
│   ├── AI Shared Services Account
│   │   ├── Model Registry (S3 + DynamoDB)
│   │   ├── AI Gateway (Solo + TrueFoundry)
│   │   ├── MCP Hub (Central Tool Registry)
│   │   └── AgentCore Runtime
│   ├── AI Training Account
│   │   ├── SageMaker Training Jobs
│   │   ├── GPU/Trainium Clusters
│   │   └── Experiment Tracking (MLflow)
│   └── AI Inference Account
│       ├── SageMaker Endpoints
│       ├── vLLM/TGI on EKS
│       └── Bedrock (managed models)
├── Workload OUs (per-BU)
│   ├── Customer Care AI Account
│   ├── Network Operations AI Account
│   ├── Platform Engineering AI Account
│   └── ... (N accounts)
└── Sandbox OU
    └── AI Experimentation Accounts
```

---

## Layer 1: AI Infrastructure

### Purpose

Layer 1 provides the compute, storage, and networking infrastructure optimized for AI workloads — GPU clusters for training, accelerated instances for inference, high-throughput storage, and low-latency networking.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LAYER 1: AI INFRASTRUCTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    GPU / ACCELERATOR COMPUTE                       │     │
│  ├───────────────────────────────────────────────────────────────────┤     │
│  │                                                                   │     │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────────┐ │     │
│  │  │  NVIDIA GPU │  │  AWS Custom │  │  On-Premises             │ │     │
│  │  │  Clusters   │  │  Silicon    │  │  GPU Fleet               │ │     │
│  │  │             │  │             │  │                          │ │     │
│  │  │  p5.48xl    │  │  Trainium2  │  │  NVIDIA A100/H100       │ │     │
│  │  │  (H100 x8)  │  │  (trn2.48xl)│  │  On-prem Kubernetes     │ │     │
│  │  │             │  │             │  │                          │ │     │
│  │  │  p4d.24xl   │  │  Inferentia2│  │  vLLM CPU Serving       │ │     │
│  │  │  (A100 x8)  │  │  (inf2.48xl)│  │  (x86 AVX-512/AMX)     │ │     │
│  │  │             │  │             │  │                          │ │     │
│  │  │  g5.48xl    │  │  Graviton4  │  │  ARM NEON (inference)   │ │     │
│  │  │  (A10G x8)  │  │  (CPU inf.) │  │                          │ │     │
│  │  └─────────────┘  └─────────────┘  └──────────────────────────┘ │     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    AI-OPTIMIZED STORAGE                            │     │
│  ├───────────────────────────────────────────────────────────────────┤     │
│  │                                                                   │     │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │     │
│  │  │ FSx Lustre  │  │ S3 Express  │  │ EBS io2     │              │     │
│  │  │             │  │ One Zone    │  │ Block Exp.  │              │     │
│  │  │ Training    │  │             │  │             │              │     │
│  │  │ checkpoints │  │ Model       │  │ Local NVMe  │              │     │
│  │  │ & datasets  │  │ artifacts & │  │ cache for   │              │     │
│  │  │ (TB scale)  │  │ weights     │  │ inference   │              │     │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    AI NETWORKING                                   │     │
│  ├───────────────────────────────────────────────────────────────────┤     │
│  │                                                                   │     │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐ │     │
│  │  │ EFA (Elastic    │  │ Placement Groups │  │ VPC Endpoints    │ │     │
│  │  │ Fabric Adapter) │  │ (Cluster)        │  │ for AI Services  │ │     │
│  │  │                 │  │                  │  │                  │ │     │
│  │  │ GPU-to-GPU      │  │ Same-AZ for      │  │ Bedrock, S3,     │ │     │
│  │  │ RDMA (3200Gbps) │  │ training jobs    │  │ SageMaker,       │ │     │
│  │  │ NCCL optimized  │  │                  │  │ ECR              │ │     │
│  │  └─────────────────┘  └─────────────────┘  └──────────────────┘ │     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    CONTAINER ORCHESTRATION                         │     │
│  ├───────────────────────────────────────────────────────────────────┤     │
│  │                                                                   │     │
│  │  EKS Cluster (AI Workloads)          ECS (Serverless Agents)     │     │
│  │  ├── GPU Node Group (p5/p4d)         ├── Fargate (MCP Servers)   │     │
│  │  ├── Inference Node Group (inf2/g5)  ├── Fargate (Lightweight)   │     │
│  │  ├── General Node Group (m7i)        └── Auto-scaling policies   │     │
│  │  ├── Karpenter (auto-scaling)                                     │     │
│  │  └── NVIDIA GPU Operator + Device Plugin                          │     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Technology | Specification |
|-----------|-----------|---------------|
| **Training Compute** | p5.48xlarge (H100), trn2.48xlarge | Distributed training with FSDP/DeepSpeed |
| **Inference Compute** | inf2.48xlarge, g5.48xlarge | Cost-optimized serving with Inferentia2 or A10G |
| **CPU Inference** | r8i (Intel AMX), Graviton4 | Small model serving without GPU dependency |
| **On-Prem GPU** | NVIDIA A100/H100 on-prem K8s | Data-sovereign workloads, burst-to-cloud capable |
| **Training Storage** | FSx for Lustre (linked to S3) | 1+ TB/s throughput for distributed training |
| **Model Store** | S3 Express One Zone | Sub-millisecond first-byte for model weight loading |
| **Networking** | EFA + NCCL, Cluster Placement | 3200 Gbps GPU-to-GPU for multi-node training |
| **Orchestration** | EKS + Karpenter + GPU Operator | Auto-scaling GPU nodes with right-sizing |

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **EKS over ECS for GPU workloads** | NVIDIA GPU Operator, Karpenter GPU-aware scaling, Ray/vLLM K8s-native |
| **Trainium for training** | 40-50% cost reduction vs. equivalent NVIDIA for supported model architectures |
| **Inferentia2 for inference** | Best $/token for supported models; NVIDIA G5 for unsupported |
| **FSx Lustre over EFS** | 10x throughput for training checkpointing; EFS latency too high |
| **CPU inference tier** | Small models (<7B) serve efficiently on AMX-accelerated CPUs — no GPU queue |
| **On-prem as first-class** | Data residency requirements + GPU burst capacity for peak loads |

---

## Layer 2: Model Serving & Training

### Purpose

Layer 2 abstracts model lifecycle management — from training and fine-tuning through registry and deployment to multi-model serving with intelligent routing. The AI Gateway provides a unified API regardless of where or how models are served.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   LAYER 2: MODEL SERVING & TRAINING                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    AI GATEWAY (Unified Model Access)               │     │
│  ├───────────────────────────────────────────────────────────────────┤     │
│  │                                                                   │     │
│  │                   ┌────────────────────────┐                      │     │
│  │                   │   Solo AI Gateway      │                      │     │
│  │                   │   (Envoy-based)        │                      │     │
│  │                   │                        │                      │     │
│  │                   │  • Multi-dataplane     │                      │     │
│  │                   │  • GitLab config sync  │                      │     │
│  │                   │  • ArgoCD deployment   │                      │     │
│  │                   │  • Rate limiting       │                      │     │
│  │                   │  • Auth (Cognito)      │                      │     │
│  │                   └───────────┬────────────┘                      │     │
│  │                               │                                   │     │
│  │              ┌────────────────┼────────────────┐                  │     │
│  │              │                │                │                  │     │
│  │              ▼                ▼                ▼                  │     │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │     │
│  │  │ TrueFoundry  │ │ Bedrock      │ │ Self-Hosted              │ │     │
│  │  │ LLM Proxy    │ │ (Managed)    │ │ (vLLM/TGI on EKS)       │ │     │
│  │  │              │ │              │ │                          │ │     │
│  │  │ • Virtual    │ │ • Claude 4   │ │ • Llama 3.1 (405B)      │ │     │
│  │  │   LLMs       │ │ • Nova Pro   │ │ • Mistral Large         │ │     │
│  │  │ • Guardrails │ │ • Titan      │ │ • Custom Fine-tuned     │ │     │
│  │  │ • Team quotas│ │ • Deepseek   │ │ • Multi-LoRA adapters   │ │     │
│  │  │ • Fallbacks  │ │              │ │                          │ │     │
│  │  └──────────────┘ └──────────────┘ └──────────────────────────┘ │     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    MODEL TRAINING & FINE-TUNING                    │     │
│  ├───────────────────────────────────────────────────────────────────┤     │
│  │                                                                   │     │
│  │  ┌─────────────────┐  ┌────────────────┐  ┌──────────────────┐  │     │
│  │  │ SageMaker       │  │ LoRA Fine-     │  │ Evaluation       │  │     │
│  │  │ Training Jobs   │  │ Tuning         │  │ Pipeline         │  │     │
│  │  │                 │  │                │  │                  │  │     │
│  │  │ • Distributed   │  │ • QLoRA (4-bit)│  │ • MMLU/MT-Bench  │  │     │
│  │  │ • FSDP/DeepSpeed│  │ • LoRA rank    │  │ • Domain-specific│  │     │
│  │  │ • Spot training │  │   optimization │  │ • A/B comparison │  │     │
│  │  │ • Checkpointing │  │ • Adapter mgmt │  │ • Human eval     │  │     │
│  │  └─────────────────┘  └────────────────┘  └──────────────────┘  │     │
│  │                                                                   │     │
│  │  Training Flow:                                                   │     │
│  │  Dataset (S3) → Preprocessing → Training Job → Evaluation →      │     │
│  │  Model Registry → Deployment (vLLM multi-LoRA or SageMaker)      │     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    MODEL REGISTRY & LIFECYCLE                      │     │
│  ├───────────────────────────────────────────────────────────────────┤     │
│  │                                                                   │     │
│  │  ┌─────────────────────────────────────────────────────────────┐ │     │
│  │  │  Model Registry (S3 + DynamoDB + SageMaker Model Registry)  │ │     │
│  │  │                                                             │ │     │
│  │  │  model-registry/                                            │ │     │
│  │  │  ├── base-models/                                           │ │     │
│  │  │  │   ├── llama-3.1-405b-instruct/                          │ │     │
│  │  │  │   ├── mistral-large-2407/                               │ │     │
│  │  │  │   └── deepseek-r1/                                      │ │     │
│  │  │  ├── fine-tuned/                                            │ │     │
│  │  │  │   ├── network-ops-assistant-v2.1/                       │ │     │
│  │  │  │   └── customer-care-agent-v1.3/                         │ │     │
│  │  │  └── lora-adapters/                                         │ │     │
│  │  │      ├── telecom-terminology-adapter/                       │ │     │
│  │  │      ├── incident-response-adapter/                         │ │     │
│  │  │      └── cost-analysis-adapter/                             │ │     │
│  │  └─────────────────────────────────────────────────────────────┘ │     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### AI Gateway — Deep Dive

The AI Gateway is the **single entry point** for all LLM interactions across the enterprise. It decouples consumers from providers.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  AI GATEWAY ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Consumers              Control Plane         Data Planes          │
│                                                                     │
│  ┌──────────┐     ┌─────────────────────┐                          │
│  │ Agents   │────▶│  TrueFoundry CP     │    ┌────────────────┐   │
│  │ (Layer 3)│     │                     │    │ Data Plane 1   │   │
│  └──────────┘     │  • Admin UI         │    │ (aitools-np)   │   │
│                    │  • Policy Engine    │────▶│                │   │
│  ┌──────────┐     │  • Config Store     │    │ • Cluster 1    │   │
│  │ Apps     │────▶│  • Cognito IdP      │    │ • vLLM pods    │   │
│  │ (Layer 4)│     │  • Guardrails       │    │ • Bedrock proxy│   │
│  └──────────┘     │  • Virtual LLMs     │    └────────────────┘   │
│                    │  • Team Quotas      │                          │
│  ┌──────────┐     │  • Agent/MCP Reg.   │    ┌────────────────┐   │
│  │ Pipelines│────▶│                     │    │ Data Plane 2   │   │
│  │ (CI/CD)  │     │  Config Sync:       │────▶│ (pe-auto-np)   │   │
│  └──────────┘     │  GitLab → ArgoCD    │    │                │   │
│                    │  → Helm → K8s       │    │ • Cluster 2    │   │
│                    └─────────────────────┘    │ • TGI pods     │   │
│                                               │ • SageMaker EP │   │
│                                               └────────────────┘   │
│                                                                     │
│  Solo AI Gateway (Envoy) sits in front of each Data Plane:         │
│  • mTLS termination       • Request routing                        │
│  • Rate limiting          • Model version pinning                  │
│  • Token counting         • Failover/retry logic                   │
│  • Cost attribution       • Response streaming                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Multi-LoRA Serving Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│              vLLM MULTI-LORA SERVING ON EKS                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Request: POST /v1/chat/completions                                 │
│           {"model": "llama-3.1-70b+telecom-adapter-v2"}            │
│                                                                     │
│           ┌─────────────────────────────────────┐                   │
│           │         vLLM Inference Server        │                   │
│           │                                     │                   │
│           │  Base Model: Llama-3.1-70B          │                   │
│           │  (loaded once in GPU memory)        │                   │
│           │                                     │                   │
│           │  ┌───────────┐ ┌───────────┐       │                   │
│           │  │ LoRA      │ │ LoRA      │       │                   │
│           │  │ Adapter 1 │ │ Adapter 2 │ ...   │                   │
│           │  │ telecom   │ │ incident  │       │                   │
│           │  │ (16MB)    │ │ (16MB)    │       │                   │
│           │  └───────────┘ └───────────┘       │                   │
│           │                                     │                   │
│           │  Hot-swap adapters per request       │                   │
│           │  (no model reload, <1ms overhead)   │                   │
│           └─────────────────────────────────────┘                   │
│                                                                     │
│  Benefits:                                                          │
│  • 1 GPU cluster serves dozens of specialized models                │
│  • Adapters are <0.1% of base model size                           │
│  • Zero-downtime adapter deployment                                 │
│  • Per-team/use-case customization without dedicated infra          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Solo AI Gateway (Envoy) + TrueFoundry** | Envoy for L7 traffic management + TrueFoundry for LLM-specific policy/routing |
| **Multi-dataplane architecture** | Workload isolation, different security zones, independent scaling |
| **GitLab → ArgoCD config sync** | GitOps for AI Gateway config; audit trail, rollback, PR-based approval |
| **vLLM for self-hosted serving** | Best throughput (continuous batching, PagedAttention), multi-LoRA native |
| **LoRA over full fine-tuning** | 100x cheaper, same base model serves many use cases, rapid iteration |
| **Bedrock as primary + self-hosted fallback** | Bedrock for frontier models (Claude, Nova); self-hosted for data-sovereign/custom |
| **SageMaker for managed endpoints** | Auto-scaling, A/B testing, shadow deployment without K8s complexity |

---

## Layer 3: AI Orchestration

### Purpose

Layer 3 is the **brain** of the platform — it provides multi-agent orchestration, tool integration via MCP, inter-agent communication via A2A, and the runtime environment where AI agents execute complex workflows.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LAYER 3: AI ORCHESTRATION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │              AWS BEDROCK AGENTCORE RUNTIME & GATEWAY               │     │
│  ├───────────────────────────────────────────────────────────────────┤     │
│  │                                                                   │     │
│  │  User (SSO) ──→ UI/CLI ──→ AgentCore Gateway ──→ Agent Runtime   │     │
│  │                                     │                             │     │
│  │                          ┌──────────┼──────────┐                  │     │
│  │                          ▼          ▼          ▼                  │     │
│  │                  ┌───────────┐ ┌─────────┐ ┌───────────┐         │     │
│  │                  │ Agent     │ │ Session │ │ Identity  │         │     │
│  │                  │ Registry  │ │ Mgmt    │ │ & Auth    │         │     │
│  │                  │           │ │         │ │           │         │     │
│  │                  │ • Agent   │ │ • State │ │ • IAM     │         │     │
│  │                  │   cards   │ │ • Memory│ │ • Cognito │         │     │
│  │                  │ • Versions│ │ • Context│ │ • RBAC   │         │     │
│  │                  │ • Health  │ │         │ │           │         │     │
│  │                  └───────────┘ └─────────┘ └───────────┘         │     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │              MCP SERVER FLEET (Model Context Protocol)             │     │
│  ├───────────────────────────────────────────────────────────────────┤     │
│  │                                                                   │     │
│  │  ┌───────────────────────────────────────────────────────────┐   │     │
│  │  │  MCP Hub (Central Tool Registry)                           │   │     │
│  │  │  Clusters: aitools-np | pe-automation-np                   │   │     │
│  │  └───────────────────────────────────────────────────────────┘   │     │
│  │                           │                                       │     │
│  │       ┌───────────────────┼───────────────────┐                  │     │
│  │       ▼                   ▼                   ▼                  │     │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │     │
│  │  │ AWS Infra   │  │ Security &  │  │ Domain-Specific         │ │     │
│  │  │ MCP Servers │  │ Compliance  │  │ MCP Servers             │ │     │
│  │  │             │  │             │  │                         │ │     │
│  │  │ • Lambda    │  │ • Security  │  │ • Customer Care         │ │     │
│  │  │ • Cost Exp. │  │   Hub       │  │ • Network Ops           │ │     │
│  │  │ • S3        │  │ • IAM       │  │ • Billing               │ │     │
│  │  │ • CloudWatch│  │ • GuardDuty │  │ • Incident Mgmt         │ │     │
│  │  │ • Neptune   │  │ • Config    │  │ • Knowledge Base        │ │     │
│  │  │ • EC2       │  │ • Macie     │  │ • Jira/ServiceNow       │ │     │
│  │  │ • EKS       │  │             │  │                         │ │     │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘ │     │
│  │                                                                   │     │
│  │  MCP Server Standard:                                             │     │
│  │  • FastAPI + FastMCP framework                                    │     │
│  │  • /mcp/healthz health check                                      │     │
│  │  • aws_{service}_ tool naming convention                          │     │
│  │  • Cross-account assume-role (400+ accounts)                      │     │
│  │  • Deployed on EKS (Fargate or managed nodes)                     │     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │              AGENT FRAMEWORKS & RUNTIME                            │     │
│  ├───────────────────────────────────────────────────────────────────┤     │
│  │                                                                   │     │
│  │  ┌──────────────────────────────────────────────────────────────┐│     │
│  │  │  Framework Selection Matrix                                   ││     │
│  │  │                                                              ││     │
│  │  │  ┌───────────┐  ┌───────────┐  ┌───────────┐               ││     │
│  │  │  │ LangGraph │  │ AWS       │  │ LangChain │               ││     │
│  │  │  │           │  │ Strands   │  │           │               ││     │
│  │  │  │ Complex   │  │           │  │ Simple    │               ││     │
│  │  │  │ stateful  │  │ AWS-native│  │ chains &  │               ││     │
│  │  │  │ workflows │  │ lightweight│  │ RAG pipes │               ││     │
│  │  │  │           │  │ agents    │  │           │               ││     │
│  │  │  │ • Graph   │  │           │  │ • LCEL    │               ││     │
│  │  │  │   exec.   │  │ • Bedrock │  │ • Retriev.│               ││     │
│  │  │  │ • HITL    │  │   native  │  │ • Memory  │               ││     │
│  │  │  │ • Checkpt │  │ • MCP/A2A │  │ • Tools   │               ││     │
│  │  │  │ • Stream  │  │ • Minimal │  │           │               ││     │
│  │  │  └───────────┘  └───────────┘  └───────────┘               ││     │
│  │  │                                                              ││     │
│  │  │  Use LangGraph for: Production business-critical agents      ││     │
│  │  │  Use Strands for: AWS-native, simple tool-calling agents     ││     │
│  │  │  Use LangChain for: RAG pipelines, simple chains             ││     │
│  │  └──────────────────────────────────────────────────────────────┘│     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │              INTER-AGENT COMMUNICATION (A2A)                       │     │
│  ├───────────────────────────────────────────────────────────────────┤     │
│  │                                                                   │     │
│  │  Protocol: JSON-RPC 2.0 over HTTP/SSE                             │     │
│  │                                                                   │     │
│  │  ┌─────────┐    A2A Protocol     ┌─────────┐                    │     │
│  │  │ Agent A │ ◄──────────────────▶ │ Agent B │                    │     │
│  │  │ (Orch.) │    • Agent Cards     │ (Worker)│                    │     │
│  │  │         │    • Task Lifecycle  │         │                    │     │
│  │  │ Invokes │    • Streaming       │ Executes│                    │     │
│  │  │ sub-task│    • Push Notify     │ & returns│                   │     │
│  │  └─────────┘                      └─────────┘                    │     │
│  │                                                                   │     │
│  │  Agent Discovery: Agent Registry → Agent Cards (capabilities,     │     │
│  │  skills, auth requirements, input/output schemas)                 │     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │              GUARDRAILS & POLICY ENGINE                            │     │
│  ├───────────────────────────────────────────────────────────────────┤     │
│  │                                                                   │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │     │
│  │  │ Input        │  │ Output       │  │ Tool Execution       │   │     │
│  │  │ Guardrails   │  │ Guardrails   │  │ Guardrails           │   │     │
│  │  │              │  │              │  │                      │   │     │
│  │  │ • PII detect │  │ • Toxicity   │  │ • Allow/deny lists   │   │     │
│  │  │ • Prompt inj.│  │ • Halluc.    │  │ • Rate limits        │   │     │
│  │  │ • Topic block│  │ • Data leak  │  │ • Approval workflows │   │     │
│  │  │ • Content    │  │ • Format     │  │ • Blast radius       │   │     │
│  │  │   classify.  │  │   enforce    │  │   controls           │   │     │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘   │     │
│  │                                                                   │     │
│  │  Bedrock Guardrails + Custom Policy Engine (TrueFoundry)          │     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### MCP Server Fleet — Detail

The MCP Server Fleet provides **tool access** to AI agents across 400+ AWS accounts:

```
┌─────────────────────────────────────────────────────────────────────┐
│              MCP SERVER FLEET — CROSS-ACCOUNT PATTERN               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Agent Request: "Get Lambda errors in account X for last 24h"      │
│                                                                     │
│  ┌─────────────┐         ┌──────────────────────┐                  │
│  │ AI Agent    │──MCP───▶│ dish-aws-lambda-mcp  │                  │
│  │ (LangGraph) │         │                      │                  │
│  └─────────────┘         │ app/server.py        │                  │
│                           │ ├── /mcp/healthz     │                  │
│                           │ └── tools/           │                  │
│                           │     ├── aws_lambda_* │                  │
│                           │     └── ...          │                  │
│                           └──────────┬───────────┘                  │
│                                      │                              │
│                           STS AssumeRole (cross-account)            │
│                                      │                              │
│                    ┌─────────────────┼─────────────────┐            │
│                    ▼                 ▼                 ▼            │
│           ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│           │ Account 001  │  │ Account 002  │  │ Account 400+ │    │
│           │ (Team Alpha) │  │ (Team Beta)  │  │ (Team ...)   │    │
│           │              │  │              │  │              │    │
│           │ IAM Role:    │  │ IAM Role:    │  │ IAM Role:    │    │
│           │ mcp-lambda-  │  │ mcp-lambda-  │  │ mcp-lambda-  │    │
│           │ read-role    │  │ read-role    │  │ read-role    │    │
│           └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                                     │
│  Deployed MCP Servers (21 per cluster, 2 clusters):                 │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │ aws-api, aws-lambda, aws-cost, aws-s3, aws-security-hub,      ││
│  │ aws-neptune, aws-cloudwatch, aws-ec2, aws-eks, aws-config,    ││
│  │ aws-guardduty, aws-iam, aws-wellarchitected, aws-bedrock,     ││
│  │ aws-scan, aws-graph, dynatrace, jira, servicenow,             ││
│  │ knowledge-base, incident-management                            ││
│  └────────────────────────────────────────────────────────────────┘│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **AgentCore as orchestration layer** | AWS-managed runtime with built-in session management, A2A protocol support |
| **MCP over custom tool APIs** | Open standard; tools reusable across any MCP-compatible agent framework |
| **LangGraph for complex agents** | Deterministic graph execution, PostgreSQL checkpointing, interrupt() for HITL |
| **Strands for simple agents** | AWS-native, minimal overhead, good for single-purpose Bedrock agents |
| **FastAPI + FastMCP standard** | Consistent developer experience, health checks, OpenAPI docs automatic |
| **Cross-account STS assume-role** | Single MCP server serves all accounts; no per-account deployment needed |
| **Agent Registry with Agent Cards** | Discoverability + capability-based routing for multi-agent orchestration |
| **Dual-cluster deployment** | High availability; both clusters independently serve MCP requests |

---

## Layer 4: AI Applications

### Purpose

Layer 4 is where business value is realized — the AI-powered applications that end users interact with. These applications consume the orchestration layer (Layer 3) and are built by development teams using self-service patterns.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LAYER 4: AI APPLICATIONS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    COPILOTS & ASSISTANTS                           │     │
│  │                                                                   │     │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │     │
│  │  │ Developer        │  │ Ops Engineer    │  │ Customer Care   │  │     │
│  │  │ Copilot          │  │ Copilot         │  │ Assistant       │  │     │
│  │  │                  │  │                 │  │                 │  │     │
│  │  │ • Code review    │  │ • Incident      │  │ • Account       │  │     │
│  │  │ • Architecture   │  │   triage        │  │   lookup        │  │     │
│  │  │ • Cost analysis  │  │ • Runbook exec  │  │ • Billing help  │  │     │
│  │  │ • Security scan  │  │ • Alert enrich  │  │ • Troubleshoot  │  │     │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘  │     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    AUTONOMOUS AGENTS                               │     │
│  │                                                                   │     │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │     │
│  │  │ Cost            │  │ Security        │  │ Infrastructure  │  │     │
│  │  │ Optimization    │  │ Remediation     │  │ Self-Healing    │  │     │
│  │  │ Agent           │  │ Agent           │  │ Agent           │  │     │
│  │  │                 │  │                 │  │                 │  │     │
│  │  │ • Idle resource │  │ • Auto-fix      │  │ • Pod restart   │  │     │
│  │  │   detection     │  │   findings      │  │ • Scale events  │  │     │
│  │  │ • RI/SP recs    │  │ • Compliance    │  │ • Failover      │  │     │
│  │  │ • Right-sizing  │  │   enforcement   │  │ • Capacity plan │  │     │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘  │     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    RAG APPLICATIONS                                │     │
│  │                                                                   │     │
│  │  ┌─────────────────────────────────────────────────────────────┐ │     │
│  │  │                                                             │ │     │
│  │  │  Query ──→ Embedding ──→ Vector Search ──→ Reranking ──→   │ │     │
│  │  │            (Titan/     (OpenSearch/     (Cross-encoder)     │ │     │
│  │  │             Cohere)     Bedrock KB)                         │ │     │
│  │  │                                          ──→ LLM ──→ Answer│ │     │
│  │  │                                                             │ │     │
│  │  │  Knowledge Sources:                                         │ │     │
│  │  │  • Runbooks & SOPs          • Architecture docs             │ │     │
│  │  │  • Incident postmortems     • API documentation             │ │     │
│  │  │  • Confluence/Wiki          • Slack conversations           │ │     │
│  │  │                                                             │ │     │
│  │  └─────────────────────────────────────────────────────────────┘ │     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    OBSERVABILITY AGENTS                            │     │
│  │                                                                   │     │
│  │  ┌─────────────────────────────────────────────────────────────┐ │     │
│  │  │                                                             │ │     │
│  │  │  Alert ──→ Context Enrichment ──→ Root Cause Analysis ──→  │ │     │
│  │  │            (Dynatrace +         (Multi-tool reasoning)     │ │     │
│  │  │             CloudWatch)                                     │ │     │
│  │  │                                          ──→ Remediation    │ │     │
│  │  │                                              Recommendation │ │     │
│  │  │                                                             │ │     │
│  │  │  Integrations:                                              │ │     │
│  │  │  • Dynatrace MCP ←→ Problem detection                      │ │     │
│  │  │  • CloudWatch MCP ←→ Metrics & logs                        │ │     │
│  │  │  • PagerDuty/ServiceNow ←→ Incident lifecycle              │ │     │
│  │  │                                                             │ │     │
│  │  └─────────────────────────────────────────────────────────────┘ │     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │          SELF-SERVICE DEPLOYMENT (AIOps Platform)                  │     │
│  ├───────────────────────────────────────────────────────────────────┤     │
│  │                                                                   │     │
│  │  Developer Experience:                                            │     │
│  │                                                                   │     │
│  │  1. Choose framework (LangGraph / Strands / LangChain)            │     │
│  │  2. Define agent (graph, tools, prompts)                          │     │
│  │  3. Select MCP servers from registry                              │     │
│  │  4. Configure guardrails & model                                  │     │
│  │  5. Push to GitLab → Auto-deploy via ArgoCD                       │     │
│  │  6. Agent live with observability, scaling, auth                   │     │
│  │                                                                   │     │
│  │  No platform tickets. No infra provisioning. Ship in hours.       │     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Application Patterns

| Pattern | Framework | Use Case | Complexity |
|---------|-----------|----------|------------|
| **Copilot** | LangGraph + HITL | Interactive assistant with approval gates | Medium |
| **Autonomous Agent** | LangGraph (no interrupt) | Scheduled/event-driven automation | High |
| **RAG Pipeline** | LangChain LCEL | Knowledge retrieval & synthesis | Low |
| **Tool Agent** | Strands | Single-purpose AWS tool caller | Low |
| **Multi-Agent Workflow** | AgentCore + A2A | Cross-domain orchestration | Very High |
| **Observability Agent** | LangGraph + Dynatrace MCP | Alert enrichment & remediation | High |

---

## Cross-Cutting Concerns

### Security & Governance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SECURITY & GOVERNANCE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐   │
│  │ IDENTITY &     │  │ DATA           │  │ MODEL ACCESS               │   │
│  │ ACCESS         │  │ PROTECTION     │  │ CONTROL                    │   │
│  │                │  │                │  │                            │   │
│  │ • SSO via      │  │ • Classification│  │ • Per-team model allow    │   │
│  │   Identity Ctr │  │   (S3 tags)    │  │   lists                   │   │
│  │ • Machine-to-  │  │ • Encryption   │  │ • Token quotas per team   │   │
│  │   machine via  │  │   at rest/     │  │ • Model version pinning   │   │
│  │   IAM roles    │  │   transit      │  │ • Prompt/response logging │   │
│  │ • Cognito for  │  │ • PII masking  │  │   (opt-in/opt-out)       │   │
│  │   app users    │  │   in prompts   │  │ • Data residency rules    │   │
│  │ • RBAC for     │  │ • DLP in       │  │                            │   │
│  │   MCP tools    │  │   guardrails   │  │                            │   │
│  └────────────────┘  └────────────────┘  └────────────────────────────┘   │
│                                                                             │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐   │
│  │ AUDIT &        │  │ NETWORK        │  │ SUPPLY CHAIN               │   │
│  │ COMPLIANCE     │  │ SECURITY       │  │ SECURITY                   │   │
│  │                │  │                │  │                            │   │
│  │ • CloudTrail   │  │ • Zero-trust   │  │ • Model provenance         │   │
│  │   for all AI   │  │   (no public   │  │   tracking                 │   │
│  │   API calls    │  │   AI endpoints)│  │ • Container image signing  │   │
│  │ • AI-specific  │  │ • PrivateLink  │  │   (ECR)                    │   │
│  │   audit events │  │   for all      │  │ • Dependency scanning      │   │
│  │ • Guardrail    │  │   model APIs   │  │ • SBOM for AI components   │   │
│  │   violation    │  │ • WAF for      │  │ • Adapter integrity        │   │
│  │   tracking     │  │   public chat  │  │   verification             │   │
│  └────────────────┘  └────────────────┘  └────────────────────────────┘   │
│                                                                             │
│  Security Architecture Principle:                                            │
│  "AI agents have the MINIMUM permissions needed to execute their tools.     │
│   Every cross-account action is audited. Every model interaction is         │
│   traced. Every tool invocation is logged."                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Observability

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AI OBSERVABILITY STACK                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    THREE PILLARS FOR AI                            │     │
│  ├───────────────────────────────────────────────────────────────────┤     │
│  │                                                                   │     │
│  │  TRACES (End-to-End Agent Execution)                              │     │
│  │  ├── OpenTelemetry SDK (instrumented in agent code)               │     │
│  │  ├── Spans: Agent → Tool → MCP → AWS API → Response              │     │
│  │  ├── LLM-specific attributes: model, tokens, latency, cost       │     │
│  │  └── Collector → Dynatrace / X-Ray / Jaeger                      │     │
│  │                                                                   │     │
│  │  METRICS (Performance & Cost)                                     │     │
│  │  ├── Token throughput (tokens/sec per model)                      │     │
│  │  ├── Latency percentiles (p50, p95, p99)                         │     │
│  │  ├── Error rates (by agent, tool, model)                          │     │
│  │  ├── Cost per request (attributed to team/agent)                  │     │
│  │  ├── GPU utilization (training & inference)                       │     │
│  │  └── Queue depth (pending requests per model)                     │     │
│  │                                                                   │     │
│  │  LOGS (Structured AI Events)                                      │     │
│  │  ├── Agent decisions (reasoning traces, tool selections)          │     │
│  │  ├── Guardrail triggers (blocked content, policy violations)      │     │
│  │  ├── Model responses (opt-in, for debugging/eval)                 │     │
│  │  └── Tool execution results (success/failure, latency)            │     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    OBSERVABILITY PIPELINE                          │     │
│  │                                                                   │     │
│  │  Agent Code ──→ OTel SDK ──→ OTel Collector ──→ Backends         │     │
│  │                                      │                            │     │
│  │                            ┌─────────┼─────────┐                  │     │
│  │                            ▼         ▼         ▼                  │     │
│  │                     ┌──────────┐ ┌────────┐ ┌──────────┐         │     │
│  │                     │Dynatrace │ │X-Ray   │ │CloudWatch│         │     │
│  │                     │(APM+AI)  │ │(traces)│ │(metrics) │         │     │
│  │                     └──────────┘ └────────┘ └──────────┘         │     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  Key Dashboards:                                                            │
│  • Agent Performance: latency, success rate, cost per interaction           │
│  • Model Health: throughput, error rate, latency degradation                │
│  • Platform Capacity: GPU utilization, queue depth, scaling events          │
│  • Cost Attribution: per-team, per-agent, per-model spend                   │
│  • Security: guardrail violations, anomalous access patterns                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Cost Management (FinOps for AI)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AI FINOPS — COST MANAGEMENT                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    COST OPTIMIZATION LEVERS                        │     │
│  ├───────────────────────────────────────────────────────────────────┤     │
│  │                                                                   │     │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │     │
│  │  │ COMPUTE COSTS    │  │ MODEL COSTS      │  │ STORAGE COSTS  │ │     │
│  │  │                  │  │                  │  │                │ │     │
│  │  │ • Spot for       │  │ • Token-level    │  │ • Tiered model │ │     │
│  │  │   training       │  │   cost tracking  │  │   storage      │ │     │
│  │  │ • Inferentia2    │  │ • Model routing  │  │ • Checkpoint   │ │     │
│  │  │   over NVIDIA    │  │   (cheap→expen.) │  │   lifecycle    │ │     │
│  │  │ • Right-size     │  │ • Caching        │  │ • S3 Glacier   │ │     │
│  │  │   GPU instances  │  │   (semantic)     │  │   for old      │ │     │
│  │  │ • Scale-to-zero  │  │ • Batch vs.      │  │   models       │ │     │
│  │  │   inference      │  │   real-time      │  │                │ │     │
│  │  └──────────────────┘  └──────────────────┘  └────────────────┘ │     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    CHARGEBACK MODEL                                │     │
│  ├───────────────────────────────────────────────────────────────────┤     │
│  │                                                                   │     │
│  │  Cost Attribution Flow:                                           │     │
│  │                                                                   │     │
│  │  Request → Team Tag → Agent Tag → Model Used → Tokens → $Cost    │     │
│  │                                                                   │     │
│  │  Chargeback Dimensions:                                           │     │
│  │  • Per-team monthly AI spend (model inference + compute)          │     │
│  │  • Per-agent operational cost (tools + models + infrastructure)   │     │
│  │  • Shared platform costs (AI Gateway, MCP fleet) → distributed   │     │
│  │  • Training costs → attributed to requesting team                 │     │
│  │                                                                   │     │
│  │  Tools: AWS Cost Explorer MCP + CUR + QuickSight dashboards       │     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  Cost Optimization Strategies:                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ Strategy                          │ Savings    │ Complexity        │    │
│  ├───────────────────────────────────┼────────────┼───────────────────┤    │
│  │ Inferentia2 over NVIDIA (inf.)    │ 40-50%     │ Medium (Neuron)   │    │
│  │ Trainium over NVIDIA (training)   │ 40-60%     │ High (compat.)    │    │
│  │ Spot instances for training       │ 60-90%     │ Low (w/ checkpt)  │    │
│  │ Semantic caching                  │ 20-30%     │ Low               │    │
│  │ Model routing (small→large)       │ 30-50%     │ Medium            │    │
│  │ Multi-LoRA over dedicated models  │ 80-90%     │ Medium            │    │
│  │ Scale-to-zero inference           │ Variable   │ Medium            │    │
│  │ Batch inference (non-real-time)   │ 50%        │ Low               │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### CI/CD & MLOps

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CI/CD & MLOPS PIPELINE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    AGENT DEPLOYMENT PIPELINE                       │     │
│  │                                                                   │     │
│  │  Developer                                                        │     │
│  │     │                                                             │     │
│  │     ▼                                                             │     │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐ │     │
│  │  │ GitLab   │──▶│ CI/CD    │──▶│ ArgoCD   │──▶│ EKS Deploy   │ │     │
│  │  │ Push     │   │ Pipeline │   │ Sync     │   │ (Agent Live) │ │     │
│  │  └──────────┘   └──────────┘   └──────────┘   └──────────────┘ │     │
│  │                       │                                           │     │
│  │                       ▼                                           │     │
│  │              ┌─────────────────┐                                  │     │
│  │              │ Pipeline Steps  │                                  │     │
│  │              │                 │                                  │     │
│  │              │ 1. Lint & Test  │                                  │     │
│  │              │ 2. Build Image  │                                  │     │
│  │              │ 3. Scan (Trivy) │                                  │     │
│  │              │ 4. Push to ECR  │                                  │     │
│  │              │ 5. Update Helm  │                                  │     │
│  │              │ 6. ArgoCD sync  │                                  │     │
│  │              │ 7. Health check │                                  │     │
│  │              │ 8. Smoke test   │                                  │     │
│  │              └─────────────────┘                                  │     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    MODEL TRAINING PIPELINE                         │     │
│  │                                                                   │     │
│  │  ┌────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  ┌───────┐ │     │
│  │  │ Data   │─▶│ Preproc. │─▶│ Training │─▶│ Eval   │─▶│ Regis.│ │     │
│  │  │ Ingest │  │ & Valid. │  │ Job      │  │ Suite  │  │ & Tag │ │     │
│  │  └────────┘  └──────────┘  └──────────┘  └────────┘  └───────┘ │     │
│  │                                                     │            │     │
│  │                                                     ▼            │     │
│  │                                              ┌────────────┐      │     │
│  │                                              │ Deploy     │      │     │
│  │                                              │ (if passes │      │     │
│  │                                              │  threshold)│      │     │
│  │                                              └────────────┘      │     │
│  │                                                                   │     │
│  │  Training Pipeline Tools:                                         │     │
│  │  • SageMaker Pipelines (orchestration)                            │     │
│  │  • MLflow (experiment tracking)                                   │     │
│  │  • DVC (data versioning)                                          │     │
│  │  • Weights & Biases (training visualization)                      │     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    MODEL PROMOTION WORKFLOW                        │     │
│  │                                                                   │     │
│  │  Development → Staging → Production                               │     │
│  │       │            │           │                                  │     │
│  │  • Unit tests  • Integration  • Canary deployment                 │     │
│  │  • Eval suite  • Load testing • Gradual rollout                   │     │
│  │  • Prompt      • Shadow mode  • A/B testing                       │     │
│  │    testing     • Guardrail    • Full traffic                      │     │
│  │                  validation                                       │     │
│  │                                                                   │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Patterns

### Multi-Account Topology for AI Workloads

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              MULTI-ACCOUNT AI DEPLOYMENT TOPOLOGY                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    ┌─────────────────────────┐                              │
│                    │    NETWORK HUB          │                              │
│                    │    Transit Gateway      │                              │
│                    │    + Direct Connect     │                              │
│                    └────────────┬────────────┘                              │
│                                 │                                           │
│         ┌───────────────────────┼───────────────────────┐                  │
│         │                       │                       │                  │
│         ▼                       ▼                       ▼                  │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐            │
│  │ AI PLATFORM  │      │ AI WORKLOAD  │      │ AI WORKLOAD  │            │
│  │ SHARED SVCS  │      │ NON-PROD     │      │ PRODUCTION   │            │
│  │              │      │              │      │              │            │
│  │ • AI Gateway │◄────▶│ • Dev agents │◄────▶│ • Prod agents│            │
│  │ • MCP Hub    │      │ • Training   │      │ • Serving    │            │
│  │ • AgentCore  │      │ • Experiment │      │ • Customer-  │            │
│  │ • Model Reg. │      │ • Sandbox    │      │   facing     │            │
│  │ • Tool Reg.  │      │              │      │              │            │
│  └──────────────┘      └──────────────┘      └──────────────┘            │
│         │                                           │                     │
│         │                                           │                     │
│         ▼                                           ▼                     │
│  ┌──────────────────────────────────────────────────────────────┐        │
│  │                 SPOKE ACCOUNTS (400+)                         │        │
│  │                                                              │        │
│  │  Each spoke account has:                                     │        │
│  │  • IAM roles for MCP cross-account access                    │        │
│  │  • CloudTrail forwarding to security account                 │        │
│  │  • Config rules for AI governance                            │        │
│  │  • Tags for cost attribution                                 │        │
│  │                                                              │        │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐   │        │
│  │  │Acct 001│ │Acct 002│ │Acct 003│ │  ...   │ │Acct 400│   │        │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘   │        │
│  └──────────────────────────────────────────────────────────────┘        │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Hub-Spoke Model Management

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              HUB-SPOKE MODEL MANAGEMENT                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    ┌──────────────────────────┐                             │
│                    │      MODEL HUB           │                             │
│                    │   (AI Platform Account)  │                             │
│                    │                          │                             │
│                    │  ┌────────────────────┐  │                             │
│                    │  │ Model Registry     │  │                             │
│                    │  │ (S3 + DynamoDB)    │  │                             │
│                    │  │                    │  │                             │
│                    │  │ • Base models      │  │                             │
│                    │  │ • Fine-tuned       │  │                             │
│                    │  │ • LoRA adapters    │  │                             │
│                    │  │ • Metadata/lineage │  │                             │
│                    │  └────────────────────┘  │                             │
│                    │                          │                             │
│                    │  ┌────────────────────┐  │                             │
│                    │  │ AI Gateway         │  │                             │
│                    │  │ (Routing Layer)    │  │                             │
│                    │  └────────────────────┘  │                             │
│                    └─────────────┬────────────┘                             │
│                                  │                                          │
│              ┌───────────────────┼───────────────────┐                     │
│              │                   │                   │                     │
│              ▼                   ▼                   ▼                     │
│     ┌────────────────┐  ┌────────────────┐  ┌────────────────┐           │
│     │ SPOKE 1        │  │ SPOKE 2        │  │ SPOKE 3        │           │
│     │ (Team A)       │  │ (Team B)       │  │ (On-Prem)      │           │
│     │                │  │                │  │                │           │
│     │ Consumes:      │  │ Consumes:      │  │ Consumes:      │           │
│     │ • Claude 4     │  │ • Llama 405B   │  │ • Llama 70B    │           │
│     │ • Custom LoRA  │  │ • Nova Pro     │  │ • Custom FT    │           │
│     │                │  │ • Cost adapter │  │ • Local vLLM   │           │
│     │ Via: AI GW API │  │ Via: AI GW API │  │ Via: Direct    │           │
│     └────────────────┘  └────────────────┘  └────────────────┘           │
│                                                                             │
│  Hub Responsibilities:                                                      │
│  • Model approval & security scanning                                       │
│  • Centralized access control & quotas                                      │
│  • Usage tracking & cost attribution                                        │
│  • Model lifecycle (deprecation, rotation)                                  │
│                                                                             │
│  Spoke Responsibilities:                                                    │
│  • Agent development & deployment                                           │
│  • Domain-specific fine-tuning (request to hub)                             │
│  • Usage within allocated quotas                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### On-Prem ↔ Cloud Hybrid Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              HYBRID DEPLOYMENT PATTERN                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────┐   │
│  │      ON-PREMISES            │    │          AWS CLOUD              │   │
│  │                             │    │                                 │   │
│  │  ┌──────────────────────┐  │    │  ┌───────────────────────────┐ │   │
│  │  │ Kubernetes Cluster   │  │    │  │ EKS Cluster               │ │   │
│  │  │                      │  │    │  │                           │ │   │
│  │  │ ┌──────────────────┐ │  │    │  │ ┌───────────────────────┐│ │   │
│  │  │ │ vLLM (Llama 70B) │ │  │    │  │ │ vLLM (Llama 405B)    ││ │   │
│  │  │ │ CPU + A100 GPU   │ │  │    │  │ │ p5.48xlarge (H100)   ││ │   │
│  │  │ └──────────────────┘ │  │    │  │ └───────────────────────┘│ │   │
│  │  │                      │  │    │  │                           │ │   │
│  │  │ ┌──────────────────┐ │  │    │  │ ┌───────────────────────┐│ │   │
│  │  │ │ MCP Servers      │ │  │    │  │ │ MCP Servers (full)    ││ │   │
│  │  │ │ (subset)         │ │  │    │  │ │ 21 servers/cluster    ││ │   │
│  │  │ └──────────────────┘ │  │    │  │ └───────────────────────┘│ │   │
│  │  │                      │  │    │  │                           │ │   │
│  │  │ ┌──────────────────┐ │  │    │  │ ┌───────────────────────┐│ │   │
│  │  │ │ Agents (local)   │ │  │    │  │ │ AgentCore Runtime     ││ │   │
│  │  │ │ Data-sovereign   │ │  │    │  │ │ Full orchestration    ││ │   │
│  │  │ └──────────────────┘ │  │    │  │ └───────────────────────┘│ │   │
│  │  └──────────────────────┘  │    │  └───────────────────────────┘ │   │
│  │                             │    │                                 │   │
│  └──────────────┬──────────────┘    └────────────────┬────────────────┘   │
│                  │                                    │                    │
│                  └──────── Direct Connect ────────────┘                    │
│                           (Private, encrypted)                             │
│                                                                             │
│  HYBRID PATTERNS:                                                           │
│                                                                             │
│  Pattern 1: CLOUD-PRIMARY with ON-PREM FALLBACK                            │
│  • All traffic routes to cloud; on-prem activates for:                      │
│    - Data residency requirements (PII workloads)                            │
│    - Cloud region outage (business continuity)                              │
│    - Cost optimization (sustained baseline load)                            │
│                                                                             │
│  Pattern 2: SPLIT BY DATA CLASSIFICATION                                    │
│  • PUBLIC/INTERNAL data → Cloud (Bedrock, SageMaker)                       │
│  • CONFIDENTIAL/RESTRICTED → On-prem (local vLLM)                          │
│  • Routing decided by data classification tag                               │
│                                                                             │
│  Pattern 3: TRAIN IN CLOUD, SERVE EVERYWHERE                               │
│  • Training always in cloud (GPU scale, Spot pricing)                       │
│  • Model weights synced to on-prem model cache                              │
│  • Inference happens closest to data/user                                   │
│                                                                             │
│  Pattern 4: CPU INFERENCE ON-PREM, GPU IN CLOUD                            │
│  • Small models (<7B) on CPU (Intel AMX / ARM NEON on-prem)                │
│  • Large models (70B+) on cloud GPU (p5, inf2)                              │
│  • Automatic routing based on model size                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Maturity Model

### AI Platform Maturity: Crawl → Walk → Run → Fly

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AI PLATFORM MATURITY MODEL                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 4: FLY ✈️  (18+ months)                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │ • Autonomous multi-agent systems making business decisions        │     │
│  │ • Self-optimizing platform (auto-scaling, auto-model-selection)   │     │
│  │ • AI-generated AI agents (meta-agents)                            │     │
│  │ • Full self-service: any team deploys agents in <1 hour           │     │
│  │ • Cross-org agent marketplace (A2A federation)                    │     │
│  │ • Real-time model adaptation (online learning + LoRA hot-swap)    │     │
│  │ • Revenue-generating AI products                                  │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                            ▲                                │
│  PHASE 3: RUN 🏃 (9-18 months)            │                                │
│  ┌─────────────────────────────────────────┼─────────────────────────┐     │
│  │ • Multi-agent orchestration (A2A) in production                   │     │
│  │ • Fine-tuned domain models deployed via multi-LoRA                │     │
│  │ • Full observability pipeline (OTel → Dynatrace)                  │     │
│  │ • FinOps chargeback operational                                   │     │
│  │ • Hybrid on-prem/cloud model serving                              │     │
│  │ • 50+ agents in production across multiple teams                  │     │
│  │ • Automated evaluation & model promotion pipelines                │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                            ▲                                │
│  PHASE 2: WALK 🚶 (3-9 months)            │  ← CURRENT POSITION           │
│  ┌─────────────────────────────────────────┼─────────────────────────┐     │
│  │ • AI Gateway operational (Solo + TrueFoundry)                     │     │
│  │ • MCP Server Fleet deployed (21 servers × 2 clusters)             │     │
│  │ • AgentCore Runtime & Gateway in non-prod                         │     │
│  │ • Self-service agent deployment platform live                     │     │
│  │ • LangGraph + Strands frameworks supported                        │     │
│  │ • Guardrails (input/output) enforced                              │     │
│  │ • Cross-account MCP (400+ accounts accessible)                    │     │
│  │ • GitOps (GitLab → ArgoCD) for all AI deployments                │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                            ▲                                │
│  PHASE 1: CRAWL 🐛 (0-3 months)           │                                │
│  ┌─────────────────────────────────────────┼─────────────────────────┐     │
│  │ • Cloud foundation (Control Tower, multi-account) established     │     │
│  │ • Bedrock enabled in target accounts                              │     │
│  │ • First MCP servers (Lambda, Cost Explorer) deployed              │     │
│  │ • Single LLM proxy (basic routing)                                │     │
│  │ • First agent prototype (single tool, single model)               │     │
│  │ • EKS cluster provisioned for AI workloads                        │     │
│  │ • Basic IAM roles for cross-account access                        │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Maturity Assessment Criteria

| Dimension | Crawl | Walk | Run | Fly |
|-----------|-------|------|-----|-----|
| **Agents in Prod** | 0-2 PoCs | 5-20 agents | 20-100 agents | 100+ agents |
| **Teams Served** | 1 (platform team) | 3-5 teams | 10+ teams | All engineering |
| **Model Strategy** | Bedrock only | + Self-hosted | + Fine-tuned | + Online learning |
| **MCP Servers** | 2-3 basic | 10-20 servers | 20-40 servers | 50+ (marketplace) |
| **Deployment** | Manual | GitOps (ArgoCD) | Self-service | Auto-generated |
| **Observability** | CloudWatch | + Tracing | Full OTel stack | AI-powered AIOps |
| **Cost Model** | Untracked | Basic attribution | Chargeback | Self-optimizing |
| **Governance** | Manual review | Guardrails | Policy-as-code | Autonomous compliance |
| **Hybrid** | Cloud only | Cloud + on-prem GPU | Intelligent routing | Edge + IoT serving |

---

## Reference Implementations

### Production Systems

| System | Description | Stack |
|--------|-------------|-------|
| **AgentCore Gateway** | Multi-agent orchestration platform | AWS Bedrock AgentCore, A2A/JSON-RPC 2.0, SSO, Agent Registry |
| **MCP Server Fleet** | 21 MCP servers × 2 clusters covering 400+ AWS accounts | FastAPI, FastMCP, EKS, cross-account STS |
| **AI Gateway** | Multi-dataplane LLM proxy with GitOps config | Solo AI Gateway (Envoy), TrueFoundry CP, ArgoCD, Cognito |
| **AIOps Platform** | Self-service AI agent deployment | LangGraph, Strands, LangChain, EKS, GitLab CI |
| **Multi-LoRA Serving** | Domain-adapted models on shared infrastructure | vLLM, SageMaker, LoRA adapters, S3 model registry |
| **Hybrid LLM Strategy** | On-prem CPU/GPU + Cloud GPU serving | vLLM (CPU: AVX-512/AMX, GPU: CUDA), Direct Connect |

### MCP Server Catalog (Production)

```
┌─────────────────────────────────────────────────────────────────┐
│  MCP SERVER FLEET — PRODUCTION CATALOG                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  AWS Infrastructure          Security & Compliance              │
│  ├── dish-aws-api-mcp        ├── dish-aws-security-hub-mcp     │
│  ├── dish-aws-lambda-mcp     ├── dish-aws-guardduty-mcp        │
│  ├── dish-aws-cost-mcp       ├── dish-aws-iam-mcp              │
│  ├── dish-aws-s3-mcp         ├── dish-aws-config-mcp           │
│  ├── dish-aws-ec2-mcp        └── dish-aws-wellarchitected-mcp  │
│  ├── dish-aws-eks-mcp                                           │
│  ├── dish-aws-cloudwatch-mcp Domain / Integration               │
│  ├── dish-aws-neptune-mcp    ├── dish-dynatrace-mcp            │
│  ├── dish-aws-bedrock-mcp    ├── dish-jira-mcp                 │
│  ├── dish-aws-scan-mcp       ├── dish-servicenow-mcp           │
│  └── dish-aws-graph-mcp      ├── dish-knowledge-base-mcp       │
│                               └── dish-incident-mgmt-mcp        │
│                                                                 │
│  Standard Pattern (all servers):                                │
│  • FastAPI + FastMCP                                            │
│  • /mcp/healthz endpoint                                        │
│  • aws_{service}_ tool naming                                   │
│  • Pydantic Settings (env var config)                           │
│  • Cross-account assume-role                                    │
│  • Deployed on EKS (2 clusters: aitools-np, pe-automation-np)   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Customer Care AI (Domain Example)

```
┌─────────────────────────────────────────────────────────────────┐
│  CUSTOMER CARE AI — DOMAIN ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Customer → Chat UI → AgentCore Gateway → Care Agent            │
│                                               │                 │
│                              ┌────────────────┼──────────┐      │
│                              ▼                ▼          ▼      │
│                     ┌──────────────┐  ┌───────────┐  ┌───────┐ │
│                     │ Account MCP  │  │ Billing   │  │ Equip.│ │
│                     │ • Lookup     │  │ MCP       │  │ MCP   │ │
│                     │ • History    │  │ • Balance │  │ • Plan│ │
│                     │ • Status     │  │ • Payment│  │ • Swap│ │
│                     └──────────────┘  └───────────┘  └───────┘ │
│                                                                 │
│  Model: Claude 4 (via AI Gateway)                               │
│  Guardrails: PII masking, topic restriction, tone enforcement   │
│  Framework: LangGraph (stateful, multi-turn with HITL)          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technology Selection Summary

### Complete Technology Stack

| Layer | Category | Primary Choice | Alternative | Rationale |
|-------|----------|---------------|-------------|-----------|
| **L0** | Account Management | Control Tower + AFT | — | AWS-native multi-account governance |
| **L0** | Networking | Transit Gateway + PrivateLink | — | Hub-spoke at 400+ account scale |
| **L0** | Identity | IAM Identity Center + Cognito | Okta | SSO for humans, Cognito for apps |
| **L1** | Training Compute | Trainium2 (trn2.48xl) | p5.48xl (H100) | 40-60% cheaper; NVIDIA for incompatible models |
| **L1** | Inference Compute | Inferentia2 (inf2.48xl) | g5.48xl (A10G) | Best $/token; G5 for unsupported architectures |
| **L1** | CPU Inference | r8i (Intel AMX) | Graviton4 (ARM) | AVX-512/AMX for BF16 acceleration |
| **L1** | Storage | FSx Lustre + S3 Express | EFS | Throughput for training; latency for serving |
| **L1** | Orchestration | EKS + Karpenter | ECS Fargate | GPU Operator, Ray, vLLM K8s-native |
| **L2** | AI Gateway | Solo (Envoy) + TrueFoundry | Kong AI Gateway | Multi-dataplane, GitOps config, LLM-native policies |
| **L2** | Model Serving | vLLM | TGI (HuggingFace) | Continuous batching, PagedAttention, multi-LoRA |
| **L2** | Managed Serving | SageMaker Endpoints | Bedrock | Auto-scaling, A/B, shadow deployment |
| **L2** | Frontier Models | Bedrock (Claude, Nova) | OpenAI API | AWS-native, PrivateLink, no data retention |
| **L2** | Fine-Tuning | LoRA / QLoRA | Full fine-tune | 100x cheaper, multi-LoRA serving, rapid iteration |
| **L3** | Orchestration | AWS AgentCore | — | Managed A2A, session management, agent registry |
| **L3** | Agent Framework | LangGraph | Strands / LangChain | Deterministic graphs, HITL, checkpointing |
| **L3** | Tool Protocol | MCP (FastMCP) | Custom REST | Open standard, framework-agnostic, discoverability |
| **L3** | Inter-Agent | A2A (JSON-RPC 2.0) | Custom gRPC | Standard protocol, agent cards, task lifecycle |
| **L3** | Guardrails | Bedrock Guardrails + TF | NeMo Guardrails | AWS-native + custom policy engine |
| **L4** | RAG | Bedrock KB + OpenSearch | Pinecone | AWS-native, integrated with Bedrock |
| **CC** | Observability | OTel + Dynatrace | Datadog | Enterprise APM, AI-specific dashboards |
| **CC** | CI/CD | GitLab + ArgoCD | GitHub Actions | Enterprise GitOps, approval workflows |
| **CC** | Cost | Cost Explorer MCP + CUR | Kubecost | Native AWS + custom AI cost attribution |

---

## Integration Pattern Reference

### Pattern 1: Agent → MCP → AWS (Cross-Account)

```
Sequence:
1. Agent receives user query
2. LLM selects appropriate MCP tool(s)
3. Agent invokes MCP server via HTTP/SSE
4. MCP server authenticates (JWT bearer)
5. MCP server assumes role in target account (STS)
6. MCP server calls AWS API
7. Result returned to agent
8. Agent synthesizes response

Latency Budget:
• MCP call overhead: ~50ms
• Cross-account STS: ~100ms
• AWS API call: 200-2000ms (varies)
• Total tool round-trip: 350-2150ms
```

### Pattern 2: Multi-Agent Delegation (A2A)

```
Sequence:
1. Orchestrator agent receives complex task
2. Decomposes into sub-tasks via LLM reasoning
3. Discovers capable agents via Agent Registry
4. Creates A2A tasks (JSON-RPC 2.0)
5. Worker agents execute in parallel
6. Results streamed back via SSE
7. Orchestrator synthesizes final response

Protocol:
• Discovery: GET /.well-known/agent.json → Agent Card
• Task Create: POST /tasks/send (JSON-RPC 2.0)
• Task Status: GET /tasks/{id}
• Streaming: SSE for long-running tasks
```

### Pattern 3: GitOps Model Deployment

```
Sequence:
1. Training job completes → model weights in S3
2. Evaluation pipeline runs (automated)
3. If passes threshold → PR to model-configs repo
4. PR approved by ML lead
5. GitLab CI updates Helm values (model version)
6. ArgoCD detects drift → syncs to EKS
7. vLLM pods pick up new model/adapter
8. Health check passes → traffic shifted
9. Canary monitoring for 30 min
10. Full rollout or auto-rollback
```

---

## Conclusion

This reference architecture represents a **production-proven approach** to enterprise AI platform engineering. It is not a theoretical exercise — every layer, component, and pattern described here operates at scale across a Fortune 200 telecommunications enterprise.

### Key Takeaways

1. **Start with foundation** — Multi-account governance and networking must exist before AI workloads
2. **Invest in the AI Gateway** — It's the highest-leverage component; controls cost, security, and routing
3. **MCP is the integration standard** — Build tools once, use everywhere; framework-agnostic
4. **LoRA over full fine-tuning** — Multi-LoRA serving delivers 90% of the benefit at 1% of the cost
5. **Self-service is non-negotiable** — If teams need platform tickets to deploy agents, adoption stalls
6. **Observe everything** — AI systems fail silently; full tracing is not optional
7. **Hybrid is realistic** — Pure cloud is aspirational; enterprises need on-prem for compliance

### What's Next

The AI platform landscape evolves weekly. Key areas of active development:

- **Agentic memory systems** — Long-term agent memory beyond session context
- **Agent evaluation frameworks** — Automated testing of agent behavior at scale
- **Real-time model adaptation** — Online learning with LoRA hot-swap
- **Multi-cloud agent federation** — A2A across AWS, GCP, and on-prem
- **Edge AI deployment** — Model serving at network edge (5G infrastructure)

---

*This document is maintained as a living reference. Architecture decisions are revisited quarterly as the AI ecosystem evolves.*

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **A2A** | Agent-to-Agent protocol — Google's open standard for inter-agent communication |
| **MCP** | Model Context Protocol — Anthropic's open standard for tool/context integration |
| **AgentCore** | AWS Bedrock AgentCore — managed multi-agent runtime and gateway |
| **LoRA** | Low-Rank Adaptation — parameter-efficient fine-tuning technique |
| **vLLM** | High-performance LLM inference engine with continuous batching |
| **TGI** | Text Generation Inference — HuggingFace's model serving solution |
| **HITL** | Human-in-the-Loop — approval gates within agent workflows |
| **EFA** | Elastic Fabric Adapter — AWS high-performance networking for HPC/ML |
| **FSDP** | Fully Sharded Data Parallel — PyTorch distributed training strategy |
| **STS** | Security Token Service — AWS service for temporary cross-account credentials |

## Appendix B: Recommended Reading

| Topic | Resource |
|-------|----------|
| MCP Protocol Specification | https://modelcontextprotocol.io |
| A2A Protocol Specification | https://google.github.io/A2A |
| AWS Bedrock AgentCore | https://docs.aws.amazon.com/bedrock/latest/userguide/agents-core.html |
| vLLM Multi-LoRA Serving | https://docs.vllm.ai/en/latest/serving/lora.html |
| LangGraph Documentation | https://langchain-ai.github.io/langgraph/ |
| AWS Strands Framework | https://strandsagents.com |
| Solo AI Gateway | https://docs.solo.io/gateway/ |
| TrueFoundry AI Gateway | https://www.truefoundry.com/docs/ai-gateway |
| OpenTelemetry for GenAI | https://opentelemetry.io/docs/specs/semconv/gen-ai/ |

---

*© 2026 — Enterprise AI Platform Reference Architecture. Published for educational and knowledge-sharing purposes.*
