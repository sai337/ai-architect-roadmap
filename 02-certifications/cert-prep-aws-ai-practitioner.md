# AWS Certified AI Practitioner (AIF-C01) — Comprehensive Prep Guide

> **Last Updated:** June 2026  
> **Target Audience:** Cloud professionals preparing for the AIF-C01 exam  
> **Estimated Study Time:** 2–3 weeks (for experienced AWS practitioners)

---

## Table of Contents

1. [Exam Overview](#1-exam-overview)
2. [Domain-by-Domain Breakdown](#2-domain-by-domain-breakdown)
3. [Key AWS AI/ML Services](#3-key-aws-aiml-services)
4. [Intensive Study Plan (2–3 Weeks)](#4-intensive-study-plan-23-weeks)
5. [Top Resources](#5-top-resources)
6. [Sample Questions with Explanations](#6-sample-questions-with-explanations)
7. [Tips for Experienced AWS Architects](#7-tips-for-experienced-aws-architects)

---

## 1. Exam Overview

### Exam Specifications

| Specification | Detail |
|---|---|
| **Exam Code** | AIF-C01 |
| **Level** | Foundational |
| **Format** | Multiple choice, multiple response, ordering, matching, case study |
| **Total Questions** | 65 (50 scored + 15 unscored) |
| **Duration** | 90 minutes |
| **Passing Score** | 700 / 1000 (scaled) |
| **Cost** | $100 USD |
| **Validity** | 3 years |
| **Delivery** | Pearson VUE (test center or online proctored) |
| **Prerequisites** | None (recommended: 6 months exposure to AI/ML on AWS) |
| **Retake Wait** | 14 days between attempts |
| **Languages** | English, Japanese, Korean, Simplified Chinese, and more |

### Domain Weightings

| Domain | Weight | Approx. Scored Questions |
|---|---|---|
| 1. Fundamentals of AI and ML | 20% | ~10 |
| 2. Fundamentals of Generative AI | 24% | ~12 |
| 3. Applications of Foundation Models | 28% | ~14 |
| 4. Guidelines for Responsible AI | 14% | ~7 |
| 5. Security, Compliance, and Governance for AI Solutions | 14% | ~7 |

### Key Exam Characteristics

- **Scenario-driven questions** — Most questions present a business scenario and ask for the *best* AWS service or approach
- **No coding required** — Purely conceptual; no Python, no API calls
- **Multiple "correct" answers** — Often two or three options could work; only one is the *best fit* for the constraint in the stem
- **Watch for qualifiers** — "MOST cost-effective," "LEAST operational overhead," "without managing infrastructure," "managed service"
- **Unscored questions are invisible** — You cannot identify which 15 questions are unscored; treat every question as if it counts
- **No per-domain cutoff** — Only the overall scaled score matters

### Scoring Notes

AWS uses Item Response Theory (IRT) scaled scoring. Practically, **aim for 75–80% accuracy** in practice exams to be safely above the 700 threshold on exam day. There is no penalty for guessing — never leave a question blank.

---

## 2. Domain-by-Domain Breakdown

---

### Domain 1: Fundamentals of AI and ML (20%)

**What it covers:** Vocabulary, concepts, and the ML lifecycle that form the baseline for the entire exam.

#### Key Concepts to Know

| Topic | What to Understand |
|---|---|
| AI vs. ML vs. Deep Learning vs. GenAI | Nested hierarchy: AI ⊃ ML ⊃ Deep Learning ⊃ Generative AI |
| Supervised Learning | Labeled data → predict outcomes (classification, regression) |
| Unsupervised Learning | No labels → find patterns (clustering, anomaly detection, dimensionality reduction) |
| Reinforcement Learning | Agent learns via reward/penalty signals over time |
| Classification vs. Regression | Categories vs. continuous numeric values |
| Clustering | Grouping similar items without predefined labels |
| Anomaly Detection | Identifying outliers (fraud, defects) |
| Recommendation Systems | Personalized suggestions based on behavior |
| ML Lifecycle | Problem framing → Data collection → Data prep → Feature engineering → Training → Evaluation → Deployment → Monitoring |
| Neural Networks | Layers of nodes; basis of deep learning |
| Overfitting vs. Underfitting | Model memorizes training data vs. model too simple to learn patterns |
| Training/Validation/Test Split | Proper data partitioning for model evaluation |
| Bias-Variance Tradeoff | Balance between model complexity and generalization |

#### AWS Services Mapped to ML Problem Types

| Problem Type | AWS Service |
|---|---|
| NLP / Sentiment / Entities | Amazon Comprehend |
| Image/Video Analysis | Amazon Rekognition |
| Document Extraction (OCR) | Amazon Textract |
| Speech-to-Text | Amazon Transcribe |
| Text-to-Speech | Amazon Polly |
| Translation | Amazon Translate |
| Conversational AI | Amazon Lex |
| Recommendations | Amazon Personalize |
| Time-Series Forecasting | Amazon Forecast |
| Enterprise Search | Amazon Kendra |
| End-to-End ML Platform | Amazon SageMaker |

#### Common Pitfalls

- Confusing **classification** (predicting a category like spam/not-spam) with **regression** (predicting a number like house price)
- Confusing **unsupervised** (no labels, pattern discovery) with **reinforcement** learning (reward-based sequential decisions)
- Mixing up similar services: Comprehend vs. Kendra, Rekognition vs. Textract, Transcribe vs. Polly

---

### Domain 2: Fundamentals of Generative AI (24%)

**What it covers:** Conceptual understanding of generative AI, foundation models, and the AWS infrastructure that supports them.

#### Key Concepts to Know

| Topic | What to Understand |
|---|---|
| Generative AI vs. Traditional ML | Traditional = classify/predict; Generative = create new content |
| Foundation Models (FMs) | Large pre-trained models adaptable to many tasks |
| Large Language Models (LLMs) | FMs specialized for text (GPT, Claude, Llama, Titan) |
| Multimodal Models | Models that process text + images + audio |
| Tokens | Basic units of text processing; context window = max tokens |
| Embeddings | Dense vector representations of text/data |
| Vector Databases | Store and retrieve embeddings for semantic similarity |
| Context Windows | Maximum input length a model can process at once |
| Hallucinations | Model generates plausible but factually incorrect content |
| Knowledge Cutoffs | Model has no awareness of events after training data |
| Deterministic vs. Stochastic | Same input may produce different outputs due to sampling |
| Transfer Learning | Reusing pre-trained model knowledge for new tasks |

#### AWS Generative AI Services

| Service | Purpose |
|---|---|
| **Amazon Bedrock** | Managed serverless API access to FMs (Claude, Titan, Llama, etc.) + Knowledge Bases, Agents, Guardrails |
| **SageMaker JumpStart** | Pre-trained models requiring endpoint deployment (more control, more ops) |
| **Amazon Q Business** | Enterprise knowledge assistant (internal docs, data sources) |
| **Amazon Q Developer** | AI coding assistant (code generation, explanation, debugging) |
| **Amazon Titan** | AWS's own family of foundation models |

#### Critical Distinction: Bedrock vs. SageMaker JumpStart

| Aspect | Bedrock | SageMaker JumpStart |
|---|---|---|
| Infrastructure | Serverless, fully managed | Requires endpoint deployment |
| Use Case | Quick API access to FMs | Custom ML workflows, full control |
| Operational Overhead | Minimal | Moderate to high |
| When to Choose | "Without managing infrastructure" | "Need full control over training/hosting" |

#### Common Pitfalls

- Choosing **SageMaker JumpStart** when the question says "serverless" or "least operational overhead" → **Bedrock**
- Confusing **Q Developer** (code) with **Q Business** (enterprise knowledge)
- Underestimating hallucination questions — recognize this as an LLM *limitation*, not a bug to be "fixed"

---

### Domain 3: Applications of Foundation Models (28%) ⭐ HIGHEST WEIGHT

**What it covers:** Practical application of FMs — prompting, customization, evaluation, and inference configuration.

#### Key Concepts to Know

##### Prompt Engineering Techniques

| Technique | Description | When to Use |
|---|---|---|
| Zero-shot | Single instruction, no examples | Simple well-defined tasks |
| Few-shot | Instruction + examples of input/output pairs | When model needs pattern guidance |
| Chain-of-thought | "Think step by step" reasoning | Complex reasoning, math, logic |
| System prompts | Set persona/rules for the model | Consistent behavior across interactions |
| Prompt chaining | Output of one prompt feeds into next | Multi-step workflows |

##### Model Customization Spectrum

| Approach | When to Use | Cost/Effort | Data Needed |
|---|---|---|---|
| Prompt Engineering | Quick iteration, general tasks | Lowest | None (just prompt text) |
| RAG (Retrieval-Augmented Generation) | Need current/proprietary facts | Low-Medium | Document corpus |
| Fine-tuning | Consistent style/behavior change | High | Labeled examples (hundreds–thousands) |
| Continued Pre-training | Domain-specific language understanding | Highest | Large domain corpus |

##### Critical Decision: RAG vs. Fine-tuning

| Scenario | Use RAG | Use Fine-tuning |
|---|---|---|
| Need up-to-date proprietary facts | ✅ | ❌ |
| Need to change model behavior/style | ❌ | ✅ |
| Data changes frequently | ✅ | ❌ |
| Need citations/sources | ✅ | ❌ |
| Need consistent tone/format | ❌ | ✅ |

**Rule of thumb:** Changing *facts* = RAG. Changing *style/behavior* = Fine-tuning.

##### Inference Parameters

| Parameter | Effect | Higher Value | Lower Value |
|---|---|---|---|
| **Temperature** | Controls randomness | More creative/diverse | More deterministic/focused |
| **Top-p** | Nucleus sampling threshold | More diverse vocabulary | More predictable |
| **Top-k** | Limits token candidates | More choices | Fewer choices |
| **Max Tokens** | Response length limit | Longer responses | Shorter responses |
| **Stop Sequences** | Tokens that halt generation | — | — |

##### Evaluation Metrics

| Metric | Best For | What It Measures |
|---|---|---|
| **ROUGE** | Summarization | Overlap between generated and reference summaries |
| **BLEU** | Translation | N-gram precision vs. reference translations |
| **BERTScore** | Semantic similarity | Contextual embedding similarity |
| **Perplexity** | Language model quality | How "surprised" the model is by text (lower = better) |
| **Human Evaluation** | Subjective quality | Relevance, helpfulness, safety |

##### RAG Architecture (Amazon Bedrock Knowledge Bases)

```
User Query → Embedding → Vector Search → Retrieved Context → FM + Context → Response
                              ↑
                    Knowledge Base (S3, web crawl, etc.)
                    indexed into vector store (OpenSearch, etc.)
```

#### Common Pitfalls

- Choosing **fine-tuning** when the scenario describes needing current proprietary data → **RAG**
- Getting temperature direction wrong: **Higher = more random** (not more focused)
- Picking wrong evaluation metric: ROUGE ≠ translation; BLEU ≠ summarization
- Defaulting to "just use prompt engineering" when the scenario clearly needs RAG or fine-tuning

---

### Domain 4: Guidelines for Responsible AI (14%)

**What it covers:** Ethics, fairness, bias, and AWS tools for responsible AI.

#### Core Responsible AI Principles

| Principle | Meaning |
|---|---|
| **Fairness** | Model treats all groups equitably; no discriminatory outcomes |
| **Inclusivity** | Diverse and representative training data |
| **Transparency** | Stakeholders understand how AI decisions are made |
| **Explainability** | Model outputs can be interpreted and justified |
| **Safety / Robustness** | AI behaves reliably even with adversarial/unexpected inputs |
| **Privacy** | User data is protected; PII is handled appropriately |
| **Veracity** | Model outputs are truthful and grounded |
| **Human Oversight** | Humans remain in the loop for high-stakes decisions |

#### AWS Responsible AI Tools

| Tool | Purpose | Applies To |
|---|---|---|
| **SageMaker Clarify** | Bias detection + model explainability (SHAP values) | Classical ML models |
| **Bedrock Guardrails** | Content filtering, denied topics, PII redaction, grounding checks | Generative AI / Foundation models |
| **Model Cards / AI Service Cards** | Documentation of model capabilities, limitations, intended use | All AI models |

#### Critical Distinction: Clarify vs. Guardrails

| Aspect | SageMaker Clarify | Bedrock Guardrails |
|---|---|---|
| Target | Traditional ML models | Generative AI / LLMs |
| Function | Detect bias in data/predictions, explain feature importance | Filter harmful content, block topics, redact PII |
| When to Choose | "Bias in training data," "explain predictions" | "Prevent harmful outputs," "block topics," "PII in responses" |

#### Common Pitfalls

- Confusing Clarify (classical ML bias) and Guardrails (generative content filtering)
- Treating responsible AI as generic philosophy — the exam wants *specific AWS tool* answers
- Forgetting that responsible AI applies **across the entire lifecycle**, not just deployment

---

### Domain 5: Security, Compliance, and Governance for AI Solutions (14%)

**What it covers:** Securing and governing AI workloads using standard AWS security primitives.

#### Key Concepts to Know

| Topic | AWS Service/Mechanism |
|---|---|
| Access Control | IAM roles, policies, least privilege |
| Network Isolation | VPC, private subnets, security groups |
| Private Connectivity | AWS PrivateLink (keeps traffic off public internet) |
| Encryption at Rest | AWS KMS (customer-managed keys) |
| Encryption in Transit | TLS/SSL |
| API Activity Logging | AWS CloudTrail (who did what, when) |
| Metrics & Operational Monitoring | Amazon CloudWatch (performance, logs, alarms) |
| FM Invocation Logging | Bedrock model invocation logging |
| Model Drift Detection | SageMaker Model Monitor |
| Data Governance | AWS Lake Formation, S3 bucket policies |
| PII Detection & Handling | Comprehend PII detection, Bedrock Guardrails |
| Compliance Frameworks | GDPR, HIPAA, SOC, ISO — AWS Artifact for reports |
| Data/Model Lineage | SageMaker ML Lineage Tracking |

#### Critical Distinction: CloudTrail vs. CloudWatch

| Service | Purpose | Think of It As |
|---|---|---|
| **CloudTrail** | API activity auditing — *who* did *what* | Security camera footage |
| **CloudWatch** | Metrics, logs, alarms — *how is it performing* | Dashboard gauges |

#### Common Pitfalls

- Forgetting that AI workloads use the **same security primitives** as any other AWS workload
- Choosing CloudWatch when CloudTrail is correct (and vice versa)
- Missing the role of **PrivateLink** for keeping Bedrock/SageMaker traffic on the AWS backbone
- Not knowing that **Bedrock model invocation logging** can capture all prompts and responses for audit

---

## 3. Key AWS AI/ML Services

### Comprehensive Service Reference

#### Generative AI Services

| Service | What It Does | Key Features |
|---|---|---|
| **Amazon Bedrock** | Managed serverless access to FMs | Multi-provider (Anthropic, Meta, Amazon, etc.), Knowledge Bases, Agents, Guardrails, Fine-tuning |
| **Amazon Titan** | AWS's own FMs (text, embeddings, image) | Titan Text, Titan Embeddings, Titan Image Generator |
| **Amazon Q Business** | Enterprise AI assistant | Connects to internal data, RAG-based, access controls |
| **Amazon Q Developer** | AI coding assistant | Code generation, debugging, transformation, security scanning |
| **SageMaker JumpStart** | Pre-built model hub + deployment | Deploy open-source FMs on SageMaker endpoints |

#### Natural Language Processing (NLP)

| Service | What It Does | Use Cases |
|---|---|---|
| **Amazon Comprehend** | NLP analysis | Sentiment, entities, key phrases, PII detection, topic modeling, custom classification |
| **Amazon Translate** | Neural machine translation | Real-time and batch translation, custom terminology |
| **Amazon Transcribe** | Speech-to-text | Call transcription, subtitles, medical transcription |
| **Amazon Polly** | Text-to-speech | Natural-sounding voices, SSML support, neural voices |
| **Amazon Lex** | Conversational interfaces | Chatbots, voice bots, same tech as Alexa |

#### Computer Vision

| Service | What It Does | Use Cases |
|---|---|---|
| **Amazon Rekognition** | Image & video analysis | Face detection, object detection, content moderation, celebrity recognition, PPE detection |
| **Amazon Textract** | Document text extraction | OCR, form extraction, table extraction, handwriting |

#### Specialized ML Services

| Service | What It Does | Use Cases |
|---|---|---|
| **Amazon Personalize** | Recommendation engine | Product recommendations, content personalization, user segmentation |
| **Amazon Forecast** | Time-series forecasting | Demand planning, financial forecasting, resource planning |
| **Amazon Kendra** | Intelligent search | Enterprise search with natural language queries, FAQ extraction |
| **Amazon Fraud Detector** | Fraud detection | Online payment fraud, account takeover, fake accounts |

#### ML Platform

| Service | What It Does | Key Features |
|---|---|---|
| **Amazon SageMaker** | End-to-end ML platform | Notebooks, training, hosting, Studio, Canvas (no-code), Pipelines, Feature Store |
| **SageMaker Canvas** | No-code ML | Visual interface for business analysts to build models |
| **SageMaker Clarify** | Responsible AI | Bias detection, model explainability (SHAP) |
| **SageMaker Model Monitor** | Production monitoring | Data drift, model quality, bias drift detection |

#### Data & Infrastructure

| Service | Role in AI/ML |
|---|---|
| **Amazon S3** | Primary data lake storage for training data |
| **AWS Glue** | ETL and data catalog for ML data preparation |
| **AWS Lake Formation** | Data governance and fine-grained access control |
| **Amazon OpenSearch** | Vector store for RAG / semantic search |
| **Amazon DynamoDB** | Low-latency feature serving |

### Service Selection Quick Reference

> **"I need to..."**

| Need | Service |
|---|---|
| Access foundation models via API without infrastructure | **Bedrock** |
| Build and train custom ML models end-to-end | **SageMaker** |
| Detect sentiment in customer reviews | **Comprehend** |
| Extract text from scanned documents | **Textract** |
| Identify objects in images | **Rekognition** |
| Convert audio recordings to text | **Transcribe** |
| Generate natural-sounding speech | **Polly** |
| Build a chatbot | **Lex** |
| Recommend products to users | **Personalize** |
| Forecast sales demand | **Forecast** |
| Search enterprise documents with natural language | **Kendra** |
| Translate text between languages | **Translate** |
| Filter harmful AI outputs | **Bedrock Guardrails** |
| Detect bias in ML model predictions | **SageMaker Clarify** |
| Ground LLM responses in company data | **Bedrock Knowledge Bases (RAG)** |
| Automate multi-step AI workflows | **Bedrock Agents** |

---

## 4. Intensive Study Plan (2–3 Weeks)

### Prerequisites Assessment

If you already have AWS Solutions Architect or equivalent experience, you can:
- ✅ Skip basic cloud concepts, IAM, VPC, S3, KMS fundamentals
- ✅ Skim Domain 5 (security/governance) — you know this pattern
- ⚠️ Focus heavily on Domains 2 and 3 (generative AI and FM applications)
- ⚠️ Don't underestimate Domain 4 (responsible AI — new to most architects)

---

### Week 1: Foundations (Domains 1 & 2)

| Day | Focus | Activities | Hours |
|---|---|---|---|
| **Day 1** | AI/ML Vocabulary | Define all key terms (AI, ML, DL, GenAI). Map problem types to learning categories. | 2 |
| **Day 2** | ML Lifecycle & Algorithms | Study lifecycle stages. Understand classification, regression, clustering at conceptual level. | 2 |
| **Day 3** | AWS AI Services Catalog | Build a one-line cheat sheet for each AI service. Map services → use cases. | 2 |
| **Day 4** | Generative AI Fundamentals | Tokens, embeddings, context windows, hallucinations, capabilities/limitations. | 2 |
| **Day 5** | AWS GenAI Stack | Deep dive: Bedrock (features, providers), Q Business, Q Developer, SageMaker JumpStart. | 2 |
| **Day 6** | Practice Questions (D1+D2) | 50+ practice questions covering Domains 1 and 2. Review every wrong answer. | 2.5 |
| **Day 7** | Review & Gap Fill | Revisit weak areas from practice. Create flashcards for frequently-missed concepts. | 1.5 |

---

### Week 2: Applications & Responsible AI (Domains 3 & 4)

| Day | Focus | Activities | Hours |
|---|---|---|---|
| **Day 8** | Prompt Engineering | Study all techniques: zero-shot, few-shot, chain-of-thought, system prompts, chaining. | 2.5 |
| **Day 9** | RAG & Knowledge Bases | Understand RAG architecture, when to use vs. fine-tuning, Bedrock Knowledge Bases. | 2 |
| **Day 10** | Model Customization | Fine-tuning vs. continued pre-training vs. RAG vs. prompt engineering decision tree. | 2 |
| **Day 11** | Inference Parameters & Evaluation | Temperature, top-p, max tokens. ROUGE, BLEU, BERTScore, perplexity. | 2 |
| **Day 12** | Responsible AI Principles | Memorize principles. Study Clarify vs. Guardrails vs. Model Cards. | 2 |
| **Day 13** | Practice Questions (D3+D4) | 60+ practice questions. Domain 3 is the hardest — review thoroughly. | 3 |
| **Day 14** | Review & Gap Fill | Build decision trees and comparison tables for commonly confused topics. | 2 |

---

### Week 3: Security, Full-Length Mocks, and Final Review

| Day | Focus | Activities | Hours |
|---|---|---|---|
| **Day 15** | Domain 5 Security & Governance | IAM for AI, PrivateLink, KMS, CloudTrail vs. CloudWatch, Bedrock logging. | 2 |
| **Day 16** | Full Mock Exam #1 | 65 questions, 90-minute timer. Score by domain. Identify weak areas. | 2.5 |
| **Day 17** | Mock #1 Review | Deep review of every wrong answer. Add to study notes. | 2 |
| **Day 18** | Full Mock Exam #2 | Second timed mock. Track improvement per domain. | 2.5 |
| **Day 19** | Mock #2 Review + Weak Areas | Targeted study on lowest-scoring domain(s). | 2 |
| **Day 20** | Full Mock Exam #3 (if needed) | Final confidence check. Target: 80%+ overall, no domain below 70%. | 2.5 |
| **Day 21** | Light Review + Rest | Quick flashcard review. No cramming. Get good sleep. **Exam day tomorrow!** | 1 |

### Readiness Criteria

✅ **Ready to sit the exam when:**
- Overall mock exam score consistently **80%+**
- No single domain below **70%**
- You can explain the difference between RAG, fine-tuning, and prompt engineering without hesitation
- You can map every major AWS AI service to its use case in <5 seconds

---

## 5. Top Resources

### Official AWS Resources (Free)

| Resource | Link / Description |
|---|---|
| **AIF-C01 Exam Guide** | [Official exam guide](https://docs.aws.amazon.com/aws-certification/latest/examguides/ai-practitioner-01.html) — Read this first |
| **AWS Skill Builder: Exam Prep Standard** | Free exam prep course on AWS Skill Builder (search "AI Practitioner") |
| **AWS Skill Builder: AI/ML Foundations** | Free foundational courses on AI/ML concepts |
| **AWS Sample Questions** | 10 official sample questions in the exam guide PDF |
| **AWS Classroom Training** | [1-day instructor-led exam prep](https://aws.amazon.com/training/classroom/exam-prep-aws-certified-ai-practitioner-aif-c01/) (paid) |
| **AWS AI Service Documentation** | Service pages for Bedrock, SageMaker, Comprehend, etc. |
| **AWS AI Blog** | [aws.amazon.com/blogs/machine-learning](https://aws.amazon.com/blogs/machine-learning/) |

### AWS Skill Builder Recommended Courses

1. **"Exam Prep Standard: AWS Certified AI Practitioner (AIF-C01)"** — Free, aligned to exam domains
2. **"Generative AI Foundations on AWS"** — Covers foundation model concepts
3. **"Amazon Bedrock Getting Started"** — Essential for Domains 2 & 3
4. **"Responsible AI Practices"** — Covers Domain 4
5. **"Exploring Artificial Intelligence Use Cases and Applications"** — Domain 1 foundations
6. **"Amazon Q Business Getting Started"** — Understand the Q ecosystem

### Third-Party Courses

| Platform | Course | Notes |
|---|---|---|
| **Coursera** | [AWS AI Practitioner Certification Prep Specialization](https://www.coursera.org/specializations/aws-certified-ai-practitioner) | Multi-course specialization |
| **Coursera** | [AWS Certified AI Practitioner](https://www.coursera.org/learn/aws-certified-ai-practitioner) | Single focused course |
| **KodeKloud** | AWS AI Practitioner complete study guide | Hands-on labs included |
| **Tutorials Dojo** | AIF-C01 study guide + practice exams | High-quality, exam-realistic |

### Practice Exams

| Source | Details |
|---|---|
| **AWS Official Practice Exam** | 20 questions on Skill Builder (free with subscription, or pay ~$20) |
| **Tutorials Dojo** | Multiple full-length practice exams with explanations |
| **Whizlabs** | AIF-C01 practice tests with domain breakdowns |
| **FlashGenius** | [Free practice questions](https://flashgenius.net/sample-tests/aws-caip) |
| **Sailor.sh** | Mock exam bundles calibrated to domain weights |

### Whitepapers & Documentation

| Document | Why It Matters |
|---|---|
| **"Machine Learning Lens" — AWS Well-Architected** | ML best practices in AWS context |
| **"Responsible Use of Machine Learning" whitepaper** | Domain 4 core content |
| **Amazon Bedrock User Guide** | Deep understanding of Bedrock features |
| **"Prompt Engineering Best Practices" (AWS docs)** | Domain 3 essential reading |
| **"Security in Amazon Bedrock"** | Domain 5 specifics |
| **"Generative AI on AWS" overview page** | Big-picture AWS GenAI strategy |

### Community Resources

| Resource | Link |
|---|---|
| **r/AWSCertifications** | reddit.com/r/AWSCertifications — Exam experiences and tips |
| **AWS Re:Post** | Official AWS community forum |
| **GitHub Study Guides** | [madtank/aws-ai-practitioner-study-guide](https://github.com/madtank/aws-ai-practitioner-study-guide), [vicsz/aif-c01-study-notes](https://github.com/vicsz/aif-c01-study-notes) |
| **AWS Certification Discord** | Active study groups and discussion |

---

## 6. Sample Questions with Explanations

### Domain 1: Fundamentals of AI and ML

---

**Q1.** A retail company wants to predict which customers are likely to cancel their subscription next month. Which type of machine learning problem is this?

- A) Regression
- B) Clustering
- C) Classification ✅
- D) Reinforcement learning

**Explanation:** Predicting a binary outcome (will cancel / won't cancel) is a **classification** problem. Regression predicts continuous values (e.g., how much they'll spend). Clustering groups similar items without predefined labels. Reinforcement learning is about sequential decision-making with rewards.

---

**Q2.** A company wants to automatically extract key information from thousands of scanned invoices, including vendor names, amounts, and dates from table formats. Which AWS service should they use?

- A) Amazon Comprehend
- B) Amazon Rekognition
- C) Amazon Textract ✅
- D) Amazon Kendra

**Explanation:** **Amazon Textract** specializes in extracting text, forms, and tables from scanned documents. Comprehend does NLP analysis on text (not document extraction). Rekognition is for image/video analysis (faces, objects). Kendra is for enterprise search.

---

**Q3.** During the ML lifecycle, a data scientist discovers that the "city" column in the training dataset contains 500 unique text values and decides to convert them into numeric representations. Which ML lifecycle stage is this?

- A) Data collection
- B) Data preparation
- C) Feature engineering ✅
- D) Model evaluation

**Explanation:** Converting categorical data into numeric form (e.g., one-hot encoding, label encoding) is **feature engineering** — transforming raw data into features the model can use. Data preparation involves cleaning (handling missing values, removing duplicates), while data collection is gathering the data.

---

### Domain 2: Fundamentals of Generative AI

---

**Q4.** A marketing team wants to generate personalized email campaigns using AI without managing any infrastructure. They need access to multiple foundation model providers to test which works best. Which AWS service should they use?

- A) Amazon SageMaker
- B) Amazon Bedrock ✅
- C) Amazon Q Business
- D) SageMaker JumpStart

**Explanation:** **Amazon Bedrock** provides serverless API access to multiple FM providers (Anthropic, Meta, Amazon Titan, etc.) without managing infrastructure. SageMaker and JumpStart require endpoint deployment. Q Business is for enterprise knowledge Q&A, not content generation workflows.

---

**Q5.** A developer is using a large language model and notices the model confidently states that a company's Q3 revenue was $4.2 billion, when the actual figure was $3.8 billion. What is this phenomenon called?

- A) Underfitting
- B) Overfitting
- C) Hallucination ✅
- D) Bias

**Explanation:** **Hallucination** is when an LLM generates plausible-sounding but factually incorrect information with confidence. This is a fundamental limitation of LLMs — they generate statistically likely text, not verified facts. This is different from bias (systematic unfairness) or overfitting (memorizing training data).

---

**Q6.** What is the primary difference between Amazon Q Business and Amazon Q Developer?

- A) Q Business uses larger models
- B) Q Business is for enterprise knowledge queries; Q Developer is for code assistance ✅
- C) Q Developer is serverless; Q Business requires hosting
- D) Q Developer only works with Python

**Explanation:** **Q Business** connects to enterprise data sources to answer business questions. **Q Developer** assists with coding tasks (generation, debugging, transformation, security scanning). Both are serverless managed services.

---

### Domain 3: Applications of Foundation Models

---

**Q7.** A legal firm wants their AI assistant to answer questions accurately using their proprietary case law database. The information changes weekly as new cases are added. Which approach is MOST appropriate?

- A) Fine-tune the model weekly
- B) Use Retrieval-Augmented Generation (RAG) ✅
- C) Increase the model's temperature
- D) Use continued pre-training

**Explanation:** **RAG** is the best approach when you need the model to reference frequently-changing proprietary information with citations. Fine-tuning is expensive, slow, and doesn't handle frequently changing facts well. RAG retrieves relevant documents at query time and provides them as context to the FM. Increasing temperature would make outputs more random, not more accurate.

---

**Q8.** A content team wants their AI to generate product descriptions that consistently match their brand's formal, sophisticated tone. The team has 2,000 examples of previously-written descriptions. Which approach is MOST appropriate?

- A) RAG with a knowledge base
- B) Fine-tuning the model ✅
- C) Zero-shot prompting
- D) Increasing max tokens

**Explanation:** **Fine-tuning** is the right approach when you want to consistently change a model's *style or behavior* with sufficient labeled examples. RAG is for injecting facts, not style. Zero-shot prompting might approximate the tone but won't be as consistent as fine-tuning with 2,000 examples.

---

**Q9.** A developer wants their generative AI application to produce highly creative and varied marketing slogans. Which inference parameter adjustment would BEST achieve this?

- A) Decrease temperature
- B) Increase temperature ✅
- C) Decrease max tokens
- D) Add stop sequences

**Explanation:** **Higher temperature** increases randomness and creativity in model outputs. Lower temperature makes outputs more deterministic and repetitive. Max tokens controls length, not creativity. Stop sequences control where generation ends.

---

**Q10.** A company is evaluating how well their AI summarizes long legal documents by comparing generated summaries to human-written reference summaries. Which evaluation metric is MOST appropriate?

- A) BLEU
- B) ROUGE ✅
- C) Perplexity
- D) F1 Score

**Explanation:** **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) measures overlap between generated and reference summaries — designed specifically for summarization tasks. BLEU is for translation quality. Perplexity measures language model quality overall. F1 Score is a general classification metric.

---

### Domain 4: Guidelines for Responsible AI

---

**Q11.** A healthcare company has deployed an ML model for patient risk scoring. They discover the model performs significantly better for one demographic group than others. Which AWS tool should they use to investigate and quantify this bias?

- A) Amazon Bedrock Guardrails
- B) Amazon SageMaker Clarify ✅
- C) AWS CloudTrail
- D) Amazon Comprehend

**Explanation:** **SageMaker Clarify** detects and measures bias in traditional ML models and provides explainability (feature attribution via SHAP values). Bedrock Guardrails is for filtering generative AI outputs (content filtering, PII redaction), not for detecting bias in ML predictions. CloudTrail is for API audit logging.

---

**Q12.** A company is deploying a customer service chatbot using Amazon Bedrock. They want to ensure the chatbot never discusses competitor products or generates content about certain sensitive topics. Which tool should they use?

- A) SageMaker Clarify
- B) Amazon Bedrock Guardrails ✅
- C) SageMaker Model Monitor
- D) AWS Config

**Explanation:** **Bedrock Guardrails** allows you to define denied topics, content filters, and word filters that prevent the FM from generating specific types of content. Clarify is for bias in classical ML. Model Monitor watches for drift. AWS Config tracks resource configurations.

---

### Domain 5: Security, Compliance, and Governance

---

**Q13.** A financial services company needs to ensure all Amazon Bedrock API calls are logged for regulatory compliance, including which users invoked which models and when. Which AWS service provides this capability?

- A) Amazon CloudWatch
- B) AWS CloudTrail ✅
- C) Amazon GuardDuty
- D) AWS Config

**Explanation:** **CloudTrail** records API activity — who called which API, when, and from where. This is the "who did what" audit trail required for regulatory compliance. CloudWatch provides metrics and operational logs but doesn't provide the user-level API audit trail. GuardDuty is for threat detection. Config tracks resource state changes.

---

**Q14.** A company wants to ensure that data sent to Amazon Bedrock for inference never traverses the public internet. Which AWS mechanism should they implement?

- A) S3 bucket policies
- B) AWS PrivateLink ✅
- C) Security groups
- D) AWS WAF

**Explanation:** **AWS PrivateLink** creates a private connection between your VPC and the Bedrock service, ensuring traffic stays on the AWS backbone and never crosses the public internet. Security groups control inbound/outbound traffic rules within a VPC but don't change the network path. WAF is for web application firewall protection.

---

## 7. Tips for Experienced AWS Architects

### What You Can Safely Skim (~30% Time Saved)

If you already hold SA-Associate/Pro or similar certifications:

| Topic | Why You Can Skim |
|---|---|
| IAM fundamentals (roles, policies, least privilege) | You know this cold |
| VPC, subnets, security groups | Same concepts apply to AI workloads |
| S3, KMS, encryption at rest/in transit | Standard AWS security patterns |
| CloudTrail vs. CloudWatch distinction | You've used both extensively |
| AWS shared responsibility model | Identical for AI services |
| PrivateLink / VPC Endpoints | Same architecture pattern |

### What Will Surprise You (~70% of Your Study Time Here)

| Topic | Why It's New | Priority |
|---|---|---|
| **Prompt engineering techniques** | Zero-shot, few-shot, chain-of-thought — these are new | 🔴 Critical |
| **RAG architecture** | Bedrock Knowledge Bases, vector stores, embeddings | 🔴 Critical |
| **Model customization spectrum** | When to use RAG vs. fine-tuning vs. continued pre-training | 🔴 Critical |
| **Inference parameters** | Temperature, top-p, top-k, max tokens and their effects | 🔴 Critical |
| **Evaluation metrics** | ROUGE, BLEU, BERTScore, perplexity | 🟡 High |
| **Amazon Bedrock feature set** | Agents, Knowledge Bases, Guardrails, model providers | 🟡 High |
| **Foundation model concepts** | Tokens, context windows, embeddings, hallucinations | 🟡 High |
| **Responsible AI tools** | Clarify vs. Guardrails — easy to confuse | 🟡 High |
| **Amazon Q (Business + Developer)** | New services, distinct use cases | 🟡 High |
| **ML problem types** | Classification vs. regression vs. clustering | 🟢 Quick review |
| **ML lifecycle stages** | Feature engineering, model evaluation | 🟢 Quick review |

### Your Compressed 2-Week Plan

**Week 1 (8–10 hours):**
- Day 1–2: Deep dive on Amazon Bedrock (all features), generative AI concepts
- Day 3–4: Prompt engineering techniques + RAG architecture
- Day 5: Model customization decision tree (RAG vs. fine-tuning)
- Day 6: Inference parameters + evaluation metrics
- Day 7: Responsible AI (Clarify vs. Guardrails) + quick Domain 1 review

**Week 2 (6–8 hours):**
- Day 8–9: Two full mock exams + deep review
- Day 10–11: Focus on weakest domain from mocks
- Day 12: Third mock exam — target 80%+
- Day 13: Light review, flashcards on commonly confused topics
- Day 14: **Exam day** 🎯

### Common Mistakes Made by Experienced Architects

1. **Over-engineering answers** — This is a *foundational* exam. The answer is usually the simplest managed service, not a complex architecture.
2. **Defaulting to SageMaker** — Bedrock is the answer when the question says "serverless," "managed," or "without infrastructure."
3. **Skipping responsible AI** — You haven't encountered Clarify/Guardrails distinctions in architecture work. Study this fresh.
4. **Ignoring generative AI specifics** — Tokens, temperature, hallucinations, RAG — these aren't covered in SA exams.
5. **Rushing Domain 3** — It's 28% of the exam and the most technical domain. Don't assume your architecture experience covers it.

### Quick Wins for Architects

- Domain 5 (Security) should be nearly free points — you already know IAM, VPC, KMS, CloudTrail
- Service selection questions are your strength — you think in terms of "right tool for the job"
- Scenario-based question format is familiar from SA exams
- "Least operational overhead" / "managed service" patterns are your bread and butter

---

## Appendix: Quick Reference Tables

### Acronyms & Terms

| Term | Meaning |
|---|---|
| FM | Foundation Model |
| LLM | Large Language Model |
| RAG | Retrieval-Augmented Generation |
| NLP | Natural Language Processing |
| ROUGE | Recall-Oriented Understudy for Gisting Evaluation |
| BLEU | Bilingual Evaluation Understudy |
| SHAP | SHapley Additive exPlanations (explainability method) |
| PII | Personally Identifiable Information |
| IRT | Item Response Theory (AWS scoring method) |

### The "One-Line" Service Cheat Sheet

| Service | One Line |
|---|---|
| Bedrock | Serverless API access to multiple FMs with RAG, agents, and guardrails |
| SageMaker | Full ML platform: build, train, deploy any model |
| JumpStart | Pre-built models deployed on SageMaker endpoints |
| Q Business | Enterprise knowledge assistant connected to company data |
| Q Developer | AI coding assistant for developers |
| Comprehend | NLP: sentiment, entities, PII, key phrases, custom classification |
| Rekognition | Computer vision: faces, objects, content moderation |
| Textract | Extract text, forms, and tables from documents |
| Transcribe | Speech-to-text |
| Polly | Text-to-speech |
| Translate | Neural machine translation |
| Lex | Build conversational chatbots (same tech as Alexa) |
| Personalize | Real-time personalized recommendations |
| Forecast | Time-series forecasting |
| Kendra | Intelligent enterprise search with NLP queries |
| Clarify | Bias detection and model explainability for classical ML |
| Guardrails | Content filtering, PII redaction, denied topics for GenAI |

---

## Final Exam Day Checklist

- [ ] Government-issued photo ID ready
- [ ] Test environment prepared (quiet room, clear desk for online proctoring)
- [ ] Check-in 15 minutes early
- [ ] Plan: ~83 seconds per question average
- [ ] Flag uncertain questions and revisit at the end
- [ ] Never leave a question blank (no penalty for guessing)
- [ ] Read every word in question stems — qualifiers matter
- [ ] When stuck between two options, pick the more *managed* / *serverless* / *less operational overhead* option
- [ ] Trust your preparation — if you're scoring 80%+ on mocks, you're ready

---

*Good luck! 🎯*
