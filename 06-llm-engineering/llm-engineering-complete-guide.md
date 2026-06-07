# ð§  LLM Engineering â From Zero to Architect

> A complete reference guide covering every concept in Large Language Model engineering.  
> Written for someone starting fresh, building up to production architect level.

---

## Table of Contents

| # | Section | Level |
|---|---------|-------|
| 1 | [What Are Language Models?](#1-what-are-language-models) | ð¢ Beginner |
| 2 | [The Transformer â The Engine Behind Everything](#2-the-transformer--the-engine-behind-everything) | ð¢ Beginner |
| 3 | [Tokenization Deep Dive](#3-tokenization-deep-dive) | ð¢ Beginner |
| 4 | [Model Architectures: Encoder, Decoder, Encoder-Decoder](#4-model-architectures-encoder-decoder-encoder-decoder) | ð¡ Intermediate |
| 5 | [LLM vs SLM vs Foundation Models](#5-llm-vs-slm-vs-foundation-models) | ð¡ Intermediate |
| 6 | [Mixture of Experts (MoE)](#6-mixture-of-experts-moe) | ð¡ Intermediate |
| 7 | [Model Families â The Complete Landscape](#7-model-families--the-complete-landscape) | ð¡ Intermediate |
| 8 | [How to Choose a Model](#8-how-to-choose-a-model) | ð¡ Intermediate |
| 9 | [Training Pipeline: Pre-training â Fine-tuning â Alignment](#9-training-pipeline-pre-training--fine-tuning--alignment) | ð  Advanced |
| 10 | [Parameter-Efficient Fine-Tuning (PEFT): LoRA, QLoRA & Friends](#10-parameter-efficient-fine-tuning-peft-lora-qlora--friends) | ð  Advanced |
| 11 | [Quantization & Inference Optimization](#11-quantization--inference-optimization) | ð  Advanced |
| 12 | [Embeddings & Vector Search](#12-embeddings--vector-search) | ð  Advanced |
| 13 | [RAG (Retrieval-Augmented Generation)](#13-rag-retrieval-augmented-generation) | ð  Advanced |
| 14 | [Prompt Engineering & In-Context Learning](#14-prompt-engineering--in-context-learning) | ð¡ Intermediate |
| 15 | [Agents, Tool Use & Function Calling](#15-agents-tool-use--function-calling) | ð  Advanced |
| 16 | [Evaluation & Benchmarks](#16-evaluation--benchmarks) | ð´ Pro |
| 17 | [Deployment & Serving at Scale](#17-deployment--serving-at-scale) | ð´ Pro |
| 18 | [MLOps for LLMs (LLMOps)](#18-mlops-for-llms-llmops) | ð´ Pro |
| 19 | [Safety, Alignment & Guardrails](#19-safety-alignment--guardrails) | ð´ Pro |
| 20 | [The Complete LLM Engineering Stack](#20-the-complete-llm-engineering-stack) | ð´ Pro |

---

## 1. What Are Language Models?

### The Simplest Definition

A **language model** is a program that predicts the next word (token) given previous words.

```
Input:  "The cat sat on the ___"
Output: "mat" (with 73% probability)
```

That's it. Every ChatGPT response, every code completion, every translation â it's all just **next-token prediction** done incredibly well.

### The Scale Spectrum

```
ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
â                   Language Model Scale                             â
ââââââââââââ¬âââââââââââââââ¬âââââââââââââââââ¬ââââââââââââââââââââââââ¤
â  Tiny    â  Small (SLM)  â  Large (LLM)   â  Frontier             â
â  <100M   â  100M - 7B    â  7B - 70B      â  70B+ / MoE           â
â          â               â                â                       â
â DistilBERTâ Phi-3 (3.8B) â LLaMA-3 (70B)  â GPT-4 (~1.8T MoE)    â
â TinyBERT â Gemma-2 (2B)  â Mistral (7B)   â Claude (~??)          â
â          â Qwen-2 (1.5B) â DeepSeek (67B) â Gemini Ultra          â
ââââââââââââ´âââââââââââââââ´âââââââââââââââââ´ââââââââââââââââââââââââ
```

### LLM vs SLM

| | LLM (Large Language Model) | SLM (Small Language Model) |
|---|---|---|
| **Parameters** | 7B+ (often 70B-1T+) | 100M - 7B |
| **Runs on** | Cloud GPUs, clusters | Single GPU, edge devices, phones |
| **Strengths** | Broad knowledge, complex reasoning | Fast, cheap, domain-specific |
| **Examples** | GPT-4, Claude, LLaMA-70B | Phi-3, Gemma-2B, Mistral-7B |
| **Cost** | $$$$ | $ |
| **Use when** | General-purpose, complex tasks | Specific task, latency-critical, budget-limited |

### Key Insight

> "LLMs are not intelligent. They are statistical pattern matchers trained on internet-scale text that have emergent behaviors resembling intelligence."

---

## 2. The Transformer â The Engine Behind Everything

### Why This Matters

**Every modern language model is a Transformer.** GPT, BERT, LLaMA, Claude, Gemini, Mistral â ALL transformers. Before 2017, we used RNNs/LSTMs which were slow and forgot long-range context. The Transformer fixed both problems.

### The Paper That Changed Everything

**"Attention Is All You Need"** (Vaswani et al., 2017, Google)

### What IS a Transformer?

A Transformer is a neural network architecture with these key innovations:

#### 1. Self-Attention Mechanism ("Who should I pay attention to?")

```
Sentence: "The animal didn't cross the street because it was too tired"

Question: What does "it" refer to?

Self-attention lets the model look at EVERY other word simultaneously
and figure out: "it" â "animal" (not "street")

   The   animal   didn't   cross   the   street   because   it   was   too   tired
                                                             â
    0.02   0.85    0.01     0.01   0.01   0.08     0.01    1.0   0.01  0.00  0.00
           ^^^^                            ^^^^
           HIGH attention                  some attention
```

This is the **magic** â the model can relate any word to any other word regardless of distance.

#### 2. Positional Encoding ("Where am I in the sequence?")

Since attention looks at everything simultaneously (no sequential processing), we need to tell the model word ORDER matters. Positional encodings add "position information" to each token.

```
"Dog bites man" â  "Man bites dog"

Token embeddings:  [0.2, 0.5, ...] + Position info: [sin/cos patterns]
```

#### 3. Multi-Head Attention ("Look at different aspects simultaneously")

One attention head might learn syntax, another learns semantics, another learns coreference. Multiple heads = multiple perspectives on the same text.

```
Head 1: "it" â focuses on grammatical subject
Head 2: "it" â focuses on semantic meaning
Head 3: "it" â focuses on proximity
...
Combined: "it" = "animal" (high confidence)
```

#### 4. Feed-Forward Network ("Process what attention found")

After attention gathers relevant context, a feed-forward network (simple neural net) processes it further â this is where "knowledge" is stored.

#### 5. Layer Stacking

Stack these blocks (attention + FFN) many times:
- BERT-base: 12 layers
- GPT-3: 96 layers
- LLaMA-70B: 80 layers

More layers = more capacity to learn complex patterns.

### The Full Transformer Architecture

```
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
â                    TRANSFORMER                                â
â                                                              â
â  ââââââââââââââââââââââââ    âââââââââââââââââââââââââââââ  â
â  â      ENCODER          â    â        DECODER             â  â
â  â                        â    â                           â  â
â  â  Input: "Translate     â    â  Output: "Traduire        â  â
â  â         this to        â    â          ceci en           â  â
â  â         French"        â    â          franÃ§ais"         â  â
â  â                        â    â                           â  â
â  â  ââââââââââââââââââââ â    â  âââââââââââââââââââââââ  â  â
â  â  â Self-Attention    â â    â  â Masked Self-Attn    â  â  â
â  â  â (see ALL tokens)  â â    â  â (see only PAST)     â  â  â
â  â  ââââââââââ¬ââââââââââ â    â  ââââââââââââ¬âââââââââââ  â  â
â  â           â            â    â             â              â  â
â  â  ââââââââââââââââââââ â    â  âââââââââââââââââââââââ  â  â
â  â  â Feed-Forward     â â    â  â Cross-Attention     â  â  â
â  â  â Network          â â    â  â (look at encoder)   â  â  â
â  â  ââââââââââ¬ââââââââââ â    â  ââââââââââââ¬âââââââââââ  â  â
â  â           â            â    â             â              â  â
â  â  [Repeat Ã N layers]  â    â  âââââââââââââââââââââââ  â  â
â  â                        â    â  â Feed-Forward        â  â  â
â  ââââââââââââ¬ââââââââââââââ    â  ââââââââââââ¬âââââââââââ  â  â
â             â                   â             â              â  â
â             ââââââââââââââââââââââ [Repeat Ã N layers]      â  â
â                                 â                           â  â
â                                 âââââââââââââââââââââââââââââ  â
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
```

### Is Every Model a Transformer?

**In 2024-2026, yes â virtually all production LLMs are transformers.** Some alternatives are emerging:
- **Mamba / State Space Models (SSM)**: Linear scaling (vs quadratic for transformers), good for very long contexts
- **RWKV**: RNN-like but parallelizable like transformers
- **Hybrids**: Jamba (Mamba + Transformer layers)

But for now: **Transformer = the default architecture for LLMs.**

---

## 3. Tokenization Deep Dive

### Why Can't Models Just Read Characters?

```
"Hello" = 5 characters
But a 1000-word document = ~5000 characters = too many individual items to process

Solution: Group characters into meaningful chunks (tokens)
"Hello" = 1 token
"unhappiness" = ["un", "happiness"] = 2 tokens
```

### Tokenization Algorithms

| Algorithm | Used By | How It Works |
|-----------|---------|--------------|
| **BPE** (Byte-Pair Encoding) | GPT, LLaMA, Mistral | Merge most frequent character pairs iteratively |
| **WordPiece** | BERT, DistilBERT | Similar to BPE but uses likelihood-based merging |
| **SentencePiece** | LLaMA, T5, mT5 | Language-agnostic, treats input as raw bytes |
| **Unigram** | XLNet, ALBERT | Starts with large vocab, prunes least useful tokens |

### BPE Step-by-Step Example

```
Corpus: "low lower lowest"

Step 0 (characters): l o w _ l o w e r _ l o w e s t
Step 1 (merge 'l'+'o'): lo w _ lo w e r _ lo w e s t  
Step 2 (merge 'lo'+'w'): low _ low e r _ low e s t
Step 3 (merge 'e'+'r'): low _ low er _ low e s t
Step 4 (merge 'e'+'s'): low _ low er _ low es t
Step 5 (merge 'es'+'t'): low _ low er _ low est

Final vocabulary: [low, er, est, _]
"lowest" â ["low", "est"]
"lower"  â ["low", "er"]
```

### Tiktoken (OpenAI's Implementation)

```python
import tiktoken

# Get encoder for GPT-4
enc = tiktoken.encoding_for_model("gpt-4")

# Encode
tokens = enc.encode("Hello, how are you?")
# â [9906, 11, 1268, 527, 499, 30]

# Decode back
text = enc.decode(tokens)
# â "Hello, how are you?"

# Count tokens (for cost estimation)
len(enc.encode("Your long prompt here..."))
```

### Why Tokenization Matters for Engineers

1. **Cost**: APIs charge per token (~$0.01-$0.06 per 1K tokens)
2. **Context window**: Models have max token limits (128K for GPT-4)
3. **Non-English tax**: Many tokenizers were trained on English â other languages use 2-3x more tokens for the same content
4. **Code**: Code often tokenizes poorly (lots of small symbols)

---

## 4. Model Architectures: Encoder, Decoder, Encoder-Decoder

### The Three Types

```
ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
â                                                                     â
â   ENCODER-ONLY          DECODER-ONLY         ENCODER-DECODER        â
â   (Understand)          (Generate)           (Transform)            â
â                                                                     â
â   "Read & Classify"     "Write & Create"     "Read then Write"      â
â                                                                     â
â   âââââââââââââ        âââââââââââââ        âââââââ âââââââ       â
â   â  ENCODER  â        â  DECODER  â        â ENC âââ DEC â       â
â   â           â        â           â        â     â â     â       â
â   â See ALL   â        â See only  â        â ALL â âPAST â       â
â   â tokens    â        â PAST      â        â     â â     â       â
â   â at once   â        â tokens    â        â     â â     â       â
â   âââââââââââââ        âââââââââââââ        âââââââ âââââââ       â
â                                                                     â
â   BERT                  GPT-1/2/3/4          T5                     â
â   RoBERTa               LLaMA                BART                   â
â   ALBERT                Mistral              mT5                    â
â   DeBERTa               Claude               Flan-T5               â
â   DistilBERT            Gemini               UL2                    â
â                         DeepSeek                                    â
â                         Phi                                         â
â                         Qwen                                        â
â                                                                     â
â   USE FOR:              USE FOR:             USE FOR:               â
â   - Classification      - Text generation    - Translation          â
â   - NER                 - Chatbots           - Summarization        â
â   - Embeddings          - Code generation    - Question answering   â
â   - Sentiment           - Reasoning          - (Mostly legacy now)  â
â   - Search/ranking      - Everything else                           â
â                                                                     â
ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
```

### Deep Dive: Why Does Direction Matter?

**Encoder (Bidirectional) â BERT:**
```
"The [MASK] sat on the mat"

Encoder sees: The â â [MASK] â â sat â â on â â the â â mat
It can look BOTH left AND right to predict [MASK] = "cat"
```

**Decoder (Autoregressive) â GPT:**
```
"The cat sat on the ___"

Decoder sees: The â cat â sat â on â the â ???
It can ONLY look left (past) to predict next = "mat"

This is called "causal masking" or "autoregressive" generation
```

### Why Decoder-Only Won

In 2024-2026, **decoder-only models dominate** because:
1. Simple to scale (just make bigger)
2. One architecture does everything (generate, classify, translate, reason)
3. In-context learning works (few-shot prompting)
4. Instruction tuning makes them versatile

BERT-style encoders are still used for:
- Embedding models (turning text â vectors for search)
- Classification at scale (cheap, fast)
- Reranking in RAG pipelines

---

## 5. LLM vs SLM vs Foundation Models

### Definitions

| Term | Meaning | Examples |
|------|---------|----------|
| **Foundation Model** | Any large model pre-trained on broad data that can be adapted to many tasks | GPT-4, BERT, DALL-E, Whisper |
| **LLM** | Foundation model focused on language, typically 7B+ parameters | GPT-4, Claude, LLaMA-70B |
| **SLM** | Smaller language model (100M-7B), optimized for efficiency | Phi-3, Gemma-2B, TinyLlama |
| **Frontier Model** | State-of-the-art, best performance regardless of size | GPT-4o, Claude Opus, Gemini Ultra |

### The SLM Revolution (2024-2026)

Small models got surprisingly good:

```
Task: Summarize this email

âââââââââââââââââââââââââââââââââââââââââââââââ
â Model         â Quality â Speed   â Cost     â
âââââââââââââââââ¼ââââââââââ¼ââââââââââ¼âââââââââââ¤
â GPT-4         â 95/100  â 2.1s    â $0.03    â
â LLaMA-3 70B   â 92/100  â 1.8s    â $0.01    â
â Phi-3 3.8B    â 88/100  â 0.3s    â $0.001   â
â Gemma-2 2B    â 85/100  â 0.2s    â $0.0005  â
âââââââââââââââââââââââââââââââââââââââââââââââ

For many tasks, SLMs are 90%+ as good at 1/100th the cost!
```

### When to Use What

```
                        ââââââââââââââââââââ
                        â What's your task? â
                        ââââââââââ¬ââââââââââ
                                 â
                    ââââââââââââââ´âââââââââââââ
                    â                         â
            Complex reasoning?          Simple/specific task?
            Multi-step logic?           Classification?
            Creative writing?           Extraction?
                    â                         â
                    â¼                         â¼
              Use LLM (70B+)            Use SLM (1-7B)
              or Frontier API           Fine-tuned on your data
                    â                         â
              âââââââ´ââââââ            ââââââââ´âââââââ
              â           â            â             â
          Budget?    No budget?    On-device?    Cloud OK?
              â           â            â             â
              â¼           â¼            â¼             â¼
         LLaMA-70B    GPT-4/Claude  Phi-3/Gemma   Mistral-7B
         (self-host)  (API)        (quantized)    (API/host)
```

---

## 6. Mixture of Experts (MoE)

### The Problem MoE Solves

A 1 trillion parameter model would be incredibly capable BUT:
- Costs $$$ to run (every token activates ALL parameters)
- Slow inference
- Needs massive GPU clusters

### The MoE Solution

**Only activate a SUBSET of parameters for each token.**

```
ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
â                    MoE Architecture                          â
â                                                             â
â  Input Token: "programming"                                 â
â         â                                                   â
â         â¼                                                   â
â  âââââââââââââââ                                           â
â  â   ROUTER    â  â Learned gating network                  â
â  â  (decides   â    "Which experts should handle this?"     â
â  â   which     â                                           â
â  â   experts)  â                                           â
â  ââââ¬âââ¬âââ¬âââ¬ââ                                           â
â     â  â  â  â                                              â
â     â¼  â¼  â¼  â¼                                              â
â  ââââââââââââââââââââââââââââââââ                         â
â  âE1ââE2ââE3ââE4ââE5ââE6ââE7ââE8â  â 8 Expert FFNs         â
â  ââââââââââââââââââââââââââââââââ                         â
â   â       â                                                â
â   â       â     (Only 2 of 8 experts activated!)            â
â   â¼       â¼                                                â
â  âââââââââââââââ                                           â
â  â   COMBINE   â  â Weighted sum of expert outputs          â
â  âââââââââââââââ                                           â
â         â                                                   â
â         â¼                                                   â
â    Output                                                   â
ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
```

### Key Numbers

| Model | Total Params | Active Params per Token | Experts | Active Experts |
|-------|-------------|------------------------|---------|----------------|
| GPT-4 (rumored) | ~1.8T | ~220B | 16 | 2 |
| Mixtral 8x7B | 46.7B | 12.9B | 8 | 2 |
| DeepSeek-V2 | 236B | 21B | 160 | 6 |
| Grok-1 | 314B | ~86B | 8 | 2 |
| DBRX | 132B | 36B | 16 | 4 |

### Why MoE is Brilliant

```
Dense Model (LLaMA-70B):    70B params Ã every token = SLOW + EXPENSIVE
MoE Model (Mixtral 8Ã7B):  47B total but only 13B active = FAST + CHEAP

Result: MoE gets 70B-quality at 13B-cost!
```

### MoE vs Dense â When to Choose

| | Dense (LLaMA, Mistral) | MoE (Mixtral, DeepSeek) |
|---|---|---|
| **Inference speed** | Slower per-param | Faster (fewer active params) |
| **Memory** | Lower total | Higher total (all experts in RAM) |
| **Fine-tuning** | Easier, well-understood | Harder (expert balancing) |
| **Quality** | Good | Often better at same compute |
| **Best for** | Constrained memory, fine-tuning | Inference at scale |

---

## 7. Model Families â The Complete Landscape

### The Family Tree (2024-2026)

```
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
â                    LLM Family Tree                                    â
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ¤
â                                                                      â
â  PROPRIETARY (Closed-Source, API Only)                               â
â  âââ OpenAI: GPT-4o, GPT-4 Turbo, o1, o3                           â
â  âââ Anthropic: Claude 3.5 Sonnet, Claude Opus                      â
â  âââ Google: Gemini 1.5 Pro, Gemini Ultra                           â
â  âââ Amazon: Nova Pro, Nova Lite, Nova Micro                        â
â                                                                      â
â  OPEN-WEIGHT (Download & Run)                                        â
â  âââ Meta: LLaMA-3.1 (8B/70B/405B)                                 â
â  âââ Mistral: Mistral-7B, Mixtral-8x7B, Mistral Large              â
â  âââ Google: Gemma-2 (2B/9B/27B)                                    â
â  âââ Microsoft: Phi-3 (mini/small/medium)                           â
â  âââ Alibaba: Qwen-2.5 (0.5B to 72B)                               â
â  âââ DeepSeek: DeepSeek-V2, DeepSeek-Coder                         â
â  âââ Cohere: Command R+                                              â
â  âââ 01.AI: Yi (6B/34B)                                             â
â                                                                      â
â  SPECIALIZED                                                         â
â  âââ Code: StarCoder2, CodeLlama, DeepSeek-Coder, Codestral        â
â  âââ Embedding: E5, BGE, GTE, Cohere Embed, Voyage                  â
â  âââ Vision+Language: LLaVA, GPT-4V, Gemini Vision                  â
â  âââ Math/Reasoning: Llemma, DeepSeek-Math, Minerva                  â
â  âââ Medical/Legal/Finance: BioGPT, LegalBERT, FinGPT               â
â                                                                      â
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
```

### Detailed Comparison

| Family | Maker | Sizes | License | Strengths | Weaknesses |
|--------|-------|-------|---------|-----------|------------|
| **GPT-4** | OpenAI | ~1.8T MoE | Proprietary | Best reasoning, tool-use | Expensive, no self-hosting |
| **Claude** | Anthropic | Unknown | Proprietary | Long context (200K), safety, coding | API only |
| **Gemini** | Google | Unknown | Proprietary | Multimodal, long context (1M) | Inconsistent |
| **LLaMA-3** | Meta | 8B/70B/405B | Meta License | Open, huge community, very capable | 405B needs massive infra |
| **Mistral** | Mistral AI | 7B/8x7B/Large | Apache 2.0 / Proprietary | Fast, efficient, great MoE | Smaller ecosystem |
| **Phi-3** | Microsoft | 3.8B/7B/14B | MIT | Incredible for size, runs on phone | Limited for complex reasoning |
| **Qwen-2.5** | Alibaba | 0.5B-72B | Apache 2.0 | Full size range, good multilingual | Less English community |
| **DeepSeek** | DeepSeek | 7B/67B/V2-236B | MIT | Great at code, efficient MoE | Chinese-focused training |
| **Gemma-2** | Google | 2B/9B/27B | Gemma License | High quality for size | Restrictive license |

---

## 8. How to Choose a Model

### Decision Framework

```
START HERE
    â
    âââ Do you need the BEST quality regardless of cost?
    â   âââ YES â GPT-4o / Claude Opus / Gemini Ultra (API)
    â
    âââ Do you need to SELF-HOST (data privacy, air-gapped, compliance)?
    â   âââ Have lots of GPUs? â LLaMA-3 70B/405B or Mixtral
    â   âââ Single GPU? â Mistral-7B or LLaMA-3 8B (quantized)
    â   âââ Edge/mobile? â Phi-3 Mini or Gemma-2 2B
    â
    âââ Do you need FINE-TUNING on your data?
    â   âââ Lots of data + compute â Full fine-tune LLaMA-3
    â   âââ Limited compute â QLoRA on Mistral-7B or LLaMA-3 8B
    â   âââ Just need API fine-tune â OpenAI fine-tuning API
    â
    âââ What's your PRIMARY task?
    â   âââ Code generation â DeepSeek-Coder, CodeLlama, GPT-4
    â   âââ RAG/Search â Need embedding model (E5, BGE) + generation model
    â   âââ Classification â Fine-tuned BERT/DeBERTa (fast + cheap)
    â   âââ Translation â mT5, NLLB, or GPT-4
    â   âââ Vision + Text â GPT-4V, Gemini Vision, LLaVA
    â
    âââ What's your BUDGET?
        âââ $0 â Self-host quantized open model
        âââ $100/mo â OpenAI/Anthropic API with caching
        âââ $10K+/mo â Dedicated endpoints, multiple models
```

### The "Right Model" Matrix

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Startup MVP, general chatbot | GPT-4o-mini API | Cheap, fast, good enough |
| Enterprise with data privacy | LLaMA-3 70B on AWS/Azure | Self-hosted, no data leaves |
| Mobile app, on-device | Phi-3 Mini (4-bit) | Fits in 2GB RAM |
| Domain expert (medical/legal) | Fine-tuned Mistral-7B | Small enough to QLoRA, good base |
| Code assistant | DeepSeek-Coder-33B | Open, specialized, strong |
| RAG pipeline | Embed: E5-large + Gen: LLaMA/GPT | Separate models for each job |
| Agent/tool-use | GPT-4o or Claude | Best function-calling ability |

---

## 9. Training Pipeline: Pre-training â Fine-tuning â Alignment

### The Three Stages

```
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
â                                                                      â
â  STAGE 1: PRE-TRAINING                                              â
â  ââââââââââââââââââââ                                                â
â  â¢ Train on TRILLIONS of tokens (internet, books, code)             â
â  â¢ Learn language, facts, reasoning patterns                        â
â  â¢ Cost: $1M - $100M+ in compute                                    â
â  â¢ Duration: weeks to months on thousands of GPUs                   â
â  â¢ Result: "Base model" â raw text completer                        â
â  â¢ Example: LLaMA-3-base just completes text, can't chat            â
â                                                                      â
â                              â                                       â
â                                                                      â
â  STAGE 2: SUPERVISED FINE-TUNING (SFT)                              â
â  âââââââââââââââââââââââââââââââââââââ                                â
â  â¢ Train on (instruction, response) pairs                           â
â  â¢ Learn to FOLLOW instructions                                     â
â  â¢ Data: 10K-1M human-written examples                              â
â  â¢ Cost: $1K - $100K                                                â
â  â¢ Result: "Instruct model" â can chat but may be harmful/unhelpful â
â  â¢ Example: LLaMA-3-instruct                                        â
â                                                                      â
â                              â                                       â
â                                                                      â
â  STAGE 3: ALIGNMENT (RLHF / DPO / RLAIF)                           â
â  âââââââââââââââââââââââââââââââââââââââââ                            â
â  â¢ Train model to be helpful, harmless, honest                      â
â  â¢ Human raters rank model outputs (preference data)                â
â  â¢ Reinforcement learning from human feedback (RLHF)                â
â  â¢ Or Direct Preference Optimization (DPO) â simpler               â
â  â¢ Cost: $10K - $1M                                                  â
â  â¢ Result: "Chat model" â helpful and safe                          â
â  â¢ Example: ChatGPT, Claude                                         â
â                                                                      â
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
```

### RLHF Explained Simply

```
Step 1: Generate multiple responses to same prompt
  Prompt: "How do I pick a lock?"
  Response A: "Here's a step-by-step guide..." (harmful)
  Response B: "I can't help with that, but here's info on locksmiths" (safe)

Step 2: Human rates: B > A

Step 3: Train a "reward model" to predict human preferences

Step 4: Use reward model to train the LLM via reinforcement learning
  â LLM learns to generate responses humans prefer
```

### DPO (Direct Preference Optimization) â The Modern Alternative

RLHF is complex (needs reward model + RL training). DPO simplifies:

```
RLHF:  Data â Reward Model â PPO Training â Aligned Model (3 stages)
DPO:   Data â Direct Training â Aligned Model (1 stage, same quality!)
```

---

## 10. Parameter-Efficient Fine-Tuning (PEFT): LoRA, QLoRA & Friends

### The Fine-Tuning Spectrum

```
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
â                                                                  â
â  FULL FINE-TUNE              PEFT Methods            PROMPTING   â
â  ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââº    â
â                                                                  â
â  â¢ Updates ALL params        â¢ Updates <1% params    â¢ Updates   â
â  â¢ Best quality              â¢ 90-99% quality          NOTHING   â
â  â¢ Needs 8+ A100 GPUs        â¢ Needs 1 GPU          â¢ Just      â
â  â¢ $$$$$                     â¢ $                       change    â
â                                                        the input â
â  Examples:                   Examples:                            â
â  - Pre-training              - LoRA                   Examples:  â
â  - Full SFT                  - QLoRA                  - Zero-shotâ
â                              - Adapters               - Few-shot â
â                              - Prefix Tuning          - CoT      â
â                              - IA3                               â
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
```

### LoRA â The Math (Simplified)

```
Original model has weight matrix W of shape [d Ã d]

Normal fine-tuning:  W_new = W + ÎW        (ÎW is also [d Ã d] = dÂ² params!)

LoRA insight:        ÎW â A Ã B            where A is [d Ã r] and B is [r Ã d]
                                            r = "rank" (typically 8, 16, 32, 64)

If d = 4096 and r = 16:
  Full ÎW:  4096 Ã 4096 = 16,777,216 parameters
  LoRA:     (4096 Ã 16) + (16 Ã 4096) = 131,072 parameters
  
  â 128Ã fewer trainable parameters!
```

### LoRA Hyperparameters

| Parameter | What it means | Typical values |
|-----------|--------------|----------------|
| `r` (rank) | Size of adapter matrices. Higher = more capacity | 8, 16, 32, 64 |
| `alpha` | Scaling factor. Controls how much adapter affects output | 16, 32 (often alpha = 2Ãr) |
| `target_modules` | Which layers to attach adapters to | q_proj, v_proj, k_proj, o_proj |
| `dropout` | Regularization | 0.05 - 0.1 |

### QLoRA â LoRA + Quantization

```python
# QLoRA Training (conceptual)
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Step 1: Load base model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # â Quantize to 4-bit
    bnb_4bit_quant_type="nf4",            # â NormalFloat4 (best for LLMs)
    bnb_4bit_compute_dtype=torch.bfloat16 # â Compute in bf16 for accuracy
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config
)

# Step 2: Attach LoRA adapters (these stay in 16-bit for training)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05
)
model = get_peft_model(model, lora_config)

# Step 3: Train only the adapters!
# Total trainable: ~0.1% of model parameters
```

### Other PEFT Methods

| Method | How it works | When to use |
|--------|-------------|-------------|
| **LoRA** | Low-rank matrices on attention layers | Default choice |
| **QLoRA** | LoRA + 4-bit quantization | Consumer GPUs |
| **DoRA** | LoRA + direction/magnitude decomposition | Slightly better than LoRA |
| **Adapters** | Small bottleneck layers between transformer blocks | Multi-task learning |
| **Prefix Tuning** | Learnable "virtual tokens" prepended to input | Lightweight, less flexible |
| **IA3** | Learned scaling vectors (even fewer params than LoRA) | Extreme efficiency |
| **Full Fine-Tune** | Update everything | Unlimited compute, best results |

---

## 11. Quantization & Inference Optimization

### What is Quantization?

Reducing the precision of model weights from 32/16-bit to 8/4-bit:

```
Full precision (FP32):   3.14159265358979  â 32 bits per number
Half precision (FP16):   3.14159           â 16 bits per number  (2Ã smaller)
8-bit (INT8):            3.14              â 8 bits per number   (4Ã smaller)
4-bit (NF4/GPTQ):       3.1               â 4 bits per number   (8Ã smaller)

7B model memory:
  FP32: 28 GB
  FP16: 14 GB
  INT8:  7 GB
  INT4:  3.5 GB  â Fits on a laptop GPU!
```

### Quantization Methods

| Method | Type | Quality Loss | Speed | Used By |
|--------|------|-------------|-------|---------|
| **GPTQ** | Post-training, weight-only | Low | Fast inference | TheBloke models |
| **AWQ** | Post-training, activation-aware | Very low | Fast inference | vLLM default |
| **GGUF** | CPU-optimized format | Low | Good on CPU | llama.cpp, Ollama |
| **BitsAndBytes** | On-the-fly 4/8-bit | Low | Training OK | QLoRA, HuggingFace |
| **SmoothQuant** | Weights + activations | Very low | Fast | Production serving |
| **FP8** | Native hardware support | Minimal | Very fast | H100/H200 GPUs |

### Inference Optimization Stack

```
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
â            Inference Optimization Layers                â
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ¤
â                                                        â
â  1. MODEL LEVEL                                        â
â     â¢ Quantization (4-bit, 8-bit)                      â
â     â¢ Pruning (remove unimportant weights)             â
â     â¢ Distillation (train smaller model from larger)   â
â                                                        â
â  2. ATTENTION LEVEL                                    â
â     â¢ FlashAttention-2 (memory-efficient attention)    â
â     â¢ Multi-Query Attention (MQA)                      â
â     â¢ Grouped-Query Attention (GQA)                    â
â     â¢ Sliding Window Attention                         â
â                                                        â
â  3. BATCHING LEVEL                                     â
â     â¢ Continuous batching (don't wait for all to finish)â
â     â¢ Dynamic batching                                 â
â     â¢ Paged Attention (vLLM's innovation)              â
â                                                        â
â  4. SYSTEM LEVEL                                       â
â     â¢ KV-Cache (don't recompute past tokens)           â
â     â¢ Speculative Decoding (draft + verify)            â
â     â¢ Tensor Parallelism (split across GPUs)           â
â     â¢ Pipeline Parallelism                             â
â                                                        â
â  5. SERVING LEVEL                                      â
â     â¢ vLLM, TGI, TensorRT-LLM                         â
â     â¢ Prefix caching                                   â
â     â¢ Prompt caching                                   â
â                                                        â
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
```

### KV-Cache Explained

```
Without KV-Cache (wasteful):
  Generate token 1: Process [prompt]                    â "The"
  Generate token 2: Process [prompt, "The"]             â "cat"   (re-processes prompt!)
  Generate token 3: Process [prompt, "The", "cat"]      â "sat"   (re-processes everything!)

With KV-Cache (efficient):
  Generate token 1: Process [prompt], CACHE attention   â "The"
  Generate token 2: Process ["The"], USE cache          â "cat"   (only process new token!)
  Generate token 3: Process ["cat"], USE cache          â "sat"   (only process new token!)

Result: O(nÂ²) â O(n) for generation. Massive speedup.
```

---

## 12. Embeddings & Vector Search

### What are Embeddings?

Converting text into a fixed-size vector (list of numbers) that captures meaning:

```
"King"    â [0.2, 0.8, 0.1, 0.9, ...]   (768 numbers)
"Queen"   â [0.2, 0.8, 0.9, 0.9, ...]   (similar to King!)
"Banana"  â [0.9, 0.1, 0.3, 0.2, ...]   (very different)

Cosine similarity:
  King â Queen  = 0.95 (very similar)
  King â Banana = 0.12 (very different)
```

### Embedding Models (Not the same as generation LLMs!)

| Model | Dimensions | Max Tokens | Maker |
|-------|-----------|------------|-------|
| text-embedding-3-large | 3072 | 8191 | OpenAI |
| E5-large-v2 | 1024 | 512 | Microsoft |
| BGE-large-en-v1.5 | 1024 | 512 | BAAI |
| GTE-large | 1024 | 512 | Alibaba |
| Cohere Embed v3 | 1024 | 512 | Cohere |
| Voyage-2 | 1024 | 4000 | Voyage AI |
| Amazon Titan Embed v2 | 1024 | 8192 | AWS |

### Vector Databases

Store millions of embeddings and find similar ones fast:

| Database | Type | Best For |
|----------|------|----------|
| **Pinecone** | Managed cloud | Easiest, no infra |
| **Weaviate** | Self-hosted/cloud | Hybrid search |
| **Qdrant** | Self-hosted/cloud | Performance |
| **ChromaDB** | Embedded (local) | Prototyping |
| **pgvector** | PostgreSQL extension | Already using Postgres |
| **OpenSearch** | AWS managed | AWS ecosystem |
| **FAISS** | Library (in-memory) | Research, small scale |

---

## 13. RAG (Retrieval-Augmented Generation)

### The Problem RAG Solves

LLMs have **knowledge cutoff** (don't know recent events) and **hallucinate** (make up facts). RAG fixes this by giving the model real documents to reference.

### How RAG Works

```
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
â                      RAG Pipeline                                 â
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ¤
â                                                                  â
â  INDEXING (Offline, one-time)                                    â
â  âââââââââââââââââââââââââââ                                     â
â                                                                  â
â  Documents â Chunk â Embed â Store in Vector DB                  â
â                                                                  â
â  ââââââââ    ââââââââââ    ââââââââââââ    âââââââââââââ       â
â  â PDFs â    â 500    â    â [0.2,    â    â Pinecone  â       â
â  â Docs â â  â token  â â  â  0.8,   â â  â Qdrant    â       â
â  â Web  â    â chunks â    â  ...]    â    â pgvector  â       â
â  ââââââââ    ââââââââââ    ââââââââââââ    âââââââââââââ       â
â                                                                  â
â  RETRIEVAL + GENERATION (Online, per query)                      â
â  âââââââââââââââââââââââââââââââââââââââââââ                      â
â                                                                  â
â  User Query                                                      â
â      â                                                           â
â      âââ Embed query â Search vector DB â Top-K chunks           â
â      â                                                           â
â      âââ Construct prompt:                                       â
â           "Given this context: [retrieved chunks]                 â
â            Answer this question: [user query]"                   â
â                â                                                  â
â                â¼                                                  â
â           LLM generates answer grounded in retrieved docs         â
â                                                                  â
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
```

### Advanced RAG Techniques

| Technique | What it does | When to use |
|-----------|-------------|-------------|
| **Hybrid Search** | Combine vector (semantic) + keyword (BM25) search | Always (catches both exact + fuzzy matches) |
| **Reranking** | Use cross-encoder to re-score retrieved chunks | When precision matters |
| **Query Expansion** | Rephrase query multiple ways, search each | Ambiguous queries |
| **HyDE** | Generate hypothetical answer, search for that | Low-info queries |
| **Chunking Strategies** | Sentence, paragraph, semantic, recursive | Depends on document structure |
| **Parent-Child Chunks** | Retrieve child chunk, return parent for context | Better context in answers |
| **Multi-Index** | Different indexes for different doc types | Heterogeneous collections |
| **Graph RAG** | Build knowledge graph + vector search | Entity-heavy domains |
| **Self-RAG** | Model decides when to retrieve | Reduce unnecessary retrieval |
| **CRAG** | Corrective RAG â validate retrieval quality | High-stakes applications |

---

## 14. Prompt Engineering & In-Context Learning

### Prompt Strategies (Ordered by Complexity)

```
ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
â  PROMPTING HIERARCHY                                        â
ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ¤
â                                                             â
â  Zero-Shot         "Classify this text: ..."               â
â      â             (just ask, no examples)                 â
â                                                             â
â  Few-Shot          "Example 1: ... â positive              â
â      â              Example 2: ... â negative              â
â                     Now classify: ..."                      â
â                                                             â
â  Chain-of-Thought  "Think step by step..."                 â
â      â             (forces reasoning)                      â
â                                                             â
â  ReAct             "Thought: I need to search for...       â
â      â              Action: search(query)                  â
â                     Observation: ...                        â
â                     Thought: Now I know..."                 â
â                                                             â
â  Tree of Thought   Explore multiple reasoning paths        â
â      â             Evaluate each, pick best                â
â                                                             â
â  Self-Consistency  Generate N responses,                   â
â                    take majority vote                       â
â                                                             â
ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
```

### System Prompt Engineering

```
STRUCTURE:
ââââââââââââââââââââââââââââââââââââââââ
â SYSTEM PROMPT                         â
â âââ Role/Persona                      â
â âââ Context/Knowledge                 â
â âââ Instructions (what to do)         â
â âââ Constraints (what NOT to do)      â
â âââ Output Format                     â
â âââ Examples                          â
ââââââââââââââââââââââââââââââââââââââââ¤
â USER MESSAGE                          â
â âââ The actual query                  â
ââââââââââââââââââââââââââââââââââââââââ
```

---

## 15. Agents, Tool Use & Function Calling

### What are LLM Agents?

An agent is an LLM that can **take actions** â not just generate text, but call functions, browse the web, write code, etc.

```
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
â                      AGENT LOOP                              â
â                                                              â
â  User: "What's the weather in Denver and should I bring     â
â         an umbrella to my 3pm meeting?"                     â
â                                                              â
â  LLM thinks: I need (1) weather data (2) calendar info      â
â                                                              â
â  âââââââââââââââââââââââââââââââââââââââââââââââââââ        â
â  â ITERATION 1                                      â        â
â  â Thought: Need weather for Denver                 â        â
â  â Action: get_weather(city="Denver")               â        â
â  â Observation: 72Â°F, 60% chance of rain at 2-4pm  â        â
â  âââââââââââââââââââââââââââââââââââââââââââââââââââ        â
â                                                              â
â  âââââââââââââââââââââââââââââââââââââââââââââââââââ        â
â  â ITERATION 2                                      â        â
â  â Thought: Need to check if 3pm meeting is outdoor â        â
â  â Action: get_calendar(time="3pm today")           â        â
â  â Observation: "Team standup" - Room 204           â        â
â  âââââââââââââââââââââââââââââââââââââââââââââââââââ        â
â                                                              â
â  âââââââââââââââââââââââââââââââââââââââââââââââââââ        â
â  â ITERATION 3                                      â        â
â  â Thought: Indoor meeting, but walking to building â        â
â  â Action: RESPOND                                  â        â
â  â "72Â°F but 60% rain chance at 3pm. Your meeting  â        â
â  â  is indoors (Room 204) but bring an umbrella    â        â
â  â  for the walk over."                            â        â
â  âââââââââââââââââââââââââââââââââââââââââââââââââââ        â
â                                                              â
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
```

### Agent Frameworks

| Framework | Maker | Best For |
|-----------|-------|----------|
| **LangChain/LangGraph** | LangChain | Complex chains, stateful agents |
| **CrewAI** | CrewAI | Multi-agent collaboration |
| **AutoGen** | Microsoft | Multi-agent conversations |
| **Amazon Bedrock Agents** | AWS | Enterprise, managed |
| **OpenAI Assistants** | OpenAI | Simple tool-use agents |
| **Semantic Kernel** | Microsoft | Enterprise .NET/Python |
| **Haystack** | deepset | RAG-focused pipelines |

### Multi-Agent Systems

```
ââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
â              MULTI-AGENT ARCHITECTURE                  â
â                                                       â
â  âââââââââââââââââ                                   â
â  â  ORCHESTRATOR â  â Routes tasks to specialists    â
â  âââââââââ¬ââââââââ                                   â
â          â                                            â
â    âââââââ¼ââââââ¬âââââââââââ                          â
â    â¼     â¼     â¼          â¼                          â
â  âââââ âââââ âââââ    âââââ                         â
â  â R â â C â â A â    â V â                         â
â  â e â â o â â n â    â a â                         â
â  â s â â d â â a â    â l â                         â
â  â e â â e â â l â    â i â                         â
â  â a â â r â â y â    â d â                         â
â  â r â â   â â s â    â a â                         â
â  â c â â   â â t â    â t â                         â
â  â h â â   â â   â    â o â                         â
â  âââââ âââââ âââââ    âââââ                         â
â                                                       â
â  Each agent has its own:                             â
â  - System prompt (role)                              â
â  - Tools (what it can do)                            â
â  - Memory (what it remembers)                        â
ââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
```

---

## 16. Evaluation & Benchmarks

### How to Measure LLM Quality

| Benchmark | What it Tests | Example Task |
|-----------|--------------|--------------|
| **MMLU** | Knowledge (57 subjects) | Multiple-choice questions |
| **HumanEval** | Code generation | Write Python functions from docstrings |
| **GSM8K** | Math reasoning | Grade-school math word problems |
| **HellaSwag** | Common sense | Complete the sentence sensibly |
| **TruthfulQA** | Truthfulness | Avoid common misconceptions |
| **MT-Bench** | Multi-turn chat | Judge conversation quality (1-10) |
| **LMSYS Chatbot Arena** | Human preference | Blind A/B comparison by humans |
| **BigBench** | Diverse capabilities | 204 tasks testing various abilities |

### LLM-as-Judge

Use a strong LLM (GPT-4) to evaluate weaker models:

```
Prompt to GPT-4:
"Rate this response on a scale of 1-10 for:
 - Accuracy
 - Helpfulness  
 - Coherence
 
Question: {question}
Response: {model_response}
Reference: {gold_answer}"
```

### Evaluation for RAG

| Metric | What it Measures |
|--------|-----------------|
| **Faithfulness** | Does the answer stick to retrieved context? (no hallucination) |
| **Relevance** | Are retrieved chunks actually relevant to the query? |
| **Answer Correctness** | Is the final answer factually correct? |
| **Context Precision** | Are relevant chunks ranked higher? |
| **Context Recall** | Were all needed chunks retrieved? |

Tools: RAGAS, DeepEval, Langsmith, Phoenix (Arize)

---

## 17. Deployment & Serving at Scale

### Serving Frameworks

| Framework | Best For | Key Feature |
|-----------|----------|-------------|
| **vLLM** | High-throughput serving | PagedAttention, continuous batching |
| **TGI** (Text Generation Inference) | HuggingFace models | Easy deploy, streaming |
| **TensorRT-LLM** | NVIDIA GPUs, max speed | Kernel fusion, FP8 |
| **Ollama** | Local/development | One-command run |
| **llama.cpp** | CPU inference | GGUF format, runs anywhere |
| **SageMaker** | AWS managed | Auto-scaling, multi-model |
| **Bedrock** | AWS managed (API) | No infra, pay-per-token |

### Deployment Architecture

```
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
â                   PRODUCTION LLM SERVING                          â
â                                                                   â
â  Client â API Gateway â Load Balancer                            â
â                              â                                    â
â                    âââââââââââ¼ââââââââââ                         â
â                    â¼         â¼         â¼                         â
â              âââââââââââ âââââââââââ âââââââââââ               â
â              â vLLM    â â vLLM    â â vLLM    â               â
â              â Instanceâ â Instanceâ â Instanceâ               â
â              â (A100)  â â (A100)  â â (A100)  â               â
â              âââââââââââ âââââââââââ âââââââââââ               â
â                                                                   â
â  Supporting Infrastructure:                                       â
â  âââ Prompt Cache (Redis/DynamoDB)                               â
â  âââ Rate Limiter                                                 â
â  âââ Request Queue (for traffic spikes)                          â
â  âââ Model Registry (track versions)                             â
â  âââ Monitoring (latency, throughput, errors)                    â
â  âââ A/B Testing (route % to new model)                          â
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
```

### Key Serving Metrics

| Metric | What | Target |
|--------|------|--------|
| **TTFT** (Time to First Token) | Latency before streaming starts | < 500ms |
| **TPS** (Tokens Per Second) | Generation speed | 30-100+ |
| **Throughput** | Requests/second across all users | Depends on scale |
| **P99 Latency** | 99th percentile response time | < 5s |
| **GPU Utilization** | Are GPUs being fully used? | > 80% |

---

## 18. MLOps for LLMs (LLMOps)

### The LLMOps Lifecycle

```
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
â                                                                  â
â  âââââââââââ    âââââââââââââ    ââââââââââââ    ââââââââââââ  â
â  â Data    â â  â Training  â â  â Evaluate â â  â Deploy   â  â
â  â Pipelineâ    â Pipeline  â    â Pipeline â    â Pipeline â  â
â  âââââââââââ    âââââââââââââ    ââââââââââââ    ââââââââââââ  â
â       â                                                â         â
â       â         ââââââââââââââââââââââââââ             â         â
â       âââââââââââ     MONITOR            âââââââââââââââ         â
â                 â  - Quality drift        â                      â
â                 â  - Cost tracking        â                      â
â                 â  - Latency              â                      â
â                 â  - User feedback        â                      â
â                 â  - Safety violations    â                      â
â                 ââââââââââââââââââââââââââ                       â
â                                                                  â
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
```

### LLMOps Tools

| Category | Tools |
|----------|-------|
| **Experiment Tracking** | MLflow, W&B, SageMaker Experiments |
| **Data Management** | DVC, LakeFS, Delta Lake |
| **Training** | SageMaker Training, Ray Train, Anyscale |
| **Evaluation** | RAGAS, DeepEval, Promptfoo |
| **Serving** | vLLM, TGI, SageMaker Endpoints, Bedrock |
| **Monitoring** | Langsmith, Langfuse, Phoenix, Datadog |
| **Guardrails** | Guardrails AI, NeMo Guardrails, Bedrock Guardrails |
| **Orchestration** | LangChain, LangGraph, Haystack |
| **Vector DB** | Pinecone, Qdrant, OpenSearch, pgvector |

---

## 19. Safety, Alignment & Guardrails

### The Safety Stack

```
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
â              LLM SAFETY LAYERS                         â
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ¤
â                                                        â
â  Layer 1: INPUT GUARDRAILS                             â
â  â¢ Content filters (block harmful prompts)            â
â  â¢ PII detection (mask sensitive data)                â
â  â¢ Jailbreak detection                                â
â  â¢ Topic restrictions                                  â
â                                                        â
â  Layer 2: MODEL-LEVEL ALIGNMENT                        â
â  â¢ RLHF / DPO training                                â
â  â¢ Constitutional AI (Anthropic's approach)           â
â  â¢ System prompt safety instructions                   â
â                                                        â
â  Layer 3: OUTPUT GUARDRAILS                            â
â  â¢ Hallucination detection (check against sources)    â
â  â¢ Toxicity scoring                                    â
â  â¢ Factual consistency check                           â
â  â¢ PII in output detection                             â
â                                                        â
â  Layer 4: MONITORING & FEEDBACK                        â
â  â¢ Human review of flagged responses                  â
â  â¢ User feedback collection                            â
â  â¢ Red-teaming and adversarial testing                â
â                                                        â
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
```

### Key Concepts

| Term | Meaning |
|------|---------|
| **Hallucination** | Model generates confident but false information |
| **Jailbreak** | User tricks model into ignoring safety training |
| **Prompt Injection** | Malicious instructions hidden in user input or retrieved documents |
| **Red-teaming** | Adversarial testing to find vulnerabilities |
| **Constitutional AI** | Self-critique: model evaluates its own outputs against principles |
| **Guardrails** | Programmatic rules that filter/modify inputs and outputs |

---

## 20. The Complete LLM Engineering Stack

### From Theory to Production â The Full Picture

```
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
â                                                                      â
â                    THE LLM ENGINEERING STACK                          â
â                                                                      â
â  âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ    â
â  â APPLICATION LAYER                                            â    â
â  â Agents â¢ Chatbots â¢ Code Assistants â¢ RAG Apps â¢ Workflows  â    â
â  âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ    â
â                              â                                       â
â  âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ    â
â  â ORCHESTRATION LAYER                                          â    â
â  â LangChain â¢ LangGraph â¢ Haystack â¢ Semantic Kernel           â    â
â  â Prompt Management â¢ Memory â¢ Tool Routing â¢ Guardrails       â    â
â  âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ    â
â                              â                                       â
â  âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ    â
â  â MODEL LAYER                                                  â    â
â  â Foundation Models (GPT-4, Claude, LLaMA, Mistral)            â    â
â  â Embedding Models (E5, BGE, Cohere Embed)                     â    â
â  â Specialized Models (Code, Vision, Speech)                     â    â
â  âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ    â
â                              â                                       â
â  âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ    â
â  â OPTIMIZATION LAYER                                           â    â
â  â Fine-tuning (LoRA/QLoRA) â¢ Quantization (GPTQ/AWQ)          â    â
â  â Distillation â¢ Pruning â¢ FlashAttention                      â    â
â  âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ    â
â                              â                                       â
â  âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ    â
â  â SERVING LAYER                                                â    â
â  â vLLM â¢ TGI â¢ TensorRT-LLM â¢ SageMaker â¢ Bedrock            â    â
â  â KV-Cache â¢ Continuous Batching â¢ Speculative Decoding        â    â
â  âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ    â
â                              â                                       â
â  âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ    â
â  â DATA LAYER                                                   â    â
â  â Vector DBs (Pinecone, Qdrant) â¢ Document Stores              â    â
â  â Training Data â¢ Evaluation Datasets â¢ Feedback Logs          â    â
â  âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ    â
â                              â                                       â
â  âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ    â
â  â INFRASTRUCTURE LAYER                                         â    â
â  â GPUs (A100/H100) â¢ Kubernetes â¢ Auto-scaling                 â    â
â  â MLflow â¢ Monitoring â¢ CI/CD â¢ Cost Management                â    â
â  âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ    â
â                                                                      â
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
```

---

## Quick Reference: Glossary

| Term | Definition |
|------|-----------|
| **Attention** | Mechanism allowing model to focus on relevant parts of input |
| **Autoregressive** | Generating one token at a time, left to right |
| **Base model** | Pre-trained model before instruction tuning |
| **BPE** | Byte-Pair Encoding â tokenization algorithm |
| **Causal LM** | Language model that only sees past tokens (decoder) |
| **Context window** | Maximum tokens a model can process at once |
| **Cross-attention** | Decoder attending to encoder output |
| **DPO** | Direct Preference Optimization â simpler RLHF alternative |
| **Embedding** | Dense vector representation of text |
| **Few-shot** | Giving examples in the prompt |
| **Fine-tuning** | Continuing training on specific data |
| **FlashAttention** | Memory-efficient attention implementation |
| **FP16/BF16** | 16-bit floating point formats |
| **GQA** | Grouped-Query Attention (fewer KV heads) |
| **Hallucination** | Model generating false information confidently |
| **ICL** | In-Context Learning â learning from examples in the prompt |
| **Inference** | Using a trained model to generate predictions |
| **Instruct model** | Model fine-tuned to follow instructions |
| **KV-Cache** | Cached attention computations for faster generation |
| **LoRA** | Low-Rank Adaptation for efficient fine-tuning |
| **MHA** | Multi-Head Attention |
| **MoE** | Mixture of Experts â sparse activation |
| **NF4** | NormalFloat 4-bit â quantization format for QLoRA |
| **PEFT** | Parameter-Efficient Fine-Tuning (umbrella term) |
| **PPO** | Proximal Policy Optimization â RL algorithm used in RLHF |
| **Pre-training** | Initial training on massive text corpus |
| **QLoRA** | Quantized LoRA â 4-bit base + 16-bit adapters |
| **Quantization** | Reducing numerical precision of weights |
| **RAG** | Retrieval-Augmented Generation |
| **Rank (r)** | Dimension of LoRA adapter matrices |
| **RLHF** | Reinforcement Learning from Human Feedback |
| **RoPE** | Rotary Position Embedding |
| **Self-attention** | Tokens attending to other tokens in same sequence |
| **SFT** | Supervised Fine-Tuning |
| **Speculative decoding** | Draft tokens fast, verify in batch |
| **Temperature** | Controls randomness (0=deterministic, 1=creative) |
| **Token** | Basic unit of text processing (subword) |
| **Top-k** | Sample from top K most probable tokens |
| **Top-p (nucleus)** | Sample from smallest set with probability â¥ p |
| **Transformer** | The neural network architecture behind all modern LLMs |
| **Zero-shot** | No examples given, just instructions |

---

## Learning Roadmap

### Week 1-2: Foundations
- [ ] Understand tokenization (try tiktoken)
- [ ] Learn Transformer architecture (watch 3Blue1Brown)
- [ ] Run a local model with Ollama
- [ ] Use OpenAI/Anthropic APIs

### Week 3-4: Application Building
- [ ] Build a simple RAG pipeline (LangChain + ChromaDB)
- [ ] Implement prompt engineering techniques
- [ ] Build a function-calling agent
- [ ] Deploy with Streamlit/Gradio

### Week 5-6: Fine-Tuning
- [ ] Fine-tune a model with QLoRA (your llm-lora repo!)
- [ ] Create training datasets
- [ ] Evaluate fine-tuned model
- [ ] Merge adapters and deploy

### Week 7-8: Production
- [ ] Deploy with vLLM
- [ ] Implement guardrails
- [ ] Set up monitoring (Langfuse)
- [ ] Build CI/CD for models

### Week 9-10: Advanced
- [ ] Multi-agent systems
- [ ] Advanced RAG (Graph RAG, CRAG)
- [ ] Model distillation
- [ ] Cost optimization at scale

---

## Recommended Resources

| Resource | Type | Level |
|----------|------|-------|
| [Andrej Karpathy - Let's Build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Video | Beginner |
| [3Blue1Brown - Transformers](https://www.youtube.com/watch?v=wjZofJX0v4M) | Video | Beginner |
| [Attention Is All You Need (paper)](https://arxiv.org/abs/1706.03762) | Paper | Intermediate |
| [LoRA Paper](https://arxiv.org/abs/2106.09685) | Paper | Intermediate |
| [HuggingFace NLP Course](https://huggingface.co/learn/nlp-course) | Course | Beginner-Inter |
| [Full Stack LLM Bootcamp](https://fullstackdeeplearning.com/) | Course | Intermediate |
| [LLM University by Cohere](https://docs.cohere.com/docs/llmu) | Course | Beginner |
| [Chip Huyen - Building LLM Apps](https://huyenchip.com/2023/04/11/llm-engineering.html) | Blog | Advanced |
| [Your ai-architect-roadmap repo](https://github.com/sai337/ai-architect-roadmap) | Your Repo! | All |

---

*Last updated: June 2026*  
*Author: Generated for Saishiva's LLM Engineering journey*
