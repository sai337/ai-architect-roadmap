# Advanced RAG Architecture: Production Engineering Guide

> **Audience**: Cloud Architects with deep AWS/EKS expertise, existing Bedrock KB + MCP experience  
> **Focus**: Advanced patterns, production-grade implementations, and architectural decisions  
> **Last Updated**: June 2026

---

## Table of Contents

1. [RAG Evolution & Taxonomy](#1-rag-evolution--taxonomy)
2. [Chunking Strategies Deep Dive](#2-chunking-strategies-deep-dive)
3. [Embedding Models Comparison](#3-embedding-models-comparison)
4. [Vector Database Comparison](#4-vector-database-comparison)
5. [Retrieval Strategies](#5-retrieval-strategies)
6. [Reranking Architectures](#6-reranking-architectures)
7. [Knowledge Graph RAG](#7-knowledge-graph-rag)
8. [Evaluation Frameworks](#8-evaluation-frameworks)
9. [Production Patterns](#9-production-patterns)
10. [AWS Implementation](#10-aws-implementation)
11. [Architecture Diagram](#11-production-architecture-diagram)
12. [Code Snippets](#12-code-snippets)

---

## 1. RAG Evolution & Taxonomy

### The Four Generations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RAG ARCHITECTURE EVOLUTION                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Gen 1: Naive RAG          Gen 2: Advanced RAG                              │
│  ┌──────────────┐          ┌──────────────────────┐                         │
│  │ Query → Embed │          │ Query Transform      │                         │
│  │ → TopK Retrieve│         │ → Multi-stage Retrieve│                        │
│  │ → Stuff in Prompt│       │ → Rerank + Filter    │                         │
│  │ → Generate     │         │ → Compress Context   │                         │
│  └──────────────┘          │ → Generate + Cite    │                         │
│                             └──────────────────────┘                         │
│                                                                             │
│  Gen 3: Modular RAG        Gen 4: Agentic RAG                               │
│  ┌──────────────────┐      ┌────────────────────────┐                       │
│  │ Pluggable modules │      │ Autonomous Agent Loop   │                      │
│  │ • Router           │      │ • Tool-use retrieval    │                      │
│  │ • Adaptive retrieval│     │ • Self-reflection       │                      │
│  │ • Multi-source fusion│    │ • Iterative refinement  │                      │
│  │ • Feedback loops    │     │ • Multi-hop reasoning   │                      │
│  │ • Custom pipelines  │     │ • Adaptive strategies   │                      │
│  └──────────────────┘      └────────────────────────┘                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Differentiators

| Aspect | Naive RAG | Advanced RAG | Modular RAG | Agentic RAG |
|--------|-----------|--------------|-------------|-------------|
| **Retrieval** | Single-pass TopK | Hybrid + Rerank | Pipeline-based, configurable | Tool-based, autonomous |
| **Query** | Passthrough | HyDE, Multi-query | Router-selected strategy | LLM decides when/how to retrieve |
| **Context** | Stuff all chunks | Compressed, filtered | Module-fused from multi-source | Iteratively gathered |
| **Generation** | Single prompt | Structured w/ citations | Template-selected | Self-reflective w/ verification |
| **Failure Mode** | Silent hallucination | Reduced via reranking | Fallback modules | Self-correcting loops |
| **Latency** | ~1-2s | ~2-4s | ~3-6s (configurable) | ~5-15s (multi-turn) |

### When to Use What

- **Naive RAG**: Internal prototypes, low-stakes Q&A
- **Advanced RAG**: Production customer-facing, SLA-bound applications
- **Modular RAG**: Multi-domain systems needing different strategies per query type
- **Agentic RAG**: Complex research tasks, multi-hop reasoning, where latency budget > 10s

---

## 2. Chunking Strategies Deep Dive

### Strategy Comparison Matrix

| Strategy | Best For | Chunk Quality | Implementation Complexity | Retrieval Precision |
|----------|----------|---------------|--------------------------|---------------------|
| **Fixed-size** | Homogeneous docs, quick prototype | ⭐⭐ | ⭐ | ⭐⭐ |
| **Recursive Character** | General purpose, mixed content | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Semantic** | Research papers, long-form content | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Document-Aware** | Structured docs (PDF, HTML, Markdown) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Agentic/LLM-based** | Complex multi-format corpora | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### Detailed Breakdown

#### Fixed-Size Chunking
```python
# Pros: Deterministic, fast, predictable token count
# Cons: Breaks semantic units, mid-sentence splits, no context awareness
chunk_size = 512  # tokens
chunk_overlap = 64  # 12.5% overlap
```

#### Recursive Character Splitting (LangChain default)
```python
# Splits on: ["\n\n", "\n", " ", ""] — preserves paragraph boundaries
# Pros: Respects natural boundaries, configurable hierarchy
# Cons: Still content-unaware, overlap doesn't guarantee coherence
separators = ["\n\n", "\n", ". ", " ", ""]
```

#### Semantic Chunking (Embedding-based)
```python
# Computes embedding similarity between consecutive sentences
# Splits where cosine similarity drops below threshold
# Pros: Semantically coherent chunks, adaptive size
# Cons: Expensive (embedding every sentence), non-deterministic
from langchain_experimental.text_splitter import SemanticChunker

chunker = SemanticChunker(
    embeddings=bedrock_embeddings,
    breakpoint_threshold_type="percentile",  # or "standard_deviation", "interquartile"
    breakpoint_threshold_amount=75
)
```

#### Document-Aware Chunking (Structure-preserving)
```python
# Leverages document structure: headers, tables, code blocks, lists
# Maintains parent-child relationships for hierarchical retrieval
# Pros: Highest fidelity, preserves tables/code intact
# Cons: Format-specific parsers needed, complex pipeline

# Bedrock KB Custom Chunking (hierarchical)
chunking_config = {
    "chunkingStrategy": "HIERARCHICAL",
    "hierarchicalChunkingConfiguration": {
        "levelConfigurations": [
            {"maxTokens": 1500},  # Parent chunks
            {"maxTokens": 300}    # Child chunks
        ],
        "overlapTokens": 60
    }
}
```

#### Advanced: Contextual Chunking (Anthropic's approach)
```python
# Prepend document-level context to each chunk before embedding
# "This chunk is from [doc title], section [X], discussing [topic]"
# Dramatically improves retrieval precision (+49% per Anthropic's benchmarks)

CONTEXT_PROMPT = """Given the document and a chunk, provide a short context 
that situates this chunk within the overall document for retrieval purposes.
Document: {doc_title}
Chunk: {chunk_text}
Context:"""
```

### Production Recommendation

For DISH-scale workloads with mixed document types:

```
┌─────────────────────────────────────────────────┐
│         HYBRID CHUNKING PIPELINE                 │
├─────────────────────────────────────────────────┤
│ 1. Document Type Router                          │
│    ├── PDF → Document-Aware (unstructured.io)   │
│    ├── Markdown → Header-based recursive        │
│    ├── Code → AST-aware splitting               │
│    ├── HTML → DOM-structure splitting           │
│    └── Plain text → Semantic chunking           │
│                                                  │
│ 2. Post-processing                              │
│    ├── Contextual enrichment (prepend context)  │
│    ├── Metadata extraction (title, date, author)│
│    └── Parent-child linking for hierarchical    │
└─────────────────────────────────────────────────┘
```

---

## 3. Embedding Models Comparison

### Head-to-Head Comparison

| Model | Dimensions | Max Tokens | MTEB Score | Latency (p99) | Cost (1M tokens) | Best For |
|-------|-----------|------------|------------|---------------|-------------------|----------|
| **Titan Embed v2** | 256/512/1024 | 8,192 | ~63 | ~50ms | $0.02 | AWS-native, cost-optimized |
| **Cohere Embed v3** | 1024 | 512 | ~66 | ~80ms | $0.10 | Multilingual, compression |
| **OpenAI text-embedding-3-large** | 3072 (flexible) | 8,191 | ~65 | ~100ms | $0.13 | General purpose, MRL |
| **BGE-M3 (BAAI)** | 1024 | 8,192 | ~68 | ~30ms* | Self-hosted | Multi-granularity, hybrid |
| **E5-Mistral-7B** | 4096 | 32,768 | ~69 | ~150ms* | Self-hosted | Long context, instruction-tuned |
| **Jina Embeddings v3** | 1024 | 8,192 | ~67 | ~60ms | $0.02 | Task-specific LoRA adapters |
| **NV-Embed-v2** | 4096 | 32,768 | ~72 | ~200ms* | Self-hosted | SOTA quality, GPU-intensive |
| **Snowflake Arctic Embed L** | 1024 | 8,192 | ~67 | ~40ms* | Self-hosted | Efficient, permissive license |

*Self-hosted latency depends on infrastructure (EKS GPU nodes, instance type)

### Key Architectural Decisions

#### Matryoshka Representation Learning (MRL)
```python
# OpenAI & newer models support truncating embeddings without retraining
# Trade quality for storage/speed — critical for large-scale deployments
embedding_full = model.encode(text)       # 3072 dims → best quality
embedding_med = embedding_full[:1024]      # 1024 dims → ~98% quality
embedding_small = embedding_full[:256]     # 256 dims  → ~94% quality
# Use small for candidate retrieval, full for reranking
```

#### Multi-Vector Embeddings (ColBERT-style)
```python
# Instead of single vector per chunk, encode token-level embeddings
# Enables late interaction → dramatically better for complex queries
# BGE-M3 supports dense + sparse + multi-vector simultaneously
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
output = model.encode(sentences, return_dense=True, return_sparse=True, return_colbert_vecs=True)
```

#### Bedrock Titan v2 — Dimension Selection
```python
import boto3, json

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-west-2")

# Titan v2 supports 256, 512, 1024 dimensions
# Lower dims = faster search + less storage, minimal quality loss
response = bedrock_runtime.invoke_model(
    modelId="amazon.titan-embed-text-v2:0",
    body=json.dumps({
        "inputText": "Advanced RAG architectures for production systems",
        "dimensions": 512,        # 256 | 512 | 1024
        "normalize": True         # L2 normalize for cosine similarity
    })
)
embedding = json.loads(response["body"].read())["embedding"]
```

### Embedding Selection Decision Tree

```
Is your corpus multilingual?
├── Yes → Cohere Embed v3 or BGE-M3
└── No
    ├── Need AWS-native managed service?
    │   ├── Yes → Titan Embed v2 (1024d for quality, 512d for cost)
    │   └── No
    │       ├── Need long context (>8K tokens)?
    │       │   ├── Yes → E5-Mistral-7B or Jina v3 (self-host on EKS)
    │       │   └── No
    │       │       ├── Maximum quality (cost no object)?
    │       │       │   ├── Yes → NV-Embed-v2 (self-host)
    │       │       │   └── No → BGE-M3 (best balance)
    │       └── Need sparse+dense hybrid?
    │           └── Yes → BGE-M3 (native multi-vector)
```

---

## 4. Vector Database Comparison

### Feature Matrix

| Feature | OpenSearch Serverless | Pinecone | pgvector | Qdrant | ChromaDB |
|---------|---------------------|----------|----------|--------|----------|
| **Deployment** | AWS Managed | SaaS | Self-hosted/RDS | Self-hosted/Cloud | Embedded/Self-hosted |
| **Max Vectors** | Billions | Billions | ~10M practical | Billions | ~1M practical |
| **Hybrid Search** | ✅ BM25 + kNN | ✅ Sparse+Dense | ⚠️ tsvector + ivfflat | ✅ Sparse+Dense | ❌ Dense only |
| **Filtering** | ✅ Full Lucene | ✅ Metadata filters | ✅ SQL WHERE | ✅ Payload filters | ⚠️ Basic |
| **HNSW Tuning** | ef_construction, M | Managed | m, ef_construction | m, ef_construct | Managed |
| **Quantization** | ❌ | ✅ Product Quantization | ✅ halfvec, bit | ✅ Scalar, Product, Binary | ❌ |
| **Multi-tenancy** | ✅ Collection policies | ✅ Namespaces | ✅ Row-level security | ✅ Collection + payload | ⚠️ Collections |
| **Latency (p99)** | ~50-100ms | ~20-50ms | ~10-50ms (small scale) | ~10-30ms | ~5-20ms (local) |
| **Cost Model** | OCU-hours ($0.24/hr) | $0.096/1M reads | RDS instance cost | Per-node | Free (OSS) |
| **AWS Integration** | ✅ Native (Bedrock KB) | ⚠️ PrivateLink | ✅ RDS/Aurora | ⚠️ Self-managed | ❌ Dev only |

### OpenSearch Serverless — Production Configuration

```python
# For Bedrock KB integration — this is your primary path at DISH
# Key tuning: engine, space_type, ef_construction, m

index_body = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 512,  # Higher = better recall, more latency
            "number_of_shards": 2,
            "number_of_replicas": 1
        }
    },
    "mappings": {
        "properties": {
            "embedding": {
                "type": "knn_vector",
                "dimension": 1024,
                "method": {
                    "name": "hnsw",
                    "engine": "faiss",          # faiss > nmslib for filtered search
                    "space_type": "l2",         # l2 | cosinesimil | innerproduct
                    "parameters": {
                        "ef_construction": 512,  # Build-time quality (higher = better graph)
                        "m": 16                  # Connections per node (16 is sweet spot)
                    }
                }
            },
            "text": {"type": "text"},           # For BM25 hybrid
            "metadata": {"type": "object"},
            "source_doc": {"type": "keyword"},  # For filtering
            "timestamp": {"type": "date"}
        }
    }
}
```

### pgvector — When to Choose (Aurora PostgreSQL)

```sql
-- Best when: existing RDS/Aurora footprint, <10M vectors, need ACID + joins
-- HNSW index (preferred over IVFFlat for production)
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 256);

-- Halfvec for 50% storage reduction (PG16+)
ALTER TABLE documents ALTER COLUMN embedding TYPE halfvec(1024);

-- Hybrid search with RRF (Reciprocal Rank Fusion)
WITH semantic AS (
    SELECT id, 1.0 / (60 + rank) as score
    FROM (SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> $1) as rank
          FROM documents WHERE category = 'technical') sub
    LIMIT 20
), keyword AS (
    SELECT id, 1.0 / (60 + rank) as score  
    FROM (SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank(tsv, query) DESC) as rank
          FROM documents, plainto_tsquery($2) query
          WHERE tsv @@ query) sub
    LIMIT 20
)
SELECT COALESCE(s.id, k.id) as id,
       COALESCE(s.score, 0) + COALESCE(k.score, 0) as rrf_score
FROM semantic s FULL OUTER JOIN keyword k ON s.id = k.id
ORDER BY rrf_score DESC LIMIT 10;
```

### Qdrant — When to Choose (EKS self-hosted)

```python
# Best when: need maximum performance, advanced filtering, multi-vector
# Deploy on EKS with GPU nodes for quantization decode

from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://qdrant.internal:6333")

# Multi-vector collection (dense + sparse)
client.create_collection(
    collection_name="documents",
    vectors_config={
        "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(
            modifier=models.Modifier.IDF  # TF-IDF weighting
        )
    },
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=0.99,
            always_ram=True  # Keep quantized vectors in RAM, full on disk
        )
    )
)
```

---

## 5. Retrieval Strategies

### Strategy Taxonomy

```
┌─────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL STRATEGY SPACE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  QUERY TRANSFORMATION          RETRIEVAL METHOD                  │
│  ┌───────────────────┐        ┌────────────────────────┐        │
│  │ • HyDE             │        │ • Dense (ANN/kNN)      │        │
│  │ • Multi-Query      │        │ • Sparse (BM25/SPLADE) │        │
│  │ • Step-back        │        │ • Hybrid (RRF fusion)  │        │
│  │ • Sub-question     │        │ • Multi-vector (ColBERT)│       │
│  │ • Query routing    │        │ • Knowledge Graph      │        │
│  └───────────────────┘        └────────────────────────┘        │
│                                                                  │
│  POST-RETRIEVAL                ADAPTIVE                          │
│  ┌───────────────────┐        ┌────────────────────────┐        │
│  │ • Reranking        │        │ • Self-RAG             │        │
│  │ • Contextual comp. │        │ • CRAG (Corrective)    │        │
│  │ • Diversity filter │        │ • Adaptive retrieval   │        │
│  │ • Lost-in-middle   │        │ • Speculative RAG      │        │
│  │   reordering       │        │                        │        │
│  └───────────────────┘        └────────────────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Dense Retrieval (Baseline)
```python
# Standard ANN search — fast, good for semantically similar queries
# Weakness: vocabulary mismatch, exact term matching
results = vector_store.similarity_search(query_embedding, k=20)
```

### Sparse Retrieval (BM25/SPLADE)
```python
# BM25: TF-IDF based, excellent for exact matches, acronyms, codes
# SPLADE: Learned sparse representations — best of both worlds

# OpenSearch BM25 + kNN hybrid
query = {
    "size": 10,
    "query": {
        "hybrid": {
            "queries": [
                {
                    "match": {
                        "text": {
                            "query": user_query,
                            "boost": 0.3  # BM25 weight
                        }
                    }
                },
                {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": 20,
                            "boost": 0.7  # Dense weight
                        }
                    }
                }
            ]
        }
    }
}
```

### HyDE (Hypothetical Document Embeddings)
```python
# Generate a hypothetical answer, embed THAT, then retrieve
# Dramatically improves retrieval for questions where query ≠ document language

HYDE_PROMPT = """Given the question, write a detailed paragraph that would 
answer this question as if it appeared in a technical document.
Question: {query}
Hypothetical Document:"""

async def hyde_retrieve(query: str, llm, embedder, vector_store):
    # Generate hypothetical document
    hypo_doc = await llm.generate(HYDE_PROMPT.format(query=query))
    # Embed the hypothetical document (not the query!)
    hypo_embedding = embedder.encode(hypo_doc)
    # Retrieve using hypothetical embedding
    results = vector_store.similarity_search(hypo_embedding, k=20)
    return results
```

### Multi-Query Retrieval
```python
# Generate multiple query perspectives, retrieve for each, deduplicate
MULTI_QUERY_PROMPT = """Generate 3 different versions of the given question 
to retrieve relevant documents from a vector database. Provide alternative 
phrasings that capture different aspects or perspectives.
Original: {query}
Alternatives:"""

async def multi_query_retrieve(query: str, llm, retriever):
    alt_queries = await llm.generate(MULTI_QUERY_PROMPT.format(query=query))
    all_queries = [query] + parse_alternatives(alt_queries)
    
    # Parallel retrieval
    all_results = await asyncio.gather(*[
        retriever.retrieve(q, k=10) for q in all_queries
    ])
    
    # Reciprocal Rank Fusion
    return reciprocal_rank_fusion(all_results, k=60)
```

### Contextual Compression
```python
# After retrieval, compress chunks to only relevant portions
# Reduces noise, fits more context in prompt window

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

COMPRESSION_PROMPT = """Given the context and question, extract ONLY the 
parts of the context that are directly relevant to answering the question.
If nothing is relevant, return "NOT_RELEVANT".
Question: {query}
Context: {context}
Relevant Extract:"""
```

### Hybrid Search with Reciprocal Rank Fusion (RRF)
```python
def reciprocal_rank_fusion(result_lists: list[list], k: int = 60) -> list:
    """
    Fuse multiple ranked lists using RRF.
    k=60 is the standard constant (paper recommendation).
    Higher k → more weight to lower-ranked results.
    """
    fused_scores = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = doc.id
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"doc": doc, "score": 0.0}
            fused_scores[doc_id]["score"] += 1.0 / (k + rank + 1)
    
    return sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
```

### Self-RAG (Adaptive Retrieval)
```python
# LLM decides: (1) whether to retrieve, (2) if retrieved docs are relevant,
# (3) if generated answer is supported, (4) if answer is useful

class SelfRAG:
    async def generate(self, query: str):
        # Step 1: Does this query need retrieval?
        needs_retrieval = await self.llm.classify(
            f"Does answering '{query}' require external knowledge? [Yes/No]"
        )
        
        if needs_retrieval == "No":
            return await self.llm.generate(query)
        
        # Step 2: Retrieve and assess relevance
        docs = await self.retriever.retrieve(query, k=10)
        relevant_docs = []
        for doc in docs:
            is_relevant = await self.llm.classify(
                f"Is this document relevant to '{query}'?\nDoc: {doc.text}\n[Relevant/Irrelevant]"
            )
            if is_relevant == "Relevant":
                relevant_docs.append(doc)
        
        # Step 3: Generate with relevant docs
        answer = await self.llm.generate(query, context=relevant_docs)
        
        # Step 4: Verify support (hallucination check)
        is_supported = await self.llm.classify(
            f"Is this answer fully supported by the context?\nAnswer: {answer}\n[Supported/Partial/No Support]"
        )
        
        if is_supported == "No Support":
            # Retry with different retrieval strategy
            return await self.fallback_generate(query)
        
        return answer
```

---

## 6. Reranking Architectures

### Why Reranking is Non-Negotiable in Production

```
┌──────────────────────────────────────────────────────────────┐
│ Retrieval (bi-encoder): Fast but approximate                  │
│   Query → [Embed] → ANN Search → Top 100 candidates          │
│   Speed: ~10ms │ Quality: ⭐⭐⭐                              │
│                                                               │
│ Reranking (cross-encoder): Slow but precise                   │
│   (Query, Doc) pairs → Cross-attention → Relevance score      │
│   Speed: ~200ms │ Quality: ⭐⭐⭐⭐⭐                          │
│                                                               │
│ TWO-STAGE PIPELINE:                                           │
│   Retrieve Top-100 (fast) → Rerank to Top-5 (precise)        │
│   Combined: ~250ms │ Quality: ⭐⭐⭐⭐⭐                       │
└──────────────────────────────────────────────────────────────┘
```

### Reranker Comparison

| Reranker | Type | Latency (20 docs) | Quality (NDCG@10) | AWS Integration |
|----------|------|-------------------|-------------------|-----------------|
| **Cohere Rerank v3** | API | ~150ms | 0.72 | Bedrock (native) |
| **BGE-Reranker-v2-M3** | Cross-encoder | ~100ms* | 0.70 | Self-host (EKS) |
| **ColBERT v2** | Late-interaction | ~50ms | 0.68 | Self-host (EKS) |
| **Jina Reranker v2** | Cross-encoder | ~120ms | 0.71 | API |
| **FlashRank (tiny)** | Distilled | ~20ms* | 0.62 | Self-host |
| **RankGPT (LLM)** | Listwise | ~2000ms | 0.74 | Any LLM API |

*Self-hosted on g5.xlarge or inf2

### Cohere Rerank via Bedrock
```python
import boto3, json

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-west-2")

def rerank_with_cohere(query: str, documents: list[str], top_n: int = 5) -> list:
    """Rerank using Cohere Rerank v3 via Bedrock."""
    response = bedrock_runtime.invoke_model(
        modelId="cohere.rerank-v3-5:0",
        body=json.dumps({
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": True
        })
    )
    result = json.loads(response["body"].read())
    return [
        {"index": r["index"], "score": r["relevance_score"], "text": r["document"]["text"]}
        for r in result["results"]
    ]
```

### ColBERT Late-Interaction (EKS deployment)
```python
# ColBERT computes token-level embeddings at index time
# At query time: MaxSim between query tokens and document tokens
# 10-100x faster than cross-encoders with comparable quality

from colbert import Indexer, Searcher
from colbert.infra import ColBERTConfig

config = ColBERTConfig(
    nbits=2,           # Residual quantization (2-bit = 16x compression)
    doc_maxlen=300,
    query_maxlen=64,
    kmeans_niters=4,
    nranks=1           # GPU count
)

# Index once (offline)
indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", config=config)
indexer.index(name="dish_docs", collection=documents)

# Search (online) — sub-50ms even at scale
searcher = Searcher(index="dish_docs")
results = searcher.search(query, k=10)
```

### RankGPT (LLM-based Listwise Reranking)
```python
# Use when quality matters more than latency (e.g., customer-facing search)
# LLM sees all documents at once and produces optimal ranking

RANKGPT_PROMPT = """I will provide a question and {n} passages. 
Rank the passages based on their relevance to the question.
Output the ranking as a list of passage numbers from most to least relevant.

Question: {query}

{passages}

Ranking (most relevant first):"""

async def rankgpt_rerank(query: str, docs: list[str], llm) -> list[int]:
    passages = "\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(docs)])
    prompt = RANKGPT_PROMPT.format(n=len(docs), query=query, passages=passages)
    ranking = await llm.generate(prompt)
    return parse_ranking(ranking)  # Returns ordered indices
```

### Production Two-Stage Pipeline
```python
class TwoStageRetriever:
    def __init__(self, vector_store, reranker, retrieval_k=100, final_k=5):
        self.vector_store = vector_store
        self.reranker = reranker
        self.retrieval_k = retrieval_k
        self.final_k = final_k
    
    async def retrieve(self, query: str, filters: dict = None):
        # Stage 1: Fast bi-encoder retrieval (broad net)
        candidates = await self.vector_store.search(
            query, k=self.retrieval_k, filters=filters
        )
        
        # Stage 2: Precise cross-encoder reranking (narrow focus)
        reranked = await self.reranker.rerank(
            query=query,
            documents=[c.text for c in candidates],
            top_n=self.final_k
        )
        
        # Apply diversity filter (avoid near-duplicate chunks)
        diversified = self._mmr_diversity(reranked, lambda_=0.7)
        
        return diversified
    
    def _mmr_diversity(self, docs, lambda_=0.7):
        """Maximal Marginal Relevance — balance relevance with diversity"""
        selected = [docs[0]]
        candidates = docs[1:]
        
        while len(selected) < self.final_k and candidates:
            best_score = -1
            best_idx = 0
            for i, cand in enumerate(candidates):
                relevance = cand["score"]
                max_sim = max(cosine_sim(cand["embedding"], s["embedding"]) for s in selected)
                mmr_score = lambda_ * relevance - (1 - lambda_) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            selected.append(candidates.pop(best_idx))
        
        return selected
```

---

## 7. Knowledge Graph RAG

### Why Graph + Vector Hybrid

```
┌────────────────────────────────────────────────────────────────┐
│              VECTOR-ONLY vs GRAPH+VECTOR                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Vector-only limitations:                                        │
│ • No explicit relationships between entities                    │
│ • Can't answer "What teams does person X work with?"           │
│ • No multi-hop reasoning ("Who reports to the VP of Y?")       │
│ • No temporal reasoning ("What happened after event X?")        │
│                                                                 │
│ Graph+Vector strengths:                                         │
│ • Entities + relationships + attributes                         │
│ • Multi-hop traversal for complex queries                       │
│ • Structured knowledge + unstructured context                   │
│ • Better factual accuracy (grounded in explicit relations)      │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### GraphRAG Architecture (Microsoft pattern)

```
┌──────────────────────────────────────────────────────────────────┐
│                    GRAPHRAG PIPELINE                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  INDEXING (Offline):                                              │
│  ┌─────────┐    ┌───────────┐    ┌──────────────┐               │
│  │ Documents│───▶│ Entity &  │───▶│ Community    │               │
│  │          │    │ Relation  │    │ Detection    │               │
│  │          │    │ Extraction│    │ (Leiden alg) │               │
│  └─────────┘    └───────────┘    └──────────────┘               │
│                       │                   │                       │
│                       ▼                   ▼                       │
│              ┌────────────┐    ┌──────────────────┐              │
│              │ Knowledge  │    │ Community         │              │
│              │ Graph      │    │ Summaries         │              │
│              │ (entities, │    │ (hierarchical)    │              │
│              │  relations)│    │                   │              │
│              └────────────┘    └──────────────────┘              │
│                                                                   │
│  RETRIEVAL (Online):                                              │
│  • Local Search: entity neighborhoods + linked chunks             │
│  • Global Search: community summaries for broad questions         │
│  • Hybrid: route based on query type                              │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Neptune + Vector Hybrid Implementation
```python
# Amazon Neptune (graph) + OpenSearch Serverless (vector)
# Best for DISH: managed services, IAM integration, scalable

import boto3
from gremlin_python.driver import client as gremlin_client

class GraphVectorRAG:
    def __init__(self):
        self.neptune = gremlin_client.Client(
            'wss://your-neptune-endpoint:8182/gremlin', 'g'
        )
        self.opensearch = boto3.client('opensearchserverless')
        self.bedrock = boto3.client('bedrock-runtime')
    
    async def retrieve(self, query: str):
        # Step 1: Extract entities from query
        entities = await self._extract_entities(query)
        
        # Step 2: Graph traversal (find related entities + context)
        graph_context = await self._traverse_graph(entities)
        
        # Step 3: Vector search (semantic similarity)
        vector_results = await self._vector_search(query)
        
        # Step 4: Fuse graph + vector results
        fused = self._fuse_results(graph_context, vector_results)
        
        return fused
    
    async def _extract_entities(self, query: str):
        """Use LLM to extract entity mentions from query."""
        response = self.bedrock.invoke_model(
            modelId="anthropic.claude-sonnet-4-20250514-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": f"""Extract named entities from this query.
                Return as JSON: {{"entities": [{{"name": "...", "type": "..."}}]}}
                Query: {query}"""}],
                "max_tokens": 200
            })
        )
        return json.loads(response["body"].read())["content"][0]["text"]
    
    async def _traverse_graph(self, entities: list, hops: int = 2):
        """Multi-hop graph traversal from extracted entities."""
        graph_data = []
        for entity in entities:
            # Gremlin query: find entity + N-hop neighborhood
            query = f"""
            g.V().has('name', '{entity["name"]}')
              .repeat(bothE().otherV()).times({hops})
              .path().by(valueMap(true))
              .limit(50)
            """
            results = self.neptune.submit(query).all().result()
            graph_data.extend(results)
        return graph_data
    
    async def _vector_search(self, query: str, k: int = 20):
        """Standard dense retrieval from OpenSearch."""
        query_embedding = await self._embed(query)
        # ... OpenSearch kNN query ...
        return results
    
    def _fuse_results(self, graph_context, vector_results):
        """Combine graph structure with vector-retrieved text."""
        # Graph provides: entity relationships, structured facts
        # Vector provides: relevant text passages
        # Fusion: use graph to validate/augment vector results
        enhanced_results = []
        for result in vector_results:
            # Attach related graph entities to each chunk
            related_entities = self._find_overlapping_entities(
                result.text, graph_context
            )
            result.metadata["graph_entities"] = related_entities
            enhanced_results.append(result)
        return enhanced_results
```

### Neo4j + LangChain GraphRAG
```python
# Alternative: Neo4j for graph (more query flexibility, Cypher)
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain

graph = Neo4jGraph(
    url="bolt://neo4j.internal:7687",
    username="neo4j",
    password="***"
)

# Auto-generate Cypher from natural language
chain = GraphCypherQAChain.from_llm(
    llm=bedrock_claude,
    graph=graph,
    verbose=True,
    validate_cypher=True,  # LLM validates generated Cypher
    top_k=10
)

# Hybrid: Graph for structured + Vector for unstructured
class HybridGraphVectorChain:
    def __init__(self, graph_chain, vector_retriever):
        self.graph_chain = graph_chain
        self.vector_retriever = vector_retriever
        self.router = QueryRouter()  # Classifies query type
    
    async def answer(self, query: str):
        query_type = await self.router.classify(query)
        
        if query_type == "structured":  # "Who manages team X?"
            return await self.graph_chain.run(query)
        elif query_type == "unstructured":  # "Explain the RAG architecture"
            docs = await self.vector_retriever.retrieve(query)
            return await self.generate(query, docs)
        else:  # "hybrid" — combine both
            graph_answer = await self.graph_chain.run(query)
            vector_docs = await self.vector_retriever.retrieve(query)
            return await self.generate(query, vector_docs, graph_context=graph_answer)
```

---

## 8. Evaluation Frameworks

### RAGAS (Retrieval Augmented Generation Assessment)

```
┌────────────────────────────────────────────────────────────────┐
│                    RAGAS METRIC TAXONOMY                         │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GENERATION QUALITY          RETRIEVAL QUALITY                   │
│  ┌─────────────────┐        ┌────────────────────────┐         │
│  │ Faithfulness     │        │ Context Precision       │         │
│  │ (no hallucination│        │ (relevant chunks ranked │         │
│  │  vs. context)    │        │  higher?)               │         │
│  │                  │        │                         │         │
│  │ Answer Relevance │        │ Context Recall          │         │
│  │ (answers the     │        │ (all needed info        │         │
│  │  question?)      │        │  retrieved?)            │         │
│  └─────────────────┘        └────────────────────────┘         │
│                                                                  │
│  END-TO-END                  COMPONENT-LEVEL                     │
│  ┌─────────────────┐        ┌────────────────────────┐         │
│  │ Answer Semantic  │        │ Chunk Attribution       │         │
│  │ Similarity       │        │ Noise Robustness        │         │
│  │                  │        │ Information Integration  │         │
│  │ Answer Correct-  │        │ Counterfactual Robust.  │         │
│  │ ness             │        │                         │         │
│  └─────────────────┘        └────────────────────────┘         │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

### RAGAS Implementation
```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_entity_recall,
    answer_similarity
)
from datasets import Dataset

# Prepare evaluation dataset
eval_data = {
    "question": ["What is DISH's RAG architecture?", ...],
    "answer": ["DISH uses Bedrock KB with...", ...],  # Generated answers
    "contexts": [["chunk1...", "chunk2..."], ...],     # Retrieved contexts
    "ground_truth": ["The actual architecture...", ...]  # Human labels
}

dataset = Dataset.from_dict(eval_data)

# Run evaluation
results = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,          # Is answer grounded in context? (0-1)
        answer_relevancy,      # Does answer address the question? (0-1)
        context_precision,     # Are relevant chunks ranked first? (0-1)
        context_recall,        # Is all needed info retrieved? (0-1)
    ],
    llm=bedrock_claude,        # LLM-as-judge
    embeddings=titan_embeddings
)

print(results)
# {'faithfulness': 0.87, 'answer_relevancy': 0.92, 
#  'context_precision': 0.78, 'context_recall': 0.85}
```

### DeepEval Framework
```python
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    HallucinationMetric,
    ToxicityMetric
)
from deepeval.test_case import LLMTestCase

# DeepEval advantages over RAGAS:
# - Built-in pytest integration
# - More granular metrics (hallucination types)
# - Confidence intervals
# - CI/CD pipeline friendly

test_case = LLMTestCase(
    input="What is the chunking strategy for PDFs?",
    actual_output="We use document-aware chunking with...",
    expected_output="Document-aware chunking preserving...",
    retrieval_context=["chunk1...", "chunk2..."]
)

# Metrics with thresholds for CI/CD gates
metrics = [
    FaithfulnessMetric(threshold=0.8, model="gpt-4"),
    AnswerRelevancyMetric(threshold=0.7),
    ContextualPrecisionMetric(threshold=0.7),
    HallucinationMetric(threshold=0.5),  # Lower = fewer hallucinations
]

results = evaluate(test_cases=[test_case], metrics=metrics)
```

### Production Evaluation Pipeline
```python
class RAGEvaluationPipeline:
    """
    Automated evaluation pipeline for CI/CD integration.
    Runs on every deployment, gates on metric thresholds.
    """
    
    THRESHOLDS = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.80,
        "context_precision": 0.75,
        "context_recall": 0.80,
    }
    
    def __init__(self, rag_system, eval_dataset_path: str):
        self.rag = rag_system
        self.dataset = self._load_golden_dataset(eval_dataset_path)
    
    async def run_evaluation(self) -> dict:
        """Run full evaluation suite."""
        results = []
        
        for item in self.dataset:
            # Generate answer using the RAG system
            response = await self.rag.query(item["question"])
            
            results.append({
                "question": item["question"],
                "answer": response.answer,
                "contexts": response.source_chunks,
                "ground_truth": item["expected_answer"],
                "latency_ms": response.latency_ms,
                "tokens_used": response.total_tokens
            })
        
        # Compute metrics
        metrics = self._compute_metrics(results)
        
        # Gate check
        passed = all(
            metrics[k] >= v for k, v in self.THRESHOLDS.items()
        )
        
        return {
            "metrics": metrics,
            "passed": passed,
            "details": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _compute_metrics(self, results):
        """Compute RAGAS-style metrics using LLM-as-judge."""
        # Implementation using Claude as judge
        faithfulness_scores = []
        for r in results:
            score = await self._judge_faithfulness(
                r["answer"], r["contexts"]
            )
            faithfulness_scores.append(score)
        
        return {
            "faithfulness": np.mean(faithfulness_scores),
            "answer_relevancy": np.mean(relevancy_scores),
            "context_precision": np.mean(precision_scores),
            "context_recall": np.mean(recall_scores),
            "avg_latency_ms": np.mean([r["latency_ms"] for r in results]),
            "p99_latency_ms": np.percentile([r["latency_ms"] for r in results], 99),
        }
```

### Evaluation Metric Cheat Sheet

| Metric | What it Measures | When it Fails | Action |
|--------|-----------------|---------------|--------|
| **Faithfulness** ↓ | Answer grounded in context? | Hallucination | Better reranking, stricter prompting |
| **Answer Relevancy** ↓ | Answer addresses question? | Off-topic generation | Improve query understanding |
| **Context Precision** ↓ | Relevant chunks ranked high? | Noise in top-K | Better reranking, tune K |
| **Context Recall** ↓ | All info retrieved? | Missing context | More chunks, better chunking, hybrid search |
| **Latency** ↑ | Response time | SLA breach | Caching, reduce K, faster model |

---

## 9. Production Patterns

### 9.1 Semantic Caching

```python
import hashlib
import numpy as np
from datetime import timedelta

class SemanticCache:
    """
    Cache RAG responses by semantic similarity of queries.
    Avoids redundant LLM calls for paraphrased questions.
    """
    
    def __init__(self, embedder, cache_store, similarity_threshold=0.95):
        self.embedder = embedder
        self.cache = cache_store  # Redis or DynamoDB
        self.threshold = similarity_threshold
    
    async def get(self, query: str) -> Optional[dict]:
        query_embedding = await self.embedder.encode(query)
        
        # Check for semantically similar cached queries
        cached_entries = await self.cache.get_all_embeddings()
        for entry in cached_entries:
            similarity = cosine_similarity(query_embedding, entry["embedding"])
            if similarity >= self.threshold:
                # Cache hit — but check TTL and staleness
                if not self._is_stale(entry):
                    return entry["response"]
        
        return None  # Cache miss
    
    async def set(self, query: str, response: dict, ttl: timedelta = timedelta(hours=1)):
        query_embedding = await self.embedder.encode(query)
        await self.cache.store({
            "query": query,
            "embedding": query_embedding,
            "response": response,
            "created_at": datetime.utcnow(),
            "ttl": ttl
        })
    
    def _is_stale(self, entry) -> bool:
        """Check if underlying data has changed since cache entry."""
        return datetime.utcnow() > entry["created_at"] + entry["ttl"]


# DynamoDB + DAX implementation for AWS
class DynamoSemanticCache:
    def __init__(self):
        self.table = boto3.resource('dynamodb').Table('rag-cache')
        # Use DAX for sub-ms reads
        self.dax = amazondax.AmazonDaxClient(endpoint_url='dax://cache.internal:8111')
```

### 9.2 Streaming Responses with Citations

```python
import asyncio
from typing import AsyncGenerator

class StreamingRAG:
    """Stream LLM response while tracking citation spans."""
    
    async def stream_with_citations(
        self, query: str, contexts: list[dict]
    ) -> AsyncGenerator[dict, None]:
        
        prompt = self._build_prompt_with_citation_instructions(query, contexts)
        
        buffer = ""
        citation_pattern = r'\[(\d+)\]'
        
        async for chunk in self.llm.stream(prompt):
            buffer += chunk.text
            
            # Yield text chunks with inline citation markers
            yield {
                "type": "text_delta",
                "content": chunk.text
            }
        
        # After streaming, extract citation map
        citations = self._extract_citations(buffer, contexts)
        yield {
            "type": "citations",
            "content": citations
        }
    
    def _build_prompt_with_citation_instructions(self, query, contexts):
        numbered_contexts = "\n".join([
            f"[{i+1}] {ctx['text']}\nSource: {ctx['metadata']['source']}"
            for i, ctx in enumerate(contexts)
        ])
        
        return f"""Answer the question using ONLY the provided contexts.
Cite sources using [N] notation inline where you use information from context N.

Contexts:
{numbered_contexts}

Question: {query}

Answer (with inline citations):"""
```

### 9.3 Guardrails

```python
from typing import Optional

class RAGGuardrails:
    """
    Pre-retrieval and post-generation guardrails.
    Integrates with Bedrock Guardrails for managed PII/toxicity filtering.
    """
    
    def __init__(self, guardrail_id: str, guardrail_version: str):
        self.bedrock = boto3.client('bedrock-runtime')
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
    
    async def check_input(self, query: str) -> tuple[bool, Optional[str]]:
        """Pre-retrieval: block harmful/off-topic queries."""
        response = self.bedrock.apply_guardrail(
            guardrailIdentifier=self.guardrail_id,
            guardrailVersion=self.guardrail_version,
            source='INPUT',
            content=[{"text": {"text": query}}]
        )
        
        if response['action'] == 'GUARDRAIL_INTERVENED':
            return False, response['outputs'][0]['text']
        return True, None
    
    async def check_output(self, response: str, contexts: list[str]) -> tuple[bool, str]:
        """Post-generation: verify grounding + safety."""
        # Bedrock Guardrail check
        gr_response = self.bedrock.apply_guardrail(
            guardrailIdentifier=self.guardrail_id,
            guardrailVersion=self.guardrail_version,
            source='OUTPUT',
            content=[{"text": {"text": response}}]
        )
        
        if gr_response['action'] == 'GUARDRAIL_INTERVENED':
            return False, gr_response['outputs'][0]['text']
        
        # Custom grounding check — ensure claims are supported
        grounding_score = await self._check_grounding(response, contexts)
        if grounding_score < 0.7:
            return False, "Response may contain unsupported claims. Please verify."
        
        return True, response
    
    async def _check_grounding(self, response: str, contexts: list[str]) -> float:
        """NLI-based grounding verification."""
        # Decompose response into claims
        claims = await self._extract_claims(response)
        
        supported = 0
        for claim in claims:
            # Check if claim is entailed by any context
            for ctx in contexts:
                entailment = await self._nli_check(premise=ctx, hypothesis=claim)
                if entailment > 0.8:
                    supported += 1
                    break
        
        return supported / len(claims) if claims else 1.0
```

### 9.4 Metadata Filtering

```python
class MetadataFilteredRetrieval:
    """
    Apply structured filters BEFORE vector search for efficiency.
    Critical for multi-tenant, time-sensitive, or access-controlled data.
    """
    
    def build_filter(self, user_context: dict) -> dict:
        """Build OpenSearch filter from user context."""
        filters = []
        
        # Tenant isolation (critical for DISH multi-org)
        if user_context.get("org_id"):
            filters.append({"term": {"metadata.org_id": user_context["org_id"]}})
        
        # Time-based filtering (only recent docs)
        if user_context.get("recency_days"):
            filters.append({
                "range": {
                    "metadata.updated_at": {
                        "gte": f"now-{user_context['recency_days']}d"
                    }
                }
            })
        
        # Document type filtering
        if user_context.get("doc_types"):
            filters.append({
                "terms": {"metadata.doc_type": user_context["doc_types"]}
            })
        
        # Access control (RBAC-based)
        if user_context.get("user_groups"):
            filters.append({
                "terms": {"metadata.access_groups": user_context["user_groups"]}
            })
        
        return {"bool": {"filter": filters}}
    
    async def retrieve(self, query: str, user_context: dict, k: int = 20):
        """Filtered vector search."""
        query_embedding = await self.embedder.encode(query)
        pre_filter = self.build_filter(user_context)
        
        # OpenSearch filtered kNN
        body = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": k,
                        "filter": pre_filter  # Applied BEFORE ANN search
                    }
                }
            }
        }
        
        return await self.opensearch.search(index="documents", body=body)
```

### 9.5 Citation Extraction & Grounding

```python
class CitationExtractor:
    """
    Extract verifiable citations from generated responses.
    Maps each claim to specific source chunks + page numbers.
    """
    
    EXTRACTION_PROMPT = """Analyze the response and extract each factual claim.
For each claim, identify which source context(s) support it.

Response: {response}

Source Contexts:
{contexts}

Output JSON:
[{{"claim": "...", "supporting_sources": [1, 3], "confidence": 0.95}}, ...]"""
    
    async def extract(self, response: str, contexts: list[dict]) -> list[dict]:
        numbered_contexts = "\n".join([
            f"[{i+1}] {ctx['text'][:500]}" for i, ctx in enumerate(contexts)
        ])
        
        result = await self.llm.generate(
            self.EXTRACTION_PROMPT.format(
                response=response,
                contexts=numbered_contexts
            )
        )
        
        citations = json.loads(result)
        
        # Enrich with source metadata
        for citation in citations:
            citation["sources"] = [
                {
                    "chunk_id": contexts[i-1]["id"],
                    "file": contexts[i-1]["metadata"]["source"],
                    "page": contexts[i-1]["metadata"].get("page"),
                    "url": contexts[i-1]["metadata"].get("url")
                }
                for i in citation["supporting_sources"]
            ]
        
        return citations
```

---

## 10. AWS Implementation

### 10.1 Bedrock Knowledge Bases — Advanced Configuration

```python
import boto3

bedrock_agent = boto3.client('bedrock-agent', region_name='us-west-2')

# Create Knowledge Base with advanced chunking + parsing
kb_response = bedrock_agent.create_knowledge_base(
    name='dish-advanced-rag',
    description='Production RAG with hierarchical chunking',
    roleArn='arn:aws:iam::ACCOUNT:role/BedrockKBRole',
    knowledgeBaseConfiguration={
        'type': 'VECTOR',
        'vectorKnowledgeBaseConfiguration': {
            'embeddingModelArn': 'arn:aws:bedrock:us-west-2::foundation-model/amazon.titan-embed-text-v2:0',
            'embeddingModelConfiguration': {
                'bedrockEmbeddingModelConfiguration': {
                    'dimensions': 1024  # Max quality
                }
            }
        }
    },
    storageConfiguration={
        'type': 'OPENSEARCH_SERVERLESS',
        'opensearchServerlessConfiguration': {
            'collectionArn': 'arn:aws:aoss:us-west-2:ACCOUNT:collection/xyz',
            'vectorIndexName': 'dish-kb-index',
            'fieldMapping': {
                'vectorField': 'embedding',
                'textField': 'text',
                'metadataField': 'metadata'
            }
        }
    }
)

# Add data source with CUSTOM parsing & chunking
ds_response = bedrock_agent.create_data_source(
    knowledgeBaseId=kb_response['knowledgeBase']['knowledgeBaseId'],
    name='technical-docs',
    dataSourceConfiguration={
        'type': 'S3',
        's3Configuration': {
            'bucketArn': 'arn:aws:s3:::dish-rag-docs',
            'inclusionPrefixes': ['technical/', 'runbooks/']
        }
    },
    vectorIngestionConfiguration={
        'chunkingConfiguration': {
            'chunkingStrategy': 'HIERARCHICAL',
            'hierarchicalChunkingConfiguration': {
                'levelConfigurations': [
                    {'maxTokens': 1500},  # Parent (broader context)
                    {'maxTokens': 300}    # Child (precise retrieval)
                ],
                'overlapTokens': 60
            }
        },
        'parsingConfiguration': {
            'parsingStrategy': 'BEDROCK_FOUNDATION_MODEL',
            'bedrockFoundationModelConfiguration': {
                'modelArn': 'arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0',
                'parsingPrompt': {
                    'parsingPromptText': """Extract and structure the content from this document.
                    Preserve tables as markdown. Extract code blocks with language tags.
                    Maintain section hierarchy. Include figure descriptions."""
                }
            }
        },
        'customTransformationConfiguration': {
            'intermediateStorage': {
                's3Location': {'uri': 's3://dish-rag-intermediate/'}
            },
            'transformations': [{
                'stepToApply': 'POST_CHUNKING',
                'transformationFunction': {
                    'transformationLambdaConfiguration': {
                        'lambdaArn': 'arn:aws:lambda:us-west-2:ACCOUNT:function:contextual-enrichment'
                    }
                }
            }]
        }
    }
)
```

### 10.2 Custom Post-Chunking Lambda (Contextual Enrichment)

```python
# Lambda function for Bedrock KB custom transformation
# Adds document-level context to each chunk (Anthropic's contextual retrieval pattern)

import boto3
import json

bedrock_runtime = boto3.client('bedrock-runtime')

def handler(event, context):
    """
    Bedrock KB custom transformation Lambda.
    Enriches each chunk with document-level context.
    """
    s3 = boto3.client('s3')
    input_files = event['inputFiles']
    output_files = []
    
    for input_file in input_files:
        content = json.loads(
            s3.get_object(
                Bucket=input_file['contentBatches'][0]['bucket'],
                Key=input_file['contentBatches'][0]['key']
            )['Body'].read()
        )
        
        # Get document-level summary for context
        doc_title = content.get('documentMetadata', {}).get('title', 'Unknown')
        
        enriched_chunks = []
        for chunk in content['fileContents']:
            original_text = chunk['contentBody']['textContent']['data']
            
            # Generate contextual prefix using Claude
            context_prefix = generate_context(doc_title, original_text)
            
            # Prepend context to chunk
            chunk['contentBody']['textContent']['data'] = (
                f"[Context: {context_prefix}]\n\n{original_text}"
            )
            enriched_chunks.append(chunk)
        
        content['fileContents'] = enriched_chunks
        
        # Write enriched content back to S3
        output_key = f"enriched/{input_file['originalFileLocation']['key']}"
        s3.put_object(
            Bucket='dish-rag-intermediate',
            Key=output_key,
            Body=json.dumps(content)
        )
        output_files.append({'originalFileLocation': input_file['originalFileLocation']})
    
    return {'outputFiles': output_files}


def generate_context(doc_title: str, chunk_text: str) -> str:
    """Generate contextual prefix for a chunk."""
    response = bedrock_runtime.invoke_model(
        modelId='anthropic.claude-3-haiku-20240307-v1:0',
        body=json.dumps({
            'messages': [{'role': 'user', 'content': f"""Given this chunk from "{doc_title}", 
write a 1-2 sentence context that helps a retrieval system understand what this chunk is about 
and how it fits in the document.

Chunk: {chunk_text[:1000]}

Context (be specific, mention key entities/topics):"""}],
            'max_tokens': 100,
            'anthropic_version': 'bedrock-2023-05-31'
        })
    )
    return json.loads(response['body'].read())['content'][0]['text']
```

### 10.3 OpenSearch Serverless — Advanced Setup

```python
import boto3

aoss = boto3.client('opensearchserverless')

# Create encryption policy
aoss.create_security_policy(
    name='dish-rag-encryption',
    type='encryption',
    policy=json.dumps({
        "Rules": [{"ResourceType": "collection", "Resource": ["collection/dish-rag"]}],
        "AWSOwnedKey": True
    })
)

# Create network policy (VPC endpoint for EKS access)
aoss.create_security_policy(
    name='dish-rag-network',
    type='network',
    policy=json.dumps([{
        "Rules": [
            {"ResourceType": "collection", "Resource": ["collection/dish-rag"]},
            {"ResourceType": "dashboard", "Resource": ["collection/dish-rag"]}
        ],
        "AllowFromPublic": False,
        "SourceVPCEs": ["vpce-0123456789abcdef"]  # Your VPC endpoint
    }])
)

# Create data access policy (fine-grained for Bedrock KB + application)
aoss.create_access_policy(
    name='dish-rag-access',
    type='data',
    policy=json.dumps([{
        "Rules": [
            {
                "ResourceType": "index",
                "Resource": ["index/dish-rag/*"],
                "Permission": [
                    "aoss:CreateIndex", "aoss:UpdateIndex",
                    "aoss:DescribeIndex", "aoss:ReadDocument",
                    "aoss:WriteDocument"
                ]
            },
            {
                "ResourceType": "collection",
                "Resource": ["collection/dish-rag"],
                "Permission": ["aoss:CreateCollectionItems"]
            }
        ],
        "Principal": [
            "arn:aws:iam::ACCOUNT:role/BedrockKBRole",
            "arn:aws:iam::ACCOUNT:role/RAGApplicationRole"
        ]
    }])
)

# Create collection
aoss.create_collection(
    name='dish-rag',
    type='VECTORSEARCH',
    description='DISH Advanced RAG vector store'
)
```

### 10.4 Retrieval with Bedrock KB (Advanced)

```python
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name='us-west-2')

def advanced_retrieve(query: str, kb_id: str, filters: dict = None):
    """
    Advanced retrieval with metadata filtering, reranking, and hybrid search.
    """
    retrieval_config = {
        'vectorSearchConfiguration': {
            'numberOfResults': 20,  # Retrieve more, rerank later
            'overrideSearchType': 'HYBRID',  # SEMANTIC | HYBRID
        }
    }
    
    # Add metadata filters
    if filters:
        retrieval_config['vectorSearchConfiguration']['filter'] = {
            'andAll': [
                {'equals': {'key': k, 'value': v}} 
                for k, v in filters.items()
            ]
        }
    
    response = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=kb_id,
        retrievalQuery={'text': query},
        retrievalConfiguration=retrieval_config
    )
    
    results = response['retrievalResults']
    
    # Apply Cohere Rerank via Bedrock
    reranked = rerank_with_cohere(
        query=query,
        documents=[r['content']['text'] for r in results],
        top_n=5
    )
    
    return reranked


def retrieve_and_generate(query: str, kb_id: str, model_id: str):
    """
    End-to-end RAG with Bedrock KB (managed orchestration).
    """
    response = bedrock_agent_runtime.retrieve_and_generate(
        input={'text': query},
        retrieveAndGenerateConfiguration={
            'type': 'KNOWLEDGE_BASE',
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': kb_id,
                'modelArn': f'arn:aws:bedrock:us-west-2::foundation-model/{model_id}',
                'retrievalConfiguration': {
                    'vectorSearchConfiguration': {
                        'numberOfResults': 10,
                        'overrideSearchType': 'HYBRID'
                    }
                },
                'generationConfiguration': {
                    'promptTemplate': {
                        'textPromptTemplate': """You are a technical assistant for DISH Network.
Answer using ONLY the provided search results. Cite sources using [N] notation.
If you cannot answer from the context, say "I don't have enough information."

$search_results$

Question: $query$

Answer (with citations):"""
                    },
                    'guardrailConfiguration': {
                        'guardrailId': 'your-guardrail-id',
                        'guardrailVersion': '1'
                    }
                },
                'orchestrationConfiguration': {
                    'queryTransformationConfiguration': {
                        'type': 'QUERY_DECOMPOSITION'  # Auto multi-query
                    }
                }
            }
        }
    )
    
    return {
        'answer': response['output']['text'],
        'citations': response.get('citations', []),
        'session_id': response.get('sessionId')
    }
```

### 10.5 Infrastructure as Code (CDK)

```python
# Key CDK constructs for the RAG stack
from aws_cdk import (
    Stack, Duration, RemovalPolicy,
    aws_opensearchserverless as aoss,
    aws_bedrock as bedrock,
    aws_s3 as s3,
    aws_lambda as lambda_,
    aws_iam as iam,
    aws_elasticache as elasticache,
)

class AdvancedRAGStack(Stack):
    def __init__(self, scope, id, **kwargs):
        super().__init__(scope, id, **kwargs)
        
        # S3 for document storage
        self.docs_bucket = s3.Bucket(self, "DocsBucket",
            bucket_name="dish-rag-documents",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="archive-old-versions",
                    noncurrent_version_expiration=Duration.days(90)
                )
            ]
        )
        
        # ElastiCache (Redis) for semantic caching
        self.cache = elasticache.CfnReplicationGroup(self, "RAGCache",
            replication_group_description="RAG semantic cache",
            engine="redis",
            cache_node_type="cache.r6g.large",
            num_cache_clusters=2,
            automatic_failover_enabled=True,
            at_rest_encryption_enabled=True,
            transit_encryption_enabled=True
        )
        
        # Custom chunking Lambda
        self.enrichment_lambda = lambda_.Function(self, "EnrichmentLambda",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="contextual_enrichment.handler",
            code=lambda_.Code.from_asset("lambda/enrichment"),
            timeout=Duration.minutes(5),
            memory_size=1024,
            environment={
                "MODEL_ID": "anthropic.claude-3-haiku-20240307-v1:0"
            }
        )
        
        # Grant Bedrock access to Lambda
        self.enrichment_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=["arn:aws:bedrock:*::foundation-model/*"]
            )
        )
```

---

## 11. Production Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION RAG SYSTEM ARCHITECTURE                             │
│                         (DISH Network — AWS)                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────────────────┐
│   DATA SOURCES   │     │  INGESTION LAYER │     │      STORAGE LAYER           │
├──────────────────┤     ├──────────────────┤     ├──────────────────────────────┤
│                  │     │                  │     │                              │
│ ┌──────────────┐ │     │ ┌──────────────┐ │     │ ┌────────────────────────┐  │
│ │ S3 Buckets   │─┼────▶│ │ Doc Parser   │ │     │ │ OpenSearch Serverless   │  │
│ │ (PDFs, DOCX, │ │     │ │ (Bedrock FM  │ │     │ │ • Dense vectors (1024d)│  │
│ │  HTML, MD)   │ │     │ │  or Textract)│ │     │ │ • BM25 text index      │  │
│ └──────────────┘ │     │ └──────┬───────┘ │     │ │ • Metadata filters     │  │
│                  │     │        │         │     │ └────────────────────────┘  │
│ ┌──────────────┐ │     │        ▼         │     │                              │
│ │ Confluence/  │─┼────▶│ ┌──────────────┐ │     │ ┌────────────────────────┐  │
│ │ SharePoint   │ │     │ │ Chunker      │ │     │ │ Neptune (Knowledge     │  │
│ └──────────────┘ │     │ │ (Hierarchical│─┼────▶│ │ Graph)                 │  │
│                  │     │ │  + Contextual)│ │     │ │ • Entities & Relations │  │
│ ┌──────────────┐ │     │ └──────┬───────┘ │     │ │ • Multi-hop traversal  │  │
│ │ Git Repos    │─┼────▶│        │         │     │ └────────────────────────┘  │
│ │ (code, docs) │ │     │        ▼         │     │                              │
│ └──────────────┘ │     │ ┌──────────────┐ │     │ ┌────────────────────────┐  │
│                  │     │ │ Embedder     │ │     │ │ ElastiCache (Redis)    │  │
│ ┌──────────────┐ │     │ │ (Titan v2    │─┼────▶│ │ • Semantic cache       │  │
│ │ Slack/Email  │─┼────▶│ │  1024d)      │ │     │ │ • Query dedup          │  │
│ │ (structured) │ │     │ └──────────────┘ │     │ └────────────────────────┘  │
│ └──────────────┘ │     │                  │     │                              │
│                  │     │ ┌──────────────┐ │     │ ┌────────────────────────┐  │
│                  │     │ │ Entity       │ │     │ │ DynamoDB               │  │
│                  │     │ │ Extractor    │─┼────▶│ │ • Chat history         │  │
│                  │     │ │ (for KG)     │ │     │ │ • User preferences     │  │
│                  │     │ └──────────────┘ │     │ │ • Access control       │  │
│                  │     │                  │     │ └────────────────────────┘  │
└──────────────────┘     └──────────────────┘     └──────────────────────────────┘
         │                                                      │
         │              ┌───────────────────────────────────────┘
         │              │
         ▼              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          RETRIEVAL & GENERATION LAYER                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │   QUERY     │    │  RETRIEVAL  │    │  RERANKING  │    │   GENERATION    │  │
│  │  PIPELINE   │    │  PIPELINE   │    │  PIPELINE   │    │   PIPELINE      │  │
│  ├─────────────┤    ├─────────────┤    ├─────────────┤    ├─────────────────┤  │
│  │             │    │             │    │             │    │                 │  │
│  │ 1. Guard-   │    │ 1. Dense    │    │ 1. Cohere   │    │ 1. Prompt       │  │
│  │    rails    │───▶│    (kNN)    │───▶│    Rerank   │───▶│    Construction │  │
│  │    (input)  │    │             │    │    v3       │    │    + Citations  │  │
│  │             │    │ 2. Sparse   │    │             │    │                 │  │
│  │ 2. Cache    │    │    (BM25)   │    │ 2. MMR      │    │ 2. Claude 3.5   │  │
│  │    Check    │    │             │    │    Diversity │    │    Sonnet       │  │
│  │             │    │ 3. RRF      │    │             │    │    (Streaming)  │  │
│  │ 3. Query    │    │    Fusion   │    │ 3. Score    │    │                 │  │
│  │    Transform│    │             │    │    Threshold │    │ 3. Guardrails   │  │
│  │    (HyDE/   │    │ 4. Graph    │    │    Filter   │    │    (output)     │  │
│  │     Multi-Q)│    │    Traversal│    │             │    │                 │  │
│  │             │    │             │    │             │    │ 4. Citation     │  │
│  │ 4. Router   │    │ 5. Metadata │    │             │    │    Extraction   │  │
│  │    (strategy│    │    Filter   │    │             │    │                 │  │
│  │     select) │    │             │    │             │    │ 5. Feedback     │  │
│  │             │    │             │    │             │    │    Loop         │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
         │                                                      │
         ▼                                                      ▼
┌──────────────────────────────────┐    ┌──────────────────────────────────────────┐
│     APPLICATION LAYER            │    │       OBSERVABILITY & EVAL               │
├──────────────────────────────────┤    ├──────────────────────────────────────────┤
│                                  │    │                                          │
│ ┌──────────────────────────────┐ │    │ ┌──────────────────────────────────────┐ │
│ │ MCP Server (Tool Interface)  │ │    │ │ CloudWatch Metrics                   │ │
│ │ • retrieve_context           │ │    │ │ • Latency (p50/p95/p99)             │ │
│ │ • answer_question            │ │    │ │ • Cache hit rate                     │ │
│ │ • search_knowledge_graph     │ │    │ │ • Retrieval recall                   │ │
│ │ • get_citations              │ │    │ │ • Token usage                        │ │
│ └──────────────────────────────┘ │    │ └──────────────────────────────────────┘ │
│                                  │    │                                          │
│ ┌──────────────────────────────┐ │    │ ┌──────────────────────────────────────┐ │
│ │ API Gateway + ALB            │ │    │ │ RAGAS/DeepEval Pipeline              │ │
│ │ (EKS Ingress)               │ │    │ │ • Nightly eval on golden dataset     │ │
│ │ • Rate limiting              │ │    │ │ • Regression detection               │ │
│ │ • Auth (Cognito/IAM)        │ │    │ │ • Metric dashboards                  │ │
│ └──────────────────────────────┘ │    │ └──────────────────────────────────────┘ │
│                                  │    │                                          │
│ ┌──────────────────────────────┐ │    │ ┌──────────────────────────────────────┐ │
│ │ EKS Pods (Application)       │ │    │ │ LangSmith / Langfuse Tracing        │ │
│ │ • FastAPI + async            │ │    │ │ • Per-query traces                   │ │
│ │ • Connection pooling         │ │    │ │ • Cost attribution                   │ │
│ │ • Graceful degradation       │ │    │ │ • Feedback collection                │ │
│ └──────────────────────────────┘ │    │ └──────────────────────────────────────┘ │
│                                  │    │                                          │
└──────────────────────────────────┘    └──────────────────────────────────────────┘
```

### Data Flow Summary

```
User Query → API Gateway → EKS Pod → Query Pipeline
  ├── Cache Check (Redis) → HIT → Return cached response
  └── MISS → Query Transform (HyDE/Multi-Query)
       → Parallel Retrieval:
         ├── OpenSearch (Dense + BM25 Hybrid)
         ├── Neptune (Graph Traversal, if entity query)
         └── Metadata Pre-filter
       → RRF Fusion → Cohere Rerank → Top-5 Selection
       → Prompt Construction (with citations template)
       → Claude 3.5 Sonnet (Streaming)
       → Output Guardrails → Citation Extraction
       → Cache Store → Stream to User
```

---

## 12. Code Snippets

### 12.1 Complete Production RAG Service (FastAPI)

```python
"""
Production RAG Service — FastAPI on EKS
Features: Streaming, caching, guardrails, hybrid retrieval, reranking
"""

import asyncio
import json
import time
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

import boto3
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from opensearchpy import AsyncOpenSearch, AsyncHttpConnection

# ─── Configuration ───────────────────────────────────────────────
class Config:
    OPENSEARCH_ENDPOINT = "https://xyz.us-west-2.aoss.amazonaws.com"
    INDEX_NAME = "dish-rag-index"
    EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
    GENERATION_MODEL = "anthropic.claude-sonnet-4-20250514-v1:0"
    RERANK_MODEL = "cohere.rerank-v3-5:0"
    KB_ID = "YOUR_KNOWLEDGE_BASE_ID"
    GUARDRAIL_ID = "YOUR_GUARDRAIL_ID"
    CACHE_TTL_SECONDS = 3600
    RETRIEVAL_K = 50
    RERANK_TOP_N = 5
    SIMILARITY_CACHE_THRESHOLD = 0.95


# ─── Models ──────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    filters: Optional[dict] = None
    strategy: Optional[str] = "hybrid"  # hybrid | dense | graph
    stream: bool = True
    session_id: Optional[str] = None


class RAGResponse(BaseModel):
    answer: str
    citations: list[dict]
    metadata: dict


# ─── Service ─────────────────────────────────────────────────────
class RAGService:
    def __init__(self):
        self.bedrock = boto3.client('bedrock-runtime', region_name='us-west-2')
        self.bedrock_agent = boto3.client('bedrock-agent-runtime', region_name='us-west-2')
        self.opensearch = None  # Initialized in lifespan
    
    async def initialize(self):
        """Initialize async clients."""
        from opensearchpy import AWSV4SignerAsyncAuth
        credentials = boto3.Session().get_credentials()
        auth = AWSV4SignerAsyncAuth(credentials, 'us-west-2', 'aoss')
        
        self.opensearch = AsyncOpenSearch(
            hosts=[{'host': Config.OPENSEARCH_ENDPOINT.replace('https://', ''), 'port': 443}],
            http_auth=auth,
            use_ssl=True,
            connection_class=AsyncHttpConnection
        )
    
    async def embed(self, text: str) -> list[float]:
        """Generate embedding using Titan v2."""
        response = self.bedrock.invoke_model(
            modelId=Config.EMBEDDING_MODEL,
            body=json.dumps({
                "inputText": text,
                "dimensions": 1024,
                "normalize": True
            })
        )
        return json.loads(response["body"].read())["embedding"]
    
    async def hybrid_retrieve(self, query: str, filters: dict = None, k: int = 50):
        """Hybrid dense + sparse retrieval with metadata filtering."""
        query_embedding = await self.embed(query)
        
        # Build filter clause
        filter_clause = []
        if filters:
            for key, value in filters.items():
                filter_clause.append({"term": {f"metadata.{key}": value}})
        
        # Hybrid query (dense kNN + BM25)
        body = {
            "size": k,
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_embedding,
                                    "k": k
                                }
                            }
                        },
                        {
                            "match": {
                                "text": {"query": query}
                            }
                        }
                    ]
                }
            }
        }
        
        if filter_clause:
            body["query"] = {
                "bool": {
                    "must": [body["query"]],
                    "filter": filter_clause
                }
            }
        
        response = await self.opensearch.search(
            index=Config.INDEX_NAME, body=body
        )
        
        return [
            {
                "text": hit["_source"]["text"],
                "score": hit["_score"],
                "metadata": hit["_source"].get("metadata", {})
            }
            for hit in response["hits"]["hits"]
        ]
    
    async def rerank(self, query: str, documents: list[dict], top_n: int = 5):
        """Rerank using Cohere via Bedrock."""
        if not documents:
            return []
        
        response = self.bedrock.invoke_model(
            modelId=Config.RERANK_MODEL,
            body=json.dumps({
                "query": query,
                "documents": [d["text"] for d in documents],
                "top_n": top_n,
                "return_documents": True
            })
        )
        
        results = json.loads(response["body"].read())["results"]
        return [
            {
                **documents[r["index"]],
                "rerank_score": r["relevance_score"]
            }
            for r in results
        ]
    
    async def generate_stream(
        self, query: str, contexts: list[dict]
    ) -> AsyncGenerator[str, None]:
        """Stream response with Claude via Bedrock."""
        
        context_text = "\n\n".join([
            f"[{i+1}] {ctx['text']}\n(Source: {ctx['metadata'].get('source', 'unknown')})"
            for i, ctx in enumerate(contexts)
        ])
        
        prompt = f"""You are a technical assistant for DISH Network infrastructure.
Answer using ONLY the provided contexts. Cite sources using [N] notation.
If you cannot answer from the context, say so.

Contexts:
{context_text}

Question: {query}

Answer (with inline [N] citations):"""
        
        response = self.bedrock.invoke_model_with_response_stream(
            modelId=Config.GENERATION_MODEL,
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0.1,
                "anthropic_version": "bedrock-2023-05-31"
            })
        )
        
        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk["type"] == "content_block_delta":
                yield chunk["delta"].get("text", "")
    
    async def query(self, request: QueryRequest) -> AsyncGenerator[str, None]:
        """Full RAG pipeline: retrieve → rerank → generate."""
        start_time = time.time()
        
        # Step 1: Input guardrails
        # (In production, call Bedrock Guardrails API here)
        
        # Step 2: Retrieve
        candidates = await self.hybrid_retrieve(
            query=request.query,
            filters=request.filters,
            k=Config.RETRIEVAL_K
        )
        
        # Step 3: Rerank
        top_contexts = await self.rerank(
            query=request.query,
            documents=candidates,
            top_n=Config.RERANK_TOP_N
        )
        
        # Step 4: Generate (streaming)
        async for chunk in self.generate_stream(request.query, top_contexts):
            yield chunk
        
        # Step 5: Emit metadata (after stream)
        latency = (time.time() - start_time) * 1000
        yield f"\n\n---\n_Latency: {latency:.0f}ms | Sources: {len(top_contexts)}_"


# ─── FastAPI App ─────────────────────────────────────────────────
rag_service = RAGService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await rag_service.initialize()
    yield
    if rag_service.opensearch:
        await rag_service.opensearch.close()

app = FastAPI(title="DISH RAG Service", lifespan=lifespan)


@app.post("/query")
async def query_rag(request: QueryRequest):
    """Main RAG endpoint with streaming support."""
    if request.stream:
        return StreamingResponse(
            rag_service.query(request),
            media_type="text/event-stream"
        )
    else:
        # Collect full response
        chunks = []
        async for chunk in rag_service.query(request):
            chunks.append(chunk)
        return {"answer": "".join(chunks)}


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.0.0"}
```

### 12.2 Agentic RAG with Tool Use

```python
"""
Agentic RAG — LLM decides when and how to retrieve.
Integrates with MCP server pattern for tool-based retrieval.
"""

import json
from typing import Any

class AgenticRAG:
    """
    The LLM autonomously decides:
    1. Whether to retrieve (or answer from knowledge)
    2. Which retrieval strategy to use
    3. Whether retrieved context is sufficient
    4. Whether to iterate (multi-hop)
    """
    
    TOOLS = [
        {
            "name": "vector_search",
            "description": "Search documents using semantic similarity. Best for conceptual questions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "filters": {"type": "object", "description": "Metadata filters"},
                    "k": {"type": "integer", "default": 10}
                },
                "required": ["query"]
            }
        },
        {
            "name": "keyword_search",
            "description": "Search using exact keyword matching (BM25). Best for specific terms, IDs, error codes.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "fields": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["query"]
            }
        },
        {
            "name": "graph_query",
            "description": "Query knowledge graph for entity relationships. Best for 'who/what is connected to X' questions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "relation_type": {"type": "string"},
                    "hops": {"type": "integer", "default": 2}
                },
                "required": ["entity"]
            }
        },
        {
            "name": "sql_query",
            "description": "Query structured data in Aurora. Best for aggregations, counts, time-series.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Natural language question about structured data"}
                },
                "required": ["question"]
            }
        }
    ]
    
    def __init__(self, bedrock_client, retrieval_backends: dict):
        self.bedrock = bedrock_client
        self.backends = retrieval_backends
        self.max_iterations = 3
    
    async def run(self, query: str, conversation_history: list = None) -> dict:
        """Execute agentic RAG loop."""
        messages = conversation_history or []
        messages.append({"role": "user", "content": query})
        
        system_prompt = """You are an intelligent retrieval agent for DISH Network.
Your job is to find the most relevant information to answer user questions.

Strategy:
1. Analyze the question to determine what information is needed
2. Choose the appropriate tool(s) to retrieve that information
3. Assess if the retrieved information is sufficient
4. If not, formulate follow-up queries to fill gaps
5. When you have enough context, synthesize a comprehensive answer

Always cite your sources. If you cannot find sufficient information, say so."""
        
        all_contexts = []
        
        for iteration in range(self.max_iterations):
            response = self.bedrock.invoke_model(
                modelId="anthropic.claude-sonnet-4-20250514-v1:0",
                body=json.dumps({
                    "system": system_prompt,
                    "messages": messages,
                    "tools": self.TOOLS,
                    "max_tokens": 4096,
                    "anthropic_version": "bedrock-2023-05-31"
                })
            )
            
            result = json.loads(response["body"].read())
            
            # Check if model wants to use tools
            if result["stop_reason"] == "tool_use":
                tool_results = []
                for content_block in result["content"]:
                    if content_block["type"] == "tool_use":
                        tool_name = content_block["name"]
                        tool_input = content_block["input"]
                        
                        # Execute the tool
                        tool_output = await self._execute_tool(tool_name, tool_input)
                        all_contexts.extend(tool_output.get("results", []))
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block["id"],
                            "content": json.dumps(tool_output)
                        })
                
                # Add assistant message and tool results
                messages.append({"role": "assistant", "content": result["content"]})
                messages.append({"role": "user", "content": tool_results})
            
            elif result["stop_reason"] == "end_turn":
                # Model is done — extract final answer
                answer = "".join([
                    block["text"] for block in result["content"] 
                    if block["type"] == "text"
                ])
                
                return {
                    "answer": answer,
                    "contexts_used": all_contexts,
                    "iterations": iteration + 1,
                    "tools_called": self._count_tool_calls(messages)
                }
        
        return {"answer": "Max iterations reached.", "contexts_used": all_contexts}
    
    async def _execute_tool(self, tool_name: str, tool_input: dict) -> dict:
        """Route tool calls to appropriate backends."""
        backend = self.backends.get(tool_name)
        if not backend:
            return {"error": f"Unknown tool: {tool_name}"}
        return await backend.execute(tool_input)
```

### 12.3 Evaluation Runner

```python
"""
Automated RAG evaluation pipeline — runs in CI/CD.
Integrates with RAGAS for standardized metrics.
"""

import asyncio
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalResult:
    question: str
    generated_answer: str
    ground_truth: str
    contexts: list[str]
    faithfulness: float
    relevancy: float
    context_precision: float
    context_recall: float
    latency_ms: float


class RAGEvaluator:
    """Production evaluation pipeline with CI/CD integration."""
    
    GOLDEN_DATASET_PATH = "s3://dish-rag-eval/golden_dataset.jsonl"
    
    FAITHFULNESS_PROMPT = """Given the context and the answer, determine if 
every claim in the answer is supported by the context.

Context: {context}

Answer: {answer}

For each sentence in the answer, determine if it is:
- SUPPORTED: directly supported by the context
- NOT_SUPPORTED: not found in or contradicted by the context

Output JSON: {{"supported": N, "not_supported": M, "total": T}}"""

    RELEVANCY_PROMPT = """Given the question and the answer, determine if 
the answer addresses the question. Score 0-1.

Question: {question}
Answer: {answer}

Score (0.0 to 1.0):"""
    
    def __init__(self, rag_service, bedrock_client):
        self.rag = rag_service
        self.bedrock = bedrock_client
    
    async def evaluate_dataset(self, dataset_path: str = None) -> dict:
        """Run evaluation on golden dataset."""
        dataset = self._load_dataset(dataset_path or self.GOLDEN_DATASET_PATH)
        results = []
        
        for item in dataset:
            result = await self._evaluate_single(item)
            results.append(result)
        
        # Aggregate metrics
        metrics = {
            "faithfulness": np.mean([r.faithfulness for r in results]),
            "answer_relevancy": np.mean([r.relevancy for r in results]),
            "context_precision": np.mean([r.context_precision for r in results]),
            "context_recall": np.mean([r.context_recall for r in results]),
            "avg_latency_ms": np.mean([r.latency_ms for r in results]),
            "p99_latency_ms": np.percentile([r.latency_ms for r in results], 99),
            "total_questions": len(results),
        }
        
        # CI/CD gate
        thresholds = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.80,
            "context_precision": 0.75,
            "context_recall": 0.80,
            "p99_latency_ms": 5000,  # 5s SLA
        }
        
        failures = []
        for metric, threshold in thresholds.items():
            if metric == "p99_latency_ms":
                if metrics[metric] > threshold:
                    failures.append(f"{metric}: {metrics[metric]:.0f}ms > {threshold}ms")
            else:
                if metrics[metric] < threshold:
                    failures.append(f"{metric}: {metrics[metric]:.3f} < {threshold}")
        
        return {
            "metrics": metrics,
            "passed": len(failures) == 0,
            "failures": failures,
            "details": [vars(r) for r in results]
        }
    
    async def _evaluate_single(self, item: dict) -> EvalResult:
        """Evaluate a single question."""
        import time
        
        start = time.time()
        response = await self.rag.query_sync(item["question"])
        latency = (time.time() - start) * 1000
        
        # Compute metrics using LLM-as-judge
        faithfulness = await self._compute_faithfulness(
            response["answer"], response["contexts"]
        )
        relevancy = await self._compute_relevancy(
            item["question"], response["answer"]
        )
        precision = await self._compute_context_precision(
            item["question"], response["contexts"], item["ground_truth"]
        )
        recall = await self._compute_context_recall(
            response["contexts"], item["ground_truth"]
        )
        
        return EvalResult(
            question=item["question"],
            generated_answer=response["answer"],
            ground_truth=item["ground_truth"],
            contexts=[c["text"] for c in response["contexts"]],
            faithfulness=faithfulness,
            relevancy=relevancy,
            context_precision=precision,
            context_recall=recall,
            latency_ms=latency
        )
    
    async def _compute_faithfulness(self, answer: str, contexts: list) -> float:
        """Check if answer is grounded in retrieved contexts."""
        context_text = "\n".join([c["text"] for c in contexts])
        
        response = self.bedrock.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": 
                    self.FAITHFULNESS_PROMPT.format(
                        context=context_text[:3000],
                        answer=answer
                    )}],
                "max_tokens": 200,
                "anthropic_version": "bedrock-2023-05-31"
            })
        )
        
        result_text = json.loads(response["body"].read())["content"][0]["text"]
        scores = json.loads(result_text)
        return scores["supported"] / scores["total"] if scores["total"] > 0 else 0.0


# ─── CI/CD Integration ───────────────────────────────────────────
async def main():
    """Entry point for CI/CD pipeline."""
    evaluator = RAGEvaluator(rag_service=..., bedrock_client=...)
    results = await evaluator.evaluate_dataset()
    
    print(json.dumps(results["metrics"], indent=2))
    
    if not results["passed"]:
        print("\n❌ EVALUATION FAILED:")
        for failure in results["failures"]:
            print(f"  • {failure}")
        exit(1)
    else:
        print("\n✅ All metrics passed thresholds")
        exit(0)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Quick Reference Card

### Performance Benchmarks (Target SLAs)

| Metric | Target | Achievable With |
|--------|--------|-----------------|
| End-to-end latency (p50) | < 2s | Caching + streaming |
| End-to-end latency (p99) | < 5s | Pre-warm + connection pooling |
| Retrieval recall@10 | > 0.85 | Hybrid + reranking |
| Faithfulness | > 0.90 | Guardrails + grounding check |
| Cache hit rate | > 40% | Semantic caching (0.95 threshold) |
| Cost per query | < $0.01 | Haiku for eval, Sonnet for gen |

### Technology Selection Summary (DISH Context)

| Component | Primary Choice | Alternative |
|-----------|---------------|-------------|
| **Vector Store** | OpenSearch Serverless | Qdrant on EKS (performance-critical) |
| **Embeddings** | Titan Embed v2 (1024d) | BGE-M3 on EKS (hybrid sparse+dense) |
| **Generation** | Claude 3.5 Sonnet (Bedrock) | Claude 3 Haiku (low-latency) |
| **Reranking** | Cohere Rerank v3 (Bedrock) | BGE-Reranker-v2 on EKS |
| **Knowledge Graph** | Neptune + OpenSearch | Neo4j on EKS |
| **Cache** | ElastiCache Redis | DynamoDB + DAX |
| **Orchestration** | Bedrock KB (managed) | Custom (LangChain/LlamaIndex on EKS) |
| **Evaluation** | RAGAS + custom pipeline | DeepEval (pytest integration) |
| **Observability** | CloudWatch + LangSmith | Langfuse (OSS, self-hosted) |
| **Guardrails** | Bedrock Guardrails | NeMo Guardrails (custom) |

---

## Further Reading

- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — The contextual chunking paper
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag) — Community-based summarization
- [RAGAS Documentation](https://docs.ragas.io/) — Evaluation framework
- [AWS Bedrock KB Custom Chunking](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-chunking-parsing.html)
- [ColBERT v2 Paper](https://arxiv.org/abs/2112.01488) — Late-interaction retrieval
- [Self-RAG Paper](https://arxiv.org/abs/2310.11511) — Adaptive retrieval with self-reflection
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) — Position bias in long contexts
