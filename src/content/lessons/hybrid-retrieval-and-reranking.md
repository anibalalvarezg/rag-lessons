---
title: "Hybrid Retrieval y Reranking"
subtitle: "Lección 4 — Recuperación Avanzada"
pillar: recuperacion
pillarName: "Recuperación Avanzada"
lessonNum: 4
description: "Retrieve + rerank, Cohere/BGE/Jina, HyDE, Multi-Query, pipeline completo."
keywords: "hybrid retrieval, reranking, BM25, dense, Cohere, BGE, Jina, HyDE"
ogSection: "Recuperación Avanzada"
pubDate: "2026-07-24"
quizzes:

  - id: "q1"
    question: "Por que un reranker es mas preciso que un bi-encoder?"
    options:
      - text: "Porque es un modelo mas grande"
        correct: false
      - text: "Porque lee query y documento juntos con atencion completa"
        correct: true
      - text: "Porque usa BM25 en lugar de embeddings"
        correct: false
      - text: "Porque opera sobre todo el corpus"
        correct: false
  - id: "q2"
    question: "Que arregla HyDE?"
    options:
      - text: "Chunks mal divididos"
        correct: false
      - text: "Gap de vocabulario entre la pregunta y los documentos"
        correct: true
      - text: "Embeddings de baja calidad"
        correct: false
      - text: "El LLM alucina informacion"
        correct: false
  - id: "q3"
    question: "Cual es el orden correcto del pipeline de produccion?"
    options:
      - text: "Retrieve -> Transform -> Rerank -> Generate"
        correct: false
      - text: "Transform -> Retrieve -> Rerank -> Generate"
        correct: true
      - text: "Rerank -> Retrieve -> Transform -> Generate"
        correct: false
      - text: "Transform -> Rerank -> Retrieve -> Generate"
        correct: false
  - id: "q4"
    question: "Cuando usar Query Decomposition?"
    options:
      - text: "Siempre, mejora todo tipo de queries"
        correct: false
      - text: "Solo con queries ambiguas"
        correct: false
      - text: "Cuando la pregunta requiere informacion de multiples documentos"
        correct: true
      - text: "Cuando el corpus es muy grande"
        correct: false

---


## Objetivo

Al finalizar esta leccion, entenderas el patron de dos etapas (retrieve + rerank), sabras elegir un reranker, y conoceras las tecnicas de query transformation (HyDE, Multi-Query, Decomposition) para mejorar recall.

## El patron de dos etapas

En produccion, la recuperacion nunca es un solo paso. Es un funnel:

1. **Stage 1 - Retrieval rapido:** Busca 50-100 candidatos usando hybrid search (dense + BM25). Rapido pero impreciso.

2. **Stage 2 - Reranking preciso:** Un cross-encoder re-scorea cada candidato leyendo query + chunk juntos. Lento pero muy preciso. Conserva top 3-5.

<div class="callout info">
<div class="callout-title">Por que dos etapas?</div>
<p>Un bi-encoder (embedding) compara query y documento por separado: es rapido pero approximate. Un cross-encoder (reranker) los lee juntos con atencion completa: es lento pero preciso. La combinacion maximiza calidad con latencia controlada.</p>
</div>

## Rerankers: Los mejores modelos de 2026

### Hosted (API)

<small>Fuente de scores ELO: [Cohere Rerank Blog](https://cohere.com/blog/rerank) y [Ranker Arena (2026)](https://livecopilot.com/ranker-arena)</small>

| Modelo | ELO | Latencia | Costo | Mejor para |
| --- | --- | --- | --- | --- |
| **Cohere Rerank 3.5** | 1629 | ~600ms | $2.00/1k busquedas | Default sin ops, 100+ idiomas |
| **Voyage Rerank 2.5** | 1544 | ~613ms | $0.05/M tokens | Variantes dominio (codigo, legal) |

### Self-hosted (open source)
| Modelo | Parametros | Latencia (GPU) | Recall@5 lift | Mejor para |
| --- | --- | --- | --- | --- |
| **BGE Reranker v2-M3** | 568M | ~84ms | +18.4% | Multilingue, bajo costo, rapido |
| **Qwen3 Reranker 4B** | 4B | ~312ms | +27.1% | Calidad maxima local |
| **Jina Reranker v3** | 600M | ~188ms | 81.3% Hit@1 | Sub-200ms, documentos largos (131k ctx) |

<div class="callout success">
<div class="callout-title">El lever mas barato en RAG</div>
<p>El reranker es "the cheapest large win left in RAG". No necesitas re-embedder ni reconstruir el indice. Solo insertas un paso de re-scoring entre retrieval y LLM. En benchmarks de Azure, hybrid + reranker da +37% NDCG vs vector-only.

<small>Fuente: <a href="https://techcommunity.microsoft.com/blog/azure-ai-services-blog/azure-ai-search-retrieval-augmented-generation-rag-with-hybrid-search/4376267">Microsoft Azure AI Blog — Hybrid Search + Reranking benchmarks</a></small></p>
</div>

## Implementacion con LangChain
```
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# Reranker hosted
reranker = CohereRerank(model="rerank-v3.5", top_n=5)

# Envolver el retriever base
compression_retriever = ContextualCompressionRetriever(
base_compressor=reranker,
base_retriever=base_retriever  # tu hybrid retriever
)

# Ahora retrieval + reranking en un solo paso
results = compression_retriever.invoke("pregunta del usuario")
```

## Query Transformation: Arreglar la pregunta, no el indice

Muchas veces el retrieval falla no por el indice, sino por la pregunta del usuario. Las queries son cortas, ambiguas, o usan vocabulario diferente al del corpus.

### 1. HyDE (Hypothetical Document Embeddings)

Pide al LLM que escriba una respuesta hipotetica, embebe esa respuesta, y busca con ella. La intuicion: una respuesta real y una hipotetica viven en la misma region del espacio de embeddings; la pregunta vive en otra.

```
# Flujo HyDE
# 1. Usuario pregunta: "Como configurar el timeout?"
# 2. LLM genera respuesta hipotetica:
#    "El timeout se configura en settings.yaml bajo
#     network.timeout. El valor por defecto es 30 segundos..."
# 3. Embeber esa respuesta hipotetica
# 4. Buscar documentos similares al embedding hipotetico
```

- **Cuando usarlo:** Gap de vocabulario entre preguntas y documentos, queries genericas

- **Riesgo:** En dominios niche donde el LLM no sabe, la hipotesis puede ser incorrecta y empeorar retrieval

- **Costo:** +1 LLM call (~200-500ms)

### 2. Multi-Query / RAG-Fusion

Genera N parrafasis de la query, busca con cada una, fusiona resultados con Reciprocal Rank Fusion.

```
# Flujo Multi-Query
# Query original: "cancelar suscripcion"
# Parrafasis generadas:
#   - "terminar plan de servicio"
#   - "darse de baja del servicio"
#   - "cerrar cuenta activa"
# Cada una recupera top-k, se fusionan con RRF
```

- **Cuando usarlo:** Queries ambiguas, vocabulario del usuario lejos del corpus

- **Ganancia tipica:** +8-10% accuracy, +30-40% comprehensiveness

- **Costo:** Nx retrieval (N=3 es comun)

### 3. Query Decomposition

Preguntas multi-hop se dividen en sub-preguntas atomicas, cada una busca por separado.

```
# Query: "Cual es la politica de reembolsos para clientes EU vs US?"
# Decomposicion:
#   1. "Politica de reembolsos clientes EU"
#   2. "Politica de reembolsos clientes US"
# Cada sub-query busca documentos independientes
```

- **Cuando usarlo:** Preguntas que requieren informacion de multiples documentos

- **Riesgo:** Over-decomposition (dividir preguntas simples en sub-preguntas innecesarias)

## El pipeline completo de produccion
```
# Pipeline tipico 2026
query = usuario.pregunta

# 1. Query Transformation (opcional, selectivo)
if es_ambigua(query):
queries = multi_query_rewrite(query)  # 3 parrafasis
elif es_multi_hop(query):
queries = decompose(query)  # sub-queries
else:
queries = [query]

# 2. Hybrid Retrieval (dense + BM25 + RRF)
candidates = []
for q in queries:
dense_results = vector_store.similarity_search(q, k=50)
sparse_results = bm25_search(q, k=50)
candidates.extend(rrf_fusion(dense_results, sparse_results))

# 3. Reranking (cross-encoder)
reranked = reranker.rerank(query, candidates[:100], top_n=5)

# 4. Generation con citations
answer = llm.generate(
context=reranked,
prompt="Responde usando SOLO el contexto proporcionado. Cita fuentes."
)
```

<div class="callout warning">
<div class="callout-title">Regla de oro: orden correcto</div>
<p><strong>Transform -> Retrieve -> Rerank -> Generate</strong>. La transformacion amplia el recall, el reranker mejora la precision. Son complementarios, no competidores.</p>
</div>

## Practica


<div class="exercise">
<div class="exercise-title">Ejercicio 1: Agrega un reranker</div>
<p>Usando el pipeline del Capitulo 1 como base, agrega un reranker Cohere o BGE:</p>
<ul>
<li>Recupera top-50 candidatos</li>
<li>Rerankea a top-5</li>
<li>Compara la calidad de la respuesta con y sin reranker</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 2: Implementa Multi-Query</div>
<p>Usa <code>MultiQueryRetriever</code> de LangChain:</p>
<ul>
<li>Genera 3 parrafasis de una pregunta ambigua</li>
<li>Recupera con cada parrafasis</li>
<li>Fusiona resultados con RRF</li>
<li>Mide: ¿mejora el Recall@5?</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 3: Cadena completa</div>
<p>Construye el pipeline completo: Query Rewrite -> Hybrid Search -> Rerank -> Generate. Mide latencia por etapa.</p>
</div>
