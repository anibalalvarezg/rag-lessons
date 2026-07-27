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
    question: "¿Por qué un reranker es más preciso que un bi-encoder?"
    options:
      - text: "Porque es un modelo más grande"
        correct: false
      - text: "Porque lee query y documento juntos con atención completa"
        correct: true
      - text: "Porque usa BM25 en lugar de embeddings"
        correct: false
      - text: "Porque opera sobre todo el corpus"
        correct: false
  - id: "q2"
    question: "¿Qué arregla HyDE?"
    options:
      - text: "Chunks mal divididos"
        correct: false
      - text: "Gap de vocabulario entre la pregunta y los documentos"
        correct: true
      - text: "Embeddings de baja calidad"
        correct: false
      - text: "El LLM alucina información"
        correct: false
  - id: "q3"
    question: "¿Cuál es el orden correcto del pipeline de producción?"
    options:
      - text: "Retrieve → Transform → Rerank → Generate"
        correct: false
      - text: "Transform → Retrieve → Rerank → Generate"
        correct: true
      - text: "Rerank → Retrieve → Transform → Generate"
        correct: false
      - text: "Transform → Rerank → Retrieve → Generate"
        correct: false
  - id: "q4"
    question: "¿Cuándo usar Query Decomposition?"
    options:
      - text: "Siempre, mejora todo tipo de queries"
        correct: false
      - text: "Solo con queries ambiguas"
        correct: false
      - text: "Cuando la pregunta requiere información de múltiples documentos"
        correct: true
      - text: "Cuando el corpus es muy grande"
        correct: false

---


## Objetivo

Al finalizar esta lección, entenderás el patrón de dos etapas (retrieve + rerank), sabrás elegir un reranker, y conocerás las técnicas de query transformation (HyDE, Multi-Query, Decomposition) para mejorar recall.

**Prerequisitos:** [Lección 3](/rag-lessons/lessons/embeddings-and-vector-stores) — tu índice `acme_docs` ya está poblado.

<div class="callout info">
<div class="callout-title">🧵 Proyecto Acme — De dónde viene este código</div>
<p>El <code>base_retriever</code> que envuelve el reranker es el <code>vectorstore.as_retriever()</code> que construiste en la Lección 3. En esta lección además indexas el segundo documento del corpus: <code>manual-tecnico.pdf</code> (donde vive la configuración de <code>settings.yaml</code>). Los ejemplos de "¿Cómo configurar el timeout?" de aquí en adelante son consultas de empleados de Acme sobre ese manual — y son el mismo caso que verificarás en la Lección 5 y evaluarás en la Lección 6.</p>
</div>

## El patrón de dos etapas

En producción, la recuperación nunca es un solo paso. Es un funnel:

1. **Stage 1 - Retrieval rápido:** Busca 50-100 candidatos usando hybrid search (dense + BM25). Rápido pero impreciso.

2. **Stage 2 - Reranking preciso:** Un cross-encoder re-scorea cada candidato leyendo query + chunk juntos. Lento pero muy preciso. Conserva top 3-5.

<div class="callout info">
<div class="callout-title">¿Por qué dos etapas?</div>
<p>Un bi-encoder (embedding) compara query y documento por separado: es rápido pero aproximado. Un cross-encoder (reranker) los lee juntos con atención completa: es lento pero preciso. La combinación maximiza calidad con latencia controlada.</p>
</div>

## Rerankers: Los mejores modelos de 2026

### Hosted (API)

<small>Fuente de scores ELO: [Cohere Rerank Blog](https://cohere.com/blog/rerank) y [Ranker Arena (2026)](https://livecopilot.com/ranker-arena)</small>

| Modelo | ELO | Latencia | Costo | Mejor para |
| --- | --- | --- | --- | --- |
| **Cohere Rerank 3.5** | 1629 | ~600ms | $2.00/1k búsquedas | Default sin ops, 100+ idiomas |
| **Voyage Rerank 2.5** | 1544 | ~613ms | $0.05/M tokens | Variantes de dominio (código, legal) |

### Self-hosted (open source)
| Modelo | Parámetros | Latencia (GPU) | Recall@5 lift | Mejor para |
| --- | --- | --- | --- | --- |
| **BGE Reranker v2-M3** | 568M | ~84ms | +18.4% | Multilingüe, bajo costo, rápido |
| **Qwen3 Reranker 4B** | 4B | ~312ms | +27.1% | Calidad máxima local |
| **Jina Reranker v3** | 600M | ~188ms | 81.3% Hit@1 | Sub-200ms, documentos largos (131k ctx) |

<div class="callout success">
<div class="callout-title">El lever más barato en RAG</div>
<p>El reranker es "the cheapest large win left in RAG". No necesitas re-embedder ni reconstruir el índice. Solo insertas un paso de re-scoring entre retrieval y LLM. En benchmarks de Azure, hybrid + reranker da +37% NDCG vs vector-only.

<small>Fuente: <a href="https://techcommunity.microsoft.com/blog/azure-ai-services-blog/azure-ai-search-retrieval-augmented-generation-rag-with-hybrid-search/4376267">Microsoft Azure AI Blog — Hybrid Search + Reranking benchmarks</a></small></p>
</div>

## Implementación con LangChain

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# Reranker hosted
reranker = CohereRerank(model="rerank-v3.5", top_n=5)

# Envolver el retriever base (el de la Lección 3 sobre acme_docs)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever  # tu hybrid retriever
)

# Ahora retrieval + reranking en un solo paso
results = compression_retriever.invoke("¿Cómo configurar el timeout?")
```

## Query Transformation: Arreglar la pregunta, no el índice

Muchas veces el retrieval falla no por el índice, sino por la pregunta del usuario. Las queries son cortas, ambiguas, o usan vocabulario diferente al del corpus.

### 1. HyDE (Hypothetical Document Embeddings)

Pide al LLM que escriba una respuesta hipotética, embebe esa respuesta, y busca con ella. La intuición: una respuesta real y una hipotética viven en la misma región del espacio de embeddings; la pregunta vive en otra.

```python
# Flujo HyDE
# 1. Usuario pregunta: "¿Cómo configurar el timeout?"
# 2. LLM genera respuesta hipotética:
#    "El timeout se configura en settings.yaml bajo
#     network.timeout. El valor por defecto es 30 segundos..."
# 3. Embeber esa respuesta hipotética
# 4. Buscar documentos similares al embedding hipotético
```

- **Cuándo usarlo:** Gap de vocabulario entre preguntas y documentos, queries genéricas

- **Riesgo:** En dominios nicho donde el LLM no sabe, la hipótesis puede ser incorrecta y empeorar retrieval

- **Costo:** +1 LLM call (~200-500ms)

### 2. Multi-Query / RAG-Fusion

Genera N paráfrasis de la query, busca con cada una, fusiona resultados con Reciprocal Rank Fusion.

```python
# Flujo Multi-Query
# Query original: "cancelar suscripción"
# Paráfrasis generadas:
#   - "terminar plan de servicio"
#   - "darse de baja del servicio"
#   - "cerrar cuenta activa"
# Cada una recupera top-k, se fusionan con RRF
```

- **Cuándo usarlo:** Queries ambiguas, vocabulario del usuario lejos del corpus

- **Ganancia típica:** +8-10% accuracy, +30-40% comprehensiveness

- **Costo:** Nx retrieval (N=3 es común)

### 3. Query Decomposition

Preguntas multi-hop se dividen en sub-preguntas atómicas, cada una busca por separado.

```python
# Query: "¿Cuál es la política de devoluciones para clientes EU vs US?"
# Descomposición:
#   1. "Política de devoluciones clientes EU"
#   2. "Política de devoluciones clientes US"
# Cada sub-query busca documentos independientes
```

- **Cuándo usarlo:** Preguntas que requieren información de múltiples documentos

- **Riesgo:** Over-decomposition (dividir preguntas simples en sub-preguntas innecesarias)

## El pipeline completo de producción

```python
# Pipeline típico 2026
query = usuario.pregunta

# 1. Query Transformation (opcional, selectivo)
if es_ambigua(query):
    queries = multi_query_rewrite(query)  # 3 paráfrasis
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
<p><strong>Transform → Retrieve → Rerank → Generate</strong>. La transformación amplía el recall, el reranker mejora la precisión. Son complementarios, no competidores.</p>
</div>

## Práctica


<div class="exercise">
<div class="exercise-title">Ejercicio 1: Agrega un reranker</div>
<p>Usa el código de reranking de arriba como base y modifícalo para:</p>
<ul>
<li>Recupera top-50 candidatos</li>
<li>Rerankea a top-5</li>
<li>Compara la calidad de la respuesta con y sin reranker</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 2: Implementa Multi-Query</div>
<p>Usa el código de Multi-Query de arriba:</p>
<ul>
<li>Genera 3 paráfrasis de una pregunta ambigua</li>
<li>Recupera con cada paráfrasis</li>
<li>Fusiona resultados con RRF</li>
<li>Mide: ¿mejora el Recall@5?</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 3: Cadena completa</div>
<p>Construye el pipeline completo: Query Rewrite → Hybrid Search → Rerank → Generate. Mide latencia por etapa.</p>
</div>