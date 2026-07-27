---
title: "Embeddings y Vector Stores"
subtitle: "Lección 3 — Ingesta y Segmentación"
pillar: ingesta
pillarName: "Ingesta y Segmentación"
lessonNum: 3
description: "Modelos de embedding 2026, Matryoshka, hybrid retrieval, LanceDB vs pgvector."
keywords: "embeddings, vector store, Matryoshka, LanceDB, pgvector, HNSW"
ogSection: "Ingesta y Segmentación"
pubDate: "2026-07-24"
quizzes:

  - id: "q1"
    question: "¿Qué métrica mide la similitud entre dos embeddings?"
    options:
      - text: "Euclidean distance"
        correct: false
      - text: "Similitud de coseno"
        correct: true
      - text: "Jaccard index"
        correct: false
      - text: "BLEU score"
        correct: false
  - id: "q2"
    question: "¿Qué es Matryoshka en embeddings?"
    options:
      - text: "Un modelo de embedding ruso"
        correct: false
      - text: "Capacidad de reducir dimensiones al inferir, sin re-embedder"
        correct: true
      - text: "Una técnica de compresión de chunks"
        correct: false
      - text: "Un tipo de vector store"
        correct: false
  - id: "q3"
    question: "¿Cuándo elegir Qdrant sobre pgvector?"
    options:
      - text: "Siempre, es mejor tecnología"
        correct: false
      - text: "Solo para prototipos"
        correct: false
      - text: "Cuando necesitas filtrado por metadata complejo o > 5M vectores"
        correct: true
      - text: "Cuando usas Python"
        correct: false
  - id: "q4"
    question: "¿Por qué hybrid retrieval supera a búsqueda pura vectorial?"
    options:
      - text: "Porque es más rápido"
        correct: false
      - text: "Combina similitud semántica (dense) con coincidencias exactas (BM25 sparse)"
        correct: true
      - text: "Porque usa un modelo más grande"
        correct: false
      - text: "Porque reduce el costo de embedding"
        correct: false

---


## Objetivo

Al finalizar esta lección, podrás elegir un modelo de embedding y una vector store para tu caso de uso, entender cómo los embeddings representan significado, y saber cuándo migrar de pgvector a una base dedicada.

**Prerequisitos:** [Lección 2](/rag-lessons/lessons/chunking-strategies) — ya tienes la variable `chunks` del corpus Acme. La intuición de vectores y coseno está en el [Anexo A](/rag-lessons/lessons/anexo-tokens-embeddings).

<div class="callout info">
<div class="callout-title">🧵 Proyecto Acme — De dónde viene este código</div>
<p>La variable <code>chunks</code> que usa este código es la que produjiste en la Lección 2 (parsing PyMuPDF + recursive splitter sobre <code>politicas.pdf</code>). Aquí la embebes y la guardas en la colección <code>acme_docs</code> — la misma que creaste en la Lección 1. Al terminar, tu índice queda listo para el retrieval avanzado de la Lección 4.</p>
</div>

## Qué es un embedding

Un embedding es un vector numérico de alta dimensión (típicamente 768-3072 dimensiones) que representa el significado semántico de un texto. Textos con significado similar quedan cerca en el espacio vectorial.

```text
# Ejemplo conceptual
"El gato duerme en el sofá" -> [0.23, -0.15, 0.87, ..., 0.42]  # 3072 dims
"Una mascota descansa en un mueble" -> [0.21, -0.12, 0.85, ..., 0.44]  # similar!
"Python es un lenguaje de programación" -> [-0.78, 0.91, -0.34, ..., 0.12]  # diferente
```

La métrica clave es la **similitud de coseno**: mide el ángulo entre dos vectores. A mayor similitud (más cercano a 1), más relacionados están los textos.

Pruébalo con embeddings reales (es el mismo script de la Lección 0, ahora sobre el corpus Acme):

```python
import numpy as np
from openai import OpenAI

client = OpenAI()

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

oraciones = [
    "Las devoluciones se aceptan hasta 30 días después de la compra.",
    "Puedes devolver un producto si no pasaron más de 30 días.",
    "El timeout por defecto es de 30 segundos.",
]

vectors = [
    np.array(v.embedding)
    for v in client.embeddings.create(model="text-embedding-3-small", input=oraciones).data
]

print(cosine_similarity(vectors[0], vectors[1]))  # alto: misma idea, otras palabras
print(cosine_similarity(vectors[0], vectors[2]))  # bajo: mismo número "30", otro tema
```

Este es el código al que se refiere el Ejercicio 1 de práctica.

## El modelo de embedding importa MÁS que el chunking

<div class="callout warning">
<div class="callout-title">Dato que cambia prioridades</div>
<p>En benchmarks de 2026, cambiar de <code>text-embedding-3-small</code> a <code>text-embedding-3-large</code> produjo un swing de hasta 10.58% en recall. La diferencia entre recursive y semantic chunking fue de 2-9%. <strong>Optimiza el embedding antes que el chunking.</strong>

<small>Fuente: <a href="https://www.runvecta.com/blog/we-benchmarked-7-chunking-strategies-most-advice-was-wrong">Vecta Benchmark (Feb 2026) — Voyage-3-large supera a OpenAI 3-large por 10.58%</a></small></p>
</div>

## El landscape de modelos en 2026

### APIs (hosted)

<small>Fuente de scores MTEB: [MTEB Leaderboard (HuggingFace, 2026)](https://huggingface.co/spaces/mteb/leaderboard) — Los valores pueden variar según la versión del benchmark.</small>

| Modelo | MTEB | Costo/1M tokens | Dimensiones | Contexto | Mejor para |
| --- | --- | --- | --- | --- | --- |
| **Voyage-3.5** | Top retrieval | $0.06 | 1024 | 32K | Calidad general, mejor precio/calidad |
| **OpenAI text-embedding-3-large** | 64.6 | $0.13 | 3072 | 8K | Ecosistema OpenAI, MRL (reducir dimensiones) |
| **OpenAI text-embedding-3-small** | Buena | $0.02 | 1536 | 8K | Presupuesto ajustado, RAG básico |
| **Gemini Embedding 2** | ~1605 ELO | $0.15 | 3072 | 32K | Multimodal, multilingüe, documentos largos |
| **Cohere Embed v4** | 65.2 | $0.12 | 1536 | 128K | Multimodal (texto + imágenes), enterprise |

### Open Source (self-hosted)

<small>Fuente de scores MTEB: [MTEB Leaderboard (HuggingFace, 2026)](https://huggingface.co/spaces/mteb/leaderboard)</small>

| Modelo | MTEB | Parámetros | Mejor para |
| --- | --- | --- | --- |
| **Qwen3-Embedding-8B** | 70.6 | 8B | Calidad máxima, multilingüe |
| **BGE-M3** | 63.2 | 568M | Multilingüe, hybrid retrieval (dense + sparse) |
| **Nomic Embed v2** | 61.4 | 137M | Ultra-liviano, on-device |

<div class="callout info">
<div class="callout-title">Matryoshka Embeddings</div>
<p>Modelos como OpenAI 3-large y Cohere v4 soportan <strong>Matryoshka</strong>: puedes reducir dimensiones de 3072 a 1024 o 512 al momento de inferencia, sin re-embedder. Pierdes ~2% de precisión pero reduces almacenamiento 4-6x.</p>
</div>

## Dimensionalidad y almacenamiento
| Dimensiones | Precisión | Almacenamiento (1M docs) | Latencia búsqueda |
| --- | --- | --- | --- |
| 256 | 94.2% | ~1 GB | 5ms |
| 512 | 96.8% | ~2 GB | 8ms |
| 1024 | 98.1% | ~4 GB | 15ms |
| 3072 | 98.5% | ~12 GB | 42ms |

## Vector Databases: ¿Cuándo usar cada una?

### pgvector (El default para la mayoría)

Una extensión de Postgres. Los vectores viven en la misma tabla que tus datos. No necesitas un servicio nuevo.

- **Cuándo usarlo:** Ya usas Postgres, corpus &lt; 5M vectores, primer RAG system

- **Latencia:** 5-20ms p50 con HNSW (1M vectores)

- **Costo:** ~$0-25/mes en Supabase (absorbido en Postgres existente)

- **Limitación:** &gt; 10M vectores empieza a degradarse

### Pinecone (Managed, zero-ops)

Base de datos vectorial fully managed. No tocas infraestructura.

- **Cuándo usarlo:** Sin capacidad de ops, necesitas sub-20ms p95 a escala, prototipos rápidos

- **Latencia:** sub-5ms p50 (1M), 5-15ms (10M)

- **Costo:** ~$50-80/mes (1M vectores, 100K queries/mes)

- **Limitación:** Se vuelve caro a escala ($700+/mes a 10M vectores)

### Qdrant (Performance + open source)

Escrito en Rust. Filtering de payloads es el mejor de la categoría.

- **Cuándo usarlo:** Filtrado por metadata complejo, necesitas throughput alto, open source

- **Latencia:** sub-5ms p50 (1M), mejor filtrado que pgvector a escala

- **Costo:** $0 (free tier 1GB), o ~$36/mes en cloud para 1M vectores

- **Limitación:** Necesitas Docker/K8s para self-host

### Weaviate (Multimodal + hybrid search)

BM25 + vector search nativo. Módulos de vectorización built-in.

- **Cuándo usarlo:** Búsqueda híbrida nativa, multi-tenant SaaS, multimodal

- **Costo:** ~$25-150/mes (cloud)

- **Limitación:** Más pesado que Qdrant, breaking changes v3 a v4

<div class="callout success">
<div class="callout-title">Recomendación rápida 2026</div>
<p><strong>Empieza con pgvector</strong> si ya usas Postgres. Es la decisión correcta para el 80% de los casos. Migra a Qdrant o Pinecone cuando necesites más de 5M vectores o filtrado avanzado.</p>
</div>

## Implementación con LangChain

```python
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

# 1. Configurar el modelo de embedding
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1536  # Matryoshka: reducir de 3072 a 1536
)

# 2. Crear/consultar la vector store (chunks viene de la Lección 2)
vectorstore = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    connection_string="postgresql://rag:rag@localhost:5432/rag",
    collection_name="acme_docs"
)

# 3. Buscar
results = vectorstore.similarity_search(
    "¿Cuál es la política de devoluciones?",
    k=5
)
```

## El patrón de producción: Hybrid Retrieval

En 2026, la búsqueda pura vectorial se considera un antipatrón fuera de casos triviales. El patrón estándar combina:

- **Dense (vectorial):** Entiende paráfrasis y similitud semántica

- **Sparse (BM25):** Coincidencias exactas (SKUs, nombres propios, siglas)

Se fusionan con **Reciprocal Rank Fusion (RRF)**:

```python
score_final = sum(1 / (k + rank_i))  # para cada fuente (dense, sparse)
# k=60 es el parámetro común
# RRF no requiere normalizar scores entre sistemas diferentes
```

<div class="callout info">
<div class="callout-title">Ganancia medida</div>
<p>Hybrid retrieval mejora típicamente 5-15% recall@5 sobre búsqueda pura vectorial. La ganancia es mayor en queries con entidades nombradas, acrónimos, o códigos de producto.</p>
</div>

## Migración de embedding models

Cambiar el modelo de embedding requiere re-embedder TODO tu corpus. Esto puede tomar horas o días en corpus grandes.

- Versiona tus embeddings: guarda el nombre del modelo + versión junto a cada vector

- Diseña tu camino de migración: re-embedde en background mientras sirves el índice viejo, luego cambia atómicamente

- Los equipos que no planean migración quedan atrapados en su primer modelo por años

## Práctica


<div class="exercise">
<div class="exercise-title">Ejercicio 1: Explora embeddings</div>
<p>Ejecuta el código de embeddings de arriba. Calcula similitud entre oraciones.</p>
<ul>
<li>Cambia el modelo de embedding (de small a large) y observa cómo cambian los scores de similitud</li>
<li>¿Qué frases tienen alta similitud aunque usen palabras diferentes?</li>
<li>¿Cuáles tienen baja similitud aunque hablan del mismo tema?</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 2: Configura pgvector</div>
<p>Usa el código de pgvector de arriba como base:</p>
<ul>
<li>Configura una tabla con columna vector(1536)</li>
<li>Inserta 5 documentos con embeddings</li>
<li>Realiza una búsqueda por similitud con una query</li>
<li>Agrega metadata (autor, fecha) y usa filtros WHERE</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 3: Mide el impacto del modelo</div>
<p>Crea un mini-eval con 10 preguntas sobre tu corpus. Mide Recall@5 con:</p>
<ul>
<li>text-embedding-3-small</li>
<li>text-embedding-3-large</li>
</ul>
<p>Calcula la diferencia porcentual. Justifica si vale la pena el costo extra.</p>
</div>