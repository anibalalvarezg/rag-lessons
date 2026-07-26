---
title: "Embeddings y Vector Stores: La capa que determina tu calidad"
subtitle: "Lección 3 — Pilar 1: Ingesta y Segmentación (parte 2)"
pillar: ingesta
pillarName: "Ingesta"
lessonNum: 3
description: "Modelos de embedding 2026, benchmarkVecta, Matryoshka, LanceDB vs pgvector, hybrid retrieval."
keywords: "embeddings, vector store, pgvector, LanceDB, Matryoshka, hybrid retrieval"
ogSection: "Ingesta"
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

import Callout from '../../components/Callout.astro';
import Exercise from '../../components/Exercise.astro';

## Que es un embedding

Un embedding es un vector numerico de alta dimension (tipicamente 768-3072 dimensiones) que representa el significado semantico de un textos. Textos con significado similar quedan cerca en el espacio vectorial.

```python
# Ejemplo conceptual
"El gato duerme en el sofa" -> [0.23, -0.15, 0.87, ..., 0.42]  # 3072 dims
"Una mascota descansa en un mueble" -> [0.21, -0.12, 0.85, ..., 0.44]  # similar!
"Python es un lenguaje de programacion" -> [-0.78, 0.91, -0.34, ..., 0.12]  # diferente
```

La metrica clave es la **similitud de coseno**: mide el angulo entre dos vectores. A mayor similitud (mas cercano a 1), mas relacionados estan los textos.

## El modelo de embedding importa MAS que el chunking

<Callout type="warning" title="Dato que cambia prioridades">
En benchmarks de 2026, cambiar de `text-embedding-3-small` a `text-embedding-3-large` produjo un swing de hasta 10.58% en recall. La diferencia entre recursive y semantic chunking fue de 2-9%. **Optimiza el embedding antes que el chunking.**
<small>Fuente: Vecta Benchmark (Feb 2026) — Voyage-3-large supera a OpenAI 3-large por 10.58%</small>
</Callout>

## El landscape de modelos en 2026

### APIs (hosted)

<small>Fuente de scores MTEB: MTEB Leaderboard (HuggingFace, 2026) — Los valores pueden variar segun la version del benchmark.</small>

| Modelo | MTEB | Costo/1M tokens | Dimensiones | Contexto | Mejor para |
|--------|------|-----------------|-------------|----------|------------|
| **Voyage-3.5** | Top retrieval | $0.06 | 1024 | 32K | Calidad general, mejor precio/calidad |
| **OpenAI text-embedding-3-large** | 64.6 | $0.13 | 3072 | 8K | Ecosistema OpenAI, MRL (reducir dimensiones) |
| **OpenAI text-embedding-3-small** | Buena | $0.02 | 1536 | 8K | Presupuesto ajustado, RAG basico |
| **Gemini Embedding 2** | ~1605 ELO | $0.15 | 3072 | 32K | Multimodal, multilingue, documentos largos |
| **Cohere Embed v4** | 65.2 | $0.12 | 1536 | 128K | Multimodal (texto + imagenes), enterprise |

### Open Source (self-hosted)

<small>Fuente de scores MTEB: MTEB Leaderboard (HuggingFace, 2026)</small>

| Modelo | MTEB | Parametros | Mejor para |
|--------|------|------------|------------|
| **Qwen3-Embedding-8B** | 70.6 | 8B | Calidad maxima, multilingue |
| **BGE-M3** | 63.2 | 568M | Multilingue, hybrid retrieval (dense + sparse) |
| **Nomic Embed v2** | 61.4 | 137M | Ultra-liviano, on-device |

<Callout type="info" title="Matryoshka Embeddings">
Modelos como OpenAI 3-large y Cohere v4 soportan **Matryoshka**: puedes reducir dimensiones de 3072 a 1024 o 512 al momento de inferencia, sin re-embedder. Pierdes ~2% de precision pero reduces almacenamiento 4-6x.
</Callout>

## Dimensionalidad y almacenamiento

| Dimensiones | Precision | Almacenamiento (1M docs) | Latencia busqueda |
|-------------|-----------|--------------------------|-------------------|
| 256 | 94.2% | ~1 GB | 5ms |
| 512 | 96.8% | ~2 GB | 8ms |
| 1024 | 98.1% | ~4 GB | 15ms |
| 3072 | 98.5% | ~12 GB | 42ms |

## Vector Databases: Cuando usar cada una

### pgvector (El default para la mayoria)

Una extension de Postgres. Los vectores viven en la misma tabla que tus datos. No necesitas un servicio nuevo.

- **Cuando usarlo:** Ya usas Postgres, corpus < 5M vectores, primer RAG system
- **Latencia:** 5-20ms p50 con HNSW (1M vectores)
- **Costo:** ~$0-25/mes en Supabase (absorbido en Postgres existente)
- **Limitacion:** > 10M vectores empieza a degradarse

### Pinecone (Managed, zero-ops)

Base de datos vectorial fully managed. No tocas infraestructura.

- **Cuando usarlo:** Sin capacidad de ops, necesitas sub-20ms p95 a escala, prototipos rapidos
- **Latencia:** sub-5ms p50 (1M), 5-15ms (10M)
- **Costo:** ~$50-80/mes (1M vectores, 100K queries/mes)
- **Limitacion:** Se vuelve caro a escala ($700+/mes a 10M vectores)

### Qdrant (Performance + open source)

Escrito en Rust. Filtering de payloads es el mejor de la categoria.

- **Cuando usarlo:** Filtrado por metadata complejo, necesitas throughput alto, open source
- **Latencia:** sub-5ms p50 (1M), mejor filtrado que pgvector a escala
- **Costo:** $0 (free tier 1GB), o ~$36/mes en cloud para 1M vectores
- **Limitacion:** Necesitas Docker/K8s para self-host

### Weaviate (Multimodal + hybrid search)

BM25 + vector search nativo. Modulos de vectorizacion built-in.

- **Cuando usarlo:** Busqueda hibrida nativa, multi-tenant SaaS, multimodal
- **Costo:** ~$25-150/mes (cloud)
- **Limitacion:** Mas pesado que Qdrant, breaking changes v3 a v4

<Callout type="success" title="Recomendacion rapida 2026">
**Empieza con pgvector** si ya usas Postgres. Es la decision correcta para el 80% de los casos. Migra a Qdrant o Pinecone cuando necesites mas de 5M vectores o filtrado avanzado.
</Callout>

## Implementacion con LangChain

```python
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

# 1. Configurar el modelo de embedding
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1536  # Matryoshka: reducir de 3072 a 1536
)

# 2. Crear/consultar la vector store
vectorstore = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    connection_string="postgresql://user:pass@localhost/db",
    collection_name="my_rag_docs"
)

# 3. Buscar
results = vectorstore.similarity_search(
    "Cual es la politica de reembolsos?",
    k=5
)
```

## El patron de produccion: Hybrid Retrieval

En 2026, la busqueda pura vectorial se considera un anti-patron fuera de casos triviales. El patron estandar combina:

- **Dense (vectorial):** Entiende parrafasis y similitud semantica
- **Sparse (BM25):** Coincidencias exactas (SKUs, nombres propios, siglas)

Se fusionan con **Reciprocal Rank Fusion (RRF)**:

```python
score_final = SUM(1/(k + rank_i))  # para cada fuente (dense, sparse)
# k=60 es el parametro comun
# RRF no requiere normalizar scores entre sistemas diferentes
```

<Callout type="info" title="Ganancia medida">
Hybrid retrieval mejora typicalamente 5-15% recall@5 sobre busqueda pura vectorial. La ganancia es mayor en queries con entidades nombradas, acronimos, o codigos de producto.
</Callout>

## Migracion de embedding models

Cambiar el modelo de embedding requiere re-embedder TODO tu corpus. Esto puede tomar horas o dias en corpus grandes.

- Versiona tus embeddings: guarda el nombre del modelo + version junto a cada vector
- Disena tu camino de migracion: re-embedde en background mientras sirves el indice viejo, luego cambia atomicamente
- Los equipos que no planean migracion quedan atrapados en su primer modelo por anos

## Practica

<Exercise title="Ejercicio 1: Explora embeddings">
Abre `chapter2/embedding.ipynb` del repositorio. Ejecuta las celdas que calculan similitud entre oraciones.

- Cambia el modelo de embedding (de small a large) y observa como cambian los scores de similitud
- Que frases tienen alta similitud aunque usen palabras diferentes?
- Cuales tienen baja similitud aunque hablan del mismo tema?
</Exercise>

<Exercise title="Ejercicio 2: Configura pgvector">
Usando el notebook `pgvector-simple.ipynb` del capitulo 2 (no confundir con el pipeline principal del capitulo 1 que usa LanceDB):

- Configura una tabla con columna vector(1536)
- Inserta 5 documentos con embeddings
- Realiza una busqueda por similitud con una query
- Agrega metadata (autor, fecha) y usa filtros WHERE
</Exercise>

<Exercise title="Ejercicio 3: Mide el impacto del modelo">
Crea un mini-eval con 10 preguntas sobre el corpus del Capitulo 1. Mide Recall@5 con:

- text-embedding-3-small
- text-embedding-3-large

Calcula la diferencia porcentual. Justifica si vale la pena el costo extra.
</Exercise>
