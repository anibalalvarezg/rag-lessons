---
title: "Estrategias de Chunking"
subtitle: "Lección 2 — Ingesta y Segmentación"
pillar: ingesta
pillarName: "Ingesta y Segmentación"
lessonNum: 2
description: "El dilema chunk-size, 4 estrategias, Contextual Retrieval (-49% fallos), árbol de decisión."
keywords: "chunking, chunk size, overlap, Contextual Retrieval, semantic splitting"
ogSection: "Ingesta y Segmentación"
pubDate: "2026-07-24"
quizzes:

  - id: "q1"
    question: "¿Por qué el tamaño de chunk importa más que el algoritmo?"
    options:
      - text: "Porque el algoritmo es siempre el mismo"
        correct: false
      - text: "Porque un chunk de 256 vs 512 tokens cambia la señal del embedding más que recursive vs semantic"
        correct: true
      - text: "Porque los LLMs solo funcionan con ciertos tamaños"
        correct: false
      - text: "No importa, ambos factores son igual de relevantes"
        correct: false
  - id: "q2"
    question: "¿Cuál es el patrón de parent-child chunking?"
    options:
      - text: "Embeddar ambos parents y children, buscar en ambos"
        correct: false
      - text: "Buscar en children pequeños, retornar parents grandes al LLM"
        correct: true
      - text: "Embeber parents grandes, buscar en ellos, retornar children"
        correct: false
      - text: "Un solo tamaño de chunk que sirve para todo"
        correct: false
  - id: "q3"
    question: "¿Cuándo SÍ vale la pena semantic chunking?"
    options:
      - text: "Siempre, es mejor que recursive"
        correct: false
      - text: "Solo con documentos Markdown bien formateados"
        correct: false
      - text: "Prosa larga sin formato, transcripciones, cuando los boundaries semánticos no coinciden con la estructura"
        correct: true
      - text: "Cuando el presupuesto de embedding es muy bajo"
        correct: false
  - id: "q4"
    question: "¿Qué reduce más los fallos de retrieval según Anthropic?"
    options:
      - text: "Cambiar el embedding model"
        correct: false
      - text: "Aumentar el chunk size a 2048"
        correct: false
      - text: "Contextual Retrieval: prepender un resumen del documento antes de embeber cada chunk"
        correct: true
      - text: "Usar semantic chunking en lugar de recursive"
        correct: false

---


## Objetivo

Al finalizar esta lección, podrás elegir la estrategia de chunking correcta para tu corpus, implementar parent-child chunking, y justificar cada decisión de tamaño de chunk con datos.

## El dilema del tamaño de chunk

Todo sistema RAG enfrenta la misma tensión:

- **Chunks pequeños (100-300 tokens):** Embeddings precisos, retrieval preciso. Pero el LLM no tiene contexto suficiente para generar una respuesta completa.

- **Chunks grandes (1000-2000 tokens):** Contexto rico para generación. Pero el embedding se promedia sobre demasiados conceptos, y la señal de retrieval se diluye.

<div class="callout info">
<div class="callout-title">Dato clave de producción</div>
<p>El tamaño de chunk mueve las métricas más que la elección del algoritmo. En benchmarks de 2026, la diferencia entre 256 y 512 tokens fue mayor que la diferencia entre recursive y semantic chunking.

<small>Fuente: <a href="https://www.runvecta.com/blog/we-benchmarked-7-chunking-strategies-most-advice-was-wrong">Vecta Benchmark (Feb 2026) — 7 estrategias, 50 papers academicos</a></small></p>
</div>

## Las estrategias principales

### 1. Fixed-size (Tamaño fijo)

Corta cada N tokens, opcionalmente con overlap. Es rápido, predecible, y "tonto".

- **Cuándo usarlo:** FAQs, catálogos de productos, contenido uniforme y corto

- **Cuándo NO:** Documentos con estructura, código, tablas, contenido mixto

- **Configuración típica:** 512 tokens, overlap 50-64 tokens (10-15%)

### 2. Recursive character splitting (El default recomendado)

Intenta dividir en la frontera más grande disponible (párrafos), y si el chunk sigue siendo muy grande, recursa a oraciones, luego palabras. Respeta la estructura cuando puede.

<div class="callout success">
<div class="callout-title">El default de producción 2026</div>
<p><code>chunk_size=512</code>, <code>chunk_overlap=64</code>, separadores en orden: párrafo → línea → oración → palabra. En benchmarks de 50 papers académicos, recursive 512 obtuvo 69% de accuracy vs 54% de semantic chunking.

<small>Fuente: <a href="https://www.runvecta.com/blog/we-benchmarked-7-chunking-strategies-most-advice-was-wrong">Vecta Benchmark (Feb 2026) — Recursive 512: 69% accuracy, Semantic: 54% accuracy</a></small></p>
</div>

Configuración de LangChain:

```
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
chunk_size=512,
chunk_overlap=64,
separators=["\n\n", "\n", ". ", " ", ""]
)
```

### 3. Semantic chunking (División semántica)

Embedda cada oración, mide similitud coseno entre oraciones adyacentes, y crea un boundary cuando la similitud cae por debajo de un threshold. Respeta cambios de tema, no estructura.

- **Cuándo usarlo:** Prosa larga sin headers claros, transcripciones, documentos donde el formato no refleja la estructura semántica

- **Cuándo NO:** Documentos bien estructurados (Markdown, HTML), cuando el presupuesto de embedding es limitado

- **Costo:** ~4x más embedding que recursive (una llamada por oración)

Implementación conceptual:

```
def semantic_chunk(text, threshold_percentile=95):
sentences = split_into_sentences(text)
embeddings = [embed(s) for s in sentences]

chunks = []
current = [sentences[0]]

for i in range(1, len(sentences)):
sim = cosine_sim(embeddings[i-1], embeddings[i])
distances.append(1 - sim)

threshold = percentile(distances, threshold_percentile)

for i in range(1, len(sentences)):
dist = 1 - cosine_sim(embeddings[i-1], embeddings[i])
if dist > threshold:
chunks.append(" ".join(current))
current = [sentences[i]]
else:
current.append(sentences[i])

chunks.append(" ".join(current))
return chunks
```

<div class="callout warning">
<div class="callout-title">Problema conocido con defaults</div>
<p>LangChain's <code>SemanticChunker</code> con threshold al percentil 95 produce chunks de solo 43 tokens en promedio en textos académicos. Casi siempre necesitas subir el threshold a 99 o ajustar el <code>min_chunk_size</code>.</p>
</div>

### 4. Parent-child chunking (El patrón de producción)

Resuelve el dilema del tamaño desacoplando retrieval de generación:

- **Child chunks (100-300 tokens):** Se embeben y buscan. Son precisos y topicamente enfocados.

- **Parent chunks (1000-2000 tokens):** Se leen por el LLM. Dan contexto rico.

**El flujo:**

1. Usuario hace una pregunta

2. Se busca en los child chunks (preciso)

3. Se recupera el parent chunk del child ganador (contexto rico)

4. Se deduplican parents (varios children pueden apuntar al mismo parent)

5. El LLM recibe los parents, no los children

```
# LangChain ParentDocumentRetriever
# Nota: en LangChain v1+, ParentDocumentRetriever se migro a langchain-classic
# pip install langchain-classic
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore

child_splitter = RecursiveCharacterTextSplitter(chunk_size=250)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500)

retriever = ParentDocumentRetriever(
vectorstore=vectorstore,      # children embeddings
docstore=InMemoryStore(),      # parent texts
child_splitter=child_splitter,
parent_splitter=parent_splitter,
)
```

<div class="callout success">
<div class="callout-title">Impacto medido</div>
<p>Parent-child retrieval mejora típicamente: +10-20% en faithfulness, +5-15% en answer relevance. La ganancia en faithfulness es la más grande porque el LLM no necesita "adivinar" el contexto faltante.</p>
</div>

## ¿Y las tablas?

Las tablas necesitan tratamiento especial. El chunking de texto plano las destruye. Estrategia recomendada:

1. Extraer tablas por separado

2. Generar un resumen en lenguaje natural de cada tabla (usando un LLM local)

3. Embeber tanto los datos crudos de la tabla como el resumen

## Contextual Retrieval (Anthropic)

Antes de embeber cada chunk, prepender un resumen generado por LLM:

```
# Prompt para contextual retrieval
context_prompt = f"""
Here is the full document:
{full_document}

Here is one chunk:
{chunk}

Write a 50-100 token context that situates this chunk in the document.
"""
context = llm.generate(context_prompt)
chunk_with_context = f"{context}\n\n{chunk}"
```

**Resultados reportados:**

- 49% reducción en fallos de retrieval (solo contextual embedding)

- 67% con reranker encima

- Costo: ~$1.02 por millón de tokens de documentos fuente

<small>Fuente: [Anthropic Engineering Blog — Contextual Retrieval (Sep 2024)](https://www.anthropic.com/engineering/contextual-retrieval)</small>

## Matriz de decisión
| Tu situación | Estrategia | Por qué |
| --- | --- | --- |
| FAQs, catálogos, contenido uniforme | Fixed 256-512 | Rápido, simple, suficiente |
| Documentación Markdown/HTML | Recursive 512 + overlap 64 | Respeta headings, buen default |
| Prosa larga sin formato | Semantic (tuneado a 99%) | Detecta boundaries semánticos |
| Documentos densos (legal, técnico) | Parent-child (250/1500) | Precisión + contexto completo |
| Transcripciones, chat logs | Semantic o parent-child | Los boundaries no son estructurales |

## Práctica


<div class="exercise">
<div class="exercise-title">Ejercicio 1: Analiza el chunking del repo</div>
<p>Abre <code>chapter2/parse-pdf.ipynb</code> del repositorio hands-on-rag. Identifica:</p>
<ul>
<li>¿Qué tamaño de chunk se usa?</li>
<li>¿Qué estrategia de división?</li>
<li>¿Qué overlap se configuró?</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 2: Implementa parent-child</div>
<p>Usando el código del Capítulo 2 como base, modifica el pipeline para usar parent-child chunking:</p>
<ul>
<li>Child chunks: 250 tokens</li>
<li>Parent chunks: 1500 tokens</li>
<li>Embedda solo los children</li>
<li>Al recuperar, retorna el parent</li>
</ul>
<p><strong>Pista:</strong> Mira <code>ParentDocumentRetriever</code> de LangChain.</p>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 3: Compara estrategias</div>
<p>Crea un mini-eval con 5 preguntas sobre el documento del Capítulo 1 (Alice in Wonderland). Ejecuta el pipeline 3 veces:</p>
<ul>
<li>Recursive 512 sin overlap</li>
<li>Recursive 512 con overlap 64</li>
<li>Semantic chunking (percentil 95)</li>
</ul>
<p>Compara: ¿cuántos de los top-3 chunks contienen la respuesta correcta?</p>
</div>
