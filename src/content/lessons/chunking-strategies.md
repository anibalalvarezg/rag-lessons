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

**Prerequisitos:** [Lección 1](/rag-lessons/lessons/what-is-rag). Si `chunk_size=512` te hace preguntarte "¿512 qué?", lee el [Anexo A](/rag-lessons/lessons/anexo-tokens-embeddings).

<div class="callout info">
<div class="callout-title">🧵 Proyecto Acme — De dónde viene este código</div>
<p>Sigues trabajando sobre <code>corpus/politicas.pdf</code> de la Lección 1. El código de esta lección reemplaza el paso de <strong>parsing + chunking</strong> del pipeline: primero extraes el texto con PyMuPDF (<code>paginas</code>), luego lo divides con un splitter (<code>chunks</code>). La variable <code>chunks</code> que produces aquí es la que embebes y almacenas en la Lección 3.</p>
</div>

## El dilema del tamaño de chunk

Todo sistema RAG enfrenta la misma tensión:

- **Chunks pequeños (100-300 tokens):** Embeddings precisos, retrieval preciso. Pero el LLM no tiene contexto suficiente para generar una respuesta completa.

- **Chunks grandes (1000-2000 tokens):** Contexto rico para generación. Pero el embedding se promedia sobre demasiados conceptos, y la señal de retrieval se diluye.

<div class="callout info">
<div class="callout-title">Dato clave de producción</div>
<p>El tamaño de chunk mueve las métricas más que la elección del algoritmo. En benchmarks de 2026, la diferencia entre 256 y 512 tokens fue mayor que la diferencia entre recursive y semantic chunking.

<small>Fuente: <a href="https://www.runvecta.com/blog/we-benchmarked-7-chunking-strategies-most-advice-was-wrong">Vecta Benchmark (Feb 2026) — 7 estrategias, 50 papers académicos</a></small></p>
</div>

## Las estrategias principales

### 1. Fixed-size (Tamaño fijo)

Corta cada N tokens, opcionalmente con overlap. Es rápido, predecible, y "tonto".

- **Cuándo usarlo:** FAQs, catálogos de productos, contenido uniforme y corto

- **Cuándo NO:** Documentos con estructura, código, tablas, contenido mixto

- **Configuración típica:** 512 tokens, overlap 50-64 tokens (10-15%)

### Parsing de documentos: antes del chunking

Antes de dividir en chunks, necesitas extraer texto del documento fuente. Para PDFs, `PyMuPDF` (`fitz`) es rápido y preserva estructura básica:

```python
import fitz  # PyMuPDF

def extraer_texto_pdf(path: str) -> list[str]:
    """Extrae el texto de cada página de un PDF."""
    doc = fitz.open(path)
    paginas = []
    for pagina in doc:
        paginas.append(pagina.get_text("text"))
    doc.close()
    return paginas

# Uso
paginas = extraer_texto_pdf("corpus/politicas.pdf")
# paginas es una lista: una string por página del PDF
```

Ojo: **este paso no hace chunking** — solo convierte el PDF en texto. El chunking ocurre después, cuando le aplicas uno de los splitters de abajo. Este es el código al que se refiere el Ejercicio 1 de práctica.

### 2. Recursive character splitting (El default recomendado)

Intenta dividir en la frontera más grande disponible (párrafos), y si el chunk sigue siendo muy grande, recursa a oraciones, luego palabras. Respeta la estructura cuando puede.

<div class="callout success">
<div class="callout-title">El default de producción 2026</div>
<p><code>chunk_size=512</code>, <code>chunk_overlap=64</code>, separadores en orden: párrafo → línea → oración → palabra. En benchmarks de 50 papers académicos, recursive 512 obtuvo 69% de accuracy vs 54% de semantic chunking.

<small>Fuente: <a href="https://www.runvecta.com/blog/we-benchmarked-7-chunking-strategies-most-advice-was-wrong">Vecta Benchmark (Feb 2026) — Recursive 512: 69% accuracy, Semantic: 54% accuracy</a></small></p>
</div>

Configuración de LangChain:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Conectando con el parsing de arriba:
chunks = splitter.create_documents(paginas)
```

### 3. Semantic chunking (División semántica)

Embedda cada oración, mide similitud coseno entre oraciones adyacentes, y crea un boundary cuando la similitud cae por debajo de un threshold. Respeta cambios de tema, no estructura.

- **Cuándo usarlo:** Prosa larga sin headers claros, transcripciones, documentos donde el formato no refleja la estructura semántica

- **Cuándo NO:** Documentos bien estructurados (Markdown, HTML), cuando el presupuesto de embedding es limitado

- **Costo:** ~4x más embedding que recursive (una llamada por oración)

Implementación conceptual:

```python
import numpy as np

def semantic_chunk(text: str, threshold_percentile: float = 95) -> list[str]:
    sentences = split_into_sentences(text)
    embeddings = [embed(s) for s in sentences]

    # 1. Distancia coseno entre oraciones adyacentes
    distances = [
        1 - cosine_sim(embeddings[i - 1], embeddings[i])
        for i in range(1, len(sentences))
    ]
    threshold = np.percentile(distances, threshold_percentile)

    # 2. Cortar donde la distancia supera el threshold
    chunks = []
    current = [sentences[0]]
    for i in range(1, len(sentences)):
        if distances[i - 1] > threshold:
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

- **Child chunks (100-300 tokens):** Se embeben y buscan. Son precisos y tópicamente enfocados.

- **Parent chunks (1000-2000 tokens):** Se leen por el LLM. Dan contexto rico.

**El flujo:**

1. Usuario hace una pregunta

2. Se busca en los child chunks (preciso)

3. Se recupera el parent chunk del child ganador (contexto rico)

4. Se deduplican parents (varios children pueden apuntar al mismo parent)

5. El LLM recibe los parents, no los children

```python
# LangChain ParentDocumentRetriever
# Nota: en LangChain v1+, ParentDocumentRetriever se migró a langchain-classic
# pip install langchain-classic
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore

child_splitter = RecursiveCharacterTextSplitter(chunk_size=250)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,     # children embeddings (el vectorstore de la L1)
    docstore=InMemoryStore(),    # parent texts
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

```python
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
<div class="exercise-title">Ejercicio 1: Del PDF a los chunks</div>
<p>Conecta las dos piezas de código de arriba: extrae <code>corpus/politicas.pdf</code> con <code>extraer_texto_pdf</code> y pásale el resultado al <code>RecursiveCharacterTextSplitter</code> (configuración 512/64). Luego responde:</p>
<ul>
<li>¿Cuántos chunks salieron del documento completo?</li>
<li>¿Cuál es el tamaño en tokens del chunk más grande y del más chico? (usa tiktoken del Anexo A)</li>
<li>Abre dos chunks consecutivos: ¿puedes ver el overlap de 64 tokens entre ellos?</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 2: Implementa parent-child</div>
<p>Usa el código de ParentDocumentRetriever de arriba como base y modifícalo para:</p>
<ul>
<li>Child chunks: 250 tokens</li>
<li>Parent chunks: 1500 tokens</li>
<li>Embedda solo los children</li>
<li>Al recuperar, retorna el parent</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 3: Compara estrategias</div>
<p>Crea un mini-eval con 5 preguntas sobre <code>corpus/politicas.pdf</code>. Ejecuta el pipeline 3 veces:</p>
<ul>
<li>Recursive 512 sin overlap</li>
<li>Recursive 512 con overlap 64</li>
<li>Semantic chunking (percentil 95)</li>
</ul>
<p>Compara: ¿cuántos de los top-3 chunks contienen la respuesta correcta?</p>
</div>
