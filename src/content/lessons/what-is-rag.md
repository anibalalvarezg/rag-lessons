---
title: "¿Qué es RAG y por qué lo necesitas en producción?"
subtitle: "Lección 1 — Fundamentos del pipeline RAG"
pillar: fundamentos
pillarName: "Fundamentos"
lessonNum: 1
description: "Lección introductoria: qué es RAG, pipeline de 4 pasos, cuándo usar RAG vs long-context vs fine-tuning, los 5 pilares de producción."
keywords: "RAG, Retrieval Augmented Generation, pipeline, long-context, fine-tuning, LLM"
ogSection: "Fundamentos"
pubDate: "2026-07-24"
quizzes:

  - id: "quiz1"
    question: "¿Cuál es el problema principal que RAG resuelve?"
    options:
      - text: "Los LLMs son demasiado lentos para responder preguntas"
        correct: false
      - text: "Los LLMs no tienen acceso a conocimiento privado o actualizado"
        correct: true
      - text: "Los LLMs alucinan demasiado sin contexto externo"
        correct: false
      - text: "Los LLMs no pueden procesar documentos largos"
        correct: false
  - id: "quiz2"
    question: "¿Qué camino del pipeline RAG ocurre ANTES de que el usuario haga una pregunta?"
    options:
      - text: "Query → Embedding → Retrieval → Generation"
        correct: false
      - text: "Parsing → Chunking → Embedding → Storage"
        correct: true
      - text: "Generation → Citation → Validation → Response"
        correct: false
      - text: "Retrieval → Reranking → Context Assembly → Generation"
        correct: false
  - id: "quiz3"
    question: "Según la regla práctica de 2026, ¿cuándo SÍ necesitas RAG?"
    options:
      - text: "Siempre, es mejor que long context"
        correct: false
      - text: "Cuando tu corpus tiene menos de 100 páginas"
        correct: false
      - text: "Cuando tu corpus > 1M tokens, cambia frecuentemente, o necesitas citas"
        correct: true
      - text: "Solo cuando usas modelos open-source"
        correct: false

---


## Objetivo

Al finalizar esta lección, podrás explicar qué es RAG, por qué es necesario, y cuáles son los 5 pilares que separan un demo de un sistema en producción.

**Prerequisitos:** [Lección 0 — Tu entorno RAG en 30 minutos](/rag-lessons/lessons/setup-entorno). Si los términos "token" o "similitud de coseno" son nuevos para ti, lee primero el [Anexo A](/rag-lessons/lessons/anexo-tokens-embeddings).

<div class="callout info">
<div class="callout-title">🧵 Proyecto Acme — el hilo conductor</div>
<p>Todo el código de este curso construye un solo sistema: el <strong>asistente RAG interno de Acme Corp</strong>. Su corpus: <code>politicas.pdf</code> (RRHH), <code>manual-tecnico.pdf</code> (operaciones) y <code>reporte-financiero.pdf</code> (finanzas). En esta lección indexas el primer documento (<code>politicas.pdf</code>) en la colección <code>acme_docs</code> que reutilizarás hasta la Lección 9.</p>
</div>

## El problema que RAG resuelve

Los LLMs tienen un límite: solo saben lo que vieron durante su entrenamiento. Si necesitas que un chatbot responda sobre tus documentos internos, políticas de la empresa, o conocimiento que cambia constantemente, un LLM solo no puede.

La solución obvia sería re-entrenar el modelo con tus datos. Pero eso es:

- **Caro** — entrenar un modelo cuesta miles de dólares

- **Lento** — toma horas o días

- **Rígido** — cuando tus documentos cambian, necesitas re-entrenar

RAG (Retrieval-Augmented Generation) ofrece una alternativa: en lugar de meter el conocimiento en el modelo, le damos acceso a una base de datos externa en tiempo real.

<div class="callout info">
<div class="callout-title">Analogía</div>
<p>Piensa en un examen. Un LLM sin RAG es como un estudiante que solo puede usar lo que memorizó. RAG es como un estudiante que puede abrir sus apuntes y buscar la respuesta antes de responder.</p>
</div>

## Cómo funciona RAG (pipeline básico)

El flujo tiene dos caminos que se encuentran:

### Camino offline (Indexing)

1. **Parsing:** Extraer texto de documentos (PDFs, Word, HTML)

2. **Chunking:** Dividir el texto en fragmentos manejables

3. **Embedding:** Convertir cada chunk en un vector numérico

4. **Almacenamiento:** Guardar los vectores en una base de datos vectorial

### Camino online (Query)

1. **Query embedding:** Convertir la pregunta del usuario en un vector

2. **Retrieval:** Buscar los chunks más similares en la base de datos

3. **Generation:** Enviar los chunks recuperados + la pregunta al LLM

<div class="callout warning">
<div class="callout-title">Diferencia clave con producción</div>
<p>Un demo para aquí. Un sistema en producción agrega: re-ranking de resultados, guardrails de entrada/salida, evaluación continua, monitoreo, y manejo de fallos.</p>
</div>

## Los 5 pilares de RAG en producción

Tu repositorio `hands-on-rag` organiza exactamente estos pilares en sus capítulos:

| Pilar | Capítulo | Pregunta clave |
| --- | --- | --- |
| **1. Fundamentos** | 1 | ¿Qué es RAG, cuándo usarlo y cuáles son sus límites? |
| **2. Ingesta y Segmentación** | 2-3 | ¿Cómo divido mis documentos sin perder información? |
| **3. Recuperación Avanzada** | 3-4 | ¿Cómo encuentro los chunks correctos de forma precisa? |
| **4. RAGOps y Evaluación** | 6 | ¿Cómo sé si mi sistema funciona bien? |
| **5. Agentic RAG y GraphRAG** | 7-9 | ¿Cuándo debe el sistema buscar vs. responder directamente? |

## RAG vs. Long Context vs. Fine-tuning

RAG no es la única solución. En 2026, existen tres opciones principales:

| Enfoque | Cuándo usarlo | Cuándo NO usarlo |
| --- | --- | --- |
| **Long Context** | Corpus &lt; 200k tokens, no cambia mucho | Corpus grande, necesita citas verificables |
| **RAG** | Corpus &gt; 1M tokens, cambia frecuentemente, necesita citations | Preguntas sobre conocimiento general |
| **Fine-tuning** | Necesitas un estilo/formato específico, comportamiento consistente | Hechos que cambian, datos actuales |

<div class="callout success">
<div class="callout-title">Regla práctica 2026</div>
<p>Si tu corpus cabe en 200k tokens (unas 500 páginas), probablemente no necesitas RAG. Prueba primero con long context + prompt caching. RAG gana cuando: el corpus es grande, cambia frecuentemente, o necesitas citas verificables.</p>
</div>

## Pipeline mínimo en código (LangChain)

Antes de los ejercicios, aquí tienes un pipeline RAG mínimo en LangChain que implementa los 4 pasos (load → split → embed → store → query) descritos arriba. Este es el código al que se refieren los ejercicios de práctica.

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# --- Camino offline: Indexing ---

# 1. Parsing — extraer texto de un PDF (corpus Acme: políticas de RRHH)
loader = PyPDFLoader("corpus/politicas.pdf")
documents = loader.load()

# 2. Chunking — dividir en fragmentos manejables
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(documents)

# 3. Embedding + 4. Almacenamiento — vectorizar y guardar
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1536)
vectorstore = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    connection_string="postgresql://rag:rag@localhost:5432/rag",  # Docker de la Lección 0
    collection_name="acme_docs",  # la colección de todo el curso
)

# --- Camino online: Query ---

# 1. Query embedding + 2. Retrieval + 3. Generation
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template(
    "Responde la pregunta usando SOLO el contexto. Cita la fuente.\n\n"
    "Contexto: {context}\n\nPregunta: {question}"
)

def responder(question: str) -> str:
    docs = retriever.invoke(question)              # 1 + 2
    context = "\n\n".join(d.page_content for d in docs)
    return llm.invoke(prompt.invoke({"context": context, "question": question})).content  # 3

print(responder("¿Cuál es la política de devoluciones?"))
```

<div class="callout warning">
<div class="callout-title">Esto es un demo</div>
<p>Si completaste la Lección 0, este código corre tal cual contra tu Postgres en Docker. Es un pipeline mínimo: en las siguientes lecciones reemplazarás cada componente por su versión de producción. Observa las 4 funciones clave: <code>loader.load()</code> (parse), <code>splitter.split_documents()</code> (chunk), <code>PGVector.from_documents()</code> (embed + store) y <code>retriever.invoke()</code> (retrieve). El LLM (<code>ChatOpenAI</code>) hace la generación al final. El prompt se desarma pieza por pieza en el <a href="/rag-lessons/lessons/anexo-prompt-rag">Anexo B</a>.</p>
</div>

## Práctica


<div class="exercise">
<div class="exercise-title">Ejercicio 1</div>
<p>Ejecuta el código del pipeline de arriba en tu entorno local. Identifica los 4 componentes del pipeline RAG.</p>
<p><strong>Pistas:</strong> Observa las funciones de LangChain que se usan (load, split, embed, store, query).</p>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 2</div>
<p>Modifica el código y responde:</p>
<ul>
<li>¿Qué modelo de embedding se usa?</li>
<li>¿Qué base de datos vectorial?</li>
<li>¿Qué tamaño de chunk se configuró?</li>
<li>¿Qué LLM genera la respuesta final?</li>
</ul>
</div>
