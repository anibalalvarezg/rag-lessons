---
title: "¿Qué es RAG y por qué lo necesitas en producción?"
subtitle: "Lección 1 — Fundamentos del pipeline RAG"
pillar: fundamentos
pillarName: "Fundamentos"
lessonNum: 1
description: "Lección introductoria: qué es RAG, pipeline de 4 pasos, cuándo usar RAG vs long-context vs fine-tuning, los 4 pilares de producción."
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

Al finalizar esta lección, podrás explicar qué es RAG, por qué es necesario, y cuáles son los 4 pilares que separan un demo de un sistema en producción.

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

## Los 4 pilares de RAG en producción

Tu repositorio `hands-on-rag` organiza exactamente estos pilares en sus capítulos:

| Pilar | Capítulo | Pregunta clave |
| --- | --- | --- |
| **1. Ingesta y Segmentación** | 2-3 | ¿Cómo divido mis documentos sin perder información? |
| **2. Recuperación Avanzada** | 3-4 | ¿Cómo encuentro los chunks correctos de forma precisa? |
| **3. Agentic RAG y GraphRAG** | 7-9 | ¿Cuándo debe el sistema buscar vs. responder directamente? |
| **4. RAGOps y Evaluación** | 6 | ¿Cómo sé si mi sistema funciona bien? |

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

## Práctica


<div class="exercise">
<div class="exercise-title">Ejercicio 1</div>
<p>Abre el repositorio <code>hands-on-rag</code> y navega al capítulo 1. Ejecuta el notebook <code>sample-rag.ipynb</code>. ¿Qué componentes del pipeline identificas?</p>
<p><strong>Pistas:</strong> Busca las funciones de LangChain que se usan (load, split, embed, store, query).</p>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 2</div>
<p>Responde estas preguntas sobre el notebook del Capítulo 1:</p>
<ul>
<li>¿Qué modelo de embedding se usa?</li>
<li>¿Qué base de datos vectorial?</li>
<li>¿Qué tamaño de chunk se configuró?</li>
<li>¿Qué LLM genera la respuesta final?</li>
</ul>
</div>
