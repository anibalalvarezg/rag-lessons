---
title: "Anexo B: Anatomía de un prompt RAG"
subtitle: "Anexo B — Fundamentos"
pillar: anexo
pillarName: "Anexos"
lessonNum: 11
description: "Las 4 partes de un prompt RAG: rol, regla de grounding, contexto inyectado, e instrucción de citación. Por qué cada una reduce alucinaciones."
keywords: "prompt engineering, RAG prompt, grounding, citación, system prompt"
ogSection: "Anexos"
pubDate: "2026-07-27"
quizzes:

  - id: "q1"
    question: "¿Qué hace la regla \"responde usando SOLO el contexto\"?"
    options:
      - text: "Hace que el modelo responda más rápido siempre"
        correct: false
      - text: "Prohíbe al modelo usar su memoria paramétrica como fuente"
        correct: true
      - text: "Reduce el costo de tokens de cada llamada"
        correct: false
      - text: "Mejora la calidad de los embeddings del corpus"
        correct: false
  - id: "q2"
    question: "¿Qué debería hacer el modelo cuando el contexto no contiene la respuesta?"
    options:
      - text: "Responder con su conocimiento general del tema"
        correct: false
      - text: "Decir explícitamente que no tiene información suficiente"
        correct: true
      - text: "Reformular la pregunta del usuario automáticamente"
        correct: false
      - text: "Buscar la respuesta en internet en tiempo real"
        correct: false
  - id: "q3"
    question: "¿Por qué pedir citas numeradas como [1], [2] en la respuesta?"
    options:
      - text: "Para que el usuario pueda verificar cada afirmación contra la fuente"
        correct: true
      - text: "Para que el texto generado se vea más profesional"
        correct: false
      - text: "Para reducir la latencia total de la generación"
        correct: false
      - text: "Para mejorar el ranking de los chunks recuperados"
        correct: false

---


## Objetivo

Al finalizar este anexo, podrás desarmar el prompt RAG de la Lección 1 en sus 4 partes, explicar qué alucinación previene cada una, y escribir tu propia variante con citas numeradas.

## El prompt que viste en la Lección 1

```python
prompt = ChatPromptTemplate.from_template(
    "Responde la pregunta usando SOLO el contexto. Cita la fuente.\n\n"
    "Contexto: {context}\n\nPregunta: {question}"
)
```

Parece trivial, pero cada elemento está haciendo trabajo defensivo. Un prompt RAG de producción tiene 4 partes:

## Las 4 partes de un prompt RAG

### 1. Rol (quién es el modelo)

Ancla el tono y el dominio. Para el asistente de Acme:

```text
Eres el asistente interno de Acme Corp. Respondes preguntas de empleados
sobre políticas de la empresa y documentación técnica.
```

### 2. Regla de grounding (de dónde puede salir la respuesta)

La línea más importante del sistema: **prohíbe responder desde la memoria paramétrica**.

```text
Responde usando SOLO el contexto proporcionado. Si el contexto no contiene
la respuesta, di "No tengo información suficiente sobre eso".
```

<div class="callout warning">
<div class="callout-title">La mitad invisible de la regla</div>
<p>La cláusula de escape ("si no está, dilo") es tan importante como la prohibición. Sin ella, el modelo prefiere inventar una respuesta antes que admitir ignorancia — los LLMs están entrenados para ser útiles, no para abstenerse. Esta cláusula es tu primera línea de defensa contra las alucinaciones Tipo I de la Lección 5.</p>
</div>

### 3. Contexto inyectado (lo que recuperó el retriever)

Aquí van los chunks que recuperó tu pipeline. Es la **única** parte que cambia en cada request:

```python
context = "\n\n".join(
    f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)
)
```

Numerar los chunks (`[1]`, `[2]`, ...) habilita la cuarta parte:

### 4. Instrucción de citación (cómo se verifica)

```text
Cita la fuente de cada afirmación con el número del chunk: [1], [2], ...
```

Las citas convierten una respuesta de "confía en mí" en una **verificable**: el usuario puede revisar el chunk 2 y confirmar. Además, obligar al modelo a citar lo obliga a atender al contexto.

## El prompt completo de producción

```python
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template("""Eres el asistente interno de Acme Corp. Respondes preguntas de empleados sobre políticas de la empresa y documentación técnica.

Reglas:
- Responde usando SOLO el contexto proporcionado.
- Si el contexto no contiene la respuesta, di "No tengo información suficiente sobre eso".
- Cita la fuente de cada afirmación con el número del chunk: [1], [2], ...

Contexto:
{context}

Pregunta: {question}
""")
```

<div class="callout success">
<div class="callout-title">Conexión con el curso</div>
<p>Este es el prompt que evoluciona a lo largo del Proyecto Acme: en la Lección 5 lo envuelves con guardrails de salida (¿cumplió su propia regla de grounding?), y en la Lección 6 mides su <strong>faithfulness</strong> — qué tan bien respeta la regla "SOLO el contexto". Un prompt no se evalúa por cómo se lee, sino por cómo se comporta medido.</p>
</div>

## Práctica


<div class="exercise">
<div class="exercise-title">Ejercicio 1: Rompe tu prompt</div>
<p>Usa el pipeline de la Lección 1 con el corpus Acme y experimenta:</p>
<ul>
<li>Quita la regla de grounding y haz una pregunta fuera del corpus — ¿alucina?</li>
<li>Quita la cláusula de escape y pregunta algo que el contexto no cubre — ¿admite ignorancia o inventa?</li>
<li>Restaura el prompt completo y repite — ¿se abstiene correctamente?</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 2: Citas verificables</div>
<p>Modifica el pipeline para que:</p>
<ul>
<li>Los chunks se numeren al inyectarse en el contexto</li>
<li>La respuesta incluya citas [1], [2] por afirmación</li>
<li>Imprimas junto a la respuesta el texto exacto de cada chunk citado</li>
</ul>
</div>
