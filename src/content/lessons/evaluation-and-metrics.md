---
title: "Evaluación y Métricas"
subtitle: "Lección 6 — RAGOps y Evaluación"
pillar: ragops
pillarName: "RAGOps y Evaluación"
lessonNum: 6
description: "RAG Triad, RAGAS, LLM-as-Judge, 8 métricas enterprise, CI/CD para RAG."
keywords: "evaluation, RAGAS, metrics, LLM-as-Judge, CI/CD"
ogSection: "RAGOps y Evaluación"
pubDate: "2026-07-24"
quizzes:

  - id: "q1"
    question: "¿Cuál es la métrica más crítica para detectar hallucinations en RAG?"
    options:
      - text: "Answer Relevance"
        correct: false
      - text: "Faithfulness"
        correct: true
      - text: "Context Precision"
        correct: false
      - text: "MRR"
        correct: false
  - id: "q2"
    question: "Baja context precision indica un problema con..."
    options:
      - text: "El LLM generator"
        correct: false
      - text: "El retriever (chunks irrelevantes o demasiado grandes)"
        correct: true
      - text: "La pregunta del usuario"
        correct: false
      - text: "El embedding model"
        correct: false
  - id: "q3"
    question: "¿Cómo mide RAGAS Answer Relevance sin ground truth?"
    options:
      - text: "Compara con la respuesta de referencia"
        correct: false
      - text: "Genera N preguntas desde la respuesta y mide similaridad con la pregunta original"
        correct: true
      - text: "Usa cosine similarity entre respuesta y contexto"
        correct: false
      - text: "Pide al LLM que dé un score del 1 al 10"
        correct: false
  - id: "q4"
    question: "Un sistema con faithfulness 0.95 puede dar respuestas incorrectas. ¿Por qué?"
    options:
      - text: "Porque faithfulness no es una métrica confiable"
        correct: false
      - text: "Porque el contexto recuperado puede estar obsoleto o ser incorrecto business-mente"
        correct: true
      - text: "Porque el LLM hallucina de todas formas"
        correct: false
      - text: "Porque faithfulness solo mide longitud de la respuesta"
        correct: false
  - id: "q5"
    question: "¿Qué porcentaje de alineación con juicio humano debería tener un juez LLM?"
    options:
      - text: "50% (azar)"
        correct: false
      - text: "70% (aceptable)"
        correct: false
      - text: ">85% (producción)"
        correct: true
      - text: "100% (necesario)"
        correct: false

---


## Objetivo

Al finalizar, sabrás medir la calidad de un sistema RAG usando el RAG Triad, implementar evaluación con LLM-as-a-judge, configurar RAGAS, y construir un pipeline de evaluación continua para CI/CD.

## El RAG Triad: 3 relaciones, 3 métricas

El framework diagnóstico más útil para RAG mide 3 relaciones entre los componentes del sistema:

<small>Fuente: [arXiv:2309.15217 — RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)</small>

<div class="callout info">
<div class="callout-title">Las 3 relaciones críticas</div>
<p><strong>1. Context Relevance</strong> (query → contexto): ¿El retriever encontró chunks relevantes?

<strong>2. Faithfulness</strong> (contexto → respuesta): ¿La respuesta está soportada por los chunks?

<strong>3. Answer Relevance</strong> (query → respuesta): ¿La respuesta realmente responde la pregunta?</p>
</div>

### Métrica 1: Faithfulness (la más crítica)

Mide qué proporción de afirmaciones de la respuesta están soportadas por el contexto recuperado. Es tu herramienta principal contra hallucinations.

```
# Fórmula de Faithfulness (RAGAS)
faithfulness = claims_supported / total_claims

# Ejemplo:
# Contexto: "Einstein nació el 14 de marzo de 1879 en Alemania"
# Respuesta: "Einstein nació en Alemania el 20 de marzo de 1879"
#
# Afirmaciones extraídas:
#   1. "Einstein nació en Alemania" -> SUPPORTED
#   2. "Einstein nació el 20 de marzo de 1879" -> NOT SUPPORTED
# Faithfulness = 1/2 = 0.5
```

- **Rango:** 0.0 (nada soportado) a 1.0 (todo soportado)

- **Umbral típico:** >= 0.85 para production

- **Interpretación:** 0.95 = 95% de las afirmaciones trazan al contexto recuperado. 0.6 = 40% viene de memoria paramétrica o fabricación

### Métrica 2: Context Relevance (Precisión del retriever)

Mide qué proporción del contexto recuperado es realmente útil para responder la pregunta. Penaliza información redundante o irrelevante.

```
# Fórmula de Context Relevance (RAGAS)
# LLM extrae oraciones "cruciales" del contexto
context_relevance = |extracted_sentences| / |total_sentences|

# Si recuperaste 10 oraciones pero solo 3 son útiles:
# Context Relevance = 3/10 = 0.3 -> PROBLEMA en el retriever
```

- **Baja context relevance** = chunks demasiado grandes, o embeddings mal alineados

- **Acción:** Reducir chunk size, mejorar reranking, agregar metadata filters

### Métrica 3: Answer Relevance

Mide si la respuesta realmente responde la pregunta. Penaliza respuestas incompletas, redundantes o tangenciales.

```
# Método de RAGAS: ingeniería inversa de preguntas
# 1. Dada la respuesta, generar N preguntas que podría responder
generated_questions = llm.generate_questions(answer, n=3)
# ["¿En qué país nació Einstein?", "¿Cuál es la nacionalidad de Einstein?", ...]

# 2. Calcular similaridad coseno promedio con la pregunta original
answer_relevance = mean(cosine_sim(q_original, q_generated))
```

## Métricas de Retrieval (para afinar el retriever)

| Métrica | Qué mide | Cuándo usarla | Ejemplo |
| --- | --- | --- | --- |
| **Recall@K** | Proporción de docs relevantes que aparecen en top-K | Dominios donde missing un doc = respuesta incompleta (legal, médico) | 5 docs relevantes, 3 en top-10 = Recall@10 = 0.6 |
| **Precision@K** | Proporción de top-K que son realmente relevantes | Cuando el contexto window es limitado (costo tokens) | 10 docs, 3 útiles = Precision@10 = 0.3 |
| **MRR** | Rango del primer doc relevante | Single-answer lookup (FAQ, entity search) | Mejor doc siempre 1ro = MRR = 1.0 |
| **NDCG** | Relevancia gradiente (no binaria) | Queries complejas donde distintos docs contribuyen diferente | Pondera posición con discount logarítmico |

## LLM-as-a-Judge: El paradigma dominante

Las métricas BLEU y ROUGE miden overlap de tokens — no capturan equivalencia semántica. El paradigma actual usa un LLM como juez para evaluar calidad semántica.

### Cómo funciona
```
# Patrón básico de LLM-as-Judge
judge_prompt = f"""
Evalúa si la respuesta está soportada por el contexto.

Contexto: {retrieved_context}
Respuesta: {generated_answer}

Para cada afirmación en la respuesta, indica si es:
- supported: inferida del contexto
- unsupported: no se puede inferir
- contradicted: contradice el contexto

Output: JSON con lista de veredictos y score de faithfulness.
"""
```

### Frameworks de evaluación

| Framework | Método | CI/CD | Mejor para |
| --- | --- | --- | --- |
| **RAGAS** | Reference-free, 4 métricas core | Moderado | Eval rápida de RAG, sin ground truth |
| **DeepEval** | LLM-as-judge + custom metrics | Fuerte | CI/CD pipelines, pytest integration |
| **TruLens** | Feedback functions + tracing | Moderado | Debugging con visibilidad de traces |
| **LangSmith** | Offline + online eval | Moderado | Equipos usando LangChain |
| **Opik** | G-Eval + LiteLLM | Moderado | Judge flexible (cualquier LLM) |

### Implementación con RAGAS

```
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Preparar dataset de evaluación
eval_data = Dataset.from_dict({
    "question": ["¿Cuál es el timeout default?"],
    "answer": ["El timeout default es 30 segundos"],
    "contexts": [["El timeout se configura en settings.yaml. Default: 30s"]],
    "ground_truth": ["30 segundos"]
})

# Ejecutar evaluación
results = evaluate(
    eval_data,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

# Resultados
print(results)
# {'faithfulness': 0.95, 'answer_relevancy': 0.88,
#  'context_precision': 0.82, 'context_recall': 0.91}
```

## Alinear el juez: El paso que la mayoría olvida

Un juez desalineado es como una brújula apuntando al norte incorrecto. La alineación del juez contra juicio humano experto es el fundamento de evaluación confiable.

<div class="callout warning">
<div class="callout-title">El problema común</div>
<p>Un juez baseline tiene ~75.6% de alineación con expertos. Todos los falsos positivos son el juez siendo "too lenient" — deja pasar respuestas que omiten conceptos críticos. La solución: refinar el prompt del juez con criteria específicos y reglas críticas.

<small>Fuente: <a href="https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/faithfulness/">RAGAS Docs — Judge Alignment Process</a></small></p>
</div>

```
# Proceso de alineación (de Ragas docs)
# 1. Recopilar 100-200 ejemplos con juicio humano experto
# 2. Evaluar juez baseline -> medir alineación
alignment = judge.evaluate(dataset, human_labels)
# baseline: 75.6% alignment

# 3. Analizar errores (qué tipo de falsos positivos?)
#    - El juez no detecta conceptos omitidos
#    - El juez es too permisivo con parciales

# 4. Refinar prompt con reglas específicas
enhanced_prompt = """
Evalúa si la respuesta cubre TODOS los conceptos clave.
- Acepta equivalentes semánticos
- Cuenta conceptos ausentes
- 0 conceptos faltantes = PASS
- 1+ conceptos faltantes = FAIL
...
"""

# 5. Re-evaluar -> medir mejora
# enhanced: 86.9% alignment (+11.3% improvement)
```

## Métricas enterprise: Más allá del RAG Triad

Para sistemas enterprise multi-turn, 8 métricas operacionales (de arxiv:2602.20379):

<small>Fuente: [arXiv:2602.20379 — Case-Aware Evaluation for Enterprise RAG](https://arxiv.org/abs/2602.20379)</small>

| Métrica | Qué evalúa | Peso |
| --- | --- | --- |
| **Hallucination / Grounding Fidelity** | Afirmaciones soportadas por el contexto | 0.20 |
| **Retrieval Correctness** | Versión, entorno, doc correcto | 0.15 |
| **Answer Helpfulness** | Accionalidad y claridad | 0.15 |
| **Context Sufficiency** | Contexto cubre todo lo necesario | 0.10 |
| **Answer Type Fit** | Tipo de respuesta adecuado (diagnóstico vs instrucción) | 0.10 |
| **Identifier Integrity** | Error codes, IDs, paths preservados correctamente | 0.10 |
| **Case Issue Identification** | Identifica correctamente el problema del caso | 0.10 |
| **Resolution Alignment** | Pasos satisfacen restricciones del workflow | 0.10 |

<div class="callout info">
<div class="callout-title">Por qué las métricas genéricas no bastan</div>
<p>En un benchmark enterprise, Llama tenía mejor faithfulness (0.84) que GPT-oss (0.58), pero GPT-oss tenía mejor Answer Type Fit y Resolution Alignment. Las métricas genéricas sugieren que Llama es mejor, pero GPT-oss resolvía mejor los casos de uso reales.</p>
</div>

## Evaluación continua: Pre, en, y post-production
```
# Pipeline de evaluación continua

# 1. PRE-PRODUCTION: Eval offline con dataset curado
eval_dataset = load_eval_set("golden_100.json")
results = evaluate(eval_dataset, metrics=[faithfulness, ...])

# 2. CI/CD: Gating en pipeline de deploy
if results["faithfulness"] < 0.85:
    raise ValueError("Faithfulness regression detected")
if results["answer_relevancy"] < 0.75:
    raise ValueError("Answer relevance regression detected")

# 3. PRODUCTION: Sampling continuo
# - Evaluar 5-10% de tráfico real
# - Monitor drift semanal
# - Alertas cuando métricas caen

# 4. POST-INCIDENT: Debugging con traces
# LangSmith/TruLens: ver retrieved chunks + prompt + respuesta
# para cada request con score bajo
```

## El ciego: Context Trustworthiness

Las 4 métricas core asumen que el índice es confiable. Ninguna mide si el contenido recuperado es correcto en sentido business.

- **Freshness:** ¿Cuándo fue la última vez que se verificó este contenido?

- **Ownership:** ¿Hay un dueño responsable de mantenerlo exacto?

- **Lineage integrity:** ¿Se puede trazar a la fuente canónica?

- **Canonical alignment:** ¿Es consistente con la definición canónica de la organización?

<div class="callout success">
<div class="callout-title">La paradoja de faithfulness alta + respuesta incorrecta</div>
<p>Un sistema puede tener faithfulness 0.95 (la respuesta refleja fielmente lo recuperado) y aun así estar equivocado porque el contenido recuperado está obsoleto, inconsistente con la fuente canónica, o no alineado con definiciones organizacionales. La confianza del índice es la dimensión oculta.</p>
</div>

## Práctica


<div class="exercise">
<div class="exercise-title">Ejercicio 1: Evalúa tu pipeline</div>
<p>Usa el código de RAGAS de arriba para evaluar tu pipeline con las 4 métricas core:</p>
<ul>
<li>Crea un dataset de 20 preguntas con ground truth</li>
<li>Ejecuta faithfulness, answer_relevancy, context_precision, context_recall</li>
<li>Identifica: ¿el problema está en el retriever o en el generator?</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 2: Alinea tu juez</div>
<p>Selecciona 50 respuestas, evaluadas por un experto humano (pass/fail):</p>
<ul>
<li>Mide alineación del juez LLM vs humano</li>
<li>Analiza falsos positivos (juez dice pass, humano dice fail)</li>
<li>Refina el prompt del juez y re-mide</li>
<li>Objetivo: >85% alineación</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 3: CI gate</div>
<p>Integra RAGAS en un script de testing:</p>
<ul>
<li>Define umbrales: faithfulness >= 0.85, answer_relevancy >= 0.75</li>
<li>Ejecuta contra tu dataset de eval</li>
<li>Retorna exit code != 0 si cualquier métrica falla</li>
<li>Wire it a tu pipeline de deploy</li>
</ul>
</div>