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
    question: "Cual es la metrica mas critica para detectar hallucinations en RAG?"
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
    question: "Como mide RAGAS Answer Relevance sin ground truth?"
    options:
      - text: "Compara con la respuesta de referencia"
        correct: false
      - text: "Genera N preguntas desde la respuesta y mide similaridad con la pregunta original"
        correct: true
      - text: "Usa cosine similarity entre respuesta y contexto"
        correct: false
      - text: "Pide al LLM que de un score del 1 al 10"
        correct: false
  - id: "q4"
    question: "Un sistema con faithfulness 0.95 puede dar respuestas incorrectas. Por que?"
    options:
      - text: "Porque faithfulness no es una metrica confiable"
        correct: false
      - text: "Porque el contexto recuperado puede estar obsoleto o ser incorrecto business-mente"
        correct: true
      - text: "Porque el LLM hallucina de todas formas"
        correct: false
      - text: "Porque faithfulness solo mide longitud de la respuesta"
        correct: false
  - id: "q5"
    question: "Que porcentaje de alineacion con juicio humano deberia tener un juez LLM?"
    options:
      - text: "50% (azar)"
        correct: false
      - text: "70% (aceptable)"
        correct: false
      - text: ">85% (produccion)"
        correct: true
      - text: "100% (necesario)"
        correct: false

---


## Objetivo

Al finalizar, sabras medir la calidad de un sistema RAG usando el RAG Triad, implementar evaluacion con LLM-as-a-judge, configurar RAGAS, y construir un pipeline de evaluacion continua para CI/CD.

## El RAG Triad: 3 relaciones, 3 metricas

El framework diagnostico mas util para RAG mide 3 relaciones entre los componentes del sistema:

<small>Fuente: [arXiv:2309.15217 — RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)</small>

<div class="callout info">
<div class="callout-title">Las 3 relaciones criticas</div>
<p><strong>1. Context Relevance</strong> (query -> contexto): El retriever encontro chunks relevantes?

<strong>2. Faithfulness</strong> (contexto -> respuesta): La respuesta esta soportada por los chunks?

<strong>3. Answer Relevance</strong> (query -> respuesta): La respuesta realmente responde la pregunta?</p>
</div>

### Metrica 1: Faithfulness (la mas critica)

Mide que proporcion de afirmaciones de la respuesta estan soportadas por el contexto recuperado. Es tu herramienta principal contra hallucinations.

```
# Formula de Faithfulness (RAGAS)
faithfulness = claims_supported / total_claims

# Ejemplo:
# Contexto: "Einstein nacio el 14 de marzo de 1879 en Alemania"
# Respuesta: "Einstein nacio en Alemania el 20 de marzo de 1879"
#
# Afirmaciones extraidas:
#   1. "Einstein nacio en Alemania" -> SUPPORTED
#   2. "Einstein nacio el 20 de marzo de 1879" -> NOT SUPPORTED
# Faithfulness = 1/2 = 0.5
```

- **Rango:** 0.0 (nada soportado) a 1.0 (todo soportado)

- **Umbral tipico:** >= 0.85 para production

- **Interpretacion:** 0.95 = 95% de las afirmaciones trazan al contexto recuperado. 0.6 = 40% viene de memoria parametrica o fabricacion

### Metrica 2: Context Relevance (Precision del retriever)

Mide que proporcion del contexto recuperado es realmente util para responder la pregunta. Penaliza informacion redundante o irrelevante.

```
# Formula de Context Relevance (RAGAS)
# LLM extrae oraciones "cruciales" del contexto
context_relevance = |extracted_sentences| / |total_sentences|

# Si recuperaste 10 oraciones pero solo 3 son utiles:
# Context Relevance = 3/10 = 0.3 -> PROBLEMA en el retriever
```

- **Baja context relevance** = chunks demasiado grandes, o embeddings mal alineados

- **Accion:** Reducir chunk size, mejorar reranking, agregar metadata filters

### Metrica 3: Answer Relevance

Mide si la respuesta realmente responde la pregunta. Penaliza respuestas incompletas, redundantes o tangenciales.

```
# Metodo de RAGAS: ingenieria inversa de preguntas
# 1. Dada la respuesta, generar N preguntas que podria responder
generated_questions = llm.generate_questions(answer, n=3)
# ["En que pais nacio Einstein?", "Cual es la nacionalidad de Einstein?", ...]

# 2. Calcular similaridad coseno promedio con la pregunta original
answer_relevance = mean(cosine_sim(q_original, q_generated))
```

## Metricas de Retrieval (para afinar el retriever)

| Metrica | Que mide | Cuando usarla | Ejemplo |
| --- | --- | --- | --- |
| **Recall@K** | Proporcion de docs relevantes que aparecen en top-K | Dominios donde missing un doc = respuesta incompleta (legal, medico) | 5 docs relevantes, 3 en top-10 = Recall@10 = 0.6 |
| **Precision@K** | Proporcion de top-K que son realmente relevantes | Cuando el contexto window es limitado (costo tokens) | 10 docs, 3 utiles = Precision@10 = 0.3 |
| **MRR** | Rango del primer doc relevante | Single-answer lookup (FAQ, entity search) | Mejor doc siempre 1ro = MRR = 1.0 |
| **NDCG** | Relevancia gradiente (no binaria) | Queries complejas donde distintos docs contribuyen diferente | Pondera posicion con discount logaritmico |

## LLM-as-a-Judge: El paradigma dominante

Los metricos BLEU y ROUGE miden overlap de tokens — no capturan equivalencia semantica. El paradigma actual usa un LLM como juez para evaluar calidad semantica.

### Como funciona
```
# Patron basico de LLM-as-Judge
judge_prompt = f"""
Evalua si la respuesta esta soportada por el contexto.

Contexto: {retrieved_context}
Respuesta: {generated_answer}

Para cada afirmacion en la respuesta, indica si es:
- supported: inferida del contexto
- unsupported: no se puede inferir
- contradicted: contradice el contexto

Output: JSON con lista de veredictos y score de faithfulness.
"""
```

### Frameworks de evaluacion

| Framework | Metodo | CI/CD | Mejor para |
| --- | --- | --- | --- |
| **RAGAS** | Reference-free, 4 metricas core | Moderado | Eval rapida de RAG, sin ground truth |
| **DeepEval** | LLM-as-judge + custom metrics | Fuerte | CI/CD pipelines, pytest integration |
| **TruLens** | Feedback functions + tracing | Moderado | Debugging con visibilidad de traces |
| **LangSmith** | Offline + online eval | Moderado | Equipos usando LangChain |
| **Opik** | G-Eval + LiteLLM | Moderado | Judge flexible (cualquier LLM) |

### Implementacion con RAGAS

```
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Preparar dataset de evaluacion
eval_data = Dataset.from_dict({
"question": ["Cual es el timeout default?"],
"answer": ["El timeout default es 30 segundos"],
"contexts": [["El timeout se configura en settings.yaml. Default: 30s"]],
"ground_truth": ["30 segundos"]
})

# Ejecutar evaluacion
results = evaluate(
eval_data,
metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

# Resultados
print(results)
# {'faithfulness': 0.95, 'answer_relevancy': 0.88,
#  'context_precision': 0.82, 'context_recall': 0.91}
```

## Alinear el juez: El paso que la mayoria olvida

Un juez desalineado es como una brujula apuntando al norte incorrecto. La alineacion del juez contra juicio humano experto es el fundamento de evaluacion confiable.

<div class="callout warning">
<div class="callout-title">El problema comun</div>
<p>Un juez baseline tiene ~75.6% de alineacion con expertos. Todos los falsos positivos son el juez siendo "too lenient" — deja pasar respuestas que omiten conceptos criticos. La solucion: refinar el prompt del juez con criteria especificos y reglas criticas.

<small>Fuente: <a href="https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/faithfulness/">RAGAS Docs — Judge Alignment Process</a></small></p>
</div>

```
# Proceso de alineacion (de Ragas docs)
# 1. Recopilar 100-200 ejemplos con juicio humano experto
# 2. Evaluar juez baseline -> medir alineacion
alignment = judge.evaluate(dataset, human_labels)
# baseline: 75.6% alignment

# 3. Analizar errores (que tipo de falsos positivos?)
#    - El juez no detecta conceptos omitidos
#    - El juez es too permisivo con parciales

# 4. Refinar prompt con reglas especificas
enhanced_prompt = """
Evalua si la respuesta cubre TODOS los conceptos clave.
- Acepta equivalentes semanticos
- Cuenta conceptos ausentes
- 0 conceptos faltantes = PASS
- 1+ conceptos faltantes = FAIL
...
"""

# 5. Re-evaluar -> medir mejora
# enhanced: 86.9% alignment (+11.3% improvement)
```

## Metricas enterprise: Mas alla del RAG Triad

Para sistemas enterprise multi-turn, 8 metricas operacionales (de arxiv:2602.20379):

<small>Fuente: [arXiv:2602.20379 — Case-Aware Evaluation for Enterprise RAG](https://arxiv.org/abs/2602.20379)</small>

| Metrica | Que evalua | Peso |
| --- | --- | --- |
| **Hallucination / Grounding Fidelity** | Afirmaciones soportadas por el contexto | 0.20 |
| **Retrieval Correctness** | Version, entorno, doc correcto | 0.15 |
| **Answer Helpfulness** | Accionalidad y claridad | 0.15 |
| **Context Sufficiency** | Contexto cubre todo lo necesario | 0.10 |
| **Answer Type Fit** | Tipo de respuesta adecuado (diagnostico vs instruccion) | 0.10 |
| **Identifier Integrity** | Error codes, IDs, paths preservados correctamente | 0.10 |
| **Case Issue Identification** | Identifica correctamente el problema del caso | 0.10 |
| **Resolution Alignment** | Pasos satisfacen restricciones del workflow | 0.10 |

<div class="callout info">
<div class="callout-title">Por que las metricas genericas no bastan</div>
<p>En un benchmark enterprise, Llama tenia mejor faithfulness (0.84) que GPT-oss (0.58), pero GPT-oss tenia mejor Answer Type Fit y Resolution Alignment. Las metricas genericas sugirian que Llama es mejor, pero GPT-oss resolvia mejor los casos de uso reales.</p>
</div>

## Evaluacion continua: Pre, en, y post-production

```
# Pipeline de evaluacion continua

# 1. PRE-PRODUCTION: Eval offline con dataset curado
eval_dataset = load_eval_set("golden_100.json")
results = evaluate(eval_dataset, metrics=[faithfulness, ...])

# 2. CI/CD: Gating en pipeline de deploy
if results["faithfulness"] < 0.85:
raise ValueError("Faithfulness regression detected")
if results["answer_relevancy"] < 0.75:
raise ValueError("Answer relevance regression detected")

# 3. PRODUCTION: Sampling continuo
# - Evaluar 5-10% de trafico real
# - Monitor drift semanal
# - Alertas cuando metricas caen

# 4. POST-INCIDENT: Debugging con traces
# LangSmith/TruLens: ver retrieved chunks + prompt + respuesta
# para cada request con score bajo
```

## El ciego: Context Trustworthiness

Las 4 metricas core asumen que el indice es confiable. Ninguna mide si el contenido recuperado es correcto en sentido business.

- **Freshness:** Cuando fue la ultima vez que se verifico este contenido?

- **Ownership:** Hay un dueno responsable de mantenerlo exacto?

- **Lineage integrity:** Se puede trazar a la fuente canonica?

- **Canonical alignment:** Es consistente con la definicion canonica de la organizacion?

<div class="callout success">
<div class="callout-title">La paradoja de faithfulness alta + respuesta incorrecta</div>
<p>Un sistema puede tener faithfulness 0.95 (la respuesta refleja fielmente lo recuperado) y aun asi estar equivocado porque el contenido recuperado esta obsoleto, inconsistente con la fuente canonica, o no alineado con definiciones organizacionales. La confianza del indice es la dimension oculta.</p>
</div>

## Practica


<div class="exercise">
<div class="exercise-title">Ejercicio 1: Evalua tu pipeline</div>
<p>Usando RAGAS, evalua el pipeline del Capitulo 1 con las 4 metricas core:</p>
<ul>
<li>Crea un dataset de 20 preguntas con ground truth</li>
<li>Ejecuta faithfulness, answer_relevancy, context_precision, context_recall</li>
<li>Identifica: ¿el problema esta en el retriever o en el generator?</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 2: Alinea tu juez</div>
<p>Selecciona 50 respuestas, evaluadas por un experto humano (pass/fail):</p>
<ul>
<li>Mide alineacion del juez LLM vs humano</li>
<li>Analiza falsos positivos (juez dice pass, humano dice fail)</li>
<li>Refina el prompt del juez y re-mide</li>
<li>Objetivo: >85% alineacion</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 3: CI gate</div>
<p>Integra RAGAS en un script de testing:</p>
<ul>
<li>Define umbrales: faithfulness >= 0.85, answer_relevancy >= 0.75</li>
<li>Ejecuta contra tu dataset de eval</li>
<li>Retorna exit code != 0 si cualquier metrica falla</li>
<li>Wire it a tu pipeline de deploy</li>
</ul>
<p>## Verifica tu comprension</p>
<p><p>1. Cual es la metrica mas critica para detectar hallucinations en RAG?</p></p>
<p>Answer Relevance</p>
<p>Faithfulness</p>
<p>Context Precision</p>
<p>MRR</p>
<p><p>2. Baja context precision indica un problema con...</p></p>
<p>El LLM generator</p>
<p>El retriever (chunks irrelevantes o demasiado grandes)</p>
<p>La pregunta del usuario</p>
<p>El embedding model</p>
<p><p>3. Como mide RAGAS Answer Relevance sin ground truth?</p></p>
<p>Compara con la respuesta de referencia</p>
<p>Genera N preguntas desde la respuesta y mide similaridad con la pregunta original</p>
<p>Usa cosine similarity entre respuesta y contexto</p>
<p>Pide al LLM que de un score del 1 al 10</p>
<p><p>4. Un sistema con faithfulness 0.95 puede dar respuestas incorrectas. Por que?</p></p>
<p>Porque faithfulness no es una metrica confiable</p>
<p>Porque el contexto recuperado puede estar obsoleto o ser incorrecto business-mente</p>
<p>Porque el LLM hallucina de todas formas</p>
<p>Porque faithfulness solo mide longitud de la respuesta</p>
<p><p>5. Que porcentaje de alineacion con juicio humano deberia tener un juez LLM?</p></p>
<p>50% (azar)</p>
<p>70% (aceptable)</p>
<p>&gt;85% (produccion)</p>
<p>100% (necesario)</p>
</div>
