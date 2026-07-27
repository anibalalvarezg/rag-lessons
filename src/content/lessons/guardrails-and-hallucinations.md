---
title: "Guardrails y Detección de Alucinaciones"
subtitle: "Lección 5 — RAGOps y Evaluación"
pillar: ragops
pillarName: "RAGOps y Evaluación"
lessonNum: 5
description: "Taxonomía de alucinaciones (3 tipos), 4 señales de detección, HALO, 41.77% fallas de agentes."
keywords: "guardrails, hallucinations, alucinaciones, HALO, factuality, faithfulness"
ogSection: "RAGOps y Evaluación"
pubDate: "2026-07-24"
quizzes:

  - id: "q1"
    question: "¿Qué tipo de hallucination NO es detectable por métodos geométricos/embeddings?"
    options:
      - text: "Tipo I: Unfaithfulness"
        correct: false
      - text: "Tipo II: Confabulation"
        correct: false
      - text: "Tipo III: Factual error within frame"
        correct: true
      - text: "Todos son detectables"
        correct: false
  - id: "q2"
    question: "¿Cuál es la señal de detección MÁS fuerte en RAG?"
    options:
      - text: "Token confidence scores"
        correct: false
      - text: "Grounding check (verificar cada claim contra el contexto)"
        correct: true
      - text: "Self-consistency con multi-sampling"
        correct: false
      - text: "Cosine similarity entre respuesta y contexto"
        correct: false
  - id: "q3"
    question: "¿Por qué los métodos basados en embeddings fallan contra modelos RLHF?"
    options:
      - text: "Porque los embeddings son de baja calidad"
        correct: false
      - text: "Porque RLHF cambia el espacio de embeddings"
        correct: false
      - text: "Porque RLHF produce hallucinations semánticamente plausibles que preservan similaridad con la fuente"
        correct: true
      - text: "Porque los embeddings no soportan texto largo"
        correct: false
  - id: "q4"
    question: "En el funnel de detección, ¿qué señal se ejecuta en TODAS las requests?"
    options:
      - text: "Grounding check completo (LLM-as-judge)"
        correct: false
      - text: "Señal barata (SGI, token confidence) como pre-filter"
        correct: true
      - text: "Self-consistency con 3 muestras"
        correct: false
      - text: "Ninguna, solo en high-stakes"
        correct: false
  - id: "q5"
    question: "¿Qué es \"semantic laziness\"?"
    options:
      - text: "El modelo responde muy lento"
        correct: false
      - text: "Las hallucinations se quedan angularmente cerca de la pregunta en vez de moverse hacia el contexto"
        correct: true
      - text: "El embedding model es demasiado simple"
        correct: false
      - text: "El retrieval retorna chunks irrelevantes"
        correct: false

---


## Objetivo

Al finalizar, sabrás cómo clasificar hallucinations en 3 tipos, implementar guardrails de entrada/salida, y construir un pipeline de detección que combine múltiples señales baratas para decidir si confiar, abstenerse o escalar.

**Prerequisitos:** [Lección 4](/rag-lessons/lessons/hybrid-retrieval-and-reranking) — las señales de esta lección envuelven ese pipeline.

<div class="callout info">
<div class="callout-title">🧵 Proyecto Acme — De dónde viene este código</div>
<p>Las verificaciones de esta lección se insertan <em>después</em> del pipeline de la Lección 4 (Transform → Retrieve → Rerank → Generate): toman la <code>query</code>, los <code>chunks</code> rerankeados y la <code>response</code> generada, y deciden si la respuesta del asistente Acme se publica, se corrige o se abstiene. El ejemplo "timeout / settings.yaml / 30 segundos" es la misma consulta al <code>manual-tecnico.pdf</code> que usaste en la Lección 4 — aquí aprendes a detectar cuándo el sistema la responde mal.</p>
</div>

## Taxonomía de hallucinations: 3 tipos, 3 firmas

No todas las hallucinations son iguales. La taxonomía de Marin (2026) identifica 3 tipos con firmas geométricas distintas:

<small>Fuente: [arXiv:2602.13224 — Marin et al., "Hallucination Taxonomy for RAG" (2026)](https://arxiv.org/abs/2602.13224)</small>

| Tipo | Nombre | Qué pasa | Detectable? | Ejemplo |
| --- | --- | --- | --- | --- |
| **Tipo I** | Unfaithfulness (query-proximate) | El modelo ignora el contexto recuperado y genera desde memoria paramétrica | SÍ, vía SGI (AUC 0.80-0.95) | El usuario pregunta sobre el contrato X, el LLM responde con información genérica del dominio ignorando los chunks |
| **Tipo II** | Confabulation | El modelo inventa contenido que cae fuera de la región de plausibilidad | SÍ, vía DGI (AUC 0.85-0.99) | Pregunta sobre CRISPR, el LLM describe "protein-folding correction" en lugar de DNA cleavage |
| **Tipo III** | Factual error within frame | Error factual dentro del marco conceptual correcto | NO, geométricamente invisible | "El sol causa estaciones por su distancia" vs "por inclinación del eje" — misma región del espacio de embeddings |

<div class="callout warning">
<div class="callout-title">La ilusión semántica</div>
<p>Los embeddings capturan semántica superficial (vocabulario, topicalidad) pero NO capturan verdad. Dos afirmaciones sobre el mismo tema ocupan la misma región del espacio de embeddings, sean correctas o no. Los métodos basados en embeddings fallan 100% en HaluEval y 88% en RAGTruth contra modelos RLHF. Solo el razonamiento (LLM-as-judge) puede detectar Tipo III.</p>
</div>

## Guardrails: Defensa en capas

Un guardrail es un control estructurado que valida input y output en cada etapa del pipeline. No es un solo check — es un funnel de múltiples señales.

### 1. Input Guardrails (antes del retrieval)

- **Validación de schema:** Verifica que el input del usuario sea del tipo esperado (formato, longitud, campos requeridos)

- **Detección de prompt injection:** Patrón de regex + LLM classifier para detectar intentos de override

- **Clasificación de dominio:** Redirige queries fuera del ámbito del sistema

- **Filtros de PII:** Detecta y redacta datos sensibles antes de procesar

### 2. Retrieval Guardrails (durante la búsqueda)

- **Relevance threshold:** No pasar al LLM chunks con score bajo un umbral

- **Metadata filters:** RBAC — solo acceder a fuentes autorizadas para el usuario

- **Deduplicación:** Evitar que el contexto contenga información contradictoria

### 3. Output Guardrails (después de la generación)

Aquí es donde ocurre la detección de hallucinations. El patrón práctico es un funnel de coste:

<div class="callout info">
<div class="callout-title">El funnel de detección</div>
<p>Ejecuta la señal barata en TODAS las requests. Solo ejecuta el check caro cuando la señal barata detecta ruido o el request es high-stakes.</p>
</div>

## Señal 1: Grounding Check (la más fuerte en RAG)

Dado que en RAG siempre tienes el contexto recuperado, la verificación más poderosa es simple: **¿cada afirmación de la respuesta está soportada por los chunks recuperados?**

```python
# Flujo de grounding check
# 1. Descomponer respuesta en afirmaciones atómicas
claims = extract_claims(response)
# ["El timeout default es 30 segundos",
#  "Se configura en settings.yaml",
#  "El campo se llama network.timeout"]

# 2. Verificar cada afirmación contra el contexto
for claim in claims:
    verdict = judge_model.verify(
        claim=claim,
        context=retrieved_chunks,
        # Retorna: supported | unsupported | contradicted
    )
    if verdict != "supported":
        flag(claim, verdict)

# 3. Decidir acción basado en % de afirmaciones soportadas
```

- **Implementación:** LLM-as-judge pasando contexto + respuesta + instrucciones de verificación

- **Costo:** +1 LLM call (~200-600ms)

- **Limitación:** El juez también puede hallucinar. Mantener el check enfocado (una afirmación a la vez, output estructurado)

## Señal 2: Semantic Grounding Index (SGI)

Mide si la respuesta se movió hacia el contexto o se quedó cerca de la pregunta. Basado en geometría angular en la hiperesfera unitaria.

<small>Fuente: [arXiv:2512.13771 — Semantic Grounding Index (SGI) Paper](https://arxiv.org/abs/2512.13771)</small>

```python
# SGI = theta(response, question) / theta(response, context)
# SGI > 1: respuesta cerca del contexto (grounded) -> PASS
# SGI < 1: respuesta cerca de la pregunta (potentially hallucinated) -> FLAG

from groundlens import compute_sgi

result = compute_sgi(
    question="¿Cómo configurar el timeout?",
    context="El timeout se configura en settings.yaml bajo network.timeout...",
    response="El timeout se configura en settings.yaml con un valor de 30s.",
    model="all-MiniLM-L6-v2"  # rápido, barato
)
# result.value = 1.35 -> PASS
# result.flagged = False
```

| Zona | SGI | Acción |
| --- | --- | --- |
| Pass fuerte | SGI >= 1.20 | Aceptar |
| Parcial | 0.95 <= SGI < 1.20 | Revisar |
| Flagged | SGI < 0.95 | Revisión humana obligatoria |

- **Ventaja:** Single embedding call, ~10ms, zero LLM cost

- **Limitación:** Detecta Tipo I (unfaithfulness) pero NO Tipo III (factual errors within frame). No mide correctness, solo engagement con el contexto

## Señal 3: Self-Consistency (para queries sin contexto)

Cuando no hay contexto para verificar contra, genera N respuestas con temperature > 0 y compara. Los hechos reales son consistentes entre muestras; los inventados varían.

```python
# Generar 3 muestras con temperature=0.7
samples = [llm.generate(query, temp=0.7) for _ in range(3)]
# Comparar hechos clave entre muestras
consistency = compare_facts(samples)
# Alta consistencia -> probablemente correcto
# Baja consistencia -> probablemente hallucination
```

## Señal 4: Token Confidence (señal barata, ruidosa)

Los log-probabilities de tokens indican incertidumbre del modelo. Tokens con baja probabilidad frecuentemente coinciden con spans fabricados.

- **Ventaja:** Gratis si el modelo expone logprobs

- **Limitación:** Noisy — un modelo puede ser confidentemente incorrecto. Usar como pre-filter, nunca como único gate

## Estrategias de respuesta: Qué hacer cuando detectas una hallucination

| Acción | Qué ve el usuario | Cuándo usar |
| --- | --- | --- |
| **Pass** | Respuesta completa | Todas las señales en verde |
| **Disclaim** | Respuesta + "verificar, no estoy seguro" | Riesgo medio: respuesta probablemente útil pero shaky |
| **Retry/Repair** | Nueva respuesta después de re-retrieval | Falla arreglable (chunks malos, contexto incompleto) |
| **Abstain** | "No tengo una respuesta confiable para eso" | Dominios high-stakes donde una respuesta mala es peor que ninguna |
| **Escalate** | Handoff a humano | Legal, médico, facturación, acciones críticas |

## Herramientas de producción

| Herramienta | Tipo | Qué hace | Mejor para |
| --- | --- | --- | --- |
| **Guardrails AI** | Open source | Valida outputs contra schemas definidos | Output estructurado, validación de formato |
| **NVIDIA NeMo Guardrails** | Open source | Topical rails, safety, jailbreak detection | Chatbots, restricción de dominio |
| **HalluGuard (4B)** | Modelo especializado | Clasifica document-claim como grounded/hallucinated con justificación | RAG pipelines enterprise, 84.4% BAcc vs 75.9% GPT-4o. arXiv:2506.10525 |
| **groundlens (SGI/DGI)** | Geométrico | Detección basada en embeddings, single-pass | Pre-filter rápido (~10ms, zero LLM cost). PyPI: groundlens |
| **Vectara FaithJudge** | Leaderboard + API | Benchmarks LLM faithfulness con anotaciones humanas | Evaluación offline, selección de modelos |
| **LangChain OutputParser** | Framework | Output parsing con retry automático | Validación de formato + retry |

## Pipeline completo de guardrails en producción

```python
# HALO-inspired architecture (6 capas)
# Ref: arxiv.org/abs/2607.17883

class RAGGuardrails:
    def process(self, query, user_context):
        # CAPA 1: Input validation
        query = self.validate_input(query)  # schema, injection, PII
        if query.rejected: return self.abstain("Invalid input")

        # CAPA 2: Grounded retrieval
        chunks = self.retrieve(query, user_context.rbac)
        if not chunks: return self.abstain("No relevant context")

        # CAPA 3: Generation
        response = self.generate(query, chunks)

        # CAPA 4: Multi-signal verification
        signals = {
            "sgi": compute_sgi(query, chunks, response),        # ~10ms
            "grounding": self.grounding_check(query, chunks, response),  # ~400ms
            "confidence": self.token_confidence(response),       # free
        }

        # CAPA 5: Decision funnel
        risk = self.assess_risk(signals, query.stakes)
        if risk == "high" and signals["grounding"].unsupported > 0.2:
            return self.retry_or_abstain(query, chunks)
        elif risk == "medium" and signals["sgi"].flagged:
            return self.add_disclaimer(response)
        else:
            return self.accept(response)

        # CAPA 6: Logging + continuous oversight
        self.log(query, response, signals, decision)
        self.monitor_drift(signals)
```

<div class="callout success">
<div class="callout-title">El principio fundamental</div>
<p>"Zero hallucination" no es una propiedad del modelo — es una propiedad del sistema. Los LLMs son, por construcción, capaces de generar texto no soportado. Ninguna cantidad de escala elimina esta posibilidad. El sistema debe contener la falla, no eliminarla.</p>
</div>

## Práctica


<div class="exercise">
<div class="exercise-title">Ejercicio 1: Grounding check</div>
<p>Usa el código de grounding check de arriba como base e impleméntalo para:</p>
<ul>
<li>Descomponga la respuesta en afirmaciones atómicas</li>
<li>Verifique cada una contra los chunks recuperados</li>
<li>Reporte % de afirmaciones soportadas vs no soportadas</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 2: SGI como pre-filter</div>
<p>Usa el código de SGI de arriba como filtro rápido antes del grounding check completo:</p>
<ul>
<li>SGI >= 1.20: aceptar directo (sin LLM call extra)</li>
<li>0.95 <= SGI < 1.20: ejecutar grounding check completo</li>
<li>SGI < 0.95: abstenerse o re-retrieve</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 3: Taxonomía en acción</div>
<p>Genera 3 tipos de hallucination intencionales:</p>
<ul>
<li>Tipo I: Ignora el contexto y responde de memoria</li>
<li>Tipo II: Inventa un concepto que no existe en el corpus</li>
<li>Tipo III: Cambia un dato numérico (30s -> 60s) pero mantiene el marco</li>
</ul>
<p>Mide qué señales detectan cada tipo. Confirma que Tipo III escapa a SGI y embeddings.</p>
</div>