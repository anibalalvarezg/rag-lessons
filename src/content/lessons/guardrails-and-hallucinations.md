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
    question: "Que tipo de hallucination NO es detectable por metodos geometricos/embeddings?"
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
    question: "Cual es el signal de deteccion MAS fuerte en RAG?"
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
    question: "Por que los metodos basados en embeddings fallan contra modelos RLHF?"
    options:
      - text: "Porque los embeddings son de baja calidad"
        correct: false
      - text: "Porque RLHF cambia el espacio de embeddings"
        correct: false
      - text: "Porque RLHF produce hallucinations semanticamente plausibles que preservan similaridad con la fuente"
        correct: true
      - text: "Porque los embeddings no soportan texto largo"
        correct: false
  - id: "q4"
    question: "En el funnel de deteccion, que senal se ejecuta en TODAS las requests?"
    options:
      - text: "Grounding check completo (LLM-as-judge)"
        correct: false
      - text: "Signal barato (SGI, token confidence) como pre-filter"
        correct: true
      - text: "Self-consistency con 3 muestras"
        correct: false
      - text: "Ninguna, solo en high-stakes"
        correct: false
  - id: "q5"
    question: "Que es \"semantic laziness\"?"
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

Al finalizar, sabras como clasificar hallucinations en 3 tipos, implementar guardrails de entrada/salida, y construir un pipeline de deteccion que combine multiples senales baratas para decidir si confiar, abstenerse o escalar.

## Taxonomia de hallucinations: 3 tipos, 3 firmas

No todas las hallucinations son iguales. La taxonomy de Marin (2026) identifica 3 tipos con firmas geometricas distintas:

<small>Fuente: [arXiv:2602.13224 — Marin et al., "Hallucination Taxonomy for RAG" (2026)](https://arxiv.org/abs/2602.13224)</small>

| Tipo | Nombre | Que pasa | Detectable? | Ejemplo |
| --- | --- | --- | --- | --- |
| **Tip I** | Unfaithfulness (query-proximate) | El modelo ignora el contexto recuperado y genera desde memoria parametrica | SI SGI (AUC 0.80-0.95) | El usuario pregunta sobre el contrato X, el LLM responde con informacion generica del dominio ignorando los chunks |
| **Tip II** | Confabulation | El modelo inventa contenido que cae fuera de la region de plausibilidad | SI DGI (AUC 0.85-0.99) | Pregunta sobre CRISPR, el LLM describe "protein-folding correction" en lugar de DNA cleavage |
| **Tip III** | Factual error within frame | Error factual dentro del marco conceptual correcto | NO geometricamente invisible | "El sol causa estaciones por su distancia" vs "por inclinacion del eje" — misma region del espacio de embeddings |

<div class="callout warning">
<div class="callout-title">La ilusion semantica</div>
<p>Los embeddings capturan semantica superficial (vocabulario, topicalidad) pero NO capturan verdad. Dos afirmaciones sobre el mismo tema ocupan la misma region del espacio de embeddings, sean correctas o no. Los metodos basados en embeddings fallan 100% en HaluEval y 88% en RAGTruth contra modelos RLHF. Solo el razonamiento (LLM-as-judge) puede detectar Tipo III.</p>
</div>

## Guardrails: Defensa en capas

Un guardrail es un control estructurado que valida input y output en cada etapa del pipeline. No es un solo check — es un funnel de multiples senales.

### 1. Input Guardrails (antes del retrieval)

- **Validacion de schema:** Verifica que el input del usuario sea del tipo esperado (formato, longitud, campos requeridos)

- **Deteccion de prompt injection:** Patron de regex + LLM classifier para detectar intentos de override

- **Clasificacion de dominio:** Redirige queries fuera del ambito del sistema

- **Filtros de PII:** Detecta y redacta datos sensibles antes de procesar

### 2. Retrieval Guardrails (durante la busqueda)

- **Relevance threshold:** No pasar al LLM chunks con score bajo un umbral

- **Metadata filters:** RBAC — solo acceder a fuentes autorizadas para el usuario

- **Deduplicacion:** Evitar que el contexto contenga informacion contradictoria

### 3. Output Guardrails (despues de la generacion)

Aqui es donde ocurre la deteccion de hallucinations. El patron practico es un funnel de coste:

<div class="callout info">
<div class="callout-title">El funnel de deteccion</div>
<p>Ejecuta el signal barato en TODAS las requests. Solo ejecuta el check caro cuando el signal barato detecta ruido o el request es high-stakes.</p>
</div>

## Signal 1: Grounding Check (el mas fuerte en RAG)

Dado que en RAG siempre tienes el contexto recuperado, la verificacion mas poderosa es simple: **cada afirmacion de la respuesta esta soportada por los chunks recuperados?**

```
# Flujo de grounding check
# 1. Descomponer respuesta en afirmaciones atomicas
claims = extract_claims(response)
# ["El timeout default es 30 segundos",
#  "Se configura en settings.yaml",
#  "El campo se llama network.timeout"]

# 2. Verificar cada afirmacion contra el contexto
for claim in claims:
verdict = judge_model.verify(
claim=claim,
context=retrieved_chunks,
# Retorna: supported | unsupported | contradicted
)
if verdict != "supported":
flag(claim, verdict)

# 3. Decidir accion basado en % de afirmaciones soportadas
```

- **Implementacion:** LLM-as-judge pasando contexto + respuesta + instrucciones de verificacion

- **Costo:** +1 LLM call (~200-600ms)

- **Limitacion:** El juez tambien puede hallucinar. Mantener el check enfocado (una afirmacion a la vez, output estructurado)

## Signal 2: Semantic Grounding Index (SGI)

Mide si la respuesta se movio hacia el contexto o se quedo cerca de la pregunta. Basado en geometria angular en el hiperesfera unitario.

<small>Fuente: [arXiv:2512.13771 — Semantic Grounding Index (SGI) Paper](https://arxiv.org/abs/2512.13771)</small>

```
# SGI = theta(response, question) / theta(response, context)
# SGI > 1: respuesta cerca del contexto (grounded) -> PASS
# SGI < 1: respuesta cerca de la pregunta (potentially hallucinated) -> FLAG

from groundlens import compute_sgi

result = compute_sgi(
question="Como configurar el timeout?",
context="El timeout se configura en settings.yaml bajo network.timeout...",
response="El timeout se configura en settings.yaml con un valor de 30s.",
model="all-MiniLM-L6-v2"  # rapido, barato
)
# result.value = 1.35 -> PASS
# result.flagged = False
```

| Zona | SGI | Accion |
| --- | --- | --- |
| Pass fuerte | SGI >= 1.20 | Aceptar |
| Parcial | 0.95 <= SGI < 1.20 | Revisar |
| Flagged | SGI < 0.95 | Revision humana obligatoria |

- **Ventaja:** Single embedding call, ~10ms, zero LLM cost

- **Limitacion:** Detecta Tipo I (unfaithfulness) pero NO Tipo III (factual errors within frame). No mide correctness, solo engagement con el contexto

## Signal 3: Self-Consistency (para queries sin contexto)

Cuando no hay contexto para verificar contra, genera N respuestas con temperature > 0 y compara. Los hechos reales son consistentes entre muestras; los inventados varian.

```
# Generar 3 muestras con temperature=0.7
samples = [llm.generate(query, temp=0.7) for _ in range(3)]
# Comparar hechos clave entre muestras
consistency = compare_facts(samples)
# Alta consistencia -> probablemente correcto
# Baja consistencia -> probablemente hallucination
```

## Signal 4: Token Confidence (signal barato, ruidoso)

Los log-probabilities de tokens indican incertidumbre del modelo. Tokens con baja probabilidad frecuentemente coinciden con spans fabricados.

- **Ventaja:** Gratis si el modelo expose logprobs

- **Limitacion:** Noisy — un modelo puede ser confidentemente incorrecto. Usar como pre-filter, nunca como unico gate

## Estrategias de respuesta: Que hacer cuando detectas una hallucination

| Accion | Que ve el usuario | Cuando usar |
| --- | --- | --- |
| **Pass** | Respuesta completa | Todos los signals en verde |
| **Disclaim** | Respuesta + "verificar, no estoy seguro" | Riesgo medio: respuesta probablemente util pero shaky |
| **Retry/Repair** | Nueva respuesta despues de re-retrieval | Falla arreglable (chunks malos, contexto incompleto) |
| **Abstain** | "No tengo una respuesta confiable para eso" | Dominios high-stakes donde una respuesta mala es peor que ninguna |
| **Escalate** | Handoff a humano | Legal, medico, facturacion, acciones criticas |

## Herramientas de produccion

| Herramienta | Tipo | Que hace | Mejor para |
| --- | --- | --- | --- |
| **Guardrails AI** | Open source | Valida outputs contra schemas definidos | Output estructurado, validacion de formato |
| **NVIDIA NeMo Guardrails** | Open source | Topical rails, safety, jailbreak detection | Chatbots, restriccion de dominio |
| **HalluGuard (4B)** | Modelo especializado | Clasifica document-claim como grounded/hallucinated con justificacion | RAG pipelines enterprise, 84.4% BAcc vs 75.9% GPT-4o. arXiv:2506.10525 |
| **groundlens (SGI/DGI)** | Geometrico | Deteccion basada en embeddings, single-pass | Pre-filter rapido (~10ms, zero LLM cost). PyPI: groundlens |
| **Vectara FaithJudge** | Leaderboard + API | Benchmarks LLM faithfulness con annotaciones humanas | Evaluacion offline, seleccion de modelos |
| **LangChain OutputParser** | Framework | Output parsing con retry automatico | Validacion de formato + retry |

## Pipeline completo de guardrails en produccion

```
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
<p>"Zero hallucination" no es una propiedad del modelo — es una propiedad del sistema. Los LLMs son, por construccion, capaces de generar texto no soportado. Ninguna cantidad de escala elimina esta posibilidad. El sistema debe contener la falla, no eliminarla.</p>
</div>

## Practica


<div class="exercise">
<div class="exercise-title">Ejercicio 1: Grounding check</div>
<p>Usando el pipeline del Capitulo 1, implementa un grounding check que:</p>
<ul>
<li>Descomponga la respuesta en afirmaciones atomicas</li>
<li>Verifique cada una contra los chunks recuperados</li>
<li>Reporte % de afirmaciones soportadas vs no soportadas</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 2: SGI como pre-filter</div>
<p>Instala <code>groundlens</code> y usa SGI como filtro rapido antes del grounding check completo:</p>
<ul>
<li>SGI >= 1.20: aceptar directo (sin LLM call extra)</li>
<li>0.95 <= SGI < 1.20: ejecutar grounding check completo</li>
<li>SGI < 0.95: abstenerse o re-retrieve</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 3: Taxonomia en accion</div>
<p>Genera 3 tipos de hallucination intencionales:</p>
<ul>
<li>Tipo I: Ignora el contexto y responde de memoria</li>
<li>Tipo II: Inventa un concepto que no existe en el corpus</li>
<li>Tipo III: Cambia un dato numerico (30s -> 60s) pero mantiene el marco</li>
</ul>
<p>Mide que senales detectan cada tipo. Confirma que Tipo III escapa a SGI y embeddings.</p>
<p>## Verifica tu comprension</p>
<p><p>1. Que tipo de hallucination NO es detectable por metodos geometricos/embeddings?</p></p>
<p>Tipo I: Unfaithfulness</p>
<p>Tipo II: Confabulation</p>
<p>Tipo III: Factual error within frame</p>
<p>Todos son detectables</p>
<p><p>2. Cual es el signal de deteccion MAS fuerte en RAG?</p></p>
<p>Token confidence scores</p>
<p>Grounding check (verificar cada claim contra el contexto)</p>
<p>Self-consistency con multi-sampling</p>
<p>Cosine similarity entre respuesta y contexto</p>
<p><p>3. Por que los metodos basados en embeddings fallan contra modelos RLHF?</p></p>
<p>Porque los embeddings son de baja calidad</p>
<p>Porque RLHF cambia el espacio de embeddings</p>
<p>Porque RLHF produce hallucinations semanticamente plausibles que preservan similaridad con la fuente</p>
<p>Porque los embeddings no soportan texto largo</p>
<p><p>4. En el funnel de deteccion, que senal se ejecuta en TODAS las requests?</p></p>
<p>Grounding check completo (LLM-as-judge)</p>
<p>Signal barato (SGI, token confidence) como pre-filter</p>
<p>Self-consistency con 3 muestras</p>
<p>Ninguna, solo en high-stakes</p>
<p><p>5. Que es "semantic laziness"?</p></p>
<p>El modelo responde muy lento</p>
<p>Las hallucinations se quedan angularmente cerca de la pregunta en vez de moverse hacia el contexto</p>
<p>El embedding model es demasiado simple</p>
<p>El retrieval retorna chunks irrelevantes</p>
</div>
