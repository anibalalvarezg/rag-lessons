---
title: "Agentic RAG"
subtitle: "Lección 7 — Agentic RAG y GraphRAG"
pillar: agentic
pillarName: "Agentic RAG y GraphRAG"
lessonNum: 7
description: "ReAct, Self-Corrective RAG, MA-RAG, 8 patrones de arquitectura, 5-agentes Google (+34% factuality)."
keywords: "agentic RAG, ReAct, Self-Correcting, MA-RAG, tool use"
ogSection: "Agentic RAG y GraphRAG"
pubDate: "2026-07-24"
quizzes:

  - id: "q1"
    question: "¿Qué patrón de agente deberías usar como default?"
    options:
      - text: "Multi-agent swarm"
        correct: false
      - text: "Supervisor-worker"
        correct: false
      - text: "ReAct (single-agent)"
        correct: true
      - text: "Graph orchestration"
        correct: false
  - id: "q2"
    question: "¿Qué hace el Sufficient Context Agent de Google?"
    options:
      - text: "Genera la respuesta final"
        correct: false
      - text: "Identifica qué información falta y da feedback específico para re-buscar"
        correct: true
      - text: "Califica documentos como relevant/irrelevant"
        correct: false
      - text: "Decide qué data source usar"
        correct: false
  - id: "q3"
    question: "¿Qué porcentaje de fallos en agentes viene de problemas de diseño, no del modelo?"
    options:
      - text: "10%"
        correct: false
      - text: "25%"
        correct: false
      - text: "41.77%"
        correct: true
      - text: "60%"
        correct: false
  - id: "q4"
    question: "¿Qué es \"context poisoning\"?"
    options:
      - text: "El contexto es demasiado largo"
        correct: false
      - text: "Un error entra al contexto y se usa como ground truth en pasos subsiguientes"
        correct: true
      - text: "El agente elige el wrong tool"
        correct: false
      - text: "Nueva info contradice info existente"
        correct: false
  - id: "q5"
    question: "¿Cuándo justificas multi-agent sobre single-agent?"
    options:
      - text: "Siempre, es más avanzado"
        correct: false
      - text: "Cuando tienes más de 3 tools"
        correct: false
      - text: "Cuando midas un failure mode que un solo agente no pueda resolver"
        correct: true
      - text: "Cuando el LLM es demasiado lento"
        correct: false

---


## Objetivo

Al finalizar, entenderás los patrones de agentes RAG (ReAct, self-corrective, multi-agent), sabrás cuándo escalar de un solo agente a múltiples, y conocerás los frameworks de producción (LangGraph, LlamaIndex, CrewAI).

**Prerequisitos:** [Lección 6](/rag-lessons/lessons/evaluation-and-metrics) — un agente sin evaluación es un agente que falla en silencio.

<div class="callout info">
<div class="callout-title">🧵 Proyecto Acme — De dónde viene este código</div>
<p>El <code>retriever</code> y el <code>llm</code> de los nodos de LangGraph son los que construiste en las Lecciones 3-4 (índice <code>acme_docs</code> + reranker). Aquí el asistente de Acme deja de ser un pipeline lineal: el router decide si la pregunta del empleado va al vectorstore (políticas y manual técnico), a web search (noticias externas) o se responde directo; y el loop self-corrective re-busca cuando los documentos no son relevantes. Los guardrails de la Lección 5 siguen activos en el nodo de generación.</p>
</div>

## De RAG vanilla a Agentic RAG

RAG vanilla es un pipeline lineal: retrieve → generate. Agentic RAG agrega razonamiento iterativo: el sistema decide *cuándo* buscar, *cómo* reformular, y *cuándo* dejar de buscar.

| Aspecto | RAG vanilla | Agentic RAG |
| --- | --- | --- |
| Flujo | Lineal (una pasada) | Cíclico (loops de razonamiento) |
| Decisiones | Ninguna | Cuándo buscar, qué reformular, cuándo parar |
| Self-correction | No | Sí — re-escribe queries, recalifica docs |
| Fuentes | Una (vector store) | Multi-fuente (vector, web, APIs, grafo) |
| Complejidad | Baja | Alta — más puntos de fallo, más debugging |

## El patrón ReAct: Think → Act → Observe

El patrón fundamental de agentes. El LLM alterna entre razonamiento (thought), acción (action), y observación (observation) en un loop hasta completar la tarea.

```python
# El loop ReAct
while not task_complete:
    # THINK: Razona sobre el estado actual
    thought = llm.reason(
        question=query,
        context=current_context,
        history=previous_steps
    )
    # "Necesito buscar información sobre X antes de responder"

    # ACT: Ejecuta una tool call
    observation = tool.execute(thought.action)
    # Retrieval de docs, web search, API call, etc.

    # OBSERVE: Procesa el resultado
    current_context.append(observation)
    # Si la respuesta es suficiente -> termina
    # Si no -> repite el ciclo
```

<div class="callout info">
<div class="callout-title">Por qué ReAct funciona</div>
<p>Interleaving reasoning con action mantiene al agente grounded. Un approach solo de razonamiento puede derivar en conocimiento interno. Un approach solo de acción carece de síntesis. ReAct combina ambos.</p>
</div>

## Self-Corrective RAG: El pipeline que se arregla solo

El patrón más importante en Agentic RAG. Agrega nodos de verificación que permiten al sistema re-intentar cuando la calidad es baja.

```python
# Flujo de Self-Corrective RAG (LangGraph)
#
# [Start] -> [Retrieve] -> [Grade Documents]
#                               |
#                    +----------+----------+
#                    |                     |
#               Relevant?            Not Relevant?
#                    |                     |
#               [Generate]         [Rewrite Query]
#                    |                     |
#                  [End]            [Retrieve] (loop)
#
# El agente puede:
# 1. Calificar cada documento como relevant/irrelevant
# 2. Si todos son irrelevantes -> re-escribir la query
# 3. Re-buscar con la query reformulada
# 4. Limitar reintentos (max 3) para evitar loops infinitos
```

### Nodos del pipeline
| Nodo | Qué hace | Herramienta |
| --- | --- | --- |
| **Retrieve** | Busca documentos del vector store | Vector store + embeddings |
| **Grade Documents** | LLM califica relevancia de cada doc | Binary classifier LLM |
| **Rewrite Query** | Reformula la pregunta para mejor retrieval | LLM prompt |
| **Generate** | Genera respuesta con contexto filtrado | LLM generador |
| **Web Search** | Suplementa con búsqueda web (CRAG) | Tavily/SerpAPI |

## Query Routing: Dirigir al data source correcto

Un router clasifica la query y la dirige al retriever apropiado. Evita buscar en vector store cuando la query requiere datos en tiempo real, o viceversa.

```python
# Router pattern
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search", "direct_response"]
    reasoning: str

# Clasificación
router = llm.with_structured_output(RouteQuery)
result = router.invoke([
    {"role": "system", "content": """Clasifica la query:
    - vectorstore: preguntas sobre documentos internos de Acme
      (políticas de RRHH, manual técnico)
    - web_search: eventos actuales, datos en tiempo real
    - direct_response: conocimiento general"""},
    {"role": "user", "content": query}
])

# Fan-out paralelo (multi-source)
if result.datasource == "vectorstore":
    docs = vector_retriever.invoke(query)
elif result.datasource == "web_search":
    docs = web_search.invoke(query)
```

## Multi-Agent RAG: Especialización y coordinación

Cuando un solo agente no puede manejar la complejidad, se decomponen roles en agentes especializados que colaboran.

### Arquitectura típica de Google Agentic RAG
```python
# 5 agentes especializados (Google Enterprise Agent Platform)
#
# 1. Orchestrator: Evalúa la complejidad, delega
# 2. Planner: Mapea rutas de información
#    "Pregunta sobre budget AND timeline ->
#     primero finance DB, luego project management logs"
# 3. Query Rewriter: Transforma la query en sub-queries
#    "¿Qué pasa con Project X?" ->
#    ["Status report Project X Q3", "Key blockers Project X"]
# 4. Search Fanout: Busca en paralelo en múltiples fuentes
# 5. Sufficient Context Agent: VERIFICADOR DE CALIDAD
#    - Examina los chunks recuperados
#    - Revisa un draft intermedio
#    - Identifica EXACTAMENTE qué falta
#    - Si falta info -> feedback específico -> re-search
```

<div class="callout success">
<div class="callout-title">El Sufficient Context Agent (innovación clave)</div>
<p>No solo dice "no tengo suficiente info". Genera feedback específico: "Encontraste meds y diet, pero falta allergies. Busca por 'rashes' o 'adverse events'". Esto permite al sistema recuperar la información faltante en vez de abstenerse prematuramente. +34% en factuality vs vanilla RAG.

<small>Fuente: <a href="https://research.google/blog/unlocking-dependable-responses-with-gemini-enterprise-agent-platforms-agentic-rag/">Google Research Blog — Agentic RAG with Sufficient Context Agent (Jun 2026)</a></small></p>
</div>

### MA-RAG: 4 agentes colaborativos
| Agente | Responsabilidad |
| --- | --- |
| **Planner** | Descompone la query en sub-tareas con CoT |
| **Step Definer** | Genera sub-queries ejecutables para cada paso |
| **Extractor** | Filtra y agrega evidencia de los docs recuperados |
| **QA Agent** | Sintetiza la respuesta final con los pasos acumulados |

## 8 patrones de arquitectura de agentes

La taxonomía de 2026 organiza los patrones en 4 cuadrantes:

<small>Fuente: [8 Agent Architecture Patterns Taxonomy (Digital Applied, 2026)](https://www.digitalapplied.com/blog/agent-architecture-patterns-taxonomy-2026)</small>

| # | Patrón | Cuándo usar | Producción? |
| --- | --- | --- | --- |
| 1 | **ReAct** | Default general-purpose, &lt;30 steps | Sí |
| 2 | **Reflexión** | Failure modes repetidos (+30% latencia, +10-30% quality) | Sí |
| 3 | **Plan-and-Execute** | Planning es bottleneck, tareas pre-decomponibles | Sí |
| 4 | **Supervisor-Worker** | Descomposición clara de tareas, roles especializados | Sí |
| 5 | **Multi-Agent Debate** | Perspectivas múltiples, stress-testing adversarial | Moderado |
| 6 | **Verifier-Critic** | Output quality crítico (legal, médico, code) | Sí |
| 7 | **Graph Orchestration** | Control flow condicional, observabilidad | Sí |
| 8 | **Swarm/Blackboard** | Exploratorio, research-mode | No |

<div class="callout warning">
<div class="callout-title">La regla de oro: empieza simple</div>
<p><strong>Empieza con ReAct (single-agent). Escala a multi-agent solo cuando midas un failure mode que un solo agente no pueda resolver.</strong> El 41.77% de los fallos en agentes vienen de problemas de especificación y diseño de sistema, no de limitaciones del modelo. Multi-agent agrega 2-5x de overhead de coordinación.

<small>Fuente: <a href="https://arxiv.org/abs/2503.13657">arXiv:2503.13657 — MAST: Multi-Agent System Failure Taxonomy (Cemri et al., 2025)</a></small></p>
</div>

## Por qué los agentes fallan en producción

| Failure Mode | Qué pasa | Solución |
| --- | --- | --- |
| **Context Poisoning** | Un error entra al contexto y se referencian repetidamente | Validación de outputs, grounding checks |
| **Context Distraction** | Contexto crece tanto que el modelo ignora training | Context compression, ventanas limitadas |
| **Context Confusion** | Info superflua causa que el modelo elija wrong tool | Limitar tools a &lt;20, mejor routing |
| **Context Clash** | Nueva info contradice info existente | Deduplicación, conflict resolution |
| **Runaway Loops** | El agente nunca decide parar | Bounded execution (max steps) |

## Frameworks de producción

| Framework | Fuerza | Mejor para |
| --- | --- | --- |
| **LangGraph** | Graph orchestration, state machines | Workflows condicionales, observabilidad |
| **LlamaIndex** | Query transforms, routers, sub-questions | RAG avanzado, multi-source retrieval |
| **CrewAI** | Role-based multi-agent | Equipos de agentes con roles definidos |
| **OpenAI Agents SDK** | Handoffs, tool calling nativo | Agentes con herramientas OpenAI |
| **AutoGen** | Multi-agent debate, group chat | Research, prototyping |

## Implementación con LangGraph

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal

class GraphState(TypedDict):
    question: str
    documents: list
    generation: str
    relevance: str

# Definir nodos
def retrieve(state):
    docs = retriever.invoke(state["question"])
    return {"documents": docs}

def grade_documents(state):
    scores = [llm.grade(doc, state["question"]) for doc in state["documents"]]
    relevant = [doc for doc, score in zip(state["documents"], scores) if score == "relevant"]
    return {"documents": relevant, "relevance": "relevant" if relevant else "irrelevant"}

def rewrite_query(state):
    new_query = llm.rewrite(state["question"])
    return {"question": new_query}

def generate(state):
    answer = llm.generate(state["question"], state["documents"])
    return {"generation": answer}

# Construir el grafo
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade_documents)
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", lambda s: "generate" if s["relevance"] == "relevant" else "rewrite")
workflow.add_edge("rewrite", "retrieve")  # Loop back
workflow.add_edge("generate", END)

app = workflow.compile()
```

## Práctica


<div class="exercise">
<div class="exercise-title">Ejercicio 1: Self-corrective RAG</div>
<p>Construye un pipeline con LangGraph que:</p>
<ul>
<li>Recupere documentos</li>
<li>Los califique con un LLM (relevant/irrelevant)</li>
<li>Si son irrelevantes, re-escriba la query (max 3 reintentos)</li>
<li>Genere la respuesta final</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 2: Multi-source router</div>
<p>Crea un router que dirija queries a:</p>
<ul>
<li>Vector store (documentos internos)</li>
<li>Web search (eventos actuales)</li>
<li>Direct LLM (conocimiento general)</li>
</ul>
<p>Usa <code>Send</code> de LangGraph para ejecución paralela cuando aplique.</p>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 3: Bounded execution</div>
<p>Agrega guardrails al agente:</p>
<ul>
<li>Max 5 iteraciones del loop ReAct</li>
<li>Max 10 tool calls por sesión</li>
<li>Circuit breaker: si 3 tool calls consecutivos fallan, abortar</li>
</ul>
</div>