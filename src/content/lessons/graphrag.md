---
title: "GraphRAG"
subtitle: "Lección 9 — Agentic RAG y GraphRAG"
pillar: agentic
pillarName: "Agentic RAG y GraphRAG"
lessonNum: 9
description: "Microsoft GraphRAG, Neo4j, 4 estrategias de búsqueda, Dynamic Community Selection (-77% costo)."
keywords: "GraphRAG, Neo4j, knowledge graph, community detection"
ogSection: "Agentic RAG y GraphRAG"
pubDate: "2026-07-24"
quizzes:

  - id: "q1"
    question: "¿Qué tipo de pregunta resuelve Global Search que baseline RAG no puede?"
    options:
      - text: "\"¿Qué es la relatividad?\""
        correct: false
      - text: "\"¿Cuáles son los 5 temas principales del corpus?\""
        correct: true
      - text: "\"¿Quién descubrió la penicilina?\""
        correct: false
      - text: "\"¿Cuál es la capital de Francia?\""
        correct: false
  - id: "q2"
    question: "¿Qué algoritmo usa GraphRAG para community detection?"
    options:
      - text: "PageRank"
        correct: false
      - text: "Leiden (clustering jerárquico)"
        correct: true
      - text: "k-means"
        correct: false
      - text: "DBSCAN"
        correct: false
  - id: "q3"
    question: "¿Cuánto reduce Dynamic Community Selection el costo vs Global Search estático?"
    options:
      - text: "25%"
        correct: false
      - text: "50%"
        correct: false
      - text: "77%"
        correct: true
      - text: "90%"
        correct: false
  - id: "q4"
    question: "¿Qué método de búsqueda es el más costoso?"
    options:
      - text: "Basic Search"
        correct: false
      - text: "Local Search"
        correct: false
      - text: "Global Search"
        correct: false
      - text: "DRIFT Search (global + múltiples locals)"
        correct: true
  - id: "q5"
    question: "¿Qué es entity resolution y por qué es crítica?"
    options:
      - text: "Extraer entidades del texto"
        correct: false
      - text: "Fusionar variantes de la misma entidad (IBM, \"International Business Machines\")"
        correct: true
      - text: "Eliminar entidades irrelevantes"
        correct: false
      - text: "Asignar embeddings a entidades"
        correct: false

---


## Objetivo

Al finalizar, entenderás cómo GraphRAG supera a baseline RAG en preguntas globales y multi-hop, sabrás construir un knowledge graph con LLMs, y conocerás las 4 estrategias de búsqueda (Global, Local, DRIFT, Basic).

**Prerequisitos:** [Lección 7](/rag-lessons/lessons/agentic-rag) — GraphRAG es otro data source que un agente puede enrutar.

<div class="callout info">
<div class="callout-title">🧵 Proyecto Acme — De dónde viene este código</div>
<p>Los <code>chunks</code> del corpus Acme (políticas + manual técnico) son la entrada del <code>LLMGraphTransformer</code>: aquí construyes un knowledge graph de las entidades de la empresa (políticas, configuraciones, personas, sistemas) sobre Neo4j. El <code>llm</code> es el mismo de siempre. Cuando una pregunta del empleado sea multi-hop ("¿qué política aplica al sistema que tiene el timeout en 30s?"), el router de la Lección 7 puede dirigirla a este grafo en vez del vectorstore.</p>
</div>

## Por qué baseline RAG falla

Baseline RAG usa vector similarity en snippets de texto. Funciona para preguntas factuales simples, pero falla en dos categorías críticas:

| Tipo de pregunta | Ejemplo | Por qué falla baseline RAG |
| --- | --- | --- |
| **Global / Holística** | "¿Cuáles son los 5 temas principales del corpus?" | No hay nada en la query que dirija al info correcta; vector search busca similaridad semántica, no agregación |
| **Multi-hop** | "¿Cómo está conectada la política X con el resultado Y a través de Z?" | Requiere travesar relaciones entre entidades; vector search no sigue edges del grafo |
| **Resumen del corpus** | "Resume las actualizaciones de las últimas 2 semanas" | Información dispersa en muchos chunks; ningún chunk individual contiene la respuesta |

## Qué es GraphRAG

GraphRAG (Microsoft Research, 2024) combina extracción de knowledge graph, clustering jerárquico, y resumen de comunidades para responder preguntas que requieren comprensión holística del dataset.

<small>Fuente: [arXiv:2404.16130 — GraphRAG: Community Summaries for RAG (Microsoft Research)](https://arxiv.org/abs/2404.16130) | [Documentación oficial](https://microsoft.github.io/graphrag/)</small>

<div class="callout info">
<div class="callout-title">Pipeline de indexing</div>
<p><strong>1. Chunking</strong> → <strong>2. Entity/Relationship Extraction</strong> (LLM) → <strong>3. Knowledge Graph</strong> (entidades + relaciones) → <strong>4. Community Detection</strong> (Leiden algorithm) → <strong>5. Community Summaries</strong> (LLM resume cada comunidad) → <strong>6. Hierarchical structure</strong></p>
</div>

### Paso 2: Entity Extraction con LLMs
```python
# LangChain LLMGraphTransformer
# Nota: en v1+, se movió de langchain-experimental a langchain-neo4j
from langchain_neo4j import LLMGraphTransformer

transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Person", "Organization", "Concept", "Event"],
    allowed_relationships=["WORKS_AT", "CAUSES", "MENTIONS", "RELATED_TO"],
    node_properties=["description", "importance"],
)

graph_docs = transformer.convert_to_graph_documents(chunks)
# Nodes: [Person("Einstein"), Organization("CERN"), ...]
# Relationships: [Einstein -WORKS_AT-> CERN, ...]
```

### Paso 4: Community Detection (Leiden Algorithm)

El algoritmo de Leiden detecta comunidades jerárquicas en el grafo. Cada nivel de la jerarquía representa un nivel de abstracción diferente:

- **Nivel 0 (hojas):** Entidades individuales con sus relaciones directas

- **Nivel 1:** Clusters de entidades relacionadas (temas específicos)

- **Nivel 2 (raíz):** Temas de alto nivel del corpus completo

### Paso 5: Community Summaries
```python
# Para cada comunidad, el LLM genera un resumen
# Ejemplo de community report:
"""
COMMUNITY: "Quantum Computing Research"
ENTITIES: IBM, Google, IonQ, Rigetti, Harvard Quantum Initiative
SUMMARY: "Major players in quantum computing include IBM (127-qubit Eagle),
Google (Sycamore), and IonQ (trapped ion). Harvard Quantum Initiative
is a key academic contributor. Key themes: error correction,
quantum advantage, and commercial applications."
"""
```

## Las 4 estrategias de búsqueda

### 1. Global Search (para preguntas holísticas)

Usa community summaries en un patrón **map-reduce**:

```python
# MAP: Cada community report genera puntos con rating de importancia
for batch in shuffled_community_reports:
    intermediate = llm.map(query, batch)
    # ["quantum error correction es importante (rating: 8)",
    #  "commercial applications creciendo (rating: 7)", ...]

# REDUCE: Los puntos más importantes se agregan en la respuesta final
top_points = rank_and_filter(intermediate_responses)
final_answer = llm.reduce(query, top_points)
```

- **Cuándo usar:** "¿Cuáles son los temas principales?", "Resume el corpus", "¿Qué tendencias hay?"

- **Ventaja:** Única forma de responder preguntas que requieren comprensión de todo el corpus

- **Costo:** Alto (muchas llamadas LLM en map-reduce)

### 2. Local Search (para entidades específicas)

Combina knowledge graph estructurado con texto no estructurado:

```python
# Flujo de Local Search:
# 1. Extraer entidades de la query
entities = extract_entities(query)
# ["Einstein", "relatividad"]

# 2. Buscar en el grafo: entidades, relaciones, covariates, community reports
graph_context = traverse_graph(entities)
# Entidades vecinas, relationships, community context

# 3. Buscar text chunks relevantes asociados a esas entidades
text_chunks = retrieve_chunks(entities)

# 4. Priorizar y filtrar para caber en el context window
context = prioritize_and_filter(graph_context + text_chunks)

# 5. Generar respuesta
answer = llm.generate(query, context)
```

- **Cuándo usar:** "¿Qué propiedades curativas tiene la manzanilla?", "¿Quién es X y qué relación tiene con Y?"

- **Ventaja:** Multi-hop traversal, entity disambiguation, contexto comunitario

- **Latencia:** Moderada

### 3. DRIFT Search (exploratorio)

Combina amplitud de Global con profundidad de Local en un árbol de exploración iterativo:

```python
# DRIFT: 3 fases
# FASE 1 - Primer (Global): context amplio
top_communities = retrieve_top_k_community_reports(query)
global_answer = llm.generate(query, top_communities)
follow_ups = llm.generate_follow_ups(query, global_answer)
# ["¿Cuáles son los subtemas de X?", "¿Qué entidades clave están involucradas?"]

# FASE 2 - Follow-up (Local): profundizar iterativamente
while budget_remaining and confidence_high:
    best_question = select_highest_confidence(follow_ups)
    local_answer = local_search(best_question)
    new_follow_ups = llm.generate_follow_ups(best_question, local_answer)
    follow_ups.extend(new_follow_ups)

# FASE 3 - Output: árbol jerárquico rankeado
# Question -> Global Answer -> Follow-up 1 -> Local Answer -> ...
#            -> Follow-up 2 -> Local Answer -> ...
```

- **Cuándo usar:** "Cuéntame sobre [tópico amplio]", "¿Qué debería saber sobre [dominio]?"

- **Ventaja:** Balanced breadth + depth, adaptive exploration

- **Costo:** El más alto (global + múltiples locals)

### 4. Basic Search (baseline vector)

Vector similarity search simple. Para cuando la pregunta es factual directa.

- **Cuándo usar:** "¿Qué es X?", "Busca menciones de Y"

- **Ventaja:** Más rápido, menor costo

## Decisión: Qué método elegir
```python
# Árbol de decisión
if es_pregunta_sobre_el_corpus_completo():
    return Global_Search
elif es_sobre_entidades_especificas():
    return Local_Search
elif es_exploratorio_o_desconocido():
    return DRIFT_Search
else:
    return Basic_Search
```

| Pregunta | Método |
| --- | --- |
| "¿Cuáles son los temas principales?" | Global |
| "¿Qué propiedades tiene X?" | Local |
| "Cuéntame sobre [tópico]" | DRIFT |
| "¿Qué es Y?" | Basic |
| "Resume las actualizaciones recientes" | Global |
| "¿Cómo está A conectado con B?" | Local |

## Dynamic Community Selection: -77% costo

Global Search estático procesa TODOS los community reports. Dynamic Selection los califica primero con un modelo barato (GPT-4o-mini) y solo procesa los relevantes.

<div class="callout success">
<div class="callout-title">Resultado</div>
<p>Calidad comparable a Global Search estático, pero con <strong>77% menos tokens</strong>. De ~1500 reports a ~470. La operación es clasificación (más barata que summarization).

<small>Fuente: <a href="https://arxiv.org/abs/2410.04361">arXiv:2410.04361 — Microsoft GraphRAG: Dynamic Community Selection</a></small></p>
</div>

## Implementación con Neo4j + LangChain
```python
# 1. Construir el knowledge graph
# Nota: en v1+, LLMGraphTransformer vive en langchain-neo4j (ver Paso 2 arriba)
from langchain_neo4j import Neo4jGraph, Neo4jVector, LLMGraphTransformer
from langchain_openai import OpenAIEmbeddings

graph = Neo4jGraph(url="neo4j+s://...", username="neo4j", password="...")

# Extraer entidades y relaciones de los chunks del corpus Acme
transformer = LLMGraphTransformer(llm=llm)
graph_docs = transformer.convert_to_graph_documents(chunks)
graph.add_graph_documents(graph_docs, include_source=True)

# 2. Vector index sobre el grafo
vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    index_name="entities",
    node_label="Entity",
    text_node_properties=["id", "description"],
    embedding_node_property="embedding"
)

# 3. Graph traversal retriever
def graph_retriever(question: str) -> str:
    entities = entity_chain.invoke({"question": question})
    relations = []
    for entity in entities:
        rows = graph.query("""
        CALL db.index.fulltext.queryNodes('entity', $query)
        YIELD node, score
        CALL { MATCH (node)-[r]->(neighbor)
        RETURN node.id + ' -> ' + type(r) + ' -> ' + neighbor.id AS output }
        RETURN output LIMIT 50
        """, {"query": entity})
        relations.extend(row["output"] for row in rows)
    return "\n".join(relations)

# 4. Hybrid retrieval (vector + graph)
chain = (
    {"context": graph_retriever, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
```

## Entity Resolution: El paso crítico

El LLM extrae "IBM", "International Business Machines", "IBM Corp" como entidades separadas. Entity resolution las fusiona:

```python
# Pipeline de entity resolution
# 1. Calcular embeddings de nombres y descripciones
# 2. Encontrar candidatos similares (kNN, cosine > 0.95)
# 3. Usar LLM para confirmar si son la misma entidad
# 4. Merge en el grafo
```

- **Método:** Embeddings + kNN graph + LLM verification

- **Umbral típico:** Cosine similarity > 0.95 para candidatos

- **Desafío:** No hay entity resolution en el repo oficial de Microsoft

## Costo y producción
| Aspecto | Detalle |
| --- | --- |
| **Indexing** | Caro — extracción de entidades + resúmenes de comunidad son múltiples LLM calls |
| **Global Search** | Alto costo (map-reduce sobre todos los community reports) |
| **Local Search** | Moderado (una generación + graph traversal) |
| **DRIFT** | El más alto (global + múltiples locals iterativos) |
| **Re-indexing** | Necesario cuando cambia el corpus significativamente |
| **Prompt Tuning** | Recomendado — los prompts default no son óptimos para todo dominio |

## Práctica


<div class="exercise">
<div class="exercise-title">Ejercicio 1: Knowledge Graph con LangChain</div>
<p>Usa el código de Neo4j de arriba como base:</p>
<ul>
<li>Extrae entidades y relaciones de los <code>chunks</code> del corpus Acme (políticas + manual técnico)</li>
<li>Almacena en Neo4j</li>
<li>Visualiza el grafo en Neo4j Browser — ¿qué entidades conectan ambos documentos?</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 2: Local vs Global</div>
<p>Usa el código de graph query de arriba:</p>
<ul>
<li>Responde una pregunta local (sobre una entidad)</li>
<li>Responde una pregunta global (sobre temas del corpus)</li>
<li>Compara: ¿qué tipo de contexto usa cada método?</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 3: Hybrid retrieval</div>
<p>Combina vector search + graph traversal:</p>
<ul>
<li>Vector search para similaridad semántica</li>
<li>Graph traversal para relaciones multi-hop</li>
<li>Merge de resultados con RRF</li>
</ul>
</div>