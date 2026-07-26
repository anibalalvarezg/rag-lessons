---
title: "GraphRAG: Knowledge Graphs + RAG para razonamiento complejo"
subtitle: "Lección 9 (Final) — Pilar 3: Agentic RAG y GraphRAG"
pillar: agentic
pillarName: "Agentic"
lessonNum: 9
description: "Microsoft GraphRAG, Neo4j, 4 estrategias de búsqueda, Dynamic Community Selection (-77% tokens), property graph model."
keywords: "GraphRAG, Microsoft GraphRAG, Neo4j, knowledge graph, dynamic community selection"
ogSection: "RAGOps"
pubDate: "2026-07-24"
quizzes:
  - id: "q1"
    question: "¿Qué tipo de pregunta resuelve Global Search que baseline RAG no puede?"
    options:
      - text: "\"Qué es la relatividad?\""
        correct: false
      - text: "\"Cuáles son los 5 temas principales del corpus?\""
        correct: true
      - text: "\"Quién descubrió la penicilina?\""
        correct: false
      - text: "\"Cuál es la capital de Francia?\""
        correct: false
  - id: "q2"
    question: "¿Qué algoritmo usa GraphRAG para community detection?"
    options:
      - text: "PageRank"
        correct: false
      - text: "Leiden (clustering jerarquico)"
        correct: true
      - text: "k-means"
        correct: false
      - text: "DBSCAN"
        correct: false
  - id: "q3"
    question: "¿Cuánto reduce Dynamic Community Selection el costo vs Global Search estatico?"
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
    question: "¿Qué metodo de busqueda es el mas costoso?"
    options:
      - text: "Basic Search"
        correct: false
      - text: "Local Search"
        correct: false
      - text: "Global Search"
        correct: false
      - text: "DRIFT Search (global + multiples locals)"
        correct: true
  - id: "q5"
    question: "¿Qué es entity resolution y por qué es critica?"
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

import Callout from '../../components/Callout.astro';
import Exercise from '../../components/Exercise.astro';

## Por que baseline RAG falla

Baseline RAG usa vector similarity en snippets de texto. Funciona para preguntas factuales simples, pero falla en dos categorias criticas:

| Tipo de pregunta | Ejemplo | Por que falla baseline RAG |
|------------------|---------|---------------------------|
| **Global / Holistica** | "Cuales son los 5 temas principales del corpus?" | No hay nada en la query que dirija al info correcta; vector search busca similaridad semantica, no agregacion |
| **Multi-hop** | "Como esta conectada la politica X con el resultado Y a traves de Z?" | Requiere travesar relaciones entre entidades; vector search no sigue edges del grafo |
| **Resumen del corpus** | "Resume las actualizaciones de las ultimas 2 semanas" | Informacion dispersa en muchos chunks; ningun chunk individual contiene la respuesta |

## Que es GraphRAG

GraphRAG (Microsoft Research, 2024) combina extraccion de knowledge graph, clustering jerarquico, y resumen de comunidades para responder preguntas que requieren comprension holistica del dataset.
<small>Fuente: arXiv:2404.16130 — GraphRAG: Community Summaries for RAG (Microsoft Research) | Documentacion oficial</small>

<Callout type="info" title="Pipeline de indexing">
**1. Chunking** -> **2. Entity/Relationship Extraction** (LLM) -> **3. Knowledge Graph** (entidades + relaciones) -> **4. Community Detection** (Leiden algorithm) -> **5. Community Summaries** (LLM resume cada comunidad) -> **6. Hierarchical structure**
</Callout>

### Paso 2: Entity Extraction con LLMs

```python
# LangChain LLMGraphTransformer
# Nota: en v1+, se movio de langchain-experimental a langchain-neo4j
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

El algoritmo de Leiden detecta comunidades jerarquicas en el grafo. Cada nivel de la jerarquia representa un nivel de abstraccion diferente:

- **Nivel 0 (hojas):** Entidades individuales con sus relaciones directas
- **Nivel 1:** Clusteres de entidades relacionadas (temas especificos)
- **Nivel 2 (raiz):** Temas de alto nivel del corpus completo

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

## Las 4 estrategias de busqueda

### 1. Global Search (para preguntas holisticas)

Usa community summaries en un patron **map-reduce**:

```python
# MAP: Cada community report genera puntos con rating de importancia
for batch in shuffled_community_reports:
    intermediate = llm.map(query, batch)
    # ["quantum error correction es importante (rating: 8)", 
    #  "commercial applications creciendo (rating: 7)", ...]

# REDUCE: Los puntos mas importantes se agregan en la respuesta final
top_points = rank_and_filter(intermediate_responses)
final_answer = llm.reduce(query, top_points)
```

- **Cuando usar:** "Cuales son los temas principales?", "Resume el corpus", "Que tendencias hay?"
- **Ventaja:** Unica forma de responder preguntas que requieren comprension de todo el corpus
- **Costo:** Alto (muchas llamadas LLM en map-reduce)

### 2. Local Search (para entidades especificas)

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

- **Cuando usar:** "Que propiedades curativas tiene la manzanilla?", "Quien es X y que relacion tiene con Y?"
- **Ventaja:** Multi-hop traversal, entity disambiguation, contexto comunitario
- **Latencia:** Moderada

### 3. DRIFT Search (exploratorio)

Combina amplitud de Global con profundidad de Local en un arbol de exploracion iterativo:

```python
# DRIFT: 3 fases
# FASE 1 - Primer (Global): context amplio
top_communities = retrieve_top_k_community_reports(query)
global_answer = llm.generate(query, top_communities)
follow_ups = llm.generate_follow_ups(query, global_answer)
# ["Cuales son los subtemas de X?", "Que entidades clave estan involucradas?"]

# FASE 2 - Follow-up (Local): profundizar iterativamente
while budget_remaining and confidence_high:
    best_question = select_highest_confidence(follow_ups)
    local_answer = local_search(best_question)
    new_follow_ups = llm.generate_follow_ups(best_question, local_answer)
    follow_ups.extend(new_follow_ups)

# FASE 3 - Output: arbol jerarquico rankeado
# Question -> Global Answer -> Follow-up 1 -> Local Answer -> ...
#            -> Follow-up 2 -> Local Answer -> ...
```

- **Cuando usar:** "Cuentame sobre [topico amplio]", "Que deberia saber sobre [dominio]?"
- **Ventaja:** Balanced breadth + depth, adaptive exploration
- **Costo:** El mas alto (global + multiples locals)

### 4. Basic Search (baseline vector)

Vector similarity search simple. Para cuando la pregunta es factual directa.

- **Cuando usar:** "Que es X?", "Busca menciones de Y"
- **Ventaja:** Mas rapido, menor costo

## Decision: Que metodo elegir

```python
# Arbol de decision
if es_pregunta_sobre_el_corpus_completo():
    return Global_Search
elif es_sobre_entidades_especificas():
    return Local_Search
elif es_exploratorio_o_desconocido():
    return DRIFT_Search
else:
    return Basic_Search
```

| Pregunta | Metodo |
|----------|--------|
| "Cuales son los temas principales?" | Global |
| "Que propiedades tiene X?" | Local |
| "Cuentame sobre [topico]" | DRIFT |
| "Que es Y?" | Basic |
| "Resume las actualizaciones recientes" | Global |
| "Como esta A conectado con B?" | Local |

## Dynamic Community Selection: -77% costo

Global Search estatico procesa TODOS los community reports. Dynamic Selection los califica primero con un modelo barato (GPT-4o-mini) y solo procesa los relevantes.

<Callout type="success" title="Resultado">
Calidad comparable a Global Search estatico, pero con **77% menos tokens**. De ~1500 reports a ~470. El rate operation es clasificacion (mas barata que summarization).
<small>Fuente: arXiv:2410.04361 — Microsoft GraphRAG: Dynamic Community Selection</small>
</Callout>

## Implementacion con Neo4j + LangChain

```python
# 1. Construir el knowledge graph
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer

graph = Neo4jGraph(url="neo4j+s://...", username="neo4j", password="...")

# Extraer entidades y relaciones
transformer = LLMGraphTransformer(llm=llm)
graph_docs = transformer.convert_to_graph_documents(documents)
graph.add_graph_documents(graph_docs, include_source=True)

# 2. Vector index sobre el grafo
from langchain_neo4j import Neo4jVector

vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    index_name="entities",
    node_label="Entity",
    text_node_properties=["id", "description"],
    embedding_node_property="embedding"
)

# 3. Graph traversal retriever
def graph_retriever(question):
    entities = entity_chain.invoke({"question": question})
    for entity in entities:
        graph.query("""
            CALL db.index.fulltext.queryNodes('entity', $query)
            YIELD node, score
            CALL { MATCH (node)-[r]->(neighbor) 
                   RETURN node.id + ' -> ' + type(r) + ' -> ' + neighbor.id }
            RETURN output LIMIT 50
        """, {"query": entity})

# 4. Hybrid retrieval (vector + graph)
chain = (
    {"context": graph_retriever, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
```

## Entity Resolution: El paso critico

El LLM extrae "IBM", "International Business Machines", "IBM Corp" como entidades separadas. Entity resolution las fusiona:

```python
# Pipeline de entity resolution
# 1. Calcular embeddings de nombres y descripciones
# 2. Encontrar candidatos similares (kNN, cosine > 0.95)
# 3. Usar LLM para confirmar si son la misma entidad
# 4. Merge en el grafo
```

- **Metodo:** Embeddings + kNN graph + LLM verification
- **Umbral tipico:** cosine similarity > 0.95 para candidatos
- **Desafio:** No hay entity resolution en el repo oficial de Microsoft

## Costo y produccion

| Aspecto | Detalle |
|---------|---------|
| **Indexing** | Caro — extraccion de entidades + resumenes de comunidad son multiples LLM calls |
| **Global Search** | Alto costo (map-reduce sobre todos los community reports) |
| **Local Search** | Moderado (una generacion + graph traversal) |
| **DRIFT** | El mas alto (global + multiples locals iterativos) |
| **Re-indexing** | Necesario cuando cambia el corpus significativamente |
| **Prompt Tuning** | Recomendado — los prompts default no son optimos para todo dominio |

## Practica

<Exercise title="Ejercicio 1: Knowledge Graph con LangChain">
Usando el notebook `create-graph.ipynb` del repo:

- Extrae entidades y relaciones de un dataset de peliculas
- Almacena en Neo4j
- Visualiza el grafo en Neo4j Browser
</Exercise>

<Exercise title="Ejercicio 2: Local vs Global">
Usando el notebook `graph-query.ipynb`:

- Responde una pregunta local (sobre una entidad)
- Responde una pregunta global (sobre temas del corpus)
- Compara: que tipo de contexto usa cada metodo?
</Exercise>

<Exercise title="Ejercicio 3: Hybrid retrieval">
Combina vector search + graph traversal:

- Vector search para similaridad semantica
- Graph traversal para relaciones multi-hop
- Merge de resultados con RRF
</Exercise>
