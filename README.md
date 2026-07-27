# RAG Production Course

Hands-on course on building production-grade RAG systems, based on the [hands-on-rag](https://github.com/ofermend/hands-on-rag) repository.

**Live site:** https://rag-lessons.netlify.app

## Hilo conductor: Proyecto Acme

Todas las lecciones construyen un solo sistema: el asistente RAG interno de **Acme Corp**. El corpus: `politicas.pdf` (RRHH, L1-L3), `manual-tecnico.pdf` (operaciones, L4-L6) y `reporte-financiero.pdf` (finanzas, L8). Cada lección indica con un callout "🧵 Proyecto Acme" de dónde vienen las variables que usa su código.

## Lessons

0. **Tu entorno RAG en 30 minutos** — Setup: venv, API keys, pgvector con Docker
1. **¿Qué es RAG?** — Fundamentos y arquitectura
2. **Chunking** — Estrategias de segmentación (Recursive, Semantic, Late Chunking)
3. **Embeddings y Vector Stores** — Modelos y bases de datos vectoriales
4. **Hybrid Retrieval y Reranking** — BM25 + denso, rerankers
5. **Guardrails y Alucinaciones** — Taxonomía, detección, mitigación
6. **Evaluación y Métricas** — 8 métricas enterprise, RAGAS, LLM-as-judge
7. **Agentic RAG** — Multi-step, tool-use, guardas
8. **Multimodal RAG** — PDFs, tablas, imágenes, routing
9. **GraphRAG** — Knowledge graphs + community detection

## Anexos

- **A: Tokens, vectores y similitud de coseno** — Intuición mínima con tiktoken y numpy
- **B: Anatomía de un prompt RAG** — Rol, grounding, contexto, citación

## Tech Stack

- **Framework:** Astro 7 + React
- **Deployment:** Netlify
- **Design:** Hallmark anti-slop skill (light/dark mode, editorial tone)
