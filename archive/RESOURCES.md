# RAG Production Resources

## Knowledge

### Libros y Repositorios Principales
- [GitHub: ofermend/hands-on-rag](https://github.com/ofermend/hands-on-rag)
  Código del libro "Hands-on RAG for Production". 9 capítulos cubriendo desde intro hasta GraphRAG. Use for: laboratorio práctico principal.
- [Anthropic: Contextual Retrieval (Sept 2024)](https://www.anthropic.com/news/contextual-retrieval)
  Técnica de prepender contexto antes de embeber chunks. 49% reducción en fallos de retrieval. Use for: mejorar chunking en producción.

### Chunking y Parsing
- [Microsoft Azure: RAG Chunking Phase](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-chunking-phase)
  Guía completa de estrategias de chunking con económicas de implementación. Use for: decidir estrategia de chunking.
- [BigDataBoutique: Semantic Chunking in RAG](https://bigdataboutique.com/blog/semantic-chunking-better-rag-results)
  Comparación honesta de semantic vs recursive chunking. Use for: entender trade-offs reales.
- [ACL Anthology: Recursive Semantic Chunking (RSC)](https://aclanthology.org/2025.icnlsp-1.15.pdf)
  Paper académico con método RSC que supera chunking tradicional. Use for: fundamentos teóricos.

### Retrieval y Reranking
- [Prompt20: RAG in Production Complete Guide (2026)](https://blog.prompt20.com/posts/rag-production-architecture/)
  Stack de producción 2026: hybrid retrieval + reranker + citations. Use for: arquitectura de referencia.
- [Cadence: Production RAG Architecture 2026](https://cadence.withremote.ai/blog/production-rag-architecture)
  Contextual retrieval, hybrid search, reranking como baseline. Use for: decisiones de stack.
- [FRE|Nxt Labs: Production-Grade RAG Pipelines](https://www.frenxt.com/research/production-rag-pipeline-guide)
  Guía práctica con tablas de comparación de chunk sizes y vector stores. Use for: tablas de referencia rápida.

### Evaluación y RAGOps
- [RAGAS Framework](https://docs.ragas.io/)
  Framework de evaluación para RAG: faithfulness, context precision, answer relevance. Use for: métricas de evaluación.
- [Arize Phoenix](https://phoenix.arize.com/)
  Observabilidad y tracing para RAG en producción. Use for: monitoreo en vivo.

### GraphRAG y Agentic RAG
- [Microsoft GraphRAG](https://microsoft.github.io/graphrag/)
  GraphRAG para preguntas globales sobre corpus grandes. Use for: cuándo RAG tradicional no alcanza.
- [LlamaIndex: Agent Workflows](https://docs.llamaindex.ai/en/stable/understanding/agent/)
  Framework para Agentic RAG con function calling. Use for: implementar agentes RAG.

## Wisdom (Communities)

- [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA)
  Comunidad activa de implementación RAG con modelos locales. Use for: preguntas de implementación práctica.
- [LangChain Discord](https://discord.gg/langchain)
  Discusiones sobre LangChain/LlamaIndex y patrones RAG. Use for: debugging y mejores prácticas.
- [r/MachineLearning](https://reddit.com/r/MachineLearning)
  Papers y discusiones sobre retrieval y evaluación. Use for: estado del arte académico.

## Gaps
- Falta documentación sobre RAG con modelos de código (para usuarios que trabajan con repositorios)
- Pocos recursos sobre RAG en español para producción
- Falta comparación empírica de costos reales entre diferentes stacks de RAG
