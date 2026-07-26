# Embeddings y Vector Stores

El estudiante comprende como los embeddings representan significado semantico y como elegir un modelo de embedding para su caso de uso. Aprendio que:

- El modelo de embedding importa MAS que el chunking (hasta 10.58% swing en recall)
- Matryoshka permite reducir dimensiones sin re-embedder
- pgvector es el default para la mayoria de casos (< 5M vectores)
- Hybrid retrieval (dense + sparse) es el patron de produccion 2026

**Evidencia:** Completó los ejercicios de embedding.ipynb y pgvector-simple.ipynb.

**Implicaciones:** Listo para avanzar a retrieval avanzado (reranking, guardrails) o a evaluacion. El siguiente paso natural es entender como recuperar los chunks correctos de forma precisa.
