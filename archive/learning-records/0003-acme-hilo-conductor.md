# Unificación del curso bajo el Proyecto Acme

El curso tenía dos casos de estudio desconectados (politicas.pdf en L1-L3, timeout/settings.yaml en L4-L6) y las lecciones 7-9 flotaban sin caso. Se unificó todo bajo un solo sistema: el asistente RAG interno de Acme Corp, con corpus de 3 documentos (politicas.pdf, manual-tecnico.pdf, reporte-financiero.pdf). Cada lección declara con un callout "🧵 Proyecto Acme" de dónde vienen sus variables.

**Evidencia:** Auditoría de los 9 archivos de lecciones encontró: ejercicios con referencias rotas (L2 Ej.1 pedía parámetros de chunking de código PyMuPDF que no chunkea; L8 Ej.1 referenciaba un snippet SigLIP inexistente), bugs reales (L2 `semantic_chunk` con `distances` sin inicializar; L9 import contradictorio de LLMGraphTransformer; L9 `graph_retriever` que descartaba resultados), e inconsistencias (collection "politicas" vs "my_rag_docs", "4 vs 5 pilares" en L1, bloques sin tag python).

**Implicaciones:** Se agregó Lección 0 (setup del entorno — el gap básico más grande: L1 pedía "ejecutar en tu entorno" sin enseñar setup) y dos anexos (tokens/coseno con numpy; anatomía del prompt RAG). Las lecciones futuras deben mantener: naming `acme_docs`, queries canónicas ("devoluciones", "timeout"), y el callout de continuidad al inicio.
