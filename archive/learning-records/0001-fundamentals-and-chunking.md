# Fundamentos RAG y Chunking Strategies

El estudiante estableció los fundamentos de RAG (pipeline básico, los 4 pilares de producción) y aprendió las estrategias de chunking principales. Quedó claro que:

- Recursive 512 con overlap 64 es el default de producción más robusto
- Semantic chunking está sobrevalorado para la mayoría de corpus estructurados
- Parent-child chunking resuelve el dilema de tamaño desacoplando retrieval de generación
- El tamaño de chunk importa más que el algoritmo elegido

**Evidencia:** El estudiante completó los ejercicios del Capítulo 2 y formuló preguntas sobre la configuración óptima de chunk sizes para su caso de uso.

**Implicaciones:** Listo para avanzar a embeddings y vector stores, o profundizar en evaluación de chunking. El siguiente paso natural es entender cómo los embeddings representan el significado de los chunks.
