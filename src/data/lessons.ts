export interface LessonMeta {
  slug: string;
  title: string;
  desc: string;
  pillar: string;
  pillarName: string;
  lessonNum: number;
  url: string;
}

export const lessons: LessonMeta[] = [
  {
    slug: 'what-is-rag',
    title: 'Qué es RAG y por qué lo necesitas',
    desc: 'Pipeline de 4 pasos, cuándo usar RAG vs long-context vs fine-tuning, los 4 pilares de producción.',
    pillar: 'pilar-1',
    pillarName: 'Fundamentos',
    lessonNum: 1,
    url: '/rag-lessons/lessons/what-is-rag',
  },
  {
    slug: 'chunking-strategies',
    title: 'Estrategias de Chunking',
    desc: 'El dilema chunk-size, 4 estrategias, Contextual Retrieval (-49% fallos), árbol de decisión.',
    pillar: 'pilar-2',
    pillarName: 'Ingesta y Segmentación',
    lessonNum: 2,
    url: '/rag-lessons/lessons/chunking-strategies',
  },
  {
    slug: 'embeddings-and-vector-stores',
    title: 'Embeddings y Vector Stores',
    desc: 'Modelos de embedding 2026, Matryoshka, hybrid retrieval, LanceDB vs pgvector.',
    pillar: 'pilar-2',
    pillarName: 'Ingesta y Segmentación',
    lessonNum: 3,
    url: '/rag-lessons/lessons/embeddings-and-vector-stores',
  },
  {
    slug: 'hybrid-retrieval-and-reranking',
    title: 'Hybrid Retrieval y Reranking',
    desc: 'Retrieve + rerank, Cohere/BGE/Jina, HyDE, Multi-Query, pipeline completo.',
    pillar: 'pilar-2',
    pillarName: 'Ingesta y Segmentación',
    lessonNum: 4,
    url: '/rag-lessons/lessons/hybrid-retrieval-and-reranking',
  },
  {
    slug: 'guardrails-and-hallucinations',
    title: 'Guardrails y Detección de Alucinaciones',
    desc: 'Taxonomía de alucinaciones (3 tipos), 4 señales de detección, HALO, 41.77% fallas de agentes.',
    pillar: 'pilar-4',
    pillarName: 'RAGOps y Evaluación',
    lessonNum: 5,
    url: '/rag-lessons/lessons/guardrails-and-hallucinations',
  },
  {
    slug: 'evaluation-and-metrics',
    title: 'Evaluación y Métricas',
    desc: 'RAG Triad, RAGAS, LLM-as-Judge, 8 métricas enterprise, CI/CD para RAG.',
    pillar: 'pilar-4',
    pillarName: 'RAGOps y Evaluación',
    lessonNum: 6,
    url: '/rag-lessons/lessons/evaluation-and-metrics',
  },
  {
    slug: 'agentic-rag',
    title: 'Agentic RAG',
    desc: 'ReAct, Self-Corrective RAG, MA-RAG, 8 patrones de arquitectura, 5-agentes Google (+34% factuality).',
    pillar: 'pilar-3',
    pillarName: 'Agentic RAG y GraphRAG',
    lessonNum: 7,
    url: '/rag-lessons/lessons/agentic-rag',
  },
  {
    slug: 'multimodal-rag',
    title: 'Multimodal RAG',
    desc: 'SigLIP, ColPali/ColQwen2, extracción de tablas, UniversalRAG, modality-aware routing.',
    pillar: 'pilar-3',
    pillarName: 'Agentic RAG y GraphRAG',
    lessonNum: 8,
    url: '/rag-lessons/lessons/multimodal-rag',
  },
  {
    slug: 'graphrag',
    title: 'GraphRAG',
    desc: 'Microsoft GraphRAG, Neo4j, 4 estrategias de búsqueda, Dynamic Community Selection (-77% costo).',
    pillar: 'pilar-3',
    pillarName: 'Agentic RAG y GraphRAG',
    lessonNum: 9,
    url: '/rag-lessons/lessons/graphrag',
  },
];

export const glossaryTerms = [
  { term: 'RAG', def: 'Retrieval-Augmented Generation. Técnica que combina recuperación de documentos externos con generación de texto por LLM para reducir alucinaciones y actualizar conocimiento.' },
  { term: 'Chunking', def: 'Proceso de dividir documentos largos en fragmentos más pequeños para su indexación y recuperación.' },
  { term: 'Embedding', def: 'Representación vectorial densa de texto que captura significado semántico en un espacio multidimensional.' },
  { term: 'Vector Store', def: 'Base de datos optimizada para almacenar y buscar vectores de embeddings por similitud.' },
  { term: 'Hybrid Retrieval', def: 'Combinación de búsqueda semántica (dense) y búsqueda léxica (sparse/BM25) para mejorar la recuperación.' },
  { term: 'Reranking', def: 'Segunda etapa de re-ordenamiento de resultados usando un cross-encoder que lee query y documento juntos.' },
  { term: 'Contextual Retrieval', def: 'Técnica de Anthropic que prepone un resumen del documento a cada chunk antes de embeber, reduciendo fallos un 49%.' },
  { term: 'Matryoshka', def: 'Propiedad de modelos de embedding que permite reducir dimensiones al inferir sin re-embedder (ej: 3072→512 dims).' },
  { term: 'HyDE', def: 'Hypothetical Document Embeddings. Genera una respuesta hipotética y la usa como query para cerrar el gap de vocabulario.' },
  { term: 'Multi-Query', def: 'Genera múltiples paráfrasis de una pregunta y fusiona resultados con RRF para mejorar recall.' },
  { term: 'RRF', def: 'Reciprocal Rank Fusion. Fórmula para combinar rankings de múltiples fuentes: score = Σ(1/(k + rank_i)).' },
  { term: 'Grounding Check', def: 'Verificación de que cada afirmación de la respuesta esté soportada por los chunks recuperados.' },
  { term: 'SGI', def: 'Semantic Grounding Index. Ratio de similitud respuesta-contexto vs respuesta-pregunta. SGI > 1 = grounded.' },
  { term: 'LLM-as-Judge', def: 'Uso de un LLM como evaluador automático de la calidad de respuestas RAG, con ~86.9% de alineación con humanos.' },
  { term: 'RAGAS', def: 'Framework de evaluación reference-free con 4 métricas core: faithfulness, answer_relevancy, context_precision, context_recall.' },
  { term: 'ReAct', def: 'Patrón de agente Think→Act→Observe. El default para empezar con agentic RAG.' },
  { term: 'Self-Corrective RAG', def: 'Pipeline que evalúa la calidad del contexto recuperado y se re-busca automáticamente si es insuficiente.' },
  { term: 'GraphRAG', def: 'RAG basado en grafos de conocimiento. Microsoft GraphRAG usa community detection para queries globales.' },
  { term: 'Dynamic Community Selection', def: 'Técnica de GraphRAG que selecciona comunidades relevantes por query, reduciendo costo un 77%.' },
];
