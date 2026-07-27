---
title: "Multimodal RAG"
subtitle: "Lección 8 — Agentic RAG y GraphRAG"
pillar: agentic
pillarName: "Agentic RAG y GraphRAG"
lessonNum: 8
description: "SigLIP, ColPali/ColQwen2, extracción de tablas, UniversalRAG, modality-aware routing."
keywords: "multimodal, SigLIP, ColPali, ColQwen2, tables, vision"
ogSection: "Agentic RAG y GraphRAG"
pubDate: "2026-07-24"
quizzes:

  - id: "q1"
    question: "¿Por qué ColPali supera a CLIP/SigLIP en document retrieval?"
    options:
      - text: "Porque es un modelo más grande"
        correct: false
      - text: "Porque usa late interaction (multi-vector) que captura layout, charts, y texto a nivel de patch"
        correct: true
      - text: "Porque fue entrenado en más datos"
        correct: false
      - text: "Porque usa OCR mejorado"
        correct: false
  - id: "q2"
    question: "¿Qué es \"modality gap\"?"
    options:
      - text: "Diferencia de velocidad entre modalidades"
        correct: false
      - text: "Inputs se agrupan por modalidad en vez de relevancia semántica en el espacio de embeddings"
        correct: true
      - text: "Falta de modelos para una modalidad"
        correct: false
      - text: "Diferencia de costo entre modalidades"
        correct: false
  - id: "q3"
    question: "¿Cuál es el patrón de 3 niveles para extraer tablas de PDFs?"
    options:
      - text: "OCR -> NER -> Embedding"
        correct: false
      - text: "pdfplumber (rápido) -> Docling (neural) -> VLM (fallback)"
        correct: true
      - text: "CLIP -> SigLIP -> ColPali"
        correct: false
      - text: "Chunking -> Reranking -> Generation"
        correct: false
  - id: "q4"
    question: "¿Cuándo deberías empezar con text-only RAG en vez de multimodal?"
    options:
      - text: "Nunca, multimodal siempre es mejor"
        correct: false
      - text: "Cuando tu corpus es puro texto, las tablas son simples, o el presupuesto es limitado"
        correct: true
      - text: "Solo para prototipos"
        correct: false
      - text: "Cuando usas GraphRAG"
        correct: false

---


## Objetivo

Al finalizar, sabrás por qué text-only RAG falla con documentos visuales, conocerás las 3 arquitecturas de Multimodal RAG, y sabrás elegir los modelos correctos (CLIP, SigLIP, ColPali) para tu caso de uso.

## El problema: RAG de texto pierde información visual

Los documentos empresariales contienen charts, tablas, diagramas y fotos que text-only RAG *descarta silenciosamente*. Un pipeline de OCR convierte una tabla compleja a texto plano y pierde la estructura. Un chart se convierte en cero contexto útil.

| Aspecto | Text-Only RAG | Multimodal RAG |
| --- | --- | --- |
| Inputs | Solo texto | Texto, imagen, video, audio, PDF |
| Evidencia preservada | Pasajes de texto | Frames, páginas, regiones, segmentos de audio |
| Embedding model | Text encoder (BGE, E5) | CLIP, SigLIP, ColPali, ImageBind |
| Generación | Text LLM | Vision-Language Model (GPT-4o, Claude, Gemini) |
| Grounding | Chunks de texto citados | Frames, páginas, timestamps citados |

## Las 3 arquitecturas de producción

### 1. Caption-and-Index (la más simple)
```
# Flujo:
# 1. Extraer imágenes del PDF/documento
# 2. Caption cada imagen con un VLM
#    "Bar chart showing Q3 revenue of $4.2B, up 12% YoY"
# 3. Embeber el caption como texto normal
# 4. Buscar con BM25 + dense (ya sabes hacer esto)

# Ventaja: reutiliza todo tu pipeline de text RAG existente
# Desventaja: lossy — el caption puede hallucinar o perder detalles
```

### 2. Unified Vision Embeddings (el punto medio)
```
# Flujo:
# 1. Embeber imágenes Y texto en el mismo espacio vectorial
# 2. Buscar con cosine similarity cross-modal
# 3. Generar con VLM que ve las imágenes directamente

# Modelos principales:
# - Cohere Embed 4: 128K context, Matryoshka (256-1536 dim)
# - voyage-multimodal-3.5: soporta video frames
# - SigLIP 2: open source, So400m, multilingüe
# - Nomic Embed Multimodal: 3B/7B, self-hosted
```

### 3. Page-as-Image con Late Interaction (la más precisa)
```
# Flujo:
# 1. Renderizar cada página del PDF como imagen (DPI alto)
# 2. ColPali/ColQwen produce embeddings multi-vector por página
#    (un vector por "patch" visual de la página)
# 3. Retrieval: MaxSim scoring (query tokens vs page patches)
# 4. Generar con VLM que recibe las imágenes de página

# Ventaja: NO necesita OCR — "ve" la página como un humano
# Captura charts, tablas, layout, texto — todo junto
# Desventaja: 100-1000x más vectores por página
```

<div class="callout info">
<div class="callout-title">El patrón de producción recomendado</div>
<p><strong>Hybrid late-fusion:</strong> Índice paralelo de texto + imagen. BM25 + dense para texto, ColPali para imágenes. Reranking cross-modal. VLM como generador final. Flexible, mejor recall en corpus mixtos.</p>
</div>

## Modelos de embedding visual: landscape 2026

| Modelo | Tipo | Dim | Mejor para |
| --- | --- | --- | --- |
| **CLIP ViT-L/14** | Dense (single vector) | 768 | General purpose, imágenes naturales |
| **SigLIP 2 So400m** | Dense | 1152 | Mejor accuracy-speed, multilingüe, documentos |
| **ColPali v1.3** | Late interaction (multi-vector) | 128xN | Documentos visuales, tablas, forms |
| **ColQwen2.5** | Late interaction | 128xN | State-of-art document retrieval |
| **Nomic Embed Vision** | Dense | 768 | Cost-efficient, edge deployment |
| **ImageBind** | Dense (6 modalidades) | 1024 | Texto + imagen + audio + depth + thermal + IMU |
| **Cohere Embed 4** | Dense (API) | 1536 | Enterprise docs, Matryoshka, 128K context |
| **voyage-multimodal-3.5** | Dense (API) | 1024 | Screenshots, slides, video frames |

## Parsing de documentos: la capa de ingesta

### El patrón de 3 niveles para tablas
```
# Production pattern para extraer tablas de PDFs:
# Nivel 1: pdfplumber (rápido, rule-based)
#   -> Si columnas consistentes y <15% celdas vacías -> usar

# Nivel 2: Docling (IBM Research, neural layout analysis)
#   -> Si pdfplumber falla validación -> Docling
#   -> 3-5x más lento pero mejor en tablas complejas

# Nivel 3: VLM extraction (GPT-4o, Claude)
#   -> Si Docling también falla -> VLM
#   -> El 5-10% más difícil

# Este patrón maneja 95%+ de tablas empresariales
```

### Docling (IBM Research)

- Análisis de layout con modelo DocLayNet

- Extrae por tipo de región: tablas, texto, imágenes, fórmulas

- Soporta: PDF, DOCX, PPTX, XLSX, HTML, imágenes, audio, video

- Integra con LangChain, LlamaIndex, Haystack

## UniversalRAG: routing modality-aware

En vez de forzar todas las modalidades en un espacio vectorial (que causa "modality gap"), UniversalRAG predice qué modalidad necesita la query y busca en el corpus especializado:

```
# 7 pathways de routing:
# None -> no necesita retrieval
# Paragraph -> texto nivel párrafo
# Document -> documento completo
# Table -> tablas específicas
# Image -> imágenes
# Clip -> segmentos de video
# Video -> video completo

router = llm.predict_modality(query)
# "¿Qué dice el chart de Q3?" -> [Image, Table]
# "Resume el documento" -> [Document]
# "¿Qué dice el audio?" -> [Clip]

results = {}
for modality, granularity in router:
    results[modality] = specialized_retriever[modality].search(query)
```

- **Ventaja:** Escala a nuevas modalidades sin modificar las existentes

- **Resultado:** +32% sobre baselines vision-centric en enterprise datasets

## Generación con VLMs

Una vez recuperada la evidencia multimodal, el generador debe ser un Vision-Language Model:

| VLM | Costo por imagen (tokens) | Mejor para |
| --- | --- | --- |
| **GPT-4o** | ~765 tokens (1024x1024) | Mejor costo-efectivo |
| **Claude 3.5/4** | ~1600 tokens | Documents complejos, razonamiento |
| **Gemini 2.5** | ~1300 tokens | Contexto largo, multi-imagen |
| **Qwen2-VL** | Open source | Self-hosted, VPC privado |

<div class="callout warning">
<div class="callout-title">Costo de imágenes en contexto</div>
<p>5 páginas de imagen pueden agregar 4K-8K tokens de input. Para documentos de 100+ páginas, el routing y priorización de páginas son críticos para controlar costo y latencia.</p>
</div>

## Cuándo NO construir Multimodal RAG

- **Tu corpus es puro texto:** text-only RAG es más barato y suficiente

- **Las tablas son simples:** OCR + Markdown puede bastar

- **Presupuesto limitado:** ColPali requiere GPU, VLM generation es caro

- **Latencia crítica:** Late interaction es 3-5x más lento que dense retrieval

<div class="callout success">
<div class="callout-title">La regla empírica</div>
<p>Si tu knowledge base contiene <strong>más del 20% de imágenes naturales</strong> (fotos, diagramas, scans), construye un approach de dos torres: SigLIP para imágenes + text embedder para texto, fusion con RRF. Si es puramente páginas de documentos, usa ColPali directamente.</p>
</div>

## Práctica


<div class="exercise">
<div class="exercise-title">Ejercicio 1: Image RAG con SigLIP</div>
<p>Usa el código de SigLIP de arriba como base:</p>
<ul>
<li>Embebe un conjunto de imágenes con SigLIP</li>
<li>Indexa en un vector store</li>
<li>Busca imágenes por query de texto</li>
<li>Genera respuestas con un VLM que recuperó las imágenes</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 2: PDF como imágenes</div>
<p>Construye un pipeline que:</p>
<ul>
<li>Renderice cada página de un PDF como imagen</li>
<li>Embeba con ColPali o SigLIP</li>
<li>Recupere las páginas más relevantes</li>
<li>Pase las imágenes de página a GPT-4o para generar respuesta</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 3: Table extraction</div>
<p>Extrae tablas de un PDF financiero:</p>
<ul>
<li>Intenta con pdfplumber primero</li>
<li>Si falla, usa Docling</li>
<li>Para las 5% más difíciles, usa VLM extraction</li>
<li>Almacena como Markdown + imagen de tabla</li>
</ul>
</div>