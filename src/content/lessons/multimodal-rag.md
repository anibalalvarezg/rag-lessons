---
title: "Multimodal RAG: Mas alla del texto — imagenes, tablas, audio y video"
subtitle: "Lección 8 — Pilar 3: Agentic RAG y GraphRAG"
pillar: agentic
pillarName: "Agentic"
lessonNum: 8
description: "SigLIP, ColPali/ColQwen2, extracción de tablas con pdfplumber/Docling, UniversalRAG, modality-aware routing."
keywords: "multimodal RAG, SigLIP, ColPali, ColQwen2, table extraction, UniversalRAG"
ogSection: "RAGOps"
pubDate: "2026-07-24"
quizzes:
  - id: "q1"
    question: "¿Por qué ColPali supera a CLIP/SigLIP en document retrieval?"
    options:
      - text: "Porque es un modelo mas grande"
        correct: false
      - text: "Porque usa late interaction (multi-vector) que captura layout, charts, y texto a nivel de patch"
        correct: true
      - text: "Porque fue entrenado en mas datos"
        correct: false
      - text: "Porque usa OCR mejorado"
        correct: false
  - id: "q2"
    question: "¿Qué es 'modality gap'?"
    options:
      - text: "Diferencia de velocidad entre modalidades"
        correct: false
      - text: "Inputs se agrupan por modalidad en vez de relevancia semantica en el espacio de embeddings"
        correct: true
      - text: "Falta de modelos para una modalidad"
        correct: false
      - text: "Diferencia de costo entre modalidades"
        correct: false
  - id: "q3"
    question: "¿Cuál es el patron de 3 niveles para extraer tablas de PDFs?"
    options:
      - text: "OCR -> NER -> Embedding"
        correct: false
      - text: "pdfplumber (rapido) -> Docling (neural) -> VLM (fallback)"
        correct: true
      - text: "CLIP -> SigLIP -> ColPali"
        correct: false
      - text: "Chunking -> Reranking -> Generation"
        correct: false
  - id: "q4"
    question: "¿Cuándo deberias empezar con text-only RAG en vez de multimodal?"
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

import Callout from '../../components/Callout.astro';
import Exercise from '../../components/Exercise.astro';

## El problema: RAG de texto pierde informacion visual

Los documentos empresariales contienen charts, tablas, diagramas y fotos que text-only RAG *descarta silenciosamente*. Un pipeline de OCR convierte una tabla compleja a texto plano y pierde la estructura. Un chart se convierte en cero contexto util.

| Aspecto | Text-Only RAG | Multimodal RAG |
|---------|---------------|----------------|
| Inputs | Solo texto | Texto, imagen, video, audio, PDF |
| Evidencia preservada | Pasajes de texto | Frames, paginas, regiones, segmentos de audio |
| Embedding model | Text encoder (BGE, E5) | CLIP, SigLIP, ColPali, ImageBind |
| Generacion | Text LLM | Vision-Language Model (GPT-4o, Claude, Gemini) |
| Grounding | Chunks de texto citados | Frames, paginas, timestamps citados |

## Las 3 arquitecturas de produccion

### 1. Caption-and-Index (la mas simple)

```python
# Flujo:
# 1. Extraer imagenes del PDF/documento
# 2. Caption cada imagen con un VLM
#    "Bar chart showing Q3 revenue of $4.2B, up 12% YoY"
# 3. Embeber el caption como texto normal
# 4. Buscar con BM25 + dense (ya sabes hacer esto)

# Ventaja: reutiliza todo tu pipeline de text RAG existente
# Desventaja: lossy — el caption puede hallucinar o perder detalles
```

### 2. Unified Vision Embeddings (el punto medio)

```python
# Flujo:
# 1. Embeber imagenes Y texto en el mismo espacio vectorial
# 2. Buscar con cosine similarity cross-modal
# 3. Generar con VLM que ve las imagenes directamente

# Modelos principales:
# - Cohere Embed 4: 128K context, Matryoshka (256-1536 dim)
# - voyage-multimodal-3.5: soporta video frames
# - SigLIP 2: open source, So400m, multilingue
# - Nomic Embed Multimodal: 3B/7B, self-hosted
```

### 3. Page-as-Image con Late Interaction (la mas precisa)

```python
# Flujo:
# 1. Renderizar cada pagina del PDF como imagen (DPI alto)
# 2. ColPali/ColQwen produce embeddings multi-vector por pagina
#    (un vector por "patch" visual de la pagina)
# 3. Retrieval: MaxSim scoring (query tokens vs page patches)
# 4. Generar con VLM que recibe las imagenes de pagina

# Ventaja: NO necesita OCR — "ve" la pagina como un humano
# Captura charts, tablas, layout, texto — todo junto
# Desventaja: 100-1000x mas vectores por pagina
```

<Callout type="info" title="El patron de produccion recomendado">
**Hybrid late-fusion:** Indice paralelo de texto + imagen. BM25 + dense para texto, ColPali para imagenes. Reranking cross-modal. VLM como generador final. Flexible, mejor recall en corpus mixtos.
</Callout>

## Modelos de embedding visual: landscape 2026

| Modelo | Tipo | Dim | Mejor para |
|--------|------|-----|------------|
| **CLIP ViT-L/14** | Dense (single vector) | 768 | General purpose, imagenes naturales |
| **SigLIP 2 So400m** | Dense | 1152 | Mejor accuracy-speed, multilingue, documentos |
| **ColPali v1.3** | Late interaction (multi-vector) | 128xN | Documentos visuales, tablas, forms |
| **ColQwen2.5** | Late interaction | 128xN | State-of-art document retrieval |
| **Nomic Embed Vision** | Dense | 768 | Cost-efficient, edge deployment |
| **ImageBind** | Dense (6 modalidades) | 1024 | Texto + imagen + audio + depth + thermal + IMU |
| **Cohere Embed 4** | Dense (API) | 1536 | Enterprise docs, Matryoshka, 128K context |
| **voyage-multimodal-3.5** | Dense (API) | 1024 | Screenshots, slides, video frames |

## Parsing de documentos: la capa de ingestion

### El patron de 3 niveles para tablas

```python
# Production pattern para extraer tablas de PDFs:
# Nivel 1: pdfplumber (rapido, rule-based)
#   -> Si columnas consistentes y <15% celdas vacias -> usar

# Nivel 2: Docling (IBM Research, neural layout analysis)
#   -> Si pdfplumber falla validacion -> Docling
#   -> 3-5x mas lento pero mejor en tablas complejas

# Nivel 3: VLM extraction (GPT-4o, Claude)
#   -> Si Docling tambien falla -> VLM
#   -> El 5-10% mas dificil

# Este patron maneja 95%+ de tablas empresariales
```

### Docling (IBM Research)

- Analisis de layout con modelo DocLayNet
- Extrae por tipo de region: tablas, texto, imagenes, formulas
- Soporta: PDF, DOCX, PPTX, XLSX, HTML, imagenes, audio, video
- Integra con LangChain, LlamaIndex, Haystack

## UniversalRAG: routing modality-aware

En vez de forzar todas las modalidades en un espacio vectorial (que causa "modality gap"), UniversalRAG predice que modalidad necesita la query y busca en el corpus especializado:

```python
# 7 pathways de routing:
# None -> no necesita retrieval
# Paragraph -> texto nivel parrafo
# Document -> documento completo
# Table -> tablas especificas
# Image -> imagenes
# Clip -> segmentos de video
# Video -> video completo

router = llm.predict_modality(query)
# "Que dice el chart de Q3?" -> [Image, Table]
# "Resume el documento" -> [Document]
# "Que dice el audio?" -> [Clip]

results = {}
for modality, granularity in router:
    results[modality] = specialized_retriever[modality].search(query)
```

- **Ventaja:** Escala a nuevas modalidades sin modificar las existentes
- **Resultado:** +32% sobre baselines vision-centric en enterprise datasets

## Generacion con VLMs

Una vez recuperada la evidencia multimodal, el generador debe ser un Vision-Language Model:

| VLM | Costo por imagen (tokens) | Mejor para |
|-----|---------------------------|------------|
| **GPT-4o** | ~765 tokens (1024x1024) | Mejor costo-efectivo |
| **Claude 3.5/4** | ~1600 tokens | Documents complejos, razonamiento |
| **Gemini 2.5** | ~1300 tokens | Contexto largo, multi-imagen |
| **Qwen2-VL** | Open source | Self-hosted, VPC privado |

<Callout type="warning" title="Costo de imagenes en contexto">
5 paginas de imagen pueden agregar 4K-8K tokens de input. Para documentos de 100+ paginas, el routing y priorizacion de paginas son criticos para controlar costo y latencia.
</Callout>

## Cuando NO construir Multimodal RAG

- **Tu corpus es puro texto:** text-only RAG es mas barato y suficiente
- **Las tablas son simples:** OCR + Markdown puede bastar
- **Presupuesto limitado:** ColPali requiere GPU, VLM generation es caro
- **Latencia critica:** Late interaction es 3-5x mas lento que dense retrieval

<Callout type="success" title="La regla empirica">
Si tu knowledge base contiene **mas del 20% de imagenes naturales** (fotos, diagramas, scans), construye un approach de dos torres: SigLIP para imagenes + text embedder para texto, fusion con RRF. Si es puramente paginas de documentos, usa ColPali directamente.
</Callout>

## Practica

<Exercise title="Ejercicio 1: Image RAG con SigLIP">
Usando el notebook `image-retrieval-siglip.ipynb` del repo:

- Embebe un conjunto de imagenes con SigLIP
- Indexa en un vector store
- Busca imagenes por query de texto
- Genera respuestas con un VLM que recupero las imagenes
</Exercise>

<Exercise title="Ejercicio 2: PDF como imagenes">
Construye un pipeline que:

- Renderice cada pagina de un PDF como imagen
- Embeba con ColPali o SigLIP
- Recupere las paginas mas relevantes
- Pase las imagenes de pagina a GPT-4o para generar respuesta
</Exercise>

<Exercise title="Ejercicio 3: Table extraction">
Extrae tablas de un PDF financiero:

- Intenta con pdfplumber primero
- Si falla, usa Docling
- Para las 5% mas dificiles, usa VLM extraction
- Almacena como Markdown + imagen de tabla
</Exercise>
