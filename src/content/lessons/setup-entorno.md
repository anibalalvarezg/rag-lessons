---
title: "Tu entorno RAG en 30 minutos"
subtitle: "Lección 0 — Fundamentos"
pillar: fundamentos
pillarName: "Fundamentos"
lessonNum: 0
description: "Setup completo del entorno del curso: venv, dependencias, API key, Postgres+pgvector con Docker, y tu primer embedding real."
keywords: "setup, entorno, venv, OpenAI API key, pgvector, Docker, primer embedding"
ogSection: "Fundamentos"
pubDate: "2026-07-27"
quizzes:

  - id: "q1"
    question: "¿Por qué usamos un entorno virtual (venv) para el curso?"
    options:
      - text: "Para que el código ejecute más rápido en producción"
        correct: false
      - text: "Para aislar las dependencias del proyecto del resto del sistema"
        correct: true
      - text: "Para poder usar Docker dentro del proyecto Python"
        correct: false
      - text: "Para que la API key sea más segura en tránsito"
        correct: false
  - id: "q2"
    question: "¿Dónde lee el cliente de OpenAI la API key por defecto?"
    options:
      - text: "De un archivo config.yaml dentro del proyecto"
        correct: false
      - text: "Del argumento api_key en cada llamada al modelo"
        correct: false
      - text: "De la variable de entorno OPENAI_API_KEY"
        correct: true
      - text: "De la base de datos Postgres que configuramos"
        correct: false
  - id: "q3"
    question: "¿Qué rol cumple Docker en este setup?"
    options:
      - text: "Ejecutar los modelos de embedding en contenedores GPU"
        correct: false
      - text: "Levantar Postgres con pgvector sin instalar Postgres a mano"
        correct: true
      - text: "Compilar el código Python a producción más rápido"
        correct: false
      - text: "Servir la aplicación web del curso en Netlify"
        correct: false

---


## Objetivo

Al finalizar esta lección, tendrás el entorno completo del curso funcionando: Python con las dependencias instaladas, una API key configurada, Postgres+pgvector corriendo en Docker, y habrás generado tu primer embedding real.

## El Proyecto Acme: el hilo conductor del curso

Todo el código de este curso construye un solo sistema: **el asistente RAG interno de Acme Corp**, una empresa ficticia. Su corpus tiene tres documentos:

| Documento | Contenido | Aparece en |
| --- | --- | --- |
| `politicas.pdf` | Políticas de RRHH (devoluciones, vacaciones) | Lecciones 1-3 |
| `manual-tecnico.pdf` | Documentación de operaciones (settings.yaml, timeout) | Lecciones 4-6 |
| `reporte-financiero.pdf` | Reporte con tablas y charts | Lección 8 |

En cada lección verás un callout **"🧵 Proyecto Acme"** que te dice qué parte del sistema estás construyendo y de dónde vienen las variables que usa el código.

## Prerequisitos

- **Python 3.11+** instalado (`python --version` para verificar)

- **Docker** instalado y corriendo ([docker.com/get-started](https://www.docker.com/get-started/))

- **Una API key de OpenAI** ([platform.openai.com/api-keys](https://platform.openai.com/api-keys)). Unas pocas horas de curso cuestan menos de $1 USD en embeddings y llamadas a `gpt-4o-mini`.

## Paso 1: Entorno virtual y dependencias

```bash
# Crear el proyecto
mkdir acme-rag && cd acme-rag

# Crear y activar el entorno virtual
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Instalar dependencias del curso
pip install langchain langchain-openai langchain-postgres langchain-community \
    pymupdf openai numpy psycopg-binary tiktoken
```

## Paso 2: Configurar la API key

```bash
# Linux/macOS — agrégala a tu ~/.bashrc o ~/.zshrc para que persista
export OPENAI_API_KEY="sk-..."
```

El cliente de OpenAI lee esta variable automáticamente. No necesitas pasarla en el código — y **nunca** la escribas en el código ni la subas a git.

<div class="callout warning">
<div class="callout-title">Regla de seguridad</div>
<p>La API key solo vive en variables de entorno (o en un archivo <code>.env</code> listado en tu <code>.gitignore</code>). Una key en un repositorio público es encontrada y explotada en minutos.</p>
</div>

## Paso 3: Postgres + pgvector con Docker

No necesitas instalar Postgres a mano. La imagen oficial de pgvector incluye todo:

```bash
docker run -d \
  --name acme-pgvector \
  -e POSTGRES_USER=rag \
  -e POSTGRES_PASSWORD=rag \
  -e POSTGRES_DB=rag \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

Tu connection string para todo el curso será:

```python
CONNECTION_STRING = "postgresql://rag:rag@localhost:5432/rag"
```

## Paso 4: Hello RAG — tu primer embedding

Este script verifica que todo funciona. Embebe tres oraciones del corpus Acme y mide qué tan cerca está cada una de una consulta, usando similitud de coseno:

```python
# hello_rag.py
import numpy as np
from openai import OpenAI

client = OpenAI()  # lee OPENAI_API_KEY del entorno

# Tres "documentos" del corpus Acme
texts = [
    "Las devoluciones se aceptan hasta 30 días después de la compra.",   # politicas.pdf
    "El timeout por defecto es de 30 segundos.",                          # manual-tecnico.pdf
    "Python es un lenguaje de programación.",                             # ruido externo
]

def embed(texts: list[str]) -> list[np.ndarray]:
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [np.array(item.embedding) for item in response.data]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

query = "¿Puedo devolver un producto?"
query_vector = embed([query])[0]
doc_vectors = embed(texts)

for text, vec in zip(texts, doc_vectors):
    print(f"{cosine_similarity(query_vector, vec):.3f}  {text}")
```

Deberías ver algo como:

```text
0.512  Las devoluciones se aceptan hasta 30 días después de la compra.
0.298  El timeout por defecto es de 30 segundos.
0.241  Python es un lenguaje de programación.
```

La oración sobre devoluciones gana aunque la consulta no comparte casi ninguna palabra con ella. **Eso es un embedding: significado convertido en geometría.** Este es exactamente el mecanismo de retrieval que usarás en la Lección 1.

<div class="callout success">
<div class="callout-title">Si llegaste aquí, estás listo</div>
<p>Este entorno es el que usan las 9 lecciones del curso: <code>langchain</code> para el pipeline, <code>pymupdf</code> para parsing (L2), <code>pgvector</code> como vector store (L1, L3), y <code>tiktoken</code>/<code>numpy</code> para los anexos. Si algo falló, revisa: ¿venv activado? ¿<code>echo $OPENAI_API_KEY</code> muestra la key? ¿<code>docker ps</code> muestra acme-pgvector?</p>
</div>

## Práctica


<div class="exercise">
<div class="exercise-title">Ejercicio 1: Verifica tu setup</div>
<p>Ejecuta <code>hello_rag.py</code> y responde:</p>
<ul>
<li>¿Qué score de similitud obtuvo la oración ganadora?</li>
<li>Cambia la query a <code>"¿cuál es el tiempo de espera configurado?"</code> — ¿qué oración gana ahora y por qué?</li>
<li>¿Cuántas dimensiones tiene cada vector? (pista: <code>len(vec)</code>)</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 2: Crea el corpus Acme</div>
<p>Crea los archivos del proyecto que usarás en todo el curso:</p>
<ul>
<li>Crea un directorio <code>corpus/</code> dentro de <code>acme-rag/</code></li>
<li>Guarda cualquier PDF de políticas como <code>corpus/politicas.pdf</code> (o genera uno simple con 2-3 páginas de texto sobre políticas de devoluciones)</li>
<li>Verifica que Docker corre: <code>docker exec -it acme-pgvector psql -U rag -c "SELECT 1"</code></li>
</ul>
</div>
