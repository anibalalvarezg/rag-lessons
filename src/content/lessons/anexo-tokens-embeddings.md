---
title: "Anexo A: Tokens, vectores y similitud de coseno"
subtitle: "Anexo A — Fundamentos"
pillar: anexo
pillarName: "Anexos"
lessonNum: 10
description: "Qué es un token (con tiktoken), qué es un vector, y cómo calcular similitud de coseno con numpy puro. La intuición mínima antes de chunking y embeddings."
keywords: "token, tiktoken, vector, similitud de coseno, numpy, embeddings"
ogSection: "Anexos"
pubDate: "2026-07-27"
quizzes:

  - id: "q1"
    question: "¿Qué es un token para un LLM?"
    options:
      - text: "Siempre una palabra completa del idioma español"
        correct: false
      - text: "Una unidad de texto de ~4 caracteres que puede ser parte de una palabra"
        correct: true
      - text: "Un carácter individual como una letra o dígito"
        correct: false
      - text: "Una oración completa terminada en punto final"
        correct: false
  - id: "q2"
    question: "Un chunk_size de 512 tokens equivale aproximadamente a..."
    options:
      - text: "512 palabras del texto original en español"
        correct: false
      - text: "512 caracteres del texto original en español"
        correct: false
      - text: "350-400 palabras del texto original en español"
        correct: true
      - text: "512 oraciones del texto original en español"
        correct: false
  - id: "q3"
    question: "¿Qué mide la similitud de coseno entre dos vectores?"
    options:
      - text: "La distancia en línea recta entre sus puntas"
        correct: false
      - text: "El ángulo entre ellos, ignorando su longitud"
        correct: true
      - text: "La cantidad de dimensiones que comparten"
        correct: false
      - text: "La diferencia entre sus valores máximos"
        correct: false

---


## Objetivo

Al finalizar este anexo, sabrás qué es un token (y por qué `chunk_size=512` no significa 512 palabras), qué es un vector, y cómo se calcula la similitud de coseno que usan todos los sistemas RAG — todo con código ejecutable de numpy y tiktoken.

## Por qué este anexo existe

En el curso verás `chunk_size=512`, `dimensions=1536` y `cosine similarity` desde la Lección 1. Este anexo te da la intuición mínima para que esos números dejen de ser magia. Son 15 minutos bien invertidos.

## Qué es un token

Los LLMs no leen palabras ni caracteres: leen **tokens**, unidades de texto de ~4 caracteres en promedio. Una palabra común es 1 token; una palabra rara se divide en varios.

```python
import tiktoken

# El tokenizer de GPT-4o / text-embedding-3
enc = tiktoken.get_encoding("cl100k_base")

text = "Las devoluciones se aceptan hasta 30 días después de la compra."
tokens = enc.encode(text)

print(len(tokens))                    # número de tokens
print([enc.decode([t]) for t in tokens])
# ['Las', ' dev', 'oluc', 'iones', ' se', ' acept', 'an', ...]
```

Observa: "devoluciones" se rompe en `dev` + `oluc` + `iones`. El español suele consumir **más tokens por palabra** que el inglés, porque el tokenizer fue entrenado mayoritariamente con texto en inglés.

<div class="callout info">
<div class="callout-title">Regla práctica</div>
<p>En español, 1 token ≈ 0.7-0.8 palabras. Entonces <code>chunk_size=512</code> ≈ 350-400 palabras ≈ media página. Cuando la Lección 2 recomienda chunks de 512 tokens, eso es lo que realmente significa en tu documento.</p>
</div>

## Qué es un vector

Un embedding es una lista de números — un punto en un espacio de alta dimensión. Para construir intuición, trabajemos en 2D con vectores inventados:

```python
import numpy as np

# Imagina que el eje X mide "qué tan relacionado con RRHH" es un texto
# y el eje Y mide "qué tan técnico" es.
politica_devoluciones = np.array([0.9, 0.1])   # muy RRHH, poco técnico
politica_vacaciones   = np.array([0.85, 0.15]) # muy RRHH, poco técnico
manual_timeout        = np.array([0.1, 0.9])   # poco RRHH, muy técnico
```

## Similitud de coseno desde cero

La similitud de coseno mide el **ángulo** entre dos vectores, ignorando su longitud. Resultado entre -1 (opuestos) y 1 (idéntica dirección):

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print(cosine_similarity(politica_devoluciones, politica_vacaciones))  # ~0.996 — mismo tema
print(cosine_similarity(politica_devoluciones, manual_timeout))       # ~0.235 — temas distintos
```

Las tres operaciones que usa la fórmula:

1. **Producto punto** (`np.dot`): cuánto apuntan en la misma dirección

2. **Norma** (`np.linalg.norm`): la longitud de cada vector

3. **División**: normalizar para que solo importe el ángulo, no la longitud

<div class="callout warning">
<div class="callout-title">Por qué coseno y no distancia euclidiana</div>
<p>Dos textos sobre el mismo tema pueden tener embeddings de distinta "longitud" (por ejemplo, un chunk corto vs uno largo). La distancia euclidiana los marcaría como lejanos; el coseno los reconoce como alineados. Por eso RAG busca por ángulo, no por distancia.</p>
</div>

## Conexión con el curso

- **Lección 0:** el script `hello_rag.py` usa exactamente esta función de coseno con embeddings reales de 1536 dimensiones — la misma matemática que acabas de ver en 2D.

- **Lección 2:** `chunk_size` y `chunk_overlap` están en tokens. Ya sabes medirlos con tiktoken.

- **Lección 3:** cuando elijas entre modelos de 1536 vs 3072 dimensiones, estarás eligiendo el tamaño de estos vectores.

## Práctica


<div class="exercise">
<div class="exercise-title">Ejercicio 1: Mide tus propios tokens</div>
<p>Toma un párrafo de <code>corpus/politicas.pdf</code> y con tiktoken responde:</p>
<ul>
<li>¿Cuántos tokens tiene? ¿Cuántas palabras?</li>
<li>¿Cuál es el ratio palabras/token en tu texto?</li>
<li>Si chunk_size=512, ¿cuántos párrafos como ese caben en un chunk?</li>
</ul>
</div>


<div class="exercise">
<div class="exercise-title">Ejercicio 2: Coseno a mano</div>
<p>Con vectores 3D inventados por ti:</p>
<ul>
<li>Crea dos vectores que apunten casi igual y verifica que el coseno da cerca de 1</li>
<li>Crea uno perpendicular (<code>[1,0,0]</code> vs <code>[0,1,0]</code>) y verifica que da 0</li>
<li>Multiplica un vector por 5 — ¿cambia el coseno? ¿Por qué?</li>
</ul>
</div>
