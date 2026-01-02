# 🔗 Mapeo Código ↔ Concepto: Script 09
🌍 [English](../en/SCRIPT_09_MAPPING.md) | 📖 [Français](../fr/SCRIPT_09_MAPPING.md) | 🇪🇸 **Español** | 🇧🇷 [Português](../pt/SCRIPT_09_MAPPING.md) | 🇸🇦 [العربية](../ar/SCRIPT_09_MAPPING.md)
> **Entiende qué código implementa qué concepto**  
> Guía de aprendizaje línea por línea

---

## 📍 Navegación Rápida

- **📖 Ver: [Recorrido Pedagógico](PEDAGOGICAL_JOURNEY.md)** - Teoría
- **🏗️ Ver: [Arquitectura](INDEX_SCRIPT_09.md)** - Estructura
- **⚡ Ver: [Inicio Rápido](QUICKSTART_SCRIPT_09.md)** - Ejecútalo
- **🌍 Otros idiomas: [English](../en/SCRIPT_09_MAPPING.md) | [Français](../fr/SCRIPT_09_MAPPING.md) | [Português](../pt/SCRIPT_09_MAPPING.md)**

---

## 🎯 Sección 1: Imports y Setup

### Concepto: Preparación del Entorno

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re
```

**Lo que enseña:**
- `numpy`: Computación numérica (embeddings, softmax)
- `cosine_similarity`: Calcular similitud entre documentos
- `defaultdict`: Estructura de datos para base de conocimientos
- `re`: Procesamiento de texto

---

## 🎯 Sección 2: Base de Conocimientos

### Concepto: Almacenamiento de Datos

```python
KNOWLEDGE_BASE = {
    'doc_1': "Un LLM es un modelo de lenguaje grande...",
    'doc_2': "Los Transformers usan mecanismos de atención...",
    'doc_3': "RAG combina recuperación con generación...",
    # ... más documentos
}
```

**Lo que enseña:**
- Cómo almacenar conocimiento de dominio
- Estructura simple de diccionario
- Escalable a miles de documentos

---

## 🎯 Sección 3: Embeddings

### Concepto: Texto → Representación Vectorial

```python
def create_embedding(text: str, dim: int = 128) -> np.ndarray:
    """Convierte texto a vector usando hash determinístico"""
    hash_val = hash(text)
    np.random.seed(abs(hash_val) % 2**32)
    return np.random.randn(dim)
```

**Lo que enseña:**
- **Producción real:** Usar SentenceTransformer
  ```python
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer('all-MiniLM-L6-v2')
  embedding = model.encode(text)
  ```
- **En esta demo:** Enfoque simplificado basado en hash para velocidad
- **Concepto clave:** Texto → vector de tamaño fijo (128 dimensiones)
- **Propiedad:** Texto similar → vectores similares

**Analogía del mundo real:**
```
Imagina: Cada documento es un punto en espacio de 128 dimensiones
Puntos cercanos = significado similar
```

---

## 🎯 Sección 4: Recuperación (RAG Parte 1)

### Concepto: Encontrar Documentos Relevantes

```python
def retrieve_documents(query: str, k: int = 3) -> list:
    """Paso 1: Embed de la consulta
       Paso 2: Comparar con todos los documentos
       Paso 3: Retornar top-k más similares
    """
    query_embedding = create_embedding(query)
    
    # Crear matriz de todos los embeddings de documentos
    doc_embeddings = np.array([
        create_embedding(doc) 
        for doc in KNOWLEDGE_BASE.values()
    ])
    
    # Calcular similitud coseno
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1), 
        doc_embeddings
    )[0]
    
    # Obtener top-k
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    results = []
    for idx in top_indices:
        doc_name = list(KNOWLEDGE_BASE.keys())[idx]
        results.append({
            'doc': doc_name,
            'content': KNOWLEDGE_BASE[doc_name],
            'similarity': similarities[idx]
        })
    
    return results
```

**Lo que enseña:**
- **Embedding:** Convertir texto a vector
- **Similitud:** Similitud coseno = ¿qué tan alineados están dos vectores?
  ```
  cosine_similarity = (A · B) / (||A|| * ||B||)
  Rango: -1 (opuesto) a 1 (idéntico)
  ```
- **Selección:** Retornar top-k (más similares) documentos
- **Complejidad:** O(n*d) donde n=docs, d=dimensiones

**Analogía del mundo real:**
```
Como un bibliotecario:
1. Lee tu pregunta
2. Compara mentalmente con todos los libros
3. Te trae los 3 libros más relevantes
```

---

## 🎯 Sección 5: Razonamiento (Chain-of-Thought)

### Concepto: Resolución Estructurada de Problemas

```python
def reasoning_phase(question: str, contexts: list) -> str:
    """Piensa paso a paso con contexto recuperado"""
    
    reasoning = f"""
    Paso 1: Analizar la Pregunta
    El usuario pregunta sobre: {question}
    
    Paso 2: Conceptos Clave
    Extraer conceptos principales de la pregunta
    
    Paso 3: Recuperar Contexto Relevante
    De los documentos recuperados:
    """
    
    for i, ctx in enumerate(contexts, 1):
        reasoning += f"\n- De {ctx['doc']}: {ctx['content'][:100]}..."
    
    reasoning += f"""
    
    Paso 4: Sintetizar una Respuesta
    Combinando el conocimiento:
    - Punto 1: [del contexto 1]
    - Punto 2: [del contexto 2]
    - Punto 3: [del contexto 3]
    
    Conclusión: Basado en lo anterior, podemos concluir...
    """
    
    return reasoning
```

**Lo que enseña:**
- **Chain-of-Thought:** Dividir problema en pasos
- **Integración de Contexto:** Usar documentos recuperados
- **Reproducibilidad:** Cada paso es visible
- **Transparencia:** Fácil de depurar el razonamiento

**Analogía del mundo real:**
```
Como mostrar tu trabajo en matemáticas:
No solo "respuesta: 42"
Sino "Paso 1: ... Paso 2: ... Paso 3: ... Respuesta: 42"
```

---

## 🎯 Sección 6: Generación con Temperatura

### Concepto: Softmax y Muestreo con Temperatura

```python
def generate_with_temperature(
    prompt: str, 
    temperature: float = 1.0
) -> str:
    """
    Simula generación de tokens con control de temperatura
    
    Temperatura:
    - 0.1: Muy enfocado (determinístico)
    - 1.0: Balanceado (softmax normal)
    - 2.0: Muy creativo (diverso)
    """
    
    # Simular logits (puntuaciones no normalizadas)
    prompt_hash = hash(prompt)
    np.random.seed(abs(prompt_hash) % 2**32)
    logits = np.random.randn(100) * 2
    
    # Aplicar escalado de temperatura
    scaled_logits = logits / temperature
    
    # Softmax para obtener probabilidades
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    probabilities = exp_logits / np.sum(exp_logits)
    
    # Muestrear token
    selected_idx = np.random.choice(100, p=probabilities)
    
    # Generar texto
    vocab = ["un", "LLM", "es", "un", "modelo", "que", 
             "genera", "texto", "usando", "redes", "neuronales"]
    response = " ".join([vocab[i % len(vocab)] for i in range(selected_idx % 20)])
    
    return response
```

**Lo que enseña:**

**Fórmula Softmax:**
```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
Resultado: distribución de probabilidad (suma = 1)
```

**Efecto de Temperatura:**
```
T = 0.1  →  [0.01, 0.98, 0.01]  ← Agudo (determinístico)
T = 1.0  →  [0.15, 0.70, 0.15]  ← Balanceado
T = 2.0  →  [0.30, 0.40, 0.30]  ← Plano (diverso)
```

**Insight clave:**
- T baja: El modelo repite el token más probable (aburrido)
- T alta: El modelo explora alternativas (creativo)

---

## 🎯 Sección 7: Bucle del Agente (ReAct)

### Concepto: Toma de Decisiones Autónoma

```python
def agent_loop(
    initial_query: str, 
    max_turns: int = 3
) -> dict:
    """
    Patrón ReAct:
    PENSAR → ACTUAR → OBSERVAR → (repetir)
    """
    
    context = initial_query
    trace = []
    turn = 0
    
    while turn < max_turns:
        turn += 1
        
        # PENSAR: Analizar estado actual
        thought = f"Turno {turn}: Analizando '{context[:50]}...'"
        trace.append(f"PENSAR: {thought}")
        
        # Decidir: ¿Continuar o Parar?
        should_continue = turn < max_turns and len(context) < 500
        
        if not should_continue:
            trace.append("PARAR: Suficiente información recopilada")
            break
        
        # ACTUAR: Recuperar documentos
        documents = retrieve_documents(context, k=2)
        trace.append(f"ACTUAR: Recuperados {len(documents)} documentos")
        
        # OBSERVAR: Procesar resultados
        context += f" [Recuperado: {documents[0]['doc']}]"
        trace.append(f"OBSERVAR: Añadido contexto de {documents[0]['doc']}")
    
    return {
        'answer': context,
        'turns': turn,
        'trace': trace
    }
```

**Lo que enseña:**

**Bucle ReAct:**
```
┌─────────────────────────────────┐
│ PENSAR (analizar estado)        │
└────────────────┬────────────────┘
                 ↓
┌─────────────────────────────────┐
│ ACTUAR (tomar acción/recuperar) │
└────────────────┬────────────────┘
                 ↓
┌─────────────────────────────────┐
│ OBSERVAR (procesar resultados)  │
└────────────────┬────────────────┘
                 ↓
        ¿Repetir o Parar?
```

**Propiedades clave:**
- Autónomo: Toma decisiones independientemente
- Observable: Cada paso está rastreado
- Iterativo: Mejora con cada turno
- Detenible: Sabe cuándo parar

---

## 🎯 Sección 8: Métricas de Evaluación

### Concepto: Evaluación de Calidad

```python
def evaluate_response(response: str, context: str) -> dict:
    """Calcula múltiples métricas de calidad"""
    
    # Métrica 1: Ratio de Longitud
    length_ratio = min(len(response), 500) / 500
    
    # Métrica 2: BLEU-like (superposición de vocabulario)
    response_words = set(response.lower().split())
    context_words = set(context.lower().split())
    overlap = len(response_words & context_words)
    vocabulary_overlap = overlap / max(len(response_words), 1)
    
    # Métrica 3: Similitud de Embeddings
    response_emb = create_embedding(response)
    context_emb = create_embedding(context)
    similarity = cosine_similarity(
        response_emb.reshape(1, -1),
        context_emb.reshape(1, -1)
    )[0][0]
    
    # Métrica 4: Coherencia (diversidad de tokens)
    tokens = response.lower().split()
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    coherence = 0.5 + 0.5 * (1 - unique_ratio)  # Balanceado
    
    # Métrica 5: Calidad General
    quality_score = (
        length_ratio * 0.2 +
        vocabulary_overlap * 0.3 +
        similarity * 0.25 +
        coherence * 0.25
    ) * 100
    
    return {
        'metrics': {
            'length_ratio': length_ratio,
            'vocabulary_overlap': vocabulary_overlap,
            'embedding_similarity': similarity,
            'coherence': coherence
        },
        'quality_score': quality_score,
        'interpretation': interpret_score(quality_score)
    }
```

**Lo que enseña:**

**Tipos de Métricas:**

1. **Ratio de Longitud**: 0-1
   - Asegura que la respuesta no sea muy corta/larga
   
2. **BLEU Score**: 0-1
   - ¿Cuántas palabras se superponen con el contexto?
   
3. **Similitud de Embeddings**: -1 a 1
   - ¿Son respuesta y contexto semánticamente similares?
   
4. **Coherencia**: 0-1
   - ¿Evita la respuesta la repetición?
   
5. **Calidad General**: 0-100
   - Combinación ponderada de las anteriores

**¿Por qué múltiples métricas?**
```
Una sola métrica = imagen incompleta
Ejemplo: Una respuesta corta y genérica podría puntuar alto en 
         vocabulary_overlap pero bajo en length_ratio
```

---

## 🎓 Lista de Verificación de Aprendizaje

Después de leer esto, deberías entender:

- [ ] Cómo el texto se convierte en vectores (embeddings)
- [ ] Cómo se calcula la similitud (similitud coseno)
- [ ] Cómo se recuperan documentos (búsqueda k-NN)
- [ ] Cómo se estructura el razonamiento (Chain-of-Thought)
- [ ] Cómo la temperatura afecta la aleatoriedad (escalado softmax)
- [ ] Cómo los agentes toman decisiones (bucle ReAct)
- [ ] Cómo se mide la calidad (múltiples métricas)
- [ ] Cómo se integran los componentes (pipeline)

---

## 🔬 Ideas de Experimentación

Intenta modificar:

```python
# 1. Cambiar dimensión de embedding
EMBEDDING_DIM = 256  # Más dimensiones = más preciso

# 2. Cambiar temperatura
temperature = 0.1    # Más enfocado
temperature = 2.0    # Más creativo

# 3. Cambiar k_documents
k = 5                # Más contexto = más lento pero más rico

# 4. Añadir más documentos
KNOWLEDGE_BASE['doc_4'] = "Tu nuevo documento..."

# 5. Cambiar pesos de evaluación
quality_score = (
    length_ratio * 0.1 +
    vocabulary_overlap * 0.5 +  # Más énfasis aquí
    similarity * 0.2 +
    coherence * 0.2
) * 100
```

---

## 📚 Lecturas Adicionales

- **Capítulo 11:** Temperatura y Generación
- **Capítulo 12:** Razonamiento Chain-of-Thought
- **Capítulo 13:** Arquitectura RAG
- **Capítulo 14:** Patrones de Agentes (ReAct)
- **Capítulo 15:** Evaluación

---

**¡Ahora entiendes el código! 🎓**
