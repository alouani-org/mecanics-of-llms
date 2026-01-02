# 🏗️ Arquitectura: El Mini Asistente Completo (Script 09)
🌍 [English](../en/INDEX_SCRIPT_09.md) | 📖 [Français](../fr/INDEX_SCRIPT_09.md) | 🇪🇸 **Español** | 🇧🇷 [Português](../pt/INDEX_SCRIPT_09.md) | 🇸🇦 [العربية](../ar/INDEX_SCRIPT_09.md)
> **Desglose completo** del proyecto integrador  
> Entendiendo la estructura técnica: capas, componentes, flujo

---

## 📍 Navegación Rápida

- **📖 Ver: [Recorrido Pedagógico](PEDAGOGICAL_JOURNEY.md)** - Cómo se conecta con los capítulos
- **⚡ Ver: [Inicio Rápido](QUICKSTART_SCRIPT_09.md)** - Ejecuta en 5 minutos
- **🔗 Ver: [Mapeo Código ↔ Conceptos](SCRIPT_09_MAPPING.md)** - Qué código enseña qué
- **🌍 Otros idiomas: [English](../en/INDEX_SCRIPT_09.md) | [Français](../fr/INDEX_SCRIPT_09.md) | [Português](../pt/INDEX_SCRIPT_09.md)**

---

## 🎯 ¿Qué Hay Dentro?

El Script 09 demuestra TODOS los conceptos de los capítulos 11-15:

| Capítulo | Concepto | Componente en Script 09 |
|----------|----------|------------------------|
| 11 | Generación + Temperatura | `generate_with_temperature()` |
| 12 | Chain-of-Thought | `reasoning_phase()` |
| 13 | RAG + Recuperación | `retrieve_documents()` |
| 14 | Agentes ReAct | `agent_loop()` |
| 15 | Evaluación | `evaluate_response()` |

---

## 🏗️ Arquitectura Técnica

### Capa 1: Capa de Datos
```
Base de Conocimientos (en memoria)
    ↓
Fragmentación de Documentos
    ↓
Embeddings Vectoriales (numpy)
```

**Responsabilidad:** Almacenar e indexar conocimiento
**Ubicación del código:** `load_knowledge_base()`, `embed_documents()`

---

### Capa 2: Capa de Recuperación (RAG)
```
Consulta del Usuario
    ↓
Embed de la Consulta
    ↓
Búsqueda por Similitud (coseno)
    ↓
Contextos Recuperados
```

**Responsabilidad:** Encontrar documentos relevantes
**Ubicación del código:** `retrieve_documents()`

**Función Clave:**
```python
def retrieve_documents(query: str, k: int = 3) -> list:
    # 1. Embed de la consulta
    # 2. Calcular similitud con todos los documentos
    # 3. Retornar top-k más relevantes
```

---

### Capa 3: Capa de Razonamiento (Chain-of-Thought)
```
Pregunta
    ↓
Paso 1: Analizar problema
Paso 2: Recuperar contexto
Paso 3: Pensar paso a paso
    ↓
Traza de Razonamiento
```

**Responsabilidad:** Estructurar el pensamiento
**Ubicación del código:** `reasoning_phase()`

---

### Capa 4: Capa de Generación (similar a LLM)
```
Traza de Razonamiento + Contexto
    ↓
Selección de Token (softmax)
    ↓
Muestreo con Temperatura
    ↓
Generación de Respuesta
```

**Responsabilidad:** Crear texto
**Ubicación del código:** `generate_with_temperature()`

---

### Capa 5: Capa de Agente (ReAct)
```
Decisión del Agente (Pensar)
    ↓
Selección de Herramienta (Actuar)
    ↓
Observar Resultado
    ↓
Bucle hasta terminar
```

**Responsabilidad:** Ejecución autónoma
**Ubicación del código:** `agent_loop()`

---

### Capa 6: Capa de Evaluación
```
Respuesta Generada
    ↓
Múltiples Métricas (BLEU, Similitud de Embeddings, Coherencia)
    ↓
Puntuación (0-100)
```

**Responsabilidad:** Evaluación de calidad
**Ubicación del código:** `evaluate_response()`

---

## 🔄 Flujo de Ejecución Completo

```
Entrada del Usuario
    ↓
embed_documents() → Vectores de documentos (128-dim)
    ↓
retrieve_documents() → Top-k documentos similares
    ↓
reasoning_phase() → Pensamiento estructurado
    ↓
generate_with_temperature() → Generación de texto
    ↓
agent_loop() → Iteración autónoma
    ↓
evaluate_response() → Métricas de calidad
    ↓
Salida al Usuario
```

**Paso a paso:**

1. **Procesamiento de Entrada**
   - Parsear consulta del usuario
   - Preparar para recuperación

2. **Recuperación (RAG)**
   - Encontrar contexto relevante de la base de conocimientos
   - Retornar top-3 documentos

3. **Razonamiento**
   - Crear cadena de pensamiento
   - Analizar problema paso a paso
   - Incluir contexto recuperado

4. **Generación**
   - Seleccionar tokens usando softmax
   - Aplicar muestreo con temperatura
   - Construir respuesta iterativamente

5. **Bucle del Agente**
   - Decidir: ¿continuar o parar?
   - Seleccionar herramienta si es necesario
   - Ejecutar y observar

6. **Evaluación**
   - Calcular 5 métricas de calidad
   - Retornar resultado con puntuación

7. **Retorno**
   - Presentar respuesta al usuario
   - Mostrar métricas y traza

---

## 📦 Funciones Principales

### `load_knowledge_base() → dict`
```python
# Retorna diccionario de documentos
{
    'doc_1': "Contenido sobre IA...",
    'doc_2': "Contenido sobre LLMs...",
    ...
}
```

---

### `embed_documents(docs: dict) → np.ndarray`
```python
# Retorna matriz (num_docs, embedding_dim)
# Simple: Embeddings basados en hash para demo
# Real: Usar embeddings de SentenceTransformer
```

---

### `retrieve_documents(query: str, k: int = 3) → list`
```python
# Entrada: "¿Qué es un LLM?"
# Salida: [
#   {'doc': 'doc_1', 'content': '...', 'similarity': 0.87},
#   {'doc': 'doc_2', 'content': '...', 'similarity': 0.76},
#   {'doc': 'doc_3', 'content': '...', 'similarity': 0.68}
# ]
```

---

### `reasoning_phase(question: str, contexts: list) → str`
```python
# Entrada: pregunta + contextos recuperados
# Salida: Traza de pensamiento estructurado
"""
Paso 1: Analizar la pregunta
El usuario pregunta sobre LLMs...

Paso 2: Identificar conceptos clave
Conceptos: arquitectura, entrenamiento, inferencia...

Paso 3: Recuperar contexto relevante
Del documento X, sabemos que...

Paso 4: Sintetizar
Combinando el conocimiento, podemos concluir...
"""
```

---

### `generate_with_temperature(prompt: str, temp: float = 1.0) → str`
```python
# Temperatura baja (0.3): determinístico, enfocado
# Temperatura media (1.0): balanceado
# Temperatura alta (2.0): creativo, diverso

# Retorna segmento de texto generado
```

---

### `agent_loop(initial_query: str, max_turns: int = 3) → dict`
```python
# Ejecución agéntica
# Cada turno: Pensar → Actuar → Observar

# Retorna: {
#   'answer': 'Respuesta final',
#   'turns': 3,
#   'trace': ['Turno 1: ...', 'Turno 2: ...', ...]
# }
```

---

### `evaluate_response(response: str, context: str) → dict`
```python
# Calcula 5 métricas:
# - Ratio de longitud
# - Superposición de vocabulario (BLEU)
# - Similitud de embeddings
# - Puntuación de coherencia
# - Calidad general (0-100)

# Retorna: {
#   'metrics': {'bleu': 0.75, 'similarity': 0.82, ...},
#   'quality_score': 79,
#   'interpretation': 'Buena respuesta...'
# }
```

---

## ⚙️ Configuración y Parámetros

| Parámetro | Default | Rango | Efecto |
|-----------|---------|-------|--------|
| `TEMPERATURE` | 1.0 | 0.0-2.0 | Control de creatividad |
| `K_DOCUMENTS` | 3 | 1-10 | Tamaño de contexto |
| `MAX_TURNS` | 3 | 1-10 | Iteraciones del agente |
| `EMBEDDING_DIM` | 128 | 64-512 | Tamaño de embedding |

**Cómo modificar:**
```python
# En script 09
TEMPERATURE = 1.5        # Más creativo
K_DOCUMENTS = 5          # Más contexto
MAX_TURNS = 5            # Más iteraciones del agente
```

---

## 💡 Detalles Clave de Implementación

### Embeddings (Demo Simplificado)
```python
# Producción real: SentenceTransformer
# Versión demo: Basado en hash (determinístico, rápido)

def simple_embedding(text: str, dim: int = 128) -> np.ndarray:
    hash_val = hash(text)
    np.random.seed(abs(hash_val) % 2**32)
    return np.random.randn(dim)
```

---

### Muestreo con Temperatura
```python
# Temperatura = factor de escala para softmax
# logits = [1.0, 2.0, 0.5]
# 
# T=0.5: softmax(logits / 0.5) → más agudo [0.1, 0.87, 0.03]
# T=1.0: softmax(logits / 1.0) → normal [0.09, 0.67, 0.24]
# T=2.0: softmax(logits / 2.0) → más plano [0.28, 0.38, 0.34]
```

---

### Prompting Chain-of-Thought
```
En lugar de: "¿Qué es X?"
Mejor:       "Pensemos paso a paso:
              1. Definir el concepto
              2. Desglosarlo
              3. Proporcionar ejemplos
              4. Concluir"
```

---

### Implementación del Bucle ReAct
```python
while not done and turns < max_turns:
    # PENSAR: Analizar estado actual
    thought = analyze_state(context)
    
    # ACTUAR: Elegir y ejecutar herramienta/acción
    action = select_action(thought)
    result = execute_action(action)
    
    # OBSERVAR: Actualizar conocimiento
    observation = observe_result(result)
    
    turns += 1
```

---

## 🎯 Resultados de Aprendizaje

Después de estudiar esta arquitectura, entiendes:

✅ Cómo RAG integra recuperación con generación  
✅ Cómo la temperatura afecta el comportamiento del modelo  
✅ Cómo Chain-of-Thought mejora el razonamiento  
✅ Cómo los agentes toman decisiones autónomas  
✅ Cómo evaluar la calidad de generación  
✅ Cómo combinar todos estos conceptos en un sistema  

---

## 🚀 Siguientes Pasos

1. **Ejecuta:** [Guía de Inicio Rápido](QUICKSTART_SCRIPT_09.md)
2. **Entiende el código:** [Mapeo Código ↔ Conceptos](SCRIPT_09_MAPPING.md)
3. **Adáptalo:** Modifica para tu caso de uso
4. **Extiéndelo:** Agrega más herramientas, mejores embeddings, etc.

---

**¿Listo para profundizar? 📚**
