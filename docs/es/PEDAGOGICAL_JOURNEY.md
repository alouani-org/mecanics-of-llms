# 🗺️ Recorrido Pedagógico Completo: Libro → Scripts → Conceptos

> **Guía completa** para navegar el proyecto "La Mecánica de los LLM"  
> Correspondencia detallada: capítulos del libro ↔ scripts Python ↔ conceptos prácticos

---

## 📍 Cómo Empezar...

### Si eres nuevo ✨

```
1. Lee esta página (estás aquí)
   ↓
2. Revisa README.md (navegación general)
   ↓
3. Abre PEDAGOGICAL_JOURNEY.md (guía de scripts)
   ↓
4. Ejecuta tu primer script
```

### Si ya leíste el libro 📖

```
1. Encuentra tu capítulo abajo
   ↓
2. Haz clic en el script correspondiente
   ↓
3. Ejecuta y experimenta
```

### Si quieres programar de inmediato 💻

```
1. Ve directamente a: 09_mini_assistant_complet.py
   ↓
2. Lee: INDEX_SCRIPT_09.md (arquitectura)
   ↓
3. Entiende y luego adapta
```

---

## 📚 Recorrido Por Capítulo del Libro

### Capítulo 1: Introducción a NLP

**Contenido del Libro:**
- ¿Qué es NLP?
- Historia: de reglas a aprendizaje a LLMs
- Dónde estamos en 2025

**Enlace de Código:**
- ❌ Sin script dedicado (teórico)
- ✅ Continúa al Capítulo 2

---

### Capítulo 2: Representación de Texto y Modelos Secuenciales

**Contenido del Libro:**
- ¿Cómo ven los modelos el texto?
- Tokens y tokenizadores (BPE, WordPiece, SentencePiece)
- Impacto en la longitud de secuencia
- RNNs, LSTMs, GRUs (los ancestros)

**👉 Script Correspondiente:**

#### [`01_tokenization_embeddings.py`](../../01_tokenization_embeddings.py)

**Lo que aprendes ejecutando:**
```python
python 01_tokenization_embeddings.py
```

- Tokenización con diferentes tokenizadores
- Impacto de la tokenización en la longitud de secuencia
- Diferencias Francés vs Inglés
- Embeddings y sus dimensiones
- Costo computacional basado en tokens

**Conceptos Clave Demostrados:**
- Tokenizadores BPE (Byte Pair Encoding)
- Vocabulario y subpalabras
- Relación Tokens ↔ costo de atención O(n²)

**Tiempo de ejecución:** ~5 segundos  
**Requisitos:** Python, `transformers`

---

### Capítulo 3: Arquitectura Transformer

**Contenido del Libro:**
- La invención del mecanismo de atención
- Self-attention y atención multi-cabezas
- Estructura encoder-decoder
- Codificación posicional
- El problema de la posición

**👉 Script Correspondiente:**

#### [`02_multihead_attention.py`](../../02_multihead_attention.py)

**Lo que aprendes ejecutando:**
```python
python 02_multihead_attention.py
```

- Arquitectura de una capa de atención
- Proyecciones Q, K, V (Query, Key, Value)
- Cálculo de puntuaciones de atención
- Multi-head: cómo cada cabeza se enfoca diferente
- Visualización: ¿quién atiende a quién?

**Conceptos Clave Demostrados:**
- Softmax y normalización de puntuaciones
- Dimensión de embedding vs número de cabezas
- Cada cabeza aprende diferentes relaciones

**Tiempo de ejecución:** ~2 segundos  
**Requisitos:** Python, `numpy`

---

### Capítulos 4-8: Arquitectura, Optimización, Pre-entrenamiento

**Contenido del Libro:**
- Cap. 4: Modelos derivados de Transformer (BERT, GPT, T5...)
- Cap. 5: Optimización de arquitectura (atención lineal, RoPE...)
- Cap. 6: Arquitectura MoE (Mixture of Experts)
- Cap. 7: Pre-entrenamiento de LLM
- Cap. 8: Optimizaciones de entrenamiento (acumulación de gradiente...)

**Enlace de Código:**
- 📖 Teórico + conceptos
- ⚡ Integrado en Script 03 (temperatura durante pre-entrenamiento)
- 🏆 Mejorado en Script 09 (mini-asistente)

---

### Capítulo 9: Fine-tuning Supervisado (SFT)

**Contenido del Libro:**
- De predicción a asistencia
- Fine-tuning supervisado (SFT)
- Calidad sobre cantidad
- Evaluación de modelos fine-tuneados
- Caso de estudio: adaptar LLaMA 7B

**👉 Script Bonus Correspondiente:**

#### [`08_lora_finetuning_example.py`](../../08_lora_finetuning_example.py) 🎁

**Lo que aprendes ejecutando:**
```python
python 08_lora_finetuning_example.py
```

- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
- Comparación: full fine-tuning vs LoRA
- Eficiencia en términos de memoria/velocidad
- Caso real SNCF (del texto del libro)

**Conceptos Clave Demostrados:**
- Adaptar modelos sin reentrenar todo
- Compromiso memoria vs calidad
- Parámetros adicionales vs ganancia

**Tiempo de ejecución:** ~3 segundos  
**Requisitos:** Python, `numpy` (demo sin LLM externo)

---

### Capítulo 11: Estrategias de Generación e Inferencia

**Contenido del Libro:**
- Prompting: guiar el modelo a través de ejemplos
- Control de temperatura
- Estrategias de muestreo (top-k, top-p, nucleus sampling)
- Optimizar latencia: KV-cache, especulación

**👉 Scripts Correspondientes:**

#### [`03_temperature_softmax.py`](../../03_temperature_softmax.py)

**Lo que aprendes ejecutando:**
```python
python 03_temperature_softmax.py
```

- Efecto de la temperatura en softmax
- T baja = determinístico (greedy)
- T alta = diversidad (creativo)
- Relación con la entropía
- Gráficos del efecto de temperatura

**Conceptos Clave Demostrados:**
- Softmax e interpretación probabilística
- Temperatura como factor de escala
- Compromiso determinismo vs creatividad

**Tiempo de ejecución:** ~2 segundos  
**Requisitos:** Python, `matplotlib` (opcional)

#### [`09_mini_assistant_complet.py`](../../09_mini_assistant_complet.py) 🏆

**Tu primer asistente con:**
- Prompting (Chain-of-Thought)
- Muestreo con temperatura
- Estrategias de generación

---

### Capítulo 12: Modelos de Razonamiento

**Contenido del Libro:**
- Prompting Chain-of-Thought (CoT)
- Tree-of-Thought (ToT)
- Código y matemáticas (demostración de razonamiento)
- Aprendizaje por Refuerzo (RL) para pensar

**👉 Scripts Correspondientes:**

#### [`05_pass_at_k_evaluation.py`](../../05_pass_at_k_evaluation.py)

**Lo que aprendes ejecutando:**
```python
python 05_pass_at_k_evaluation.py
```

- Métrica Pass@k para evaluación
- Pass^k (diferente de Pass@k)
- ¿Por qué estas métricas para razonamiento?
- Empíricos en tareas de código

**Conceptos Clave Demostrados:**
- Evaluación más allá de la simple precisión
- Múltiples intentos vs un solo intento
- Métricas específicas para razonamiento

**Tiempo de ejecución:** ~1 segundo  
**Requisitos:** Python, `numpy`

---

### Capítulo 13: Sistemas Aumentados y Agentes (RAG)

**Contenido del Libro:**
- RAG: Retrieval-Augmented Generation
- El problema de integración M:N
- Bajo el capó: implementación técnica
- Descubrimiento progresivo de herramientas

**👉 Scripts Correspondientes:**

#### [`04_rag_minimal.py`](../../04_rag_minimal.py)

**Lo que aprendes ejecutando:**
```python
python 04_rag_minimal.py
```

- Pipeline RAG mínimo (entender los pasos)
- Similitud coseno para recuperación
- Aumentación de contexto
- Calidad vs latencia

**Conceptos Clave Demostrados:**
- Fragmentación de documentos (chunking)
- Embeddings y búsqueda
- Reducción de alucinaciones

**Tiempo de ejecución:** ~3 segundos  
**Requisitos:** Python, `numpy`, `scikit-learn`

#### [`07_llamaindex_rag_advanced.py`](../../07_llamaindex_rag_advanced.py) 🎁

**Lo que aprendes ejecutando:**
```python
python 07_llamaindex_rag_advanced.py
```

- Framework RAG completo (LlamaIndex)
- 6 fases: Cargar → Indexar → RAG → Chat → Eval → Exportar
- Ingestión de documentos
- Chat con persistencia
- Evaluación automática

**Conceptos Clave Demostrados:**
- Arquitectura RAG de producción
- Estrategias de indexación
- Capa de persistencia

**Tiempo de ejecución:** ~5 segundos  
**Requisitos:** Python (demo), opcional: `llama-index`, `openai`

---

### Capítulo 14: Protocolos Agénticos (MCP)

**Contenido del Libro:**
- Agentes: autonomía y decisión
- Definición de agente
- Patrones: ReAct, Tool Use, Function Calling
- Model Context Protocol (MCP)
- Limitaciones y dificultades

**👉 Script Bonus Correspondiente:**

#### [`06_react_agent_bonus.py`](../../06_react_agent_bonus.py) 🎁

**Lo que aprendes ejecutando:**
```python
python 06_react_agent_bonus.py
```

- Patrón ReAct (Razonamiento + Acción)
- Framework genérico para crear agentes
- Registro de herramientas (tool registration)
- 3 herramientas de ejemplo
- Bucle: pensar → actuar → observar

**Conceptos Clave Demostrados:**
- Bucle de agente autónomo
- Toma de decisiones
- Composición de herramientas

**Tiempo de ejecución:** ~4 segundos  
**Requisitos:** Python, `numpy`

**Ver también:** [REACT_AGENT_INTEGRATION.md](REACT_AGENT_INTEGRATION.md)

---

### Capítulo 15: Evaluación Crítica de Flujos Agénticos

**Contenido del Libro:**
- El desafío de la medición
- Evaluar agentes: de palabras a hechos
- Métricas cuantitativas y cualitativas
- Casos de estudio

**👉 Script Integrador Completo:**

#### [`09_mini_assistant_complet.py`](../../09_mini_assistant_complet.py) 🏆

**Lo que aprendes ejecutando:**
```python
python 09_mini_assistant_complet.py
```

- Evaluación de un sistema completo
- Métricas: BLEU, similitud de embeddings, coherencia
- Trazas y debugging
- Mejora iterativa

**Conceptos Clave Demostrados:**
- Evaluación multi-criterio
- Bucles de retroalimentación
- Calidad de ejecución

**Tiempo de ejecución:** ~10 segundos  
**Requisitos:** Python (todo incluido)

**Ver también:**
- [INDEX_SCRIPT_09.md](INDEX_SCRIPT_09.md) - Arquitectura
- [QUICKSTART_SCRIPT_09.md](QUICKSTART_SCRIPT_09.md) - Inicio rápido

**¡FELICITACIONES!** 🎉 ¡Has completado el recorrido!

---

## 🎯 Rutas Aceleradas

### "Quiero entender los LLM rápidamente" (2-3 horas)

```
Leer Capítulos 1-3        (30 min)
   ↓
Ejecutar Scripts 01-02    (15 min)
   ↓
Leer Capítulos 11-12      (45 min)
   ↓
Ejecutar Scripts 03-05    (30 min)
   ↓
Leer Capítulos 13-14      (45 min)
   ↓
Ejecutar Script 09        (15 min)
```

**Resultado:** Comprensión sólida de conceptos clave ✅

### "Quiero programar una aplicación RAG + Agentes" (4-6 horas)

```
Entender RAG              (Capítulo 13)  (30 min)
   ↓
Ejecutar Scripts 04, 07   (30 min)
   ↓
Entender Agentes          (Capítulo 14)  (30 min)
   ↓
Ejecutar Script 06        (20 min)
   ↓
Estudiar Script 09        (60 min)
   ↓
Adaptar para tu caso      (variable)
```

**Resultado:** Aplicación funcional RAG + Agentes ✅

---

## 📝 Notas

- **No se requiere GPU**: todos los scripts funcionan en CPU (más lento)
- **Dependencias mínimas**: solo `numpy`, `torch`, `transformers`, `scikit-learn`
- **Código educativo**: prioriza claridad sobre optimización
- **Compatible Python 3.9+**
- **Scripts bonus** demuestran conceptos avanzados, funcionan sin LLM externo (modo simulación)

---

**¡Feliz aprendizaje! 🎓**
