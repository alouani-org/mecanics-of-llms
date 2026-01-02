
# Scripts Prácticos: Experimentando con Conceptos de LLM

🌍 [English](../en/README.md) | 📖 [Français](../fr/README.md) | 🇪🇸 **Español** | 🇧🇷 [Português](../pt/README.md) | 🇸🇦 [العربية](../ar/README.md)

Colección de **10 scripts de Python ejecutables** para experimentar con los conceptos clave del libro **"La Mecánica de los LLM"**.

> 📚 **Acerca de** : Estos scripts acompañan los capítulos del libro. Ver [Recorrido Pedagógico](PEDAGOGICAL_JOURNEY.md) para las correspondencias detalladas.

**📕 Comprar el Libro:**
- **Impreso** : [Amazon](https://amzn.eu/d/3oREERI)
- **Kindle** : [Amazon](https://amzn.eu/d/b7sG5iw)

---

## 📋 Vista General de los Scripts

| # | Script | Capítulo(s) | Conceptos | Estado |
|---|--------|-------------|-----------|--------|
| 1 | `01_tokenization_embeddings.py` | 2 | Tokenización, impacto en longitud de secuencia | ✅ |
| 2 | `02_multihead_attention.py` | 3 | Self-attention, multi-head, pesos de atención | ✅ |
| 3 | `03_temperature_softmax.py` | 7, 11 | Temperatura, softmax, entropía | ✅ |
| 4 | `04_rag_minimal.py` | 13 | Pipeline RAG, recuperación, similitud coseno | ✅ |
| 5 | `05_pass_at_k_evaluation.py` | 12 | Pass@k, Pass^k, evaluación de modelos | ✅ |
| 🎁 6 | `06_react_agent_bonus.py` | 14, 15 | **Agentes ReAct, registro de herramientas, MCP** | ✅ BONUS |
| 🎁 7 | `07_llamaindex_rag_advanced.py` | 13, 14 | **RAG avanzado, indexación, chat persistente** | ✅ BONUS |
| 🎁 8 | `08_lora_finetuning_example.py` | 9, 10 | **LoRA, QLoRA, comparación de fine-tuning** | ✅ BONUS |
| 🏆 **9** | `09_mini_assistant_complet.py` | **11-15** | **🎯 Proyecto Integrador Final** | ✅ PRINCIPAL |
| 🎁 10 | `10_activation_steering_demo.py` | 10 | **Activation Steering, 3SO, vectores de concepto** | ✅ BONUS |

---

## 📖 Descripciones Detalladas de los Scripts

### 📌 Script 01: Tokenización y Embeddings
**Archivo:** `01_tokenization_embeddings.py` | **Capítulo:** 2

**Lo que hace el script:**
- Carga un tokenizador (GPT-2 o LLaMA-2) y analiza diferentes textos
- Compara el número de tokens entre francés e inglés
- Demuestra el impacto de la longitud de secuencia en el costo computacional

**Lo que aprendes:**
- Cómo el texto se divide en tokens (BPE, WordPiece)
- Por qué "Bonjour" puede convertirse en 2-3 tokens mientras "Hello" es solo uno
- El impacto directo: más tokens = mayor costo O(n²) para la atención

**Salida esperada:**
```
Text: L'IA est utile
  Token count: 5
  Tokens: ['L', "'", 'IA', 'est', 'utile']
```

---

### 📌 Script 02: Atención Multi-Cabezas
**Archivo:** `02_multihead_attention.py` | **Capítulo:** 3

**Lo que hace el script:**
- Simula una capa de atención multi-cabezas con tensores PyTorch
- Calcula las proyecciones Q, K, V y los pesos de atención
- Muestra cómo cada cabeza "mira" la oración de manera diferente

**Lo que aprendes:**
- El mecanismo Q (Query), K (Key), V (Value)
- Por qué múltiples cabezas capturan diferentes dependencias
- Que los pesos de atención siempre suman 1 (distribución de probabilidad)

**Salida esperada:**
```
Sentence: The cat sleeps well
Head 1: Attention weights from 'cat' → 'sleeps': 0.42
Head 2: Attention weights from 'cat' → 'The': 0.38
```

---

### 📌 Script 03: Temperatura y Softmax
**Archivo:** `03_temperature_softmax.py` | **Capítulos:** 7, 11

**Lo que hace el script:**
- Aplica softmax con diferentes temperaturas (0.1, 0.5, 1.0, 2.0)
- Calcula la entropía de Shannon para cada distribución
- Genera gráficos (si matplotlib está instalado)

**Lo que aprendes:**
- T < 1: distribución "aguda" → generación determinística (greedy)
- T > 1: distribución "plana" → generación creativa/diversa
- La entropía aumenta con la temperatura (más incertidumbre)

**Salida esperada:**
```
Temperature 0.5: Token 'Paris' = 85% (agudo, determinístico)
Temperature 2.0: Token 'Paris' = 35% (plano, creativo)
```

---

### 📌 Script 04: RAG Mínimo
**Archivo:** `04_rag_minimal.py` | **Capítulo:** 13

**Lo que hace el script:**
- Crea una mini base de conocimientos (7 documentos sobre LLMs)
- Vectoriza los documentos con TF-IDF
- Realiza búsqueda por similitud coseno
- Simula la generación aumentada por el contexto recuperado

**Lo que aprendes:**
- El pipeline RAG completo: Recuperación → Aumentación → Generación
- Cómo la similitud coseno encuentra los documentos relevantes
- Por qué RAG permite responder preguntas sobre datos privados

**Salida esperada:**
```
Pregunta: "¿Cómo funciona la atención en el Transformer?"
→ Documentos recuperados: [doc_1: 0.72, doc_4: 0.65]
→ Respuesta generada con contexto
```

---

### 📌 Script 05: Evaluación Pass@k
**Archivo:** `05_pass_at_k_evaluation.py` | **Capítulo:** 12

**Lo que hace el script:**
- Simula 100 intentos de generación con una tasa de éxito del 30%
- Calcula Pass@k (al menos 1 éxito en k intentos)
- Calcula Pass^k (todos los k intentos exitosos)

**Lo que aprendes:**
- Pass@k = 1 - (1-p)^k: probabilidad de al menos un éxito
- Pass^k = p^k: probabilidad de que todos tengan éxito (muy estricto)
- Por qué Pass@10 ≈ 97% incluso con p=30% (tienes 10 oportunidades)

**Salida esperada:**
```
Pass@1  = 30%  (oportunidad con 1 intento)
Pass@5  = 83%  (oportunidad con 5 intentos)
Pass@10 = 97%  (casi seguro con 10 intentos)
```

---

### 🎁 Script 06: Agente ReAct (BONUS)
**Archivo:** `06_react_agent_bonus.py` | **Capítulos:** 14, 15

**Lo que hace el script:**
- Implementa un mini framework de agentes autónomos
- Demuestra el bucle ReAct: Thought → Action → Observation → ...
- Incluye herramientas simuladas: calculadora, búsqueda web, clima

**Lo que aprendes:**
- El patrón ReAct (Razonamiento + Acción)
- Cómo un agente decide qué acción tomar
- Auto-corrección: el agente puede reintentar si una acción falla
- La base para entender agentes MCP (Model Context Protocol)

**Salida esperada:**
```
Thought: Necesito calcular el 15% de $250
Action: calculator(250 * 0.15)
Observation: 37.5
Final Answer: La propina es de $37.50
```

---

### 🎁 Script 07: RAG Avanzado con LlamaIndex (BONUS)
**Archivo:** `07_llamaindex_rag_advanced.py` | **Capítulos:** 13, 14

**Lo que hace el script:**
- Sistema RAG completo con parsing de documentos
- Indexación y embeddings (simulados o reales con OpenAI)
- Chat con memoria conversacional
- Evaluación de calidad (Precisión, Recall, F1)

**Lo que aprendes:**
- Arquitectura RAG de producción: ingestión → indexación → recuperación → generación
- Cómo mantener el contexto a través de múltiples turnos de conversación
- Cómo evaluar la calidad de un sistema RAG

**Salida esperada:**
```
[Modo Chat]
Usuario: ¿Qué es un Transformer?
Asistente: [Contexto: 3 docs] Un Transformer es...
Usuario: ¿Y la atención multi-cabezas?
Asistente: [Memoria: pregunta anterior + 2 docs] ...
```

---

### 🎁 Script 08: Fine-tuning LoRA/QLoRA (BONUS)
**Archivo:** `08_lora_finetuning_example.py` | **Capítulos:** 9, 10

**Lo que hace el script:**
- Compara Full Fine-tuning vs LoRA vs QLoRA (cálculos numéricos)
- Muestra los ahorros de VRAM y parámetros entrenables
- Caso de uso: adaptar LLaMA-7B para un dominio empresarial (ferrocarril)

**Lo que aprendes:**
- LoRA: añade ~0.1% de parámetros vs fine-tuning completo
- QLoRA: cuantización de 4 bits + LoRA = GPU de 24GB en lugar de 140GB
- Por qué el fine-tuning eficiente democratiza los LLMs

**Salida esperada:**
```
LLaMA-7B:
  Full Fine-tuning: 28 GB VRAM, 7B params
  LoRA (rank=8):    8 GB VRAM, 4.2M params (0.06%)
  QLoRA:            6 GB VRAM, 4.2M params + base 4-bit
```

---

### � Script 10: Activation Steering & 3SO (BONUS)
**Archivo:** `10_activation_steering_demo.py` | **Capítulo:** 10

**Lo que hace el script:**
- Demuestra el steering por activaciones: inyección de vectores de concepto
- Implementa extracción de vectores por activación contrastiva
- Simula un Sparse Autoencoder (SAE) para descomposición en conceptos
- Implementa una máquina de estados finitos para 3SO (salidas JSON garantizadas)
- Compara RLHF/DPO vs Steering con tabla detallada

**Lo que aprendes:**
- El steering modifica las activaciones en inferencia: $X_{steered} = X + (c \times V)$
- Cómo extraer vectores de concepto (método contrastivo, SAE)
- Impacto del coeficiente de steering (muy bajo → nulo, óptimo → efectivo, muy alto → descarrilamiento)
- El 3SO garantiza matemáticamente una sintaxis JSON válida
- Cuándo usar alineamiento vs steering

**Salida esperada:**
```
STEP 3: Analyzing Coefficient Effect
   Coeff   Direction Δ     Perturbation    Stability
   1.0     12.5°           8.2%            ✅ stable
   5.0     45.3°           35.1%           ⚠️ moderate
   15.0    78.2°           89.4%           ❌ unstable
```

---

### �🏆 Script 09: Mini-Asistente Completo (PROYECTO FINAL)
**Archivo:** `09_mini_assistant_complet.py` | **Capítulos:** 11-15

**Lo que hace el script:**
- Integra TODOS los conceptos: RAG + Agentes + Temperatura + Evaluación
- Sistema completo con base de conocimientos, recuperación, razonamiento
- Modo interactivo para probar diferentes preguntas

**Lo que aprendes:**
- Cómo ensamblar un asistente IA completo de A a Z
- Arquitectura en capas: Datos → Recuperación → Razonamiento → Generación
- Evaluación de extremo a extremo de un sistema

**Documentación dedicada:**
- [INDEX_SCRIPT_09.md](INDEX_SCRIPT_09.md): Arquitectura completa
- [QUICKSTART_SCRIPT_09.md](QUICKSTART_SCRIPT_09.md): Inicio rápido en 5 min
- [SCRIPT_09_MAPPING.md](SCRIPT_09_MAPPING.md): Mapeo código ↔ conceptos

---

## 🚀 Inicio Rápido

### 1. Crear un Entorno Virtual (recomendado)

```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 2. Instalar Dependencias

```bash
# Instalación básica (para scripts 1-5)
pip install torch transformers numpy scikit-learn

# Instalación completa (con visualizaciones)
pip install torch transformers numpy scikit-learn matplotlib

# Para scripts bonus (opcional, funcionan en modo demo sin estas)
pip install llama-index openai python-dotenv peft bitsandbytes
```

**Nota:** Los scripts bonus (06, 07, 08) funcionan **sin dependencias externas** en modo demo.

### 3. Ejecutar un Script

```bash
python 01_tokenization_embeddings.py
python 02_multihead_attention.py
python 03_temperature_softmax.py
python 04_rag_minimal.py
python 05_pass_at_k_evaluation.py
python 06_react_agent_bonus.py
python 07_llamaindex_rag_advanced.py
python 08_lora_finetuning_example.py
python 09_mini_assistant_complet.py    # ← Proyecto integrador final
```

---

## 🏆 Proyecto Integrador: Mini-Asistente Completo

**EL script principal**: integra TODOS los conceptos de los capítulos 11-15.

- **Script:** `09_mini_assistant_complet.py`
- **Documentación:** [INDEX_SCRIPT_09.md](INDEX_SCRIPT_09.md)
- **Inicio Rápido:** [QUICKSTART_SCRIPT_09.md](QUICKSTART_SCRIPT_09.md)
- **Arquitectura:** [SCRIPT_09_MAPPING.md](SCRIPT_09_MAPPING.md)

---

## 📖 Documentación Completa

- **[Recorrido Pedagógico](PEDAGOGICAL_JOURNEY.md)**: Correspondencia capítulo por capítulo libro ↔ scripts
- **[Agentes ReAct](REACT_AGENT_INTEGRATION.md)**: Patrón ReAct e integración
- **[LlamaIndex RAG](LLAMAINDEX_GUIDE.md)**: Framework RAG avanzado

---

## 📝 Notas

- **No se requiere GPU**: todos los scripts funcionan en CPU (más lento)
- **Código educativo**: prioriza la claridad sobre la optimización
- **Compatible con Python 3.9+**

---

**¡Feliz aprendizaje! 🚀**
