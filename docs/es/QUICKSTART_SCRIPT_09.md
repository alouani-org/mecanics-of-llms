# ⚡ Inicio Rápido en 5 Minutos: Script 09

🌍 [English](../en/QUICKSTART_SCRIPT_09.md) | 📖 [Français](../fr/QUICKSTART_SCRIPT_09.md) | 🇪🇸 **Español** | 🇧🇷 [Português](../pt/QUICKSTART_SCRIPT_09.md) | 🇸🇦 [العربية](../ar/QUICKSTART_SCRIPT_09.md)

> **Ejecuta en 5 minutos**  
> Sin teoría. Solo código.

---

## 📍 Navegación Rápida

- **📖 Ver: [Recorrido Pedagógico](PEDAGOGICAL_JOURNEY.md)** - Aprende los conceptos
- **🏗️ Ver: [Arquitectura](INDEX_SCRIPT_09.md)** - Cómo está construido
- **🌍 Otros idiomas: [English](../en/QUICKSTART_SCRIPT_09.md) | [Français](../fr/QUICKSTART_SCRIPT_09.md) | [Português](../pt/QUICKSTART_SCRIPT_09.md)**

---

## Paso 1️⃣: Requisitos (30 segundos)

```bash
# ¿Ya instalado? ¡Estás listo!
# Solo necesitas esto (probablemente ya en tu sistema):
- Python 3.9+
- numpy
- scikit-learn (para similitud coseno)
```

**Verificar si está instalado:**
```bash
python --version
python -c "import numpy; print('numpy OK')"
python -c "from sklearn.metrics.pairwise import cosine_similarity; print('sklearn OK')"
```

---

## Paso 2️⃣: Navega y Ejecuta (1 minuto)

```bash
# Ve al directorio de scripts
cd c:\dev\IA-Eductation\examples

# Ejecuta el script
python 09_mini_assistant_complet.py
```

---

## Paso 3️⃣: Pruébalo (3 minutos)

Verás un menú:

```
========================================
   Mini LLM Assistant - Demo Completa
========================================

Elige una opción:
1. Hacer una pregunta
2. Probar con ejemplos
3. Ver métricas de evaluación
4. Entender arquitectura
5. Salir

Ingresa tu elección (1-5): 
```

**Prueba esto:**

```
Ingresa tu elección: 2

=== Ejecutando Ejemplos ===

Ejemplo 1: "¿Qué es un LLM?"
Pregunta: ¿Qué es un LLM?

📥 FASE DE RECUPERACIÓN
Encontrados 3 documentos:
- doc_1 (similitud: 0.85)
- doc_3 (similitud: 0.78)
- doc_2 (similitud: 0.72)

💭 FASE DE RAZONAMIENTO
[Muestra pensamiento paso a paso]

🤖 FASE DE GENERACIÓN
Respuesta: "Un LLM es un modelo de lenguaje grande..."

🎯 EVALUACIÓN
Puntuación de Calidad: 82/100
- BLEU score: 0.78
- Similitud de embeddings: 0.84
- Coherencia: 0.79

...más ejemplos...
```

---

## 💡 ¿Qué Acaba de Pasar?

Tu script:

1. **📥 Recuperó** documentos de la base de conocimientos
2. **💭 Razonó** paso a paso sobre el problema
3. **🤖 Generó** una respuesta usando muestreo con temperatura
4. **🎯 Evaluó** la calidad usando 5 métricas

Todo en `09_mini_assistant_complet.py` ✅

---

## 🎮 Modo Interactivo

Elige opción 1 para hacer tus propias preguntas:

```
Ingresa tu elección: 1

Haz tu pregunta: ¿Qué son los transformers?
Temperatura (0.1=enfocado, 1.0=balanceado, 2.0=creativo) [default 1.0]: 1.0

📥 RECUPERACIÓN: Documentos relevantes encontrados
💭 RAZONAMIENTO: Pensando paso a paso...
🤖 GENERACIÓN: Creando respuesta...
🎯 EVALUACIÓN: Evaluando calidad...

Respuesta: [Tu respuesta aquí]
Puntuación de Calidad: 78/100
```

---

## 🔧 Personalización (Avanzado)

¿Quieres cambiar el comportamiento? Edita en el script:

```python
# Cambia estas constantes al inicio del archivo:

TEMPERATURE = 1.0        # 0.1 (enfocado) a 2.0 (creativo)
K_DOCUMENTS = 3          # Cuántos documentos recuperar
MAX_TURNS = 3            # Iteraciones del agente
EMBEDDING_DIM = 128      # Dimensión de embedding
```

Luego ejecuta de nuevo.

---

## 🏆 Lo Que Estás Aprendiendo

Al ejecutar este script, estás practicando:

✅ **RAG** - Recuperar documentos relevantes  
✅ **Muestreo con Temperatura** - Controlar aleatoriedad  
✅ **Chain-of-Thought** - Razonamiento paso a paso  
✅ **Agentes ReAct** - Bucles autónomos  
✅ **Evaluación** - Medir calidad  

Todo con código educativo que puedes leer y modificar.

---

## 🆘 Solución de Problemas

**"Module not found: numpy"**
```bash
pip install numpy scikit-learn
```

**"El script no ejecuta"**
```bash
# Verifica versión de Python
python --version

# Debe ser 3.9 o superior
```

**"Ejecución lenta"**
- ¡Normal! El código demo prioriza claridad sobre velocidad
- Sistemas reales usarían aceleración GPU

---

## 🚀 Siguientes Pasos

1. ✅ Has ejecutado el script
2. 📖 [Lee la arquitectura](INDEX_SCRIPT_09.md)
3. 🔗 [Mapea código a conceptos](SCRIPT_09_MAPPING.md)
4. 💻 Modifica y experimenta
5. 🌟 Integra en tu proyecto

---

## 📚 Más Recursos

- **¿Entender conceptos?** → [Recorrido Pedagógico](PEDAGOGICAL_JOURNEY.md)
- **¿Cómo está construido?** → [Arquitectura](INDEX_SCRIPT_09.md)
- **¿Qué código enseña qué?** → [Mapeo de Código](SCRIPT_09_MAPPING.md)
- **¿Agentes en detalle?** → [Guía ReAct](REACT_AGENT_INTEGRATION.md)
- **¿RAG en detalle?** → [Guía RAG](LLAMAINDEX_GUIDE.md)

---

**¡Felicitaciones! 🎉 Estás ejecutando un mini asistente LLM.**

Prueba experimentando con diferentes preguntas y valores de temperatura. ¡Observa cómo el sistema responde de manera diferente!

**¿Preguntas? Consulta [Recorrido Pedagógico](PEDAGOGICAL_JOURNEY.md) para explicaciones detalladas.**
