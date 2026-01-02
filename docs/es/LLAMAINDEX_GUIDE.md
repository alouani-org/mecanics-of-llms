# 📚 Guía LlamaIndex para Principiantes

🌍 [English](../en/LLAMAINDEX_GUIDE.md) | 📖 [Français](../fr/LLAMAINDEX_GUIDE.md) | 🇪🇸 **Español** | 🇧🇷 [Português](../pt/LLAMAINDEX_GUIDE.md) | 🇸🇦 [العربية](../ar/LLAMAINDEX_GUIDE.md)

> **Construyendo sistemas RAG con LlamaIndex**  
> Guía Paso a Paso

---

## 📍 Navegación Rápida

- **📖 Ver: [Recorrido Pedagógico](PEDAGOGICAL_JOURNEY.md)** - Dónde encaja esto
- **⚡ Ver: [Inicio Rápido Script 09](QUICKSTART_SCRIPT_09.md)** - Usar RAG
- **🗺️ Ver: [Mapa Código ↔ Conceptos](SCRIPT_09_MAPPING.md)** - Mapeo detallado
- **🌍 Otros idiomas: [English](../en/LLAMAINDEX_GUIDE.md) | [Français](../fr/LLAMAINDEX_GUIDE.md) | [Português](../pt/LLAMAINDEX_GUIDE.md)**

---

## 🎯 ¿Qué es LlamaIndex?

**LlamaIndex** es un framework que facilita:

1. **Cargar** tus propios datos (PDF, texto, páginas web)
2. **Indexar** esos datos para búsqueda rápida
3. **Consultar** usando lenguaje natural
4. **Sintetizar** respuestas con LLMs

### Analogía

```
LlamaIndex = Tu Bibliotecario IA

1. Tú le das libros (tus documentos)
2. Él los organiza (crea índice)
3. Tú haces preguntas ("¿Dónde habla de X?")
4. Él encuentra y resume la respuesta
```

---

## 🏗️ Arquitectura LlamaIndex

```
┌─────────────────────────────────────────────┐
│              TUS DOCUMENTOS                  │
│  (PDFs, TXTs, Páginas Web, Bases de Datos)  │
└─────────────┬───────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────┐
│            CARGADORES (Loaders)              │
│  SimpleDirectoryReader, PDFReader, etc.     │
└─────────────┬───────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────┐
│              NODOS (Nodes)                   │
│  Fragmentos de texto con metadatos          │
└─────────────┬───────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────┐
│              ÍNDICE (Index)                  │
│  VectorStoreIndex, TreeIndex, etc.          │
└─────────────┬───────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────┐
│         MOTOR DE CONSULTA (Query Engine)    │
│  Recupera nodos relevantes + Genera resp.   │
└─────────────────────────────────────────────┘
```

---

## 📝 Conceptos Clave

### 1. **Documento**

Un documento es tu dato fuente:

```python
from llama_index.core import Document

# Crear documento desde texto
doc = Document(text="El cielo es azul...")

# Crear documento con metadatos
doc = Document(
    text="El cielo es azul...",
    metadata={
        "source": "mi_archivo.txt",
        "author": "Juan Pérez",
        "date": "2024-01-15"
    }
)
```

### 2. **Nodo**

Un nodo es un fragmento de documento:

```python
# Un documento grande se divide en nodos
Documento: "El cielo es azul. El océano es profundo. Las estrellas brillan."

# Se convierte en nodos:
Nodo 1: "El cielo es azul."
Nodo 2: "El océano es profundo."
Nodo 3: "Las estrellas brillan."
```

### 3. **Índice**

Un índice organiza nodos para búsqueda rápida:

```python
from llama_index.core import VectorStoreIndex

# Crear índice desde documentos
index = VectorStoreIndex.from_documents(documents)

# El índice contiene embeddings para cada nodo
# Esto permite búsqueda semántica rápida
```

### 4. **Motor de Consulta**

El motor de consulta responde preguntas:

```python
# Crear motor de consulta desde índice
query_engine = index.as_query_engine()

# Hacer pregunta
response = query_engine.query("¿De qué color es el cielo?")
print(response)  # "El cielo es azul"
```

---

## 🚀 Script 09: RAG con LlamaIndex

### Paso 1: Configurar Ambiente

```python
# Importar dependencias
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Configurar LLM y Embeddings
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)
Settings.embed_model = OpenAIEmbedding()
```

### Paso 2: Cargar Documentos

```python
# Cargar desde directorio
reader = SimpleDirectoryReader("./data")
documents = reader.load_data()

print(f"Cargados {len(documents)} documentos")

# Ver contenido de un documento
print(documents[0].text[:200])  # Primeros 200 caracteres
```

### Paso 3: Crear Índice

```python
# Crear índice vectorial
index = VectorStoreIndex.from_documents(documents)

# El índice:
# 1. Divide documentos en nodos (fragmentos)
# 2. Genera embeddings para cada nodo
# 3. Almacena en vector store
```

### Paso 4: Consultar

```python
# Crear motor de consulta
query_engine = index.as_query_engine(
    similarity_top_k=3,  # Recuperar top 3 nodos relevantes
)

# Hacer pregunta
response = query_engine.query(
    "¿Cuáles son las principales características de los LLMs?"
)

print(response.response)

# Ver fuentes utilizadas
for node in response.source_nodes:
    print(f"Fuente: {node.node.metadata.get('source', 'desconocida')}")
    print(f"Score: {node.score:.3f}")
```

---

## 🔧 Configuración Avanzada

### Personalizar Chunking (División)

```python
from llama_index.core.node_parser import SentenceSplitter

# Configurar cómo dividir documentos
node_parser = SentenceSplitter(
    chunk_size=1024,      # Tokens máximos por chunk
    chunk_overlap=200     # Superposición entre chunks
)

# Crear índice con parser personalizado
index = VectorStoreIndex.from_documents(
    documents,
    node_parser=node_parser
)
```

### Personalizar Prompt

```python
from llama_index.core import PromptTemplate

# Crear prompt personalizado
template = """
Contexto: {context_str}

Basándote en el contexto anterior, responde la siguiente pregunta.
Si no encuentras la información en el contexto, di "No tengo información suficiente".

Pregunta: {query_str}

Respuesta:
"""

qa_prompt = PromptTemplate(template)

# Usar en motor de consulta
query_engine = index.as_query_engine(
    text_qa_template=qa_prompt
)
```

### Persistir Índice

```python
# Guardar índice en disco
index.storage_context.persist(persist_dir="./storage")

# Cargar índice guardado
from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

# ¡Ahora no necesitas re-procesar documentos!
```

---

## 📊 Tipos de Índice

### 1. VectorStoreIndex (Más Común)

```python
# Usa embeddings para búsqueda semántica
index = VectorStoreIndex.from_documents(documents)

# Mejor para: Preguntas sobre contenido específico
# "¿Qué dice el documento sobre X?"
```

### 2. SummaryIndex

```python
from llama_index.core import SummaryIndex

# Almacena resúmenes de documentos
index = SummaryIndex.from_documents(documents)

# Mejor para: Preguntas que requieren visión general
# "Resume todo el documento"
```

### 3. TreeIndex

```python
from llama_index.core import TreeIndex

# Organiza en estructura de árbol
index = TreeIndex.from_documents(documents)

# Mejor para: Documentos jerárquicos
# Libros con capítulos, manuales con secciones
```

---

## ⚙️ Parámetros Importantes

### similarity_top_k

```python
# Cuántos nodos recuperar
query_engine = index.as_query_engine(similarity_top_k=5)

# k pequeño (1-3): Respuestas más enfocadas
# k grande (5-10): Más contexto, pero puede incluir ruido
```

### response_mode

```python
# Cómo sintetizar respuesta
query_engine = index.as_query_engine(
    response_mode="compact"  # Opciones: refine, compact, tree_summarize
)

# "compact": Une todo contexto, genera una respuesta
# "refine": Refina respuesta iterativamente con cada nodo
# "tree_summarize": Resume en estructura de árbol
```

### streaming

```python
# Habilitar streaming para respuestas largas
query_engine = index.as_query_engine(streaming=True)

response = query_engine.query("Explica en detalle...")

# Imprimir token por token
for text in response.response_gen:
    print(text, end="", flush=True)
```

---

## 🔍 Depuración

### Ver Qué Se Recupera

```python
# Obtener nodos sin generar respuesta
retriever = index.as_retriever(similarity_top_k=3)
nodes = retriever.retrieve("¿Qué es un LLM?")

for node in nodes:
    print(f"Score: {node.score:.3f}")
    print(f"Texto: {node.node.text[:200]}...")
    print(f"Metadatos: {node.node.metadata}")
    print("---")
```

### Logging Detallado

```python
import logging
import sys

# Habilitar logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Ahora verás todos los pasos internos
```

---

## 🎯 Mejores Prácticas

### 1. **Preparar Datos**

```python
# ✅ Bueno: Datos limpios y estructurados
documents = [
    Document(text="Capítulo 1: Introducción...", metadata={"chapter": 1}),
    Document(text="Capítulo 2: Conceptos...", metadata={"chapter": 2}),
]

# ❌ Malo: Datos sucios con mucho ruido
documents = [
    Document(text="asdfasdf Capítulo 1 ||||| Introducción.....")
]
```

### 2. **Ajustar Chunk Size**

```python
# Documentos técnicos: chunks más pequeños
node_parser = SentenceSplitter(chunk_size=512)

# Documentos narrativos: chunks más grandes
node_parser = SentenceSplitter(chunk_size=2048)
```

### 3. **Usar Metadatos**

```python
# Los metadatos ayudan a filtrar y contextualizar
doc = Document(
    text="Contenido del informe financiero Q3 2024...",
    metadata={
        "type": "financial_report",
        "quarter": "Q3",
        "year": 2024,
        "department": "finanzas"
    }
)
```

### 4. **Persistir Siempre**

```python
# No re-procesar documentos cada vez
index.storage_context.persist(persist_dir="./storage")

# Verificar si existe índice guardado
import os
if os.path.exists("./storage"):
    index = load_index_from_storage(...)
else:
    index = VectorStoreIndex.from_documents(...)
```

---

## 🐛 Errores Comunes

### Error: "Rate limit exceeded"

```python
# Problema: Demasiadas llamadas API

# Solución 1: Reducir concurrencia
Settings.num_workers = 1

# Solución 2: Añadir delays
import time
time.sleep(1)  # Entre operaciones
```

### Error: "Context length exceeded"

```python
# Problema: Documento muy grande

# Solución: Reducir chunk_size
node_parser = SentenceSplitter(chunk_size=256)
```

### Error: "Empty response"

```python
# Problema: No se encontró información relevante

# Solución 1: Aumentar similarity_top_k
query_engine = index.as_query_engine(similarity_top_k=10)

# Solución 2: Verificar que los documentos contienen la información
```

---

## 📚 Script 09: Integración Completa

Script 09 combina todo lo aprendido:

```python
# 1. Carga documentos
documents = SimpleDirectoryReader("./data").load_data()

# 2. Crea índice con configuración óptima
index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True
)

# 3. Motor de consulta configurado
query_engine = index.as_query_engine(
    similarity_top_k=3,
    response_mode="compact"
)

# 4. Ciclo interactivo
while True:
    question = input("Pregunta: ")
    if question.lower() == "salir":
        break
    
    response = query_engine.query(question)
    print(f"\nRespuesta: {response}")
    print(f"Fuentes: {len(response.source_nodes)}")
```

---

## 🎯 Resumen

| Concepto | LlamaIndex | Función |
|----------|------------|---------|
| **Document** | `Document` | Tu dato fuente |
| **Node** | Fragmento | Pieza de documento |
| **Index** | `VectorStoreIndex` | Organiza nodos |
| **Query Engine** | `as_query_engine()` | Responde preguntas |
| **Retriever** | `as_retriever()` | Busca nodos relevantes |

---

## 🚀 Próximos Pasos

1. ✅ Ejecuta Script 09 con tus propios documentos
2. ✅ Experimenta con diferentes `chunk_size`
3. ✅ Prueba diferentes `response_mode`
4. ✅ Añade metadatos a tus documentos
5. ✅ Persiste tu índice

---

**¿Listo para construir tu propio sistema RAG? 🚀**

¡Prueba Script 09 ahora!
