# 📚 Guia LlamaIndex para Iniciantes

🌍 [English](../en/LLAMAINDEX_GUIDE.md) | 📖 [Français](../fr/LLAMAINDEX_GUIDE.md) | 🇪🇸 [Español](../es/LLAMAINDEX_GUIDE.md) | 🇧🇷 **Português** | 🇸🇦 [العربية](../ar/LLAMAINDEX_GUIDE.md)

> **Construindo sistemas RAG com LlamaIndex**  
> Guia Passo a Passo

---

## 📍 Navegação Rápida

- **📖 Ver: [Jornada Pedagógica](PEDAGOGICAL_JOURNEY.md)** - Onde isso se encaixa
- **⚡ Ver: [Início Rápido Script 09](QUICKSTART_SCRIPT_09.md)** - Usar RAG
- **🗺️ Ver: [Mapa Código ↔ Conceitos](SCRIPT_09_MAPPING.md)** - Mapeamento detalhado
- **🌍 Outros idiomas: [English](../en/LLAMAINDEX_GUIDE.md) | [Français](../fr/LLAMAINDEX_GUIDE.md) | [Español](../es/LLAMAINDEX_GUIDE.md)**

---

## 🎯 O que é LlamaIndex?

**LlamaIndex** é um framework que facilita:

1. **Carregar** seus próprios dados (PDF, texto, páginas web)
2. **Indexar** esses dados para busca rápida
3. **Consultar** usando linguagem natural
4. **Sintetizar** respostas com LLMs

### Analogia

```
LlamaIndex = Seu Bibliotecário IA

1. Você dá livros para ele (seus documentos)
2. Ele organiza (cria índice)
3. Você faz perguntas ("Onde fala sobre X?")
4. Ele encontra e resume a resposta
```

---

## 🏗️ Arquitetura LlamaIndex

```
┌─────────────────────────────────────────────┐
│              SEUS DOCUMENTOS                 │
│  (PDFs, TXTs, Páginas Web, Bancos de Dados) │
└─────────────┬───────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────┐
│            CARREGADORES (Loaders)            │
│  SimpleDirectoryReader, PDFReader, etc.     │
└─────────────┬───────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────┐
│                NÓS (Nodes)                   │
│  Fragmentos de texto com metadados          │
└─────────────┬───────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────┐
│               ÍNDICE (Index)                 │
│  VectorStoreIndex, TreeIndex, etc.          │
└─────────────┬───────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────┐
│       MOTOR DE CONSULTA (Query Engine)      │
│  Recupera nós relevantes + Gera resposta    │
└─────────────────────────────────────────────┘
```

---

## 📝 Conceitos Chave

### 1. **Documento**

Um documento é seu dado fonte:

```python
from llama_index.core import Document

# Criar documento a partir de texto
doc = Document(text="O céu é azul...")

# Criar documento com metadados
doc = Document(
    text="O céu é azul...",
    metadata={
        "source": "meu_arquivo.txt",
        "author": "João Silva",
        "date": "2024-01-15"
    }
)
```

### 2. **Nó**

Um nó é um fragmento de documento:

```python
# Um documento grande é dividido em nós
Documento: "O céu é azul. O oceano é profundo. As estrelas brilham."

# Torna-se nós:
Nó 1: "O céu é azul."
Nó 2: "O oceano é profundo."
Nó 3: "As estrelas brilham."
```

### 3. **Índice**

Um índice organiza nós para busca rápida:

```python
from llama_index.core import VectorStoreIndex

# Criar índice a partir de documentos
index = VectorStoreIndex.from_documents(documents)

# O índice contém embeddings para cada nó
# Isso permite busca semântica rápida
```

### 4. **Motor de Consulta**

O motor de consulta responde perguntas:

```python
# Criar motor de consulta a partir do índice
query_engine = index.as_query_engine()

# Fazer pergunta
response = query_engine.query("Qual é a cor do céu?")
print(response)  # "O céu é azul"
```

---

## 🚀 Script 09: RAG com LlamaIndex

### Passo 1: Configurar Ambiente

```python
# Importar dependências
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Configurar LLM e Embeddings
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)
Settings.embed_model = OpenAIEmbedding()
```

### Passo 2: Carregar Documentos

```python
# Carregar de diretório
reader = SimpleDirectoryReader("./data")
documents = reader.load_data()

print(f"Carregados {len(documents)} documentos")

# Ver conteúdo de um documento
print(documents[0].text[:200])  # Primeiros 200 caracteres
```

### Passo 3: Criar Índice

```python
# Criar índice vetorial
index = VectorStoreIndex.from_documents(documents)

# O índice:
# 1. Divide documentos em nós (fragmentos)
# 2. Gera embeddings para cada nó
# 3. Armazena em vector store
```

### Passo 4: Consultar

```python
# Criar motor de consulta
query_engine = index.as_query_engine(
    similarity_top_k=3,  # Recuperar top 3 nós relevantes
)

# Fazer pergunta
response = query_engine.query(
    "Quais são as principais características dos LLMs?"
)

print(response.response)

# Ver fontes utilizadas
for node in response.source_nodes:
    print(f"Fonte: {node.node.metadata.get('source', 'desconhecida')}")
    print(f"Score: {node.score:.3f}")
```

---

## 🔧 Configuração Avançada

### Personalizar Chunking (Divisão)

```python
from llama_index.core.node_parser import SentenceSplitter

# Configurar como dividir documentos
node_parser = SentenceSplitter(
    chunk_size=1024,      # Tokens máximos por chunk
    chunk_overlap=200     # Sobreposição entre chunks
)

# Criar índice com parser personalizado
index = VectorStoreIndex.from_documents(
    documents,
    node_parser=node_parser
)
```

### Personalizar Prompt

```python
from llama_index.core import PromptTemplate

# Criar prompt personalizado
template = """
Contexto: {context_str}

Baseado no contexto acima, responda a seguinte pergunta.
Se não encontrar a informação no contexto, diga "Não tenho informação suficiente".

Pergunta: {query_str}

Resposta:
"""

qa_prompt = PromptTemplate(template)

# Usar no motor de consulta
query_engine = index.as_query_engine(
    text_qa_template=qa_prompt
)
```

### Persistir Índice

```python
# Salvar índice em disco
index.storage_context.persist(persist_dir="./storage")

# Carregar índice salvo
from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

# Agora não precisa re-processar documentos!
```

---

## 📊 Tipos de Índice

### 1. VectorStoreIndex (Mais Comum)

```python
# Usa embeddings para busca semântica
index = VectorStoreIndex.from_documents(documents)

# Melhor para: Perguntas sobre conteúdo específico
# "O que o documento diz sobre X?"
```

### 2. SummaryIndex

```python
from llama_index.core import SummaryIndex

# Armazena resumos de documentos
index = SummaryIndex.from_documents(documents)

# Melhor para: Perguntas que requerem visão geral
# "Resuma todo o documento"
```

### 3. TreeIndex

```python
from llama_index.core import TreeIndex

# Organiza em estrutura de árvore
index = TreeIndex.from_documents(documents)

# Melhor para: Documentos hierárquicos
# Livros com capítulos, manuais com seções
```

---

## ⚙️ Parâmetros Importantes

### similarity_top_k

```python
# Quantos nós recuperar
query_engine = index.as_query_engine(similarity_top_k=5)

# k pequeno (1-3): Respostas mais focadas
# k grande (5-10): Mais contexto, mas pode incluir ruído
```

### response_mode

```python
# Como sintetizar resposta
query_engine = index.as_query_engine(
    response_mode="compact"  # Opções: refine, compact, tree_summarize
)

# "compact": Une todo contexto, gera uma resposta
# "refine": Refina resposta iterativamente com cada nó
# "tree_summarize": Resume em estrutura de árvore
```

### streaming

```python
# Habilitar streaming para respostas longas
query_engine = index.as_query_engine(streaming=True)

response = query_engine.query("Explique em detalhes...")

# Imprimir token por token
for text in response.response_gen:
    print(text, end="", flush=True)
```

---

## 🔍 Depuração

### Ver O Que É Recuperado

```python
# Obter nós sem gerar resposta
retriever = index.as_retriever(similarity_top_k=3)
nodes = retriever.retrieve("O que é um LLM?")

for node in nodes:
    print(f"Score: {node.score:.3f}")
    print(f"Texto: {node.node.text[:200]}...")
    print(f"Metadados: {node.node.metadata}")
    print("---")
```

### Logging Detalhado

```python
import logging
import sys

# Habilitar logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Agora você verá todos os passos internos
```

---

## 🎯 Melhores Práticas

### 1. **Preparar Dados**

```python
# ✅ Bom: Dados limpos e estruturados
documents = [
    Document(text="Capítulo 1: Introdução...", metadata={"chapter": 1}),
    Document(text="Capítulo 2: Conceitos...", metadata={"chapter": 2}),
]

# ❌ Ruim: Dados sujos com muito ruído
documents = [
    Document(text="asdfasdf Capítulo 1 ||||| Introdução.....")
]
```

### 2. **Ajustar Chunk Size**

```python
# Documentos técnicos: chunks menores
node_parser = SentenceSplitter(chunk_size=512)

# Documentos narrativos: chunks maiores
node_parser = SentenceSplitter(chunk_size=2048)
```

### 3. **Usar Metadados**

```python
# Os metadados ajudam a filtrar e contextualizar
doc = Document(
    text="Conteúdo do relatório financeiro Q3 2024...",
    metadata={
        "type": "financial_report",
        "quarter": "Q3",
        "year": 2024,
        "department": "financeiro"
    }
)
```

### 4. **Persistir Sempre**

```python
# Não re-processar documentos toda vez
index.storage_context.persist(persist_dir="./storage")

# Verificar se existe índice salvo
import os
if os.path.exists("./storage"):
    index = load_index_from_storage(...)
else:
    index = VectorStoreIndex.from_documents(...)
```

---

## 🐛 Erros Comuns

### Erro: "Rate limit exceeded"

```python
# Problema: Muitas chamadas API

# Solução 1: Reduzir concorrência
Settings.num_workers = 1

# Solução 2: Adicionar delays
import time
time.sleep(1)  # Entre operações
```

### Erro: "Context length exceeded"

```python
# Problema: Documento muito grande

# Solução: Reduzir chunk_size
node_parser = SentenceSplitter(chunk_size=256)
```

### Erro: "Empty response"

```python
# Problema: Não encontrou informação relevante

# Solução 1: Aumentar similarity_top_k
query_engine = index.as_query_engine(similarity_top_k=10)

# Solução 2: Verificar que os documentos contêm a informação
```

---

## 📚 Script 09: Integração Completa

Script 09 combina tudo que aprendemos:

```python
# 1. Carrega documentos
documents = SimpleDirectoryReader("./data").load_data()

# 2. Cria índice com configuração ótima
index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True
)

# 3. Motor de consulta configurado
query_engine = index.as_query_engine(
    similarity_top_k=3,
    response_mode="compact"
)

# 4. Ciclo interativo
while True:
    question = input("Pergunta: ")
    if question.lower() == "sair":
        break
    
    response = query_engine.query(question)
    print(f"\nResposta: {response}")
    print(f"Fontes: {len(response.source_nodes)}")
```

---

## 🎯 Resumo

| Conceito | LlamaIndex | Função |
|----------|------------|--------|
| **Document** | `Document` | Seu dado fonte |
| **Node** | Fragmento | Pedaço de documento |
| **Index** | `VectorStoreIndex` | Organiza nós |
| **Query Engine** | `as_query_engine()` | Responde perguntas |
| **Retriever** | `as_retriever()` | Busca nós relevantes |

---

## 🚀 Próximos Passos

1. ✅ Execute Script 09 com seus próprios documentos
2. ✅ Experimente com diferentes `chunk_size`
3. ✅ Teste diferentes `response_mode`
4. ✅ Adicione metadados aos seus documentos
5. ✅ Persista seu índice

---

**Pronto para construir seu próprio sistema RAG? 🚀**

Experimente o Script 09 agora!
