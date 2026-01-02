# 🔗 Mapeamento Código ↔ Conceito: Script 09

🌍 [English](../en/SCRIPT_09_MAPPING.md) | 📖 [Français](../fr/SCRIPT_09_MAPPING.md) | 🇪🇸 [Español](../es/SCRIPT_09_MAPPING.md) | 🇧🇷 **Português** | 🇸🇦 [العربية](../ar/SCRIPT_09_MAPPING.md)

> **Entenda qual código implementa qual conceito**  
> Guia de aprendizado linha por linha

---

## 📍 Navegação Rápida

- **📖 Ver: [Jornada Pedagógica](PEDAGOGICAL_JOURNEY.md)** - Teoria
- **🏗️ Ver: [Arquitetura](INDEX_SCRIPT_09.md)** - Estrutura
- **⚡ Ver: [Início Rápido](QUICKSTART_SCRIPT_09.md)** - Execute
- **🌍 Outros idiomas: [English](../en/SCRIPT_09_MAPPING.md) | [Français](../fr/SCRIPT_09_MAPPING.md) | [Español](../es/SCRIPT_09_MAPPING.md)**

---

## 🎯 Seção 1: Imports e Setup

### Conceito: Preparação do Ambiente

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re
```

**O que ensina:**
- `numpy`: Computação numérica (embeddings, softmax)
- `cosine_similarity`: Calcular similaridade entre documentos
- `defaultdict`: Estrutura de dados para base de conhecimento
- `re`: Processamento de texto

---

## 🎯 Seção 2: Base de Conhecimento

### Conceito: Armazenamento de Dados

```python
KNOWLEDGE_BASE = {
    'doc_1': "Um LLM é um modelo de linguagem grande...",
    'doc_2': "Transformers usam mecanismos de atenção...",
    'doc_3': "RAG combina recuperação com geração...",
    # ... mais documentos
}
```

**O que ensina:**
- Como armazenar conhecimento de domínio
- Estrutura simples de dicionário
- Escalável para milhares de documentos

---

## 🎯 Seção 3: Embeddings

### Conceito: Texto → Representação Vetorial

```python
def create_embedding(text: str, dim: int = 128) -> np.ndarray:
    """Converte texto para vetor usando hash determinístico"""
    hash_val = hash(text)
    np.random.seed(abs(hash_val) % 2**32)
    return np.random.randn(dim)
```

**O que ensina:**
- **Produção real:** Usar SentenceTransformer
  ```python
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer('all-MiniLM-L6-v2')
  embedding = model.encode(text)
  ```
- **Nesta demo:** Abordagem simplificada baseada em hash para velocidade
- **Conceito chave:** Texto → vetor de tamanho fixo (128 dimensões)
- **Propriedade:** Texto similar → vetores similares

**Analogia do mundo real:**
```
Imagine: Cada documento é um ponto em espaço de 128 dimensões
Pontos próximos = significado similar
```

---

## 🎯 Seção 4: Recuperação (RAG Parte 1)

### Conceito: Encontrar Documentos Relevantes

```python
def retrieve_documents(query: str, k: int = 3) -> list:
    """Passo 1: Embed da consulta
       Passo 2: Comparar com todos os documentos
       Passo 3: Retornar top-k mais similares
    """
    query_embedding = create_embedding(query)
    
    # Criar matriz de todos os embeddings de documentos
    doc_embeddings = np.array([
        create_embedding(doc) 
        for doc in KNOWLEDGE_BASE.values()
    ])
    
    # Calcular similaridade cosseno
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1), 
        doc_embeddings
    )[0]
    
    # Obter top-k
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

**O que ensina:**
- **Embedding:** Converter texto para vetor
- **Similaridade:** Similaridade cosseno = quão alinhados estão dois vetores?
  ```
  cosine_similarity = (A · B) / (||A|| * ||B||)
  Faixa: -1 (oposto) a 1 (idêntico)
  ```
- **Seleção:** Retornar top-k (mais similares) documentos
- **Complexidade:** O(n*d) onde n=docs, d=dimensões

**Analogia do mundo real:**
```
Como um bibliotecário:
1. Lê sua pergunta
2. Compara mentalmente com todos os livros
3. Traz os 3 livros mais relevantes
```

---

## 🎯 Seção 5: Raciocínio (Chain-of-Thought)

### Conceito: Resolução Estruturada de Problemas

```python
def reasoning_phase(question: str, contexts: list) -> str:
    """Pensa passo a passo com contexto recuperado"""
    
    reasoning = f"""
    Passo 1: Analisar a Pergunta
    O usuário pergunta sobre: {question}
    
    Passo 2: Conceitos Chave
    Extrair conceitos principais da pergunta
    
    Passo 3: Recuperar Contexto Relevante
    Dos documentos recuperados:
    """
    
    for i, ctx in enumerate(contexts, 1):
        reasoning += f"\n- De {ctx['doc']}: {ctx['content'][:100]}..."
    
    reasoning += f"""
    
    Passo 4: Sintetizar uma Resposta
    Combinando o conhecimento:
    - Ponto 1: [do contexto 1]
    - Ponto 2: [do contexto 2]
    - Ponto 3: [do contexto 3]
    
    Conclusão: Baseado no acima, podemos concluir...
    """
    
    return reasoning
```

**O que ensina:**
- **Chain-of-Thought:** Dividir problema em passos
- **Integração de Contexto:** Usar documentos recuperados
- **Reprodutibilidade:** Cada passo é visível
- **Transparência:** Fácil de debugar o raciocínio

**Analogia do mundo real:**
```
Como mostrar seu trabalho em matemática:
Não apenas "resposta: 42"
Mas "Passo 1: ... Passo 2: ... Passo 3: ... Resposta: 42"
```

---

## 🎯 Seção 6: Geração com Temperatura

### Conceito: Softmax e Amostragem com Temperatura

```python
def generate_with_temperature(
    prompt: str, 
    temperature: float = 1.0
) -> str:
    """
    Simula geração de tokens com controle de temperatura
    
    Temperatura:
    - 0.1: Muito focado (determinístico)
    - 1.0: Balanceado (softmax normal)
    - 2.0: Muito criativo (diverso)
    """
    
    # Simular logits (pontuações não normalizadas)
    prompt_hash = hash(prompt)
    np.random.seed(abs(prompt_hash) % 2**32)
    logits = np.random.randn(100) * 2
    
    # Aplicar escala de temperatura
    scaled_logits = logits / temperature
    
    # Softmax para obter probabilidades
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    probabilities = exp_logits / np.sum(exp_logits)
    
    # Amostrar token
    selected_idx = np.random.choice(100, p=probabilities)
    
    # Gerar texto
    vocab = ["um", "LLM", "é", "um", "modelo", "que", 
             "gera", "texto", "usando", "redes", "neurais"]
    response = " ".join([vocab[i % len(vocab)] for i in range(selected_idx % 20)])
    
    return response
```

**O que ensina:**

**Fórmula Softmax:**
```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
Resultado: distribuição de probabilidade (soma = 1)
```

**Efeito da Temperatura:**
```
T = 0.1  →  [0.01, 0.98, 0.01]  ← Agudo (determinístico)
T = 1.0  →  [0.15, 0.70, 0.15]  ← Balanceado
T = 2.0  →  [0.30, 0.40, 0.30]  ← Plano (diverso)
```

**Insight chave:**
- T baixa: O modelo repete o token mais provável (chato)
- T alta: O modelo explora alternativas (criativo)

---

## 🎯 Seção 7: Loop do Agente (ReAct)

### Conceito: Tomada de Decisão Autônoma

```python
def agent_loop(
    initial_query: str, 
    max_turns: int = 3
) -> dict:
    """
    Padrão ReAct:
    PENSAR → AGIR → OBSERVAR → (repetir)
    """
    
    context = initial_query
    trace = []
    turn = 0
    
    while turn < max_turns:
        turn += 1
        
        # PENSAR: Analisar estado atual
        thought = f"Turno {turn}: Analisando '{context[:50]}...'"
        trace.append(f"PENSAR: {thought}")
        
        # Decidir: Continuar ou Parar?
        should_continue = turn < max_turns and len(context) < 500
        
        if not should_continue:
            trace.append("PARAR: Informação suficiente coletada")
            break
        
        # AGIR: Recuperar documentos
        documents = retrieve_documents(context, k=2)
        trace.append(f"AGIR: Recuperados {len(documents)} documentos")
        
        # OBSERVAR: Processar resultados
        context += f" [Recuperado: {documents[0]['doc']}]"
        trace.append(f"OBSERVAR: Adicionado contexto de {documents[0]['doc']}")
    
    return {
        'answer': context,
        'turns': turn,
        'trace': trace
    }
```

**O que ensina:**

**Loop ReAct:**
```
┌─────────────────────────────────┐
│ PENSAR (analisar estado)        │
└────────────────┬────────────────┘
                 ↓
┌─────────────────────────────────┐
│ AGIR (tomar ação/recuperar)     │
└────────────────┬────────────────┘
                 ↓
┌─────────────────────────────────┐
│ OBSERVAR (processar resultados) │
└────────────────┬────────────────┘
                 ↓
        Repetir ou Parar?
```

**Propriedades chave:**
- Autônomo: Toma decisões independentemente
- Observável: Cada passo é rastreado
- Iterativo: Melhora a cada turno
- Parável: Sabe quando parar

---

## 🎯 Seção 8: Métricas de Avaliação

### Conceito: Avaliação de Qualidade

```python
def evaluate_response(response: str, context: str) -> dict:
    """Calcula múltiplas métricas de qualidade"""
    
    # Métrica 1: Ratio de Comprimento
    length_ratio = min(len(response), 500) / 500
    
    # Métrica 2: BLEU-like (sobreposição de vocabulário)
    response_words = set(response.lower().split())
    context_words = set(context.lower().split())
    overlap = len(response_words & context_words)
    vocabulary_overlap = overlap / max(len(response_words), 1)
    
    # Métrica 3: Similaridade de Embeddings
    response_emb = create_embedding(response)
    context_emb = create_embedding(context)
    similarity = cosine_similarity(
        response_emb.reshape(1, -1),
        context_emb.reshape(1, -1)
    )[0][0]
    
    # Métrica 4: Coerência (diversidade de tokens)
    tokens = response.lower().split()
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    coherence = 0.5 + 0.5 * (1 - unique_ratio)  # Balanceado
    
    # Métrica 5: Qualidade Geral
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

**O que ensina:**

**Tipos de Métricas:**

1. **Ratio de Comprimento**: 0-1
   - Garante que a resposta não seja muito curta/longa
   
2. **BLEU Score**: 0-1
   - Quantas palavras se sobrepõem com o contexto?
   
3. **Similaridade de Embeddings**: -1 a 1
   - Resposta e contexto são semanticamente similares?
   
4. **Coerência**: 0-1
   - A resposta evita repetição?
   
5. **Qualidade Geral**: 0-100
   - Combinação ponderada das anteriores

**Por que múltiplas métricas?**
```
Uma única métrica = imagem incompleta
Exemplo: Uma resposta curta e genérica pode pontuar alto em 
         vocabulary_overlap mas baixo em length_ratio
```

---

## 🎓 Lista de Verificação de Aprendizado

Depois de ler isso, você deve entender:

- [ ] Como texto se torna vetores (embeddings)
- [ ] Como a similaridade é calculada (similaridade cosseno)
- [ ] Como documentos são recuperados (busca k-NN)
- [ ] Como o raciocínio é estruturado (Chain-of-Thought)
- [ ] Como a temperatura afeta a aleatoriedade (escala softmax)
- [ ] Como agentes tomam decisões (loop ReAct)
- [ ] Como a qualidade é medida (múltiplas métricas)
- [ ] Como componentes se integram (pipeline)

---

## 🔬 Ideias de Experimentação

Tente modificar:

```python
# 1. Mudar dimensão de embedding
EMBEDDING_DIM = 256  # Mais dimensões = mais preciso

# 2. Mudar temperatura
temperature = 0.1    # Mais focado
temperature = 2.0    # Mais criativo

# 3. Mudar k_documents
k = 5                # Mais contexto = mais lento mas mais rico

# 4. Adicionar mais documentos
KNOWLEDGE_BASE['doc_4'] = "Seu novo documento..."

# 5. Mudar pesos de avaliação
quality_score = (
    length_ratio * 0.1 +
    vocabulary_overlap * 0.5 +  # Mais ênfase aqui
    similarity * 0.2 +
    coherence * 0.2
) * 100
```

---

## 📚 Leituras Adicionais

- **Capítulo 11:** Temperatura e Geração
- **Capítulo 12:** Raciocínio Chain-of-Thought
- **Capítulo 13:** Arquitetura RAG
- **Capítulo 14:** Padrões de Agentes (ReAct)
- **Capítulo 15:** Avaliação

---

**Agora você entende o código! 🎓**
