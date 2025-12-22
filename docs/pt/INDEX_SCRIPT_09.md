# 🏗️ Arquitetura: O Mini Assistente Completo (Script 09)

> **Decomposição completa** do projeto integrador  
> Entendendo a estrutura técnica: camadas, componentes, fluxo

---

## 📍 Navegação Rápida

- **📖 Ver: [Jornada Pedagógica](PEDAGOGICAL_JOURNEY.md)** - Como se conecta com os capítulos
- **⚡ Ver: [Início Rápido](QUICKSTART_SCRIPT_09.md)** - Execute em 5 minutos
- **🔗 Ver: [Mapeamento Código ↔ Conceitos](SCRIPT_09_MAPPING.md)** - Qual código ensina o quê
- **🌍 Outros idiomas: [English](../en/INDEX_SCRIPT_09.md) | [Français](../fr/INDEX_SCRIPT_09.md) | [Español](../es/INDEX_SCRIPT_09.md)**

---

## 🎯 O Que Há Dentro?

O Script 09 demonstra TODOS os conceitos dos capítulos 11-15:

| Capítulo | Conceito | Componente no Script 09 |
|----------|----------|------------------------|
| 11 | Geração + Temperatura | `generate_with_temperature()` |
| 12 | Chain-of-Thought | `reasoning_phase()` |
| 13 | RAG + Recuperação | `retrieve_documents()` |
| 14 | Agentes ReAct | `agent_loop()` |
| 15 | Avaliação | `evaluate_response()` |

---

## 🏗️ Arquitetura Técnica

### Camada 1: Camada de Dados
```
Base de Conhecimento (em memória)
    ↓
Fragmentação de Documentos
    ↓
Embeddings Vetoriais (numpy)
```

**Responsabilidade:** Armazenar e indexar conhecimento
**Localização do código:** `load_knowledge_base()`, `embed_documents()`

---

### Camada 2: Camada de Recuperação (RAG)
```
Consulta do Usuário
    ↓
Embed da Consulta
    ↓
Busca por Similaridade (cosseno)
    ↓
Contextos Recuperados
```

**Responsabilidade:** Encontrar documentos relevantes
**Localização do código:** `retrieve_documents()`

**Função Chave:**
```python
def retrieve_documents(query: str, k: int = 3) -> list:
    # 1. Embed da consulta
    # 2. Calcular similaridade com todos os documentos
    # 3. Retornar top-k mais relevantes
```

---

### Camada 3: Camada de Raciocínio (Chain-of-Thought)
```
Pergunta
    ↓
Passo 1: Analisar problema
Passo 2: Recuperar contexto
Passo 3: Pensar passo a passo
    ↓
Trace de Raciocínio
```

**Responsabilidade:** Estruturar o pensamento
**Localização do código:** `reasoning_phase()`

---

### Camada 4: Camada de Geração (similar a LLM)
```
Trace de Raciocínio + Contexto
    ↓
Seleção de Token (softmax)
    ↓
Amostragem com Temperatura
    ↓
Geração de Resposta
```

**Responsabilidade:** Criar texto
**Localização do código:** `generate_with_temperature()`

---

### Camada 5: Camada de Agente (ReAct)
```
Decisão do Agente (Pensar)
    ↓
Seleção de Ferramenta (Agir)
    ↓
Observar Resultado
    ↓
Loop até terminar
```

**Responsabilidade:** Execução autônoma
**Localização do código:** `agent_loop()`

---

### Camada 6: Camada de Avaliação
```
Resposta Gerada
    ↓
Múltiplas Métricas (BLEU, Similaridade de Embeddings, Coerência)
    ↓
Pontuação (0-100)
```

**Responsabilidade:** Avaliação de qualidade
**Localização do código:** `evaluate_response()`

---

## 🔄 Fluxo de Execução Completo

```
Entrada do Usuário
    ↓
embed_documents() → Vetores de documentos (128-dim)
    ↓
retrieve_documents() → Top-k documentos similares
    ↓
reasoning_phase() → Pensamento estruturado
    ↓
generate_with_temperature() → Geração de texto
    ↓
agent_loop() → Iteração autônoma
    ↓
evaluate_response() → Métricas de qualidade
    ↓
Saída para o Usuário
```

**Passo a passo:**

1. **Processamento de Entrada**
   - Parsear consulta do usuário
   - Preparar para recuperação

2. **Recuperação (RAG)**
   - Encontrar contexto relevante da base de conhecimento
   - Retornar top-3 documentos

3. **Raciocínio**
   - Criar cadeia de pensamento
   - Analisar problema passo a passo
   - Incluir contexto recuperado

4. **Geração**
   - Selecionar tokens usando softmax
   - Aplicar amostragem com temperatura
   - Construir resposta iterativamente

5. **Loop do Agente**
   - Decidir: continuar ou parar?
   - Selecionar ferramenta se necessário
   - Executar e observar

6. **Avaliação**
   - Calcular 5 métricas de qualidade
   - Retornar resultado com pontuação

7. **Retorno**
   - Apresentar resposta ao usuário
   - Mostrar métricas e trace

---

## 📦 Funções Principais

### `load_knowledge_base() → dict`
```python
# Retorna dicionário de documentos
{
    'doc_1': "Conteúdo sobre IA...",
    'doc_2': "Conteúdo sobre LLMs...",
    ...
}
```

---

### `embed_documents(docs: dict) → np.ndarray`
```python
# Retorna matriz (num_docs, embedding_dim)
# Simples: Embeddings baseados em hash para demo
# Real: Usar embeddings do SentenceTransformer
```

---

### `retrieve_documents(query: str, k: int = 3) → list`
```python
# Entrada: "O que é um LLM?"
# Saída: [
#   {'doc': 'doc_1', 'content': '...', 'similarity': 0.87},
#   {'doc': 'doc_2', 'content': '...', 'similarity': 0.76},
#   {'doc': 'doc_3', 'content': '...', 'similarity': 0.68}
# ]
```

---

### `reasoning_phase(question: str, contexts: list) → str`
```python
# Entrada: pergunta + contextos recuperados
# Saída: Trace de pensamento estruturado
"""
Passo 1: Analisar a pergunta
O usuário pergunta sobre LLMs...

Passo 2: Identificar conceitos chave
Conceitos: arquitetura, treinamento, inferência...

Passo 3: Recuperar contexto relevante
Do documento X, sabemos que...

Passo 4: Sintetizar
Combinando o conhecimento, podemos concluir...
"""
```

---

### `generate_with_temperature(prompt: str, temp: float = 1.0) → str`
```python
# Temperatura baixa (0.3): determinístico, focado
# Temperatura média (1.0): balanceado
# Temperatura alta (2.0): criativo, diverso

# Retorna segmento de texto gerado
```

---

### `agent_loop(initial_query: str, max_turns: int = 3) → dict`
```python
# Execução agêntica
# Cada turno: Pensar → Agir → Observar

# Retorna: {
#   'answer': 'Resposta final',
#   'turns': 3,
#   'trace': ['Turno 1: ...', 'Turno 2: ...', ...]
# }
```

---

### `evaluate_response(response: str, context: str) → dict`
```python
# Calcula 5 métricas:
# - Ratio de comprimento
# - Sobreposição de vocabulário (BLEU)
# - Similaridade de embeddings
# - Pontuação de coerência
# - Qualidade geral (0-100)

# Retorna: {
#   'metrics': {'bleu': 0.75, 'similarity': 0.82, ...},
#   'quality_score': 79,
#   'interpretation': 'Boa resposta...'
# }
```

---

## ⚙️ Configuração e Parâmetros

| Parâmetro | Default | Faixa | Efeito |
|-----------|---------|-------|--------|
| `TEMPERATURE` | 1.0 | 0.0-2.0 | Controle de criatividade |
| `K_DOCUMENTS` | 3 | 1-10 | Tamanho do contexto |
| `MAX_TURNS` | 3 | 1-10 | Iterações do agente |
| `EMBEDDING_DIM` | 128 | 64-512 | Tamanho do embedding |

**Como modificar:**
```python
# No script 09
TEMPERATURE = 1.5        # Mais criativo
K_DOCUMENTS = 5          # Mais contexto
MAX_TURNS = 5            # Mais iterações do agente
```

---

## 💡 Detalhes Chave de Implementação

### Embeddings (Demo Simplificado)
```python
# Produção real: SentenceTransformer
# Versão demo: Baseado em hash (determinístico, rápido)

def simple_embedding(text: str, dim: int = 128) -> np.ndarray:
    hash_val = hash(text)
    np.random.seed(abs(hash_val) % 2**32)
    return np.random.randn(dim)
```

---

### Amostragem com Temperatura
```python
# Temperatura = fator de escala para softmax
# logits = [1.0, 2.0, 0.5]
# 
# T=0.5: softmax(logits / 0.5) → mais agudo [0.1, 0.87, 0.03]
# T=1.0: softmax(logits / 1.0) → normal [0.09, 0.67, 0.24]
# T=2.0: softmax(logits / 2.0) → mais plano [0.28, 0.38, 0.34]
```

---

### Prompting Chain-of-Thought
```
Em vez de: "O que é X?"
Melhor:    "Vamos pensar passo a passo:
            1. Definir o conceito
            2. Decompô-lo
            3. Fornecer exemplos
            4. Concluir"
```

---

### Implementação do Loop ReAct
```python
while not done and turns < max_turns:
    # PENSAR: Analisar estado atual
    thought = analyze_state(context)
    
    # AGIR: Escolher e executar ferramenta/ação
    action = select_action(thought)
    result = execute_action(action)
    
    # OBSERVAR: Atualizar conhecimento
    observation = observe_result(result)
    
    turns += 1
```

---

## 🎯 Resultados de Aprendizagem

Depois de estudar esta arquitetura, você entende:

✅ Como RAG integra recuperação com geração  
✅ Como a temperatura afeta o comportamento do modelo  
✅ Como Chain-of-Thought melhora o raciocínio  
✅ Como os agentes tomam decisões autônomas  
✅ Como avaliar a qualidade de geração  
✅ Como combinar todos esses conceitos em um sistema  

---

## 🚀 Próximos Passos

1. **Execute:** [Guia de Início Rápido](QUICKSTART_SCRIPT_09.md)
2. **Entenda o código:** [Mapeamento Código ↔ Conceitos](SCRIPT_09_MAPPING.md)
3. **Adapte:** Modifique para seu caso de uso
4. **Estenda:** Adicione mais ferramentas, melhores embeddings, etc.

---

**Pronto para aprofundar? 📚**
