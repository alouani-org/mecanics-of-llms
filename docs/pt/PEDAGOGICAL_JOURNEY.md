# 🗺️ Jornada Pedagógica Completa: Livro → Scripts → Conceitos

🌍 [English](../en/PEDAGOGICAL_JOURNEY.md) | 📖 [Français](../fr/PEDAGOGICAL_JOURNEY.md) | 🇪🇸 [Español](../es/PEDAGOGICAL_JOURNEY.md) | 🇧🇷 **Português** | 🇸🇦 [العربية](../ar/PEDAGOGICAL_JOURNEY.md)

> **Guia completo** para navegar o projeto "A Mecânica dos LLMs"  
> Correspondência detalhada: capítulos do livro ↔ scripts Python ↔ conceitos práticos

---

## 📍 Como Começar...

### Se você é novo ✨

```
1. Leia esta página (você está aqui)
   ↓
2. Confira README.md (navegação geral)
   ↓
3. Abra PEDAGOGICAL_JOURNEY.md (guia de scripts)
   ↓
4. Execute seu primeiro script
```

### Se você já leu o livro 📖

```
1. Encontre seu capítulo abaixo
   ↓
2. Clique no script correspondente
   ↓
3. Execute e experimente
```

### Se você quer programar imediatamente 💻

```
1. Vá direto para: 09_mini_assistant_complet.py
   ↓
2. Leia: INDEX_SCRIPT_09.md (arquitetura)
   ↓
3. Entenda e depois adapte
```

---

## 📚 Jornada Por Capítulo do Livro

### Capítulo 1: Introdução a NLP

**Conteúdo do Livro:**
- O que é NLP?
- História: de regras para aprendizado para LLMs
- Onde estamos em 2025

**Link de Código:**
- ❌ Sem script dedicado (teórico)
- ✅ Continue para o Capítulo 2

---

### Capítulo 2: Representação de Texto e Modelos Sequenciais

**Conteúdo do Livro:**
- Como os modelos veem o texto?
- Tokens e tokenizadores (BPE, WordPiece, SentencePiece)
- Impacto no comprimento da sequência
- RNNs, LSTMs, GRUs (os ancestrais)

**👉 Script Correspondente:**

#### [`01_tokenization_embeddings.py`](../../01_tokenization_embeddings.py)

**O que você aprende executando:**
```python
python 01_tokenization_embeddings.py
```

- Tokenização com diferentes tokenizadores
- Impacto da tokenização no comprimento da sequência
- Diferenças Francês vs Inglês
- Embeddings e suas dimensões
- Custo computacional baseado em tokens

**Conceitos Chave Demonstrados:**
- Tokenizadores BPE (Byte Pair Encoding)
- Vocabulário e subpalavras
- Relação Tokens ↔ custo de atenção O(n²)

**Tempo de execução:** ~5 segundos  
**Requisitos:** Python, `transformers`

---

### Capítulo 3: Arquitetura Transformer

**Conteúdo do Livro:**
- A invenção do mecanismo de atenção
- Self-attention e atenção multi-cabeças
- Estrutura encoder-decoder
- Codificação posicional
- O problema da posição

**👉 Script Correspondente:**

#### [`02_multihead_attention.py`](../../02_multihead_attention.py)

**O que você aprende executando:**
```python
python 02_multihead_attention.py
```

- Arquitetura de uma camada de atenção
- Projeções Q, K, V (Query, Key, Value)
- Cálculo de pontuações de atenção
- Multi-head: como cada cabeça foca diferente
- Visualização: quem atende a quem?

**Conceitos Chave Demonstrados:**
- Softmax e normalização de pontuações
- Dimensão de embedding vs número de cabeças
- Cada cabeça aprende diferentes relações

**Tempo de execução:** ~2 segundos  
**Requisitos:** Python, `numpy`

---

### Capítulos 4-8: Arquitetura, Otimização, Pré-treinamento

**Conteúdo do Livro:**
- Cap. 4: Modelos derivados do Transformer (BERT, GPT, T5...)
- Cap. 5: Otimização de arquitetura (atenção linear, RoPE...)
- Cap. 6: Arquitetura MoE (Mixture of Experts)
- Cap. 7: Pré-treinamento de LLM
- Cap. 8: Otimizações de treinamento (acumulação de gradiente...)

**Link de Código:**
- 📖 Teórico + conceitos
- ⚡ Integrado no Script 03 (temperatura durante pré-treinamento)
- 🏆 Aprimorado no Script 09 (mini-assistente)

---

### Capítulo 9: Fine-tuning Supervisionado (SFT)

**Conteúdo do Livro:**
- De predição para assistência
- Fine-tuning supervisionado (SFT)
- Qualidade sobre quantidade
- Avaliação de modelos fine-tunados
- Estudo de caso: adaptar LLaMA 7B

**👉 Script Bônus Correspondente:**

#### [`08_lora_finetuning_example.py`](../../08_lora_finetuning_example.py) 🎁

**O que você aprende executando:**
```python
python 08_lora_finetuning_example.py
```

- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
- Comparação: full fine-tuning vs LoRA
- Eficiência em termos de memória/velocidade
- Caso real SNCF (do texto do livro)

**Conceitos Chave Demonstrados:**
- Adaptar modelos sem retreinar tudo
- Tradeoff memória vs qualidade
- Parâmetros adicionais vs ganho

**Tempo de execução:** ~3 segundos  
**Requisitos:** Python, `numpy` (demo sem LLM externo)

---

### Capítulo 11: Estratégias de Geração e Inferência

**Conteúdo do Livro:**
- Prompting: guiar o modelo através de exemplos
- Controle de temperatura
- Estratégias de amostragem (top-k, top-p, nucleus sampling)
- Otimizar latência: KV-cache, especulação

**👉 Scripts Correspondentes:**

#### [`03_temperature_softmax.py`](../../03_temperature_softmax.py)

**O que você aprende executando:**
```python
python 03_temperature_softmax.py
```

- Efeito da temperatura no softmax
- T baixa = determinístico (greedy)
- T alta = diversidade (criativo)
- Relação com entropia
- Gráficos do efeito de temperatura

**Conceitos Chave Demonstrados:**
- Softmax e interpretação probabilística
- Temperatura como fator de escala
- Tradeoff determinismo vs criatividade

**Tempo de execução:** ~2 segundos  
**Requisitos:** Python, `matplotlib` (opcional)

#### [`09_mini_assistant_complet.py`](../../09_mini_assistant_complet.py) 🏆

**Seu primeiro assistente com:**
- Prompting (Chain-of-Thought)
- Amostragem com temperatura
- Estratégias de geração

---

### Capítulo 12: Modelos de Raciocínio

**Conteúdo do Livro:**
- Prompting Chain-of-Thought (CoT)
- Tree-of-Thought (ToT)
- Código e matemática (demonstração de raciocínio)
- Aprendizado por Reforço (RL) para pensar

**👉 Scripts Correspondentes:**

#### [`05_pass_at_k_evaluation.py`](../../05_pass_at_k_evaluation.py)

**O que você aprende executando:**
```python
python 05_pass_at_k_evaluation.py
```

- Métrica Pass@k para avaliação
- Pass^k (diferente de Pass@k)
- Por que essas métricas para raciocínio?
- Empíricos em tarefas de código

**Conceitos Chave Demonstrados:**
- Avaliação além da simples acurácia
- Múltiplas tentativas vs única tentativa
- Métricas específicas para raciocínio

**Tempo de execução:** ~1 segundo  
**Requisitos:** Python, `numpy`

---

### Capítulo 13: Sistemas Aumentados e Agentes (RAG)

**Conteúdo do Livro:**
- RAG: Retrieval-Augmented Generation
- O problema de integração M:N
- Por baixo do capô: implementação técnica
- Descoberta progressiva de ferramentas

**👉 Scripts Correspondentes:**

#### [`04_rag_minimal.py`](../../04_rag_minimal.py)

**O que você aprende executando:**
```python
python 04_rag_minimal.py
```

- Pipeline RAG mínimo (entender os passos)
- Similaridade cosseno para recuperação
- Aumentação de contexto
- Qualidade vs latência

**Conceitos Chave Demonstrados:**
- Fragmentação de documentos (chunking)
- Embeddings e busca
- Redução de alucinações

**Tempo de execução:** ~3 segundos  
**Requisitos:** Python, `numpy`, `scikit-learn`

#### [`07_llamaindex_rag_advanced.py`](../../07_llamaindex_rag_advanced.py) 🎁

**O que você aprende executando:**
```python
python 07_llamaindex_rag_advanced.py
```

- Framework RAG completo (LlamaIndex)
- 6 fases: Carregar → Indexar → RAG → Chat → Eval → Exportar
- Ingestão de documentos
- Chat com persistência
- Avaliação automática

**Conceitos Chave Demonstrados:**
- Arquitetura RAG de produção
- Estratégias de indexação
- Camada de persistência

**Tempo de execução:** ~5 segundos  
**Requisitos:** Python (demo), opcional: `llama-index`, `openai`

---

### Capítulo 14: Protocolos Agênticos (MCP)

**Conteúdo do Livro:**
- Agentes: autonomia e decisão
- Definição de agente
- Padrões: ReAct, Tool Use, Function Calling
- Model Context Protocol (MCP)
- Limitações e dificuldades

**👉 Script Bônus Correspondente:**

#### [`06_react_agent_bonus.py`](../../06_react_agent_bonus.py) 🎁

**O que você aprende executando:**
```python
python 06_react_agent_bonus.py
```

- Padrão ReAct (Raciocínio + Ação)
- Framework genérico para criar agentes
- Registro de ferramentas (tool registration)
- 3 ferramentas de exemplo
- Loop: pensar → agir → observar

**Conceitos Chave Demonstrados:**
- Loop de agente autônomo
- Tomada de decisões
- Composição de ferramentas

**Tempo de execução:** ~4 segundos  
**Requisitos:** Python, `numpy`

**Veja também:** [REACT_AGENT_INTEGRATION.md](REACT_AGENT_INTEGRATION.md)

---

### Capítulo 15: Avaliação Crítica de Fluxos Agênticos

**Conteúdo do Livro:**
- O desafio da medição
- Avaliar agentes: de palavras para fatos
- Métricas quantitativas e qualitativas
- Estudos de caso

**👉 Script Integrador Completo:**

#### [`09_mini_assistant_complet.py`](../../09_mini_assistant_complet.py) 🏆

**O que você aprende executando:**
```python
python 09_mini_assistant_complet.py
```

- Avaliação de um sistema completo
- Métricas: BLEU, similaridade de embeddings, coerência
- Traces e debugging
- Melhoria iterativa

**Conceitos Chave Demonstrados:**
- Avaliação multi-critério
- Loops de feedback
- Qualidade de execução

**Tempo de execução:** ~10 segundos  
**Requisitos:** Python (tudo incluído)

**Veja também:**
- [INDEX_SCRIPT_09.md](INDEX_SCRIPT_09.md) - Arquitetura
- [QUICKSTART_SCRIPT_09.md](QUICKSTART_SCRIPT_09.md) - Início rápido

**PARABÉNS!** 🎉 Você completou a jornada!

---

## 🎯 Rotas Aceleradas

### "Quero entender LLMs rapidamente" (2-3 horas)

```
Ler Capítulos 1-3         (30 min)
   ↓
Executar Scripts 01-02    (15 min)
   ↓
Ler Capítulos 11-12       (45 min)
   ↓
Executar Scripts 03-05    (30 min)
   ↓
Ler Capítulos 13-14       (45 min)
   ↓
Executar Script 09        (15 min)
```

**Resultado:** Compreensão sólida dos conceitos chave ✅

### "Quero programar uma aplicação RAG + Agentes" (4-6 horas)

```
Entender RAG              (Capítulo 13)  (30 min)
   ↓
Executar Scripts 04, 07   (30 min)
   ↓
Entender Agentes          (Capítulo 14)  (30 min)
   ↓
Executar Script 06        (20 min)
   ↓
Estudar Script 09         (60 min)
   ↓
Adaptar para seu caso     (variável)
```

**Resultado:** Aplicação funcional RAG + Agentes ✅

---

## 📝 Notas

- **GPU não é necessário**: todos os scripts funcionam em CPU (mais lento)
- **Dependências mínimas**: apenas `numpy`, `torch`, `transformers`, `scikit-learn`
- **Código educativo**: prioriza clareza sobre otimização
- **Compatível Python 3.9+**
- **Scripts bônus** demonstram conceitos avançados, funcionam sem LLM externo (modo simulação)

---

**Bom aprendizado! 🎓**
