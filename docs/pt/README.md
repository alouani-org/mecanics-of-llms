# Scripts Práticos: Experimentando com Conceitos de LLM

🌍 [English](../en/README.md) | 📖 [Français](../fr/README.md) | 🇪🇸 [Español](../es/README.md) | 🇧🇷 **Português** | 🇸🇦 [العربية](../ar/README.md)

Coleção de **10 scripts Python executáveis** para experimentar os conceitos-chave do livro **"A Mecânica dos LLMs"**.

> 📚 **Sobre**: Estes scripts acompanham os capítulos do livro. Veja [Jornada Pedagógica](PEDAGOGICAL_JOURNEY.md) para as correspondências detalhadas.

**📕 Comprar o Livro:**
- **Impresso**: [Amazon](https://amzn.eu/d/3oREERI)
- **Kindle**: [Amazon](https://amzn.eu/d/b7sG5iw)

---

## 📋 Visão Geral dos Scripts

| # | Script | Capítulo(s) | Conceitos | Status |
|---|--------|-------------|-----------|--------|
| 1 | `01_tokenization_embeddings.py` | 2 | Tokenização, impacto no comprimento da sequência | ✅ |
| 2 | `02_multihead_attention.py` | 3 | Self-attention, multi-head, pesos de atenção | ✅ |
| 3 | `03_temperature_softmax.py` | 7, 11 | Temperatura, softmax, entropia | ✅ |
| 4 | `04_rag_minimal.py` | 13 | Pipeline RAG, recuperação, similaridade cosseno | ✅ |
| 5 | `05_pass_at_k_evaluation.py` | 12 | Pass@k, Pass^k, avaliação de modelos | ✅ |
| 🎁 6 | `06_react_agent_bonus.py` | 14, 15 | **Agentes ReAct, registro de ferramentas, MCP** | ✅ BÔNUS |
| 🎁 7 | `07_llamaindex_rag_advanced.py` | 13, 14 | **RAG avançado, indexação, chat persistente** | ✅ BÔNUS |
| 🎁 8 | `08_lora_finetuning_example.py` | 9, 10 | **LoRA, QLoRA, comparação de fine-tuning** | ✅ BÔNUS |
| 🏆 **9** | `09_mini_assistant_complet.py` | **11-15** | **🎯 Projeto Integrador Final** | ✅ PRINCIPAL |
| 🎁 10 | `10_activation_steering_demo.py` | 10 | **Activation Steering, 3SO, vetores de conceito** | ✅ BÔNUS |

---

## 📖 Descrições Detalhadas dos Scripts

### 📌 Script 01: Tokenização e Embeddings
**Arquivo:** `01_tokenization_embeddings.py` | **Capítulo:** 2

**O que o script faz:**
- Carrega um tokenizador (GPT-2 ou LLaMA-2) e analisa diferentes textos
- Compara o número de tokens entre francês e inglês
- Demonstra o impacto do comprimento da sequência no custo computacional

**O que você aprende:**
- Como o texto é dividido em tokens (BPE, WordPiece)
- Por que "Bonjour" pode virar 2-3 tokens enquanto "Hello" é apenas um
- O impacto direto: mais tokens = maior custo O(n²) para atenção

**Saída esperada:**
```
Text: L'IA est utile
  Token count: 5
  Tokens: ['L', "'", 'IA', 'est', 'utile']
```

---

### 📌 Script 02: Atenção Multi-Cabeças
**Arquivo:** `02_multihead_attention.py` | **Capítulo:** 3

**O que o script faz:**
- Simula uma camada de atenção multi-cabeças com tensores PyTorch
- Calcula as projeções Q, K, V e os pesos de atenção
- Mostra como cada cabeça "olha" a frase de maneira diferente

**O que você aprende:**
- O mecanismo Q (Query), K (Key), V (Value)
- Por que múltiplas cabeças capturam diferentes dependências
- Que os pesos de atenção sempre somam 1 (distribuição de probabilidade)

**Saída esperada:**
```
Sentence: The cat sleeps well
Head 1: Attention weights from 'cat' → 'sleeps': 0.42
Head 2: Attention weights from 'cat' → 'The': 0.38
```

---

### 📌 Script 03: Temperatura e Softmax
**Arquivo:** `03_temperature_softmax.py` | **Capítulos:** 7, 11

**O que o script faz:**
- Aplica softmax com diferentes temperaturas (0.1, 0.5, 1.0, 2.0)
- Calcula a entropia de Shannon para cada distribuição
- Gera gráficos (se matplotlib estiver instalado)

**O que você aprende:**
- T < 1: distribuição "aguda" → geração determinística (greedy)
- T > 1: distribuição "plana" → geração criativa/diversa
- A entropia aumenta com a temperatura (mais incerteza)

**Saída esperada:**
```
Temperature 0.5: Token 'Paris' = 85% (agudo, determinístico)
Temperature 2.0: Token 'Paris' = 35% (plano, criativo)
```

---

### 📌 Script 04: RAG Mínimo
**Arquivo:** `04_rag_minimal.py` | **Capítulo:** 13

**O que o script faz:**
- Cria uma mini base de conhecimento (7 documentos sobre LLMs)
- Vetoriza os documentos com TF-IDF
- Realiza busca por similaridade cosseno
- Simula a geração aumentada pelo contexto recuperado

**O que você aprende:**
- O pipeline RAG completo: Recuperação → Aumentação → Geração
- Como a similaridade cosseno encontra os documentos relevantes
- Por que RAG permite responder perguntas sobre dados privados

**Saída esperada:**
```
Pergunta: "Como funciona a atenção no Transformer?"
→ Documentos recuperados: [doc_1: 0.72, doc_4: 0.65]
→ Resposta gerada com contexto
```

---

### 📌 Script 05: Avaliação Pass@k
**Arquivo:** `05_pass_at_k_evaluation.py` | **Capítulo:** 12

**O que o script faz:**
- Simula 100 tentativas de geração com taxa de sucesso de 30%
- Calcula Pass@k (pelo menos 1 sucesso em k tentativas)
- Calcula Pass^k (todas as k tentativas bem-sucedidas)

**O que você aprende:**
- Pass@k = 1 - (1-p)^k: probabilidade de pelo menos um sucesso
- Pass^k = p^k: probabilidade de todos terem sucesso (muito rigoroso)
- Por que Pass@10 ≈ 97% mesmo com p=30% (você tem 10 chances)

**Saída esperada:**
```
Pass@1  = 30%  (chance com 1 tentativa)
Pass@5  = 83%  (chance com 5 tentativas)
Pass@10 = 97%  (quase certo com 10 tentativas)
```

---

### 🎁 Script 06: Agente ReAct (BÔNUS)
**Arquivo:** `06_react_agent_bonus.py` | **Capítulos:** 14, 15

**O que o script faz:**
- Implementa um mini framework de agentes autônomos
- Demonstra o loop ReAct: Thought → Action → Observation → ...
- Inclui ferramentas simuladas: calculadora, busca web, clima

**O que você aprende:**
- O padrão ReAct (Raciocínio + Ação)
- Como um agente decide qual ação tomar
- Auto-correção: o agente pode tentar novamente se uma ação falhar
- A base para entender agentes MCP (Model Context Protocol)

**Saída esperada:**
```
Thought: Preciso calcular 15% de R$250
Action: calculator(250 * 0.15)
Observation: 37.5
Final Answer: A gorjeta é de R$37,50
```

---

### 🎁 Script 07: RAG Avançado com LlamaIndex (BÔNUS)
**Arquivo:** `07_llamaindex_rag_advanced.py` | **Capítulos:** 13, 14

**O que o script faz:**
- Sistema RAG completo com parsing de documentos
- Indexação e embeddings (simulados ou reais com OpenAI)
- Chat com memória conversacional
- Avaliação de qualidade (Precisão, Recall, F1)

**O que você aprende:**
- Arquitetura RAG de produção: ingestão → indexação → recuperação → geração
- Como manter o contexto através de múltiplos turnos de conversa
- Como avaliar a qualidade de um sistema RAG

**Saída esperada:**
```
[Modo Chat]
Usuário: O que é um Transformer?
Assistente: [Contexto: 3 docs] Um Transformer é...
Usuário: E a atenção multi-cabeças?
Assistente: [Memória: pergunta anterior + 2 docs] ...
```

---

### 🎁 Script 08: Fine-tuning LoRA/QLoRA (BÔNUS)
**Arquivo:** `08_lora_finetuning_example.py` | **Capítulos:** 9, 10

**O que o script faz:**
- Compara Full Fine-tuning vs LoRA vs QLoRA (cálculos numéricos)
- Mostra as economias de VRAM e parâmetros treináveis
- Caso de uso: adaptar LLaMA-7B para um domínio empresarial (ferroviário)

**O que você aprende:**
- LoRA: adiciona ~0.1% de parâmetros vs fine-tuning completo
- QLoRA: quantização de 4 bits + LoRA = GPU de 24GB em vez de 140GB
- Por que o fine-tuning eficiente democratiza os LLMs

**Saída esperada:**
```
LLaMA-7B:
  Full Fine-tuning: 28 GB VRAM, 7B params
  LoRA (rank=8):    8 GB VRAM, 4.2M params (0.06%)
  QLoRA:            6 GB VRAM, 4.2M params + base 4-bit
```

---

### � Script 10: Activation Steering & 3SO (BÔNUS)
**Arquivo:** `10_activation_steering_demo.py` | **Capítulo:** 10

**O que o script faz:**
- Demonstra o steering por ativações: injeção de vetores de conceito
- Implementa extração de vetores por ativação contrastiva
- Simula um Sparse Autoencoder (SAE) para decomposição em conceitos
- Implementa uma máquina de estados finitos para 3SO (saídas JSON garantidas)
- Compara RLHF/DPO vs Steering com tabela detalhada

**O que você aprende:**
- O steering modifica as ativações em inferência: $X_{steered} = X + (c \times V)$
- Como extrair vetores de conceito (método contrastivo, SAE)
- Impacto do coeficiente de steering (muito baixo → nulo, ótimo → efetivo, muito alto → descarrilamento)
- O 3SO garante matematicamente uma sintaxe JSON válida
- Quando usar alinhamento vs steering

**Saída esperada:**
```
STEP 3: Analyzing Coefficient Effect
   Coeff   Direction Δ     Perturbation    Stability
   1.0     12.5°           8.2%            ✅ stable
   5.0     45.3°           35.1%           ⚠️ moderate
   15.0    78.2°           89.4%           ❌ unstable
```

---

### �🏆 Script 09: Mini-Assistente Completo (PROJETO FINAL)
**Arquivo:** `09_mini_assistant_complet.py` | **Capítulos:** 11-15

**O que o script faz:**
- Integra TODOS os conceitos: RAG + Agentes + Temperatura + Avaliação
- Sistema completo com base de conhecimento, recuperação, raciocínio
- Modo interativo para testar diferentes perguntas

**O que você aprende:**
- Como montar um assistente IA completo de A a Z
- Arquitetura em camadas: Dados → Recuperação → Raciocínio → Geração
- Avaliação de ponta a ponta de um sistema

**Documentação dedicada:**
- [INDEX_SCRIPT_09.md](INDEX_SCRIPT_09.md): Arquitetura completa
- [QUICKSTART_SCRIPT_09.md](QUICKSTART_SCRIPT_09.md): Início rápido em 5 min
- [SCRIPT_09_MAPPING.md](SCRIPT_09_MAPPING.md): Mapeamento código ↔ conceitos

---

## 🚀 Início Rápido

### 1. Criar um Ambiente Virtual (recomendado)

```bash
# No Windows
python -m venv venv
venv\Scripts\activate

# No macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 2. Instalar Dependências

```bash
# Instalação básica (para scripts 1-5)
pip install torch transformers numpy scikit-learn

# Instalação completa (com visualizações)
pip install torch transformers numpy scikit-learn matplotlib

# Para scripts bônus (opcional, funcionam em modo demo sem estas)
pip install llama-index openai python-dotenv peft bitsandbytes
```

**Nota:** Os scripts bônus (06, 07, 08) funcionam **sem dependências externas** em modo demo.

### 3. Executar um Script

```bash
python 01_tokenization_embeddings.py
python 02_multihead_attention.py
python 03_temperature_softmax.py
python 04_rag_minimal.py
python 05_pass_at_k_evaluation.py
python 06_react_agent_bonus.py
python 07_llamaindex_rag_advanced.py
python 08_lora_finetuning_example.py
python 09_mini_assistant_complet.py    # ← Projeto integrador final
```

---

## 🏆 Projeto Integrador: Mini-Assistente Completo

**O script principal**: integra TODOS os conceitos dos capítulos 11-15.

- **Script:** `09_mini_assistant_complet.py`
- **Documentação:** [INDEX_SCRIPT_09.md](INDEX_SCRIPT_09.md)
- **Início Rápido:** [QUICKSTART_SCRIPT_09.md](QUICKSTART_SCRIPT_09.md)
- **Arquitetura:** [SCRIPT_09_MAPPING.md](SCRIPT_09_MAPPING.md)

---

## 📖 Documentação Completa

- **[Jornada Pedagógica](PEDAGOGICAL_JOURNEY.md)**: Correspondência capítulo por capítulo livro ↔ scripts
- **[Agentes ReAct](REACT_AGENT_INTEGRATION.md)**: Padrão ReAct e integração
- **[LlamaIndex RAG](LLAMAINDEX_GUIDE.md)**: Framework RAG avançado

---

## 📝 Notas

- **GPU não é necessário**: todos os scripts funcionam em CPU (mais lento)
- **Código educativo**: prioriza clareza sobre otimização
- **Compatível com Python 3.9+**

---

**Bom aprendizado! 🚀**
