# ⚡ Início Rápido em 5 Minutos: Script 09

🌍 [English](../en/QUICKSTART_SCRIPT_09.md) | 📖 [Français](../fr/QUICKSTART_SCRIPT_09.md) | 🇪🇸 [Español](../es/QUICKSTART_SCRIPT_09.md) | 🇧🇷 **Português** | 🇸🇦 [العربية](../ar/QUICKSTART_SCRIPT_09.md)

> **Execute em 5 minutos**  
> Sem teoria. Só código.

---

## 📍 Navegação Rápida

- **📖 Ver: [Jornada Pedagógica](PEDAGOGICAL_JOURNEY.md)** - Aprenda os conceitos
- **🏗️ Ver: [Arquitetura](INDEX_SCRIPT_09.md)** - Como está construído
- **🌍 Outros idiomas: [English](../en/QUICKSTART_SCRIPT_09.md) | [Français](../fr/QUICKSTART_SCRIPT_09.md) | [Español](../es/QUICKSTART_SCRIPT_09.md)**

---

## Passo 1️⃣: Requisitos (30 segundos)

```bash
# Já instalado? Você está pronto!
# Você só precisa disso (provavelmente já no seu sistema):
- Python 3.9+
- numpy
- scikit-learn (para similaridade cosseno)
```

**Verificar se está instalado:**
```bash
python --version
python -c "import numpy; print('numpy OK')"
python -c "from sklearn.metrics.pairwise import cosine_similarity; print('sklearn OK')"
```

---

## Passo 2️⃣: Navegue e Execute (1 minuto)

```bash
# Vá para o diretório de scripts
cd c:\dev\IA-Eductation\examples

# Execute o script
python 09_mini_assistant_complet.py
```

---

## Passo 3️⃣: Experimente (3 minutos)

Você verá um menu:

```
========================================
   Mini LLM Assistant - Demo Completa
========================================

Escolha uma opção:
1. Fazer uma pergunta
2. Testar com exemplos
3. Ver métricas de avaliação
4. Entender arquitetura
5. Sair

Digite sua escolha (1-5): 
```

**Tente isso:**

```
Digite sua escolha: 2

=== Executando Exemplos ===

Exemplo 1: "O que é um LLM?"
Pergunta: O que é um LLM?

📥 FASE DE RECUPERAÇÃO
Encontrados 3 documentos:
- doc_1 (similaridade: 0.85)
- doc_3 (similaridade: 0.78)
- doc_2 (similaridade: 0.72)

💭 FASE DE RACIOCÍNIO
[Mostra pensamento passo a passo]

🤖 FASE DE GERAÇÃO
Resposta: "Um LLM é um modelo de linguagem grande..."

🎯 AVALIAÇÃO
Pontuação de Qualidade: 82/100
- BLEU score: 0.78
- Similaridade de embeddings: 0.84
- Coerência: 0.79

...mais exemplos...
```

---

## 💡 O Que Acabou de Acontecer?

Seu script:

1. **📥 Recuperou** documentos da base de conhecimento
2. **💭 Raciocinou** passo a passo sobre o problema
3. **🤖 Gerou** uma resposta usando amostragem com temperatura
4. **🎯 Avaliou** a qualidade usando 5 métricas

Tudo em `09_mini_assistant_complet.py` ✅

---

## 🎮 Modo Interativo

Escolha opção 1 para fazer suas próprias perguntas:

```
Digite sua escolha: 1

Faça sua pergunta: O que são transformers?
Temperatura (0.1=focado, 1.0=balanceado, 2.0=criativo) [default 1.0]: 1.0

📥 RECUPERAÇÃO: Documentos relevantes encontrados
💭 RACIOCÍNIO: Pensando passo a passo...
🤖 GERAÇÃO: Criando resposta...
🎯 AVALIAÇÃO: Avaliando qualidade...

Resposta: [Sua resposta aqui]
Pontuação de Qualidade: 78/100
```

---

## 🔧 Personalização (Avançado)

Quer mudar o comportamento? Edite no script:

```python
# Mude estas constantes no início do arquivo:

TEMPERATURE = 1.0        # 0.1 (focado) a 2.0 (criativo)
K_DOCUMENTS = 3          # Quantos documentos recuperar
MAX_TURNS = 3            # Iterações do agente
EMBEDDING_DIM = 128      # Dimensão de embedding
```

Depois execute novamente.

---

## 🏆 O Que Você Está Aprendendo

Ao executar este script, você está praticando:

✅ **RAG** - Recuperar documentos relevantes  
✅ **Amostragem com Temperatura** - Controlar aleatoriedade  
✅ **Chain-of-Thought** - Raciocínio passo a passo  
✅ **Agentes ReAct** - Loops autônomos  
✅ **Avaliação** - Medir qualidade  

Tudo com código educativo que você pode ler e modificar.

---

## 🆘 Solução de Problemas

**"Module not found: numpy"**
```bash
pip install numpy scikit-learn
```

**"O script não executa"**
```bash
# Verifique a versão do Python
python --version

# Deve ser 3.9 ou superior
```

**"Execução lenta"**
- Normal! O código demo prioriza clareza sobre velocidade
- Sistemas reais usariam aceleração GPU

---

## 🚀 Próximos Passos

1. ✅ Você executou o script
2. 📖 [Leia a arquitetura](INDEX_SCRIPT_09.md)
3. 🔗 [Mapeie código para conceitos](SCRIPT_09_MAPPING.md)
4. 💻 Modifique e experimente
5. 🌟 Integre no seu projeto

---

## 📚 Mais Recursos

- **Entender conceitos?** → [Jornada Pedagógica](PEDAGOGICAL_JOURNEY.md)
- **Como está construído?** → [Arquitetura](INDEX_SCRIPT_09.md)
- **Qual código ensina o quê?** → [Mapeamento de Código](SCRIPT_09_MAPPING.md)
- **Agentes em detalhe?** → [Guia ReAct](REACT_AGENT_INTEGRATION.md)
- **RAG em detalhe?** → [Guia RAG](LLAMAINDEX_GUIDE.md)

---

**Parabéns! 🎉 Você está executando um mini assistente LLM.**

Experimente com diferentes perguntas e valores de temperatura. Veja como o sistema responde de maneiras diferentes!

**Dúvidas? Consulte a [Jornada Pedagógica](PEDAGOGICAL_JOURNEY.md) para explicações detalhadas.**
