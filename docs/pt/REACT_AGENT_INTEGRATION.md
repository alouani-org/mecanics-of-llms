# 🤖 Guia de Integração de Agentes ReAct

🌍 [English](../en/REACT_AGENT_INTEGRATION.md) | 📖 [Français](../fr/REACT_AGENT_INTEGRATION.md) | 🇪🇸 [Español](../es/REACT_AGENT_INTEGRATION.md) | 🇧🇷 **Português** | 🇸🇦 [العربية](../ar/REACT_AGENT_INTEGRATION.md)

> **Entendendo agentes e padrões agênticos**  
> Teoria + Implementação

---

## 📍 Navegação Rápida

- **📖 Ver: [Jornada Pedagógica](PEDAGOGICAL_JOURNEY.md)** - Onde isso se encaixa
- **⚡ Ver: [Início Rápido](QUICKSTART_SCRIPT_09.md)** - Execute Script 06
- **🌍 Outros idiomas: [English](../en/REACT_AGENT_INTEGRATION.md) | [Français](../fr/REACT_AGENT_INTEGRATION.md) | [Español](../es/REACT_AGENT_INTEGRATION.md)**

---

## 🎯 O que é um Agente?

Um **agente** é um sistema que:

1. **Observa** seu ambiente (entrada, contexto)
2. **Raciocina** sobre o que fazer
3. **Age** (toma uma ação)
4. **Observa** o resultado
5. **Repete** até alcançar o objetivo

### Agente Simples vs. Agente Inteligente

**Agente Simples:**
```
Entrada → Processar → Saída
(Um único passo, determinístico)
```

**Agente Inteligente (ReAct):**
```
Entrada → Pensar → Agir → Observar → Loop
            ↓
        Objetivo alcançado? Não → Repetir
        Objetivo alcançado? Sim → Retornar resposta
```

---

## 🏗️ Padrão ReAct

**ReAct** = **Re**asoning (Raciocínio) + **Act**ing (Ação)

### O Loop

```
┌──────────────────────────────────┐
│   INÍCIO: Receber tarefa         │
└────────────┬─────────────────────┘
             ↓
┌──────────────────────────────────┐
│ PENSAR: Qual a melhor ação?      │
│ (Raciocínio interno)             │
└────────────┬─────────────────────┘
             ↓
┌──────────────────────────────────┐
│ AGIR: Executar a ação            │
│ (Usar ferramentas, recuperar)    │
└────────────┬─────────────────────┘
             ↓
┌──────────────────────────────────┐
│ OBSERVAR: O que aconteceu?       │
│ (Processar resultado, aprender)  │
└────────────┬─────────────────────┘
             ↓
┌──────────────────────────────────┐
│ Objetivo alcançado?              │
│ SIM ↓         NÃO ↓              │
└──────┼──────────┼────────────────┘
       ↓          ↓
    RETORNAR  VOLTAR AO LOOP
             (para PENSAR)
```

### Exemplo: Responder uma Pergunta Complexa

**Usuário pergunta:** "Qual é a capital do país mais populoso?"

**O agente pensa:**
```
PENSAR: 
- Preciso encontrar o país mais populoso
- Depois encontrar sua capital
- Isso requer múltiplos passos de raciocínio
```

**O agente age:**
```
AÇÃO 1: Usar ferramenta "buscar_país" → Resultado: "Índia ou China"
AÇÃO 2: Usar ferramenta "obter_população" → Resultado: "Índia: 1.4B, China: 1.4B"
AÇÃO 3: Usar ferramenta "obter_capital" → Resultado: "Índia: Nova Delhi, China: Pequim"
```

**O agente observa:**
```
OBSERVAR:
- População da Índia: 1.4 bilhões
- População da China: 1.4 bilhões
- São aproximadamente iguais; Índia é atualmente o mais populoso
- Capital: Nova Delhi
```

**O agente raciocina novamente:**
```
PENSAR: Tenho a informação. Objetivo alcançado.
```

**Retorna:** "Nova Delhi (Índia é atualmente o país mais populoso)"

---

## 🛠️ Ferramentas em Agentes ReAct

**Uma ferramenta** é uma função que o agente pode chamar:

```python
class Tool:
    def __init__(self, name: str, function: callable, description: str):
        self.name = name
        self.function = function
        self.description = description
    
    def execute(self, *args, **kwargs):
        return self.function(*args, **kwargs)
```

### Ferramentas de Exemplo

```python
# Ferramenta 1: Busca na Internet
search_tool = Tool(
    name="search",
    function=search_internet,
    description="Buscar informação na internet"
)

# Ferramenta 2: Calculadora
calculator_tool = Tool(
    name="calculate",
    function=eval_expression,
    description="Realizar cálculos matemáticos"
)

# Ferramenta 3: Consulta de Banco de Dados
database_tool = Tool(
    name="query_db",
    function=query_database,
    description="Consultar o banco de dados da empresa"
)
```

### Como o Agente Seleciona Ferramentas

```
PENSAR: Qual ferramenta devo usar?
├─ search_tool: Boa para coletar informação
├─ calculator_tool: Boa para matemática
└─ database_tool: Boa para dados da empresa

AÇÃO: "Vou usar search_tool com consulta='capital da Índia'"

RESULTADO: "Nova Delhi"
```

---

## 📝 Formato ReAct

Os agentes se comunicam usando um formato estruturado:

```
Thought: O que devo fazer a seguir?
Action: nome_ferramenta[argumento]
Observation: [Resultado da ferramenta]

Thought: Próximo passo?
Action: nome_ferramenta[argumento]
Observation: [Resultado da ferramenta]

...

Thought: Agora sei a resposta final
Final Answer: [Resposta]
```

### Exemplo Real

```
Thought: Preciso encontrar a raiz quadrada de 144 e somar 5
Action: calculate[sqrt(144)]
Observation: 12

Thought: Agora somo 5
Action: calculate[12 + 5]
Observation: 17

Thought: Tenho a resposta
Final Answer: 17
```

---

## 🎯 Script 06: Implementação de Agente ReAct

Vamos ver como é codificado:

### 1. Definir Ferramentas

```python
def search_wiki(topic: str) -> str:
    """Simula busca na Wikipedia"""
    return f"Informação sobre {topic}..."

def calculate(expression: str) -> float:
    """Simula calculadora"""
    return eval(expression)

def get_translation(word: str, lang: str) -> str:
    """Simula ferramenta de tradução"""
    translations = {
        ('hello', 'pt'): 'olá',
        ('hello', 'es'): 'hola',
    }
    return translations.get((word, lang), 'desconhecido')

tools = {
    'search': search_wiki,
    'calculate': calculate,
    'translate': get_translation
}
```

### 2. Parsear Saída do Agente

```python
def parse_action(text: str) -> tuple:
    """Extrair nome da ferramenta e argumentos"""
    # Formato: "Action: nome_ferramenta[argumento]"
    
    import re
    match = re.search(r'Action:\s*(\w+)\[(.+?)\]', text)
    if match:
        tool_name = match.group(1)
        argument = match.group(2)
        return tool_name, argument
    return None, None
```

### 3. Loop Principal do Agente

```python
def agent_loop(query: str, max_turns: int = 5) -> str:
    """Executar loop ReAct"""
    
    context = f"Pergunta do usuário: {query}\n"
    turn = 0
    
    while turn < max_turns:
        turn += 1
        
        # PENSAR
        thought = generate_thought(context, query)
        context += f"Thought: {thought}\n"
        
        # Verificar resposta final
        if "Final Answer" in thought:
            return extract_final_answer(thought)
        
        # AGIR
        tool_name, argument = parse_action(thought)
        if tool_name and tool_name in tools:
            result = tools[tool_name](argument)
            context += f"Action: {tool_name}[{argument}]\n"
            
            # OBSERVAR
            observation = f"Observation: {result}\n"
            context += observation
        
        else:
            context += "Observation: Ferramenta inválida ou sem ação\n"
    
    return "Máximo de turnos alcançado sem resposta"
```

---

## 🔄 Tipos e Estratégias de Agentes

### Tipo 1: Agente Sequencial Simples

```
Tarefa → Passo 1 → Passo 2 → Passo 3 → Resposta
(Ordem fixa)
```

**Quando usar:** Tarefas sequenciais bem definidas

---

### Tipo 2: Agente Adaptativo (ReAct)

```
Tarefa → Avaliar → Decidir Melhor Passo → Executar → Loop
         (Tomada de decisão dinâmica)
```

**Quando usar:** Tarefas complexas e imprevisíveis

---

### Tipo 3: Sistema Multi-Agente

```
Agente 1        Agente 2        Agente 3
(Especialista)  (Especialista)  (Coordenador)
   ↓               ↓               ↓
   └───────────────┴───────────────┘
         (Colaboram)
            ↓
         Resposta
```

**Quando usar:** Tarefas muito complexas que precisam de múltiplos especialistas

---

## ⚠️ Limitações e Desafios de Agentes

### 1. **Alucinação**
```
Agente: "A capital da França é Londres"
Problema: Informação incorreta
Solução: Fundamentar agente com ferramentas verificadas
```

### 2. **Loops Infinitos**
```
Agente: Buscar → Resultado ruim → Buscar de novo → ...
Problema: Nunca termina
Solução: Adicionar limite max_turns
```

### 3. **Mau Uso de Ferramentas**
```
Agente: Usa ferramenta de busca para calcular matemática
Problema: Ferramenta errada para a tarefa
Solução: Melhores descrições de ferramentas, treinamento do agente
```

### 4. **Custo**
```
Cada chamada de ferramenta = tempo + dinheiro + latência
Problema: Muitas chamadas = lento, caro
Solução: Otimizar seleção de ferramentas
```

---

## 🎓 Quando Usar Agentes

### ✅ Bons Casos de Uso

- **Pesquisa multi-passo:** "Encontre os 3 melhores papers sobre tema X, resuma cada um"
- **Fluxos de trabalho complexos:** "Crie relatório, obtenha aprovações, envie notificação"
- **Integração de ferramentas:** Chamadas API, consultas de banco de dados, cálculos
- **Problemas adaptativos:** Número desconhecido de passos

### ❌ Não Adequado Para

- **Consultas de um passo:** "Quanto é 2+2?" (Não precisa de agente)
- **Recuperação simples:** "Encontre documento X" (Consulta direta é mais rápida)
- **Tempo real crítico:** Agentes adicionam latência
- **Operações de alto custo:** Cada decisão custa dinheiro

---

## 🚀 Estendendo Agentes

### Adicionar Nova Ferramenta

```python
def new_tool_function(arg1: str, arg2: str) -> str:
    """Sua ferramenta personalizada"""
    return f"Resultado para {arg1} e {arg2}"

# Registrar ferramenta
tools['new_tool'] = new_tool_function

# O agente agora pode chamá-la:
# "Action: new_tool[arg1_value, arg2_value]"
```

### Melhorar Tomada de Decisão

```python
# Atual: Seleção aleatória de ferramenta
# Melhor: Pontuar cada ferramenta por relevância
def score_tools(query: str, available_tools: list) -> dict:
    scores = {}
    for tool in available_tools:
        score = similarity(query, tool.description)
        scores[tool.name] = score
    return scores

best_tool = max(scores, key=scores.get)
```

### Adicionar Memória

```python
class AgentWithMemory:
    def __init__(self):
        self.memory = {}
    
    def remember(self, key: str, value: str):
        self.memory[key] = value
    
    def recall(self, key: str) -> str:
        return self.memory.get(key, "Não encontrado")
```

---

## 📊 Desempenho de Agentes

### Métricas a Rastrear

| Métrica | O Que Mede | Objetivo |
|---------|------------|----------|
| **Taxa de Sucesso** | % tarefas completadas | >90% |
| **Passos Médios** | Turnos médios para resolver | <5 |
| **Corretude** | % respostas corretas | >95% |
| **Latência** | Tempo por tarefa | <1s |
| **Custo** | Chamadas de ferramentas × custo | $0.01-0.10 |

---

## 🎯 Pontos Chave

✅ **Agentes permitem raciocínio autônomo multi-passo**  
✅ **Padrão ReAct: Pensar → Agir → Observar → Loop**  
✅ **Ferramentas estendem as capacidades do agente**  
✅ **Ferramentas fundamentadas previnem alucinações**  
✅ **Valor real para tarefas complexas e adaptativas**  
✅ **Mas adicionam complexidade e latência**  

---

## 📚 Leituras Adicionais

- Script 06: [`06_react_agent_bonus.py`](../../06_react_agent_bonus.py)
- Capítulo 14 do livro: Protocolos Agênticos (MCP)
- Capítulo 15 do livro: Avaliação Crítica
- Integração: [Script 09](QUICKSTART_SCRIPT_09.md)

---

**Pronto para construir com agentes? 🤖**

Experimente o Script 06, depois integre no Script 09!
