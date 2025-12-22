# 🤖 ReAct Agent Integration Guide

> **Understanding agents and agentic patterns**  
> Theory + Implementation

---

## 📍 Quick Navigation

- **📖 See: [Pedagogical Journey](PEDAGOGICAL_JOURNEY.md)** - Where this fits
- **⚡ See: [Quick Start](QUICKSTART_SCRIPT_09.md)** - Run Script 06
- **🌍 Français: [Version Française](../fr/REACT_AGENT_INTEGRATION.md)**

---

## 🎯 What is an Agent?

An **agent** is a system that:

1. **Observes** its environment (input, context)
2. **Reasons** about what to do
3. **Acts** (takes an action)
4. **Observes** the result
5. **Repeats** until goal achieved

### Simple Agent vs. Smart Agent

**Simple Agent:**
```
Input → Process → Output
(One-shot, deterministic)
```

**Smart Agent (ReAct):**
```
Input → Think → Act → Observe → Loop
           ↓
        Goal reached? No → Repeat
        Goal reached? Yes → Return answer
```

---

## 🏗️ ReAct Pattern

**ReAct** = **Re**asoning + **Act**ing

### The Loop

```
┌──────────────────────────────────┐
│   START: Receive task            │
└────────────┬─────────────────────┘
             ↓
┌──────────────────────────────────┐
│ THINK: What's the best action?   │
│ (Internal reasoning)             │
└────────────┬─────────────────────┘
             ↓
┌──────────────────────────────────┐
│ ACT: Execute the action          │
│ (Use tools, retrieve info, etc)  │
└────────────┬─────────────────────┘
             ↓
┌──────────────────────────────────┐
│ OBSERVE: What happened?          │
│ (Process result, learn)          │
└────────────┬─────────────────────┘
             ↓
┌──────────────────────────────────┐
│ Is goal achieved?                │
│ YES ↓         NO ↓               │
└──────┼──────────┼────────────────┘
       ↓          ↓
    RETURN    LOOP BACK
             (to THINK)
```

### Example: Answering a Complex Question

**User asks:** "What's the capital of the most populous country?"

**Agent thinks:**
```
THINK: 
- I need to find the most populous country
- Then find its capital
- This requires multiple steps of reasoning
```

**Agent acts:**
```
ACT 1: Use tool "find_country" → Result: "India or China"
ACT 2: Use tool "get_population" → Result: "India: 1.4B, China: 1.4B"
ACT 3: Use tool "get_capital" → Result: "India: New Delhi, China: Beijing"
```

**Agent observes:**
```
OBSERVE:
- India population: 1.4 billion
- China population: 1.4 billion
- They're approximately equal; India is current most populous
- Capital: New Delhi
```

**Agent reasons again:**
```
THINK: I have the information. Goal achieved.
```

**Returns:** "New Delhi (India is currently the most populous country)"

---

## 🛠️ Tools in ReAct Agents

**A tool** is a function the agent can call:

```python
class Tool:
    def __init__(self, name: str, function: callable, description: str):
        self.name = name
        self.function = function
        self.description = description
    
    def execute(self, *args, **kwargs):
        return self.function(*args, **kwargs)
```

### Example Tools

```python
# Tool 1: Internet Search
search_tool = Tool(
    name="search",
    function=search_internet,
    description="Search the internet for information"
)

# Tool 2: Calculator
calculator_tool = Tool(
    name="calculate",
    function=eval_expression,
    description="Perform mathematical calculations"
)

# Tool 3: Database Query
database_tool = Tool(
    name="query_db",
    function=query_database,
    description="Query the company database"
)
```

### How Agent Selects Tools

```
THINK: Which tool should I use?
├─ search_tool: Good for information gathering
├─ calculator_tool: Good for math
└─ database_tool: Good for company data

ACTION: "I'll use search_tool with query='capital of India'"

RESULT: "New Delhi"
```

---

## 📝 ReAct Format

Agents communicate using a structured format:

```
Thought: What should I do next?
Action: tool_name[argument]
Observation: [Result from tool]

Thought: Next step?
Action: tool_name[argument]
Observation: [Result from tool]

...

Thought: I now know the final answer
Final Answer: [Answer]
```

### Real Example

```
Thought: I need to find the square root of 144 and add 5
Action: calculate[sqrt(144)]
Observation: 12

Thought: Now I add 5
Action: calculate[12 + 5]
Observation: 17

Thought: I have the answer
Final Answer: 17
```

---

## 🎯 Script 06: ReAct Agent Implementation

Let's see how this is coded:

### 1. Define Tools

```python
def search_wiki(topic: str) -> str:
    """Simulate Wikipedia search"""
    return f"Information about {topic}..."

def calculate(expression: str) -> float:
    """Simulate calculator"""
    return eval(expression)

def get_translation(word: str, lang: str) -> str:
    """Simulate translation tool"""
    translations = {
        ('hello', 'fr'): 'bonjour',
        ('hello', 'es'): 'hola',
    }
    return translations.get((word, lang), 'unknown')

tools = {
    'search': search_wiki,
    'calculate': calculate,
    'translate': get_translation
}
```

### 2. Parsing Agent Output

```python
def parse_action(text: str) -> tuple:
    """Extract tool name and arguments"""
    # Format: "Action: tool_name[argument]"
    
    import re
    match = re.search(r'Action:\s*(\w+)\[(.+?)\]', text)
    if match:
        tool_name = match.group(1)
        argument = match.group(2)
        return tool_name, argument
    return None, None
```

### 3. Main Agent Loop

```python
def agent_loop(query: str, max_turns: int = 5) -> str:
    """Execute ReAct loop"""
    
    context = f"User question: {query}\n"
    turn = 0
    
    while turn < max_turns:
        turn += 1
        
        # THINK
        thought = generate_thought(context, query)
        context += f"Thought: {thought}\n"
        
        # Check for final answer
        if "Final Answer" in thought:
            return extract_final_answer(thought)
        
        # ACT
        tool_name, argument = parse_action(thought)
        if tool_name and tool_name in tools:
            result = tools[tool_name](argument)
            context += f"Action: {tool_name}[{argument}]\n"
            
            # OBSERVE
            observation = f"Observation: {result}\n"
            context += observation
        
        else:
            context += "Observation: Invalid tool or no action\n"
    
    return "Max turns reached without answer"
```

---

## 🔄 Agent Types & Strategies

### Type 1: Simple Sequential Agent

```
Task → Step 1 → Step 2 → Step 3 → Answer
(Fixed order)
```

**When to use:** Well-defined sequential tasks

---

### Type 2: Adaptive Agent (ReAct)

```
Task → Assess → Decide Best Step → Execute → Loop
       (Dynamic decision-making)
```

**When to use:** Complex, unpredictable tasks

---

### Type 3: Multi-Agent System

```
Agent 1        Agent 2        Agent 3
(Specialist)   (Specialist)   (Coordinator)
   ↓              ↓              ↓
   └──────────────┴──────────────┘
        (Collaborate)
           ↓
        Answer
```

**When to use:** Very complex tasks needing multiple experts

---

## 💡 Agent Decision-Making

### How Does Agent Decide Next Action?

**Option A: Rule-Based**
```python
if "math" in query:
    use_calculator()
elif "fact" in query:
    use_search()
else:
    use_reasoning()
```

**Option B: LLM-Based**
```python
next_action = llm.generate(
    prompt=f"Given context: {context}, what's the next action?"
)
```

**Option C: Hybrid**
```python
if obvious_tool_choice(query):
    use_obvious_tool()
else:
    use_llm_to_decide()
```

---

## ⚠️ Agent Limitations & Challenges

### 1. **Hallucination**
```
Agent: "The capital of France is London"
Problem: Incorrect information
Solution: Ground agent with verified tools
```

### 2. **Infinite Loops**
```
Agent: Search → Bad Result → Search Again → ...
Problem: Never terminates
Solution: Add max_turns limit
```

### 3. **Tool Misuse**
```
Agent: Uses search tool to calculate math
Problem: Wrong tool for task
Solution: Better tool descriptions, agent training
```

### 4. **Cost**
```
Each tool call = time + money + latency
Problem: Too many calls = slow, expensive
Solution: Optimize tool selection
```

---

## 🎓 When to Use Agents

### ✅ Good Use Cases

- **Multi-step research:** "Find top 3 papers on topic X, summarize each"
- **Complex workflows:** "Create report, get approvals, send notification"
- **Tool integration:** API calls, database queries, calculations
- **Adaptive problems:** Unknown number of steps

### ❌ Not Suitable For

- **Single-step queries:** "What's 2+2?" (No agent needed)
- **Simple retrieval:** "Find document X" (Direct query is faster)
- **Real-time critical:** Agents add latency
- **High-cost operations:** Every decision costs money

---

## 🚀 Extending Agents

### Add New Tool

```python
def new_tool_function(arg1: str, arg2: str) -> str:
    """Your custom tool"""
    return f"Result for {arg1} and {arg2}"

# Register tool
tools['new_tool'] = new_tool_function

# Agent can now call it:
# "Action: new_tool[arg1_value, arg2_value]"
```

### Improve Decision-Making

```python
# Current: Random tool selection
# Better: Score each tool by relevance
def score_tools(query: str, available_tools: list) -> dict:
    scores = {}
    for tool in available_tools:
        score = similarity(query, tool.description)
        scores[tool.name] = score
    return scores

best_tool = max(scores, key=scores.get)
```

### Add Memory

```python
class AgentWithMemory:
    def __init__(self):
        self.memory = {}
    
    def remember(self, key: str, value: str):
        self.memory[key] = value
    
    def recall(self, key: str) -> str:
        return self.memory.get(key, "Not found")
```

---

## 📊 Agent Performance

### Metrics to Track

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Success Rate** | % tasks completed | >90% |
| **Avg Steps** | Average turns to solve | <5 |
| **Correctness** | % correct answers | >95% |
| **Latency** | Time per task | <1s |
| **Cost** | Tool calls × cost | $0.01-0.10 |

### Example Performance

```
Task: "Find square root of 16 and multiply by 2"

Agent:
├─ Thought: Need to calculate
├─ Action: calculate[sqrt(16)]          ← 1 tool call
├─ Observation: 4
├─ Thought: Now multiply by 2
├─ Action: calculate[4 * 2]              ← 2 tool calls
├─ Observation: 8
├─ Thought: Done
└─ Final Answer: 8

Performance:
- Steps: 2
- Tool calls: 2
- Latency: ~100ms
- Correctness: ✓
```

---

## 🎯 Key Takeaways

✅ **Agents enable autonomous, multi-step reasoning**  
✅ **ReAct pattern: Think → Act → Observe → Loop**  
✅ **Tools extend agent capabilities**  
✅ **Grounding tools prevents hallucination**  
✅ **Real value for complex, adaptive tasks**  
✅ **But add complexity and latency**  

---

## 📚 Further Reading

- Script 06: [`06_react_agent_bonus.py`](../../06_react_agent_bonus.py)
- Book Chapter 14: Agentic Protocols (MCP)
- Book Chapter 15: Critical Evaluation
- Integration: [Script 09](QUICKSTART_SCRIPT_09.md)

---

**Ready to build with agents? 🤖**

Try Script 06, then integrate into Script 09!
