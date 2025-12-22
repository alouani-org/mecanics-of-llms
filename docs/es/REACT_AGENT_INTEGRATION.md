# 🤖 Guía de Integración de Agentes ReAct

> **Entendiendo agentes y patrones agénticos**  
> Teoría + Implementación

---

## 📍 Navegación Rápida

- **📖 Ver: [Recorrido Pedagógico](PEDAGOGICAL_JOURNEY.md)** - Dónde encaja esto
- **⚡ Ver: [Inicio Rápido](QUICKSTART_SCRIPT_09.md)** - Ejecuta Script 06
- **🌍 Otros idiomas: [English](../en/REACT_AGENT_INTEGRATION.md) | [Français](../fr/REACT_AGENT_INTEGRATION.md) | [Português](../pt/REACT_AGENT_INTEGRATION.md)**

---

## 🎯 ¿Qué es un Agente?

Un **agente** es un sistema que:

1. **Observa** su entorno (entrada, contexto)
2. **Razona** sobre qué hacer
3. **Actúa** (toma una acción)
4. **Observa** el resultado
5. **Repite** hasta alcanzar el objetivo

### Agente Simple vs. Agente Inteligente

**Agente Simple:**
```
Entrada → Procesar → Salida
(Un solo paso, determinístico)
```

**Agente Inteligente (ReAct):**
```
Entrada → Pensar → Actuar → Observar → Bucle
            ↓
        ¿Objetivo alcanzado? No → Repetir
        ¿Objetivo alcanzado? Sí → Retornar respuesta
```

---

## 🏗️ Patrón ReAct

**ReAct** = **Re**asoning (Razonamiento) + **Act**ing (Acción)

### El Bucle

```
┌──────────────────────────────────┐
│   INICIO: Recibir tarea          │
└────────────┬─────────────────────┘
             ↓
┌──────────────────────────────────┐
│ PENSAR: ¿Cuál es la mejor acción?│
│ (Razonamiento interno)           │
└────────────┬─────────────────────┘
             ↓
┌──────────────────────────────────┐
│ ACTUAR: Ejecutar la acción       │
│ (Usar herramientas, recuperar)   │
└────────────┬─────────────────────┘
             ↓
┌──────────────────────────────────┐
│ OBSERVAR: ¿Qué pasó?             │
│ (Procesar resultado, aprender)   │
└────────────┬─────────────────────┘
             ↓
┌──────────────────────────────────┐
│ ¿Objetivo alcanzado?             │
│ SÍ ↓          NO ↓               │
└──────┼──────────┼────────────────┘
       ↓          ↓
    RETORNAR  VOLVER AL BUCLE
             (a PENSAR)
```

### Ejemplo: Responder una Pregunta Compleja

**Usuario pregunta:** "¿Cuál es la capital del país más poblado?"

**El agente piensa:**
```
PENSAR: 
- Necesito encontrar el país más poblado
- Luego encontrar su capital
- Esto requiere múltiples pasos de razonamiento
```

**El agente actúa:**
```
ACCIÓN 1: Usar herramienta "buscar_país" → Resultado: "India o China"
ACCIÓN 2: Usar herramienta "obtener_población" → Resultado: "India: 1.4B, China: 1.4B"
ACCIÓN 3: Usar herramienta "obtener_capital" → Resultado: "India: Nueva Delhi, China: Beijing"
```

**El agente observa:**
```
OBSERVAR:
- Población de India: 1.4 mil millones
- Población de China: 1.4 mil millones
- Son aproximadamente iguales; India es actualmente el más poblado
- Capital: Nueva Delhi
```

**El agente razona de nuevo:**
```
PENSAR: Tengo la información. Objetivo alcanzado.
```

**Retorna:** "Nueva Delhi (India es actualmente el país más poblado)"

---

## 🛠️ Herramientas en Agentes ReAct

**Una herramienta** es una función que el agente puede llamar:

```python
class Tool:
    def __init__(self, name: str, function: callable, description: str):
        self.name = name
        self.function = function
        self.description = description
    
    def execute(self, *args, **kwargs):
        return self.function(*args, **kwargs)
```

### Herramientas de Ejemplo

```python
# Herramienta 1: Búsqueda en Internet
search_tool = Tool(
    name="search",
    function=search_internet,
    description="Buscar información en internet"
)

# Herramienta 2: Calculadora
calculator_tool = Tool(
    name="calculate",
    function=eval_expression,
    description="Realizar cálculos matemáticos"
)

# Herramienta 3: Consulta de Base de Datos
database_tool = Tool(
    name="query_db",
    function=query_database,
    description="Consultar la base de datos de la empresa"
)
```

### Cómo el Agente Selecciona Herramientas

```
PENSAR: ¿Qué herramienta debo usar?
├─ search_tool: Buena para recopilar información
├─ calculator_tool: Buena para matemáticas
└─ database_tool: Buena para datos de la empresa

ACCIÓN: "Usaré search_tool con consulta='capital de India'"

RESULTADO: "Nueva Delhi"
```

---

## 📝 Formato ReAct

Los agentes se comunican usando un formato estructurado:

```
Thought: ¿Qué debo hacer a continuación?
Action: nombre_herramienta[argumento]
Observation: [Resultado de la herramienta]

Thought: ¿Siguiente paso?
Action: nombre_herramienta[argumento]
Observation: [Resultado de la herramienta]

...

Thought: Ahora conozco la respuesta final
Final Answer: [Respuesta]
```

### Ejemplo Real

```
Thought: Necesito encontrar la raíz cuadrada de 144 y sumar 5
Action: calculate[sqrt(144)]
Observation: 12

Thought: Ahora sumo 5
Action: calculate[12 + 5]
Observation: 17

Thought: Tengo la respuesta
Final Answer: 17
```

---

## 🎯 Script 06: Implementación de Agente ReAct

Veamos cómo se codifica:

### 1. Definir Herramientas

```python
def search_wiki(topic: str) -> str:
    """Simula búsqueda en Wikipedia"""
    return f"Información sobre {topic}..."

def calculate(expression: str) -> float:
    """Simula calculadora"""
    return eval(expression)

def get_translation(word: str, lang: str) -> str:
    """Simula herramienta de traducción"""
    translations = {
        ('hello', 'es'): 'hola',
        ('hello', 'pt'): 'olá',
    }
    return translations.get((word, lang), 'desconocido')

tools = {
    'search': search_wiki,
    'calculate': calculate,
    'translate': get_translation
}
```

### 2. Parsear Salida del Agente

```python
def parse_action(text: str) -> tuple:
    """Extraer nombre de herramienta y argumentos"""
    # Formato: "Action: nombre_herramienta[argumento]"
    
    import re
    match = re.search(r'Action:\s*(\w+)\[(.+?)\]', text)
    if match:
        tool_name = match.group(1)
        argument = match.group(2)
        return tool_name, argument
    return None, None
```

### 3. Bucle Principal del Agente

```python
def agent_loop(query: str, max_turns: int = 5) -> str:
    """Ejecutar bucle ReAct"""
    
    context = f"Pregunta del usuario: {query}\n"
    turn = 0
    
    while turn < max_turns:
        turn += 1
        
        # PENSAR
        thought = generate_thought(context, query)
        context += f"Thought: {thought}\n"
        
        # Verificar respuesta final
        if "Final Answer" in thought:
            return extract_final_answer(thought)
        
        # ACTUAR
        tool_name, argument = parse_action(thought)
        if tool_name and tool_name in tools:
            result = tools[tool_name](argument)
            context += f"Action: {tool_name}[{argument}]\n"
            
            # OBSERVAR
            observation = f"Observation: {result}\n"
            context += observation
        
        else:
            context += "Observation: Herramienta inválida o sin acción\n"
    
    return "Máximo de turnos alcanzado sin respuesta"
```

---

## 🔄 Tipos y Estrategias de Agentes

### Tipo 1: Agente Secuencial Simple

```
Tarea → Paso 1 → Paso 2 → Paso 3 → Respuesta
(Orden fijo)
```

**Cuándo usar:** Tareas secuenciales bien definidas

---

### Tipo 2: Agente Adaptativo (ReAct)

```
Tarea → Evaluar → Decidir Mejor Paso → Ejecutar → Bucle
        (Toma de decisiones dinámica)
```

**Cuándo usar:** Tareas complejas e impredecibles

---

### Tipo 3: Sistema Multi-Agente

```
Agente 1        Agente 2        Agente 3
(Especialista)  (Especialista)  (Coordinador)
   ↓               ↓               ↓
   └───────────────┴───────────────┘
         (Colaboran)
            ↓
         Respuesta
```

**Cuándo usar:** Tareas muy complejas que necesitan múltiples expertos

---

## ⚠️ Limitaciones y Desafíos de Agentes

### 1. **Alucinación**
```
Agente: "La capital de Francia es Londres"
Problema: Información incorrecta
Solución: Fundamentar agente con herramientas verificadas
```

### 2. **Bucles Infinitos**
```
Agente: Buscar → Mal resultado → Buscar de nuevo → ...
Problema: Nunca termina
Solución: Añadir límite max_turns
```

### 3. **Mal Uso de Herramientas**
```
Agente: Usa herramienta de búsqueda para calcular matemáticas
Problema: Herramienta incorrecta para la tarea
Solución: Mejores descripciones de herramientas, entrenamiento del agente
```

### 4. **Costo**
```
Cada llamada a herramienta = tiempo + dinero + latencia
Problema: Demasiadas llamadas = lento, caro
Solución: Optimizar selección de herramientas
```

---

## 🎓 Cuándo Usar Agentes

### ✅ Buenos Casos de Uso

- **Investigación multi-paso:** "Encuentra los 3 mejores papers sobre tema X, resume cada uno"
- **Flujos de trabajo complejos:** "Crea informe, obtén aprobaciones, envía notificación"
- **Integración de herramientas:** Llamadas API, consultas de base de datos, cálculos
- **Problemas adaptativos:** Número desconocido de pasos

### ❌ No Adecuado Para

- **Consultas de un paso:** "¿Cuánto es 2+2?" (No se necesita agente)
- **Recuperación simple:** "Encuentra documento X" (Consulta directa es más rápida)
- **Tiempo real crítico:** Los agentes añaden latencia
- **Operaciones de alto costo:** Cada decisión cuesta dinero

---

## 🚀 Extendiendo Agentes

### Añadir Nueva Herramienta

```python
def new_tool_function(arg1: str, arg2: str) -> str:
    """Tu herramienta personalizada"""
    return f"Resultado para {arg1} y {arg2}"

# Registrar herramienta
tools['new_tool'] = new_tool_function

# El agente ahora puede llamarla:
# "Action: new_tool[arg1_value, arg2_value]"
```

### Mejorar Toma de Decisiones

```python
# Actual: Selección aleatoria de herramienta
# Mejor: Puntuar cada herramienta por relevancia
def score_tools(query: str, available_tools: list) -> dict:
    scores = {}
    for tool in available_tools:
        score = similarity(query, tool.description)
        scores[tool.name] = score
    return scores

best_tool = max(scores, key=scores.get)
```

### Añadir Memoria

```python
class AgentWithMemory:
    def __init__(self):
        self.memory = {}
    
    def remember(self, key: str, value: str):
        self.memory[key] = value
    
    def recall(self, key: str) -> str:
        return self.memory.get(key, "No encontrado")
```

---

## 📊 Rendimiento de Agentes

### Métricas a Rastrear

| Métrica | Qué Mide | Objetivo |
|---------|----------|----------|
| **Tasa de Éxito** | % tareas completadas | >90% |
| **Pasos Promedio** | Turnos promedio para resolver | <5 |
| **Correctitud** | % respuestas correctas | >95% |
| **Latencia** | Tiempo por tarea | <1s |
| **Costo** | Llamadas a herramientas × costo | $0.01-0.10 |

---

## 🎯 Puntos Clave

✅ **Los agentes permiten razonamiento autónomo multi-paso**  
✅ **Patrón ReAct: Pensar → Actuar → Observar → Bucle**  
✅ **Las herramientas extienden las capacidades del agente**  
✅ **Las herramientas fundamentadas previenen alucinaciones**  
✅ **Valor real para tareas complejas y adaptativas**  
✅ **Pero añaden complejidad y latencia**  

---

## 📚 Lecturas Adicionales

- Script 06: [`06_react_agent_bonus.py`](../../06_react_agent_bonus.py)
- Capítulo 14 del libro: Protocolos Agénticos (MCP)
- Capítulo 15 del libro: Evaluación Crítica
- Integración: [Script 09](QUICKSTART_SCRIPT_09.md)

---

**¿Listo para construir con agentes? 🤖**

¡Prueba Script 06, luego intégralo en Script 09!
