# 🤖 Intégration des Agents ReAct

> 🌍 **English** | 📖 **[Version Française](./REACT_AGENT_INTEGRATION.md)**

## 📍 Navigation Rapide

- **📖 Lire d'abord:** [PEDAGOGICAL_JOURNEY.md](./PEDAGOGICAL_JOURNEY.md) - Où s'intègrent les agents
- **⚡ Démarrage rapide:** [QUICKSTART_SCRIPT_09.md](./QUICKSTART_SCRIPT_09.md) - Lancer Script 06
- **🌍 English:** [English Version](../en/REACT_AGENT_INTEGRATION.md)

---

## 🎯 Qu'est-ce qu'un Agent ReAct ?

**ReAct** = **Re**asoning + **Act**ing

Un agent qui :
1. **Réfléchit** (Reasoning) - Analyse le problème
2. **Agit** (Acting) - Utilise un outil
3. **Observe** - Reçoit le résultat
4. **Boucle** - Répète jusqu'à répondre

### Exemple : Répondre à une question complexe

```
Q: "Quel est le capital du pays le plus peuplé ?"

Agent Réfléchit:
"Je dois d'abord trouver le pays le plus peuplé"

Agent Agit:
Utilise outil "search" → "L'Inde et la Chine"

Agent Observe:
"Les deux ont ~1.4 milliards d'habitants"

Agent Réfléchit:
"L'Inde est actuellement le plus peuplé"

Agent Agit:
Utilise outil "get_capital" → "New Delhi"

Agent Observe:
"La réponse est New Delhi"

Agent: "Le capital du pays le plus peuplé est New Delhi"
```

---

## 🏗️ Architecture de la Boucle ReAct

```
┌─────────────────────┐
│  Question initiale  │
└──────────┬──────────┘
           │
    ┌──────▼──────┐
    │ THINK       │ ← Analyser l'état
    │ (Reasoning) │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ ACT         │ ← Choisir outil
    │ (Tool call) │
    └──────┬──────┘
           │
    ┌──────▼──────────┐
    │ OBSERVE         │ ← Traiter résultat
    │ (Observation)   │
    └──────┬──────────┘
           │
    Fini ? │
    ┌──────┴─────┐
    │ OUI  │ NON │
    │      │     │ → Retour à THINK
    │      │     │
    ▼      ▼
 RÉPONSE  BOUCLE
```

---

## 🛠️ Outils dans Script 06 & 09

### Outil 1: Calculator (Calcul)
```python
def tool_calculator(expression: str) -> str:
    """Effectue des calculs mathématiques"""
    try:
        result = eval(expression)
        return f"Résultat: {result}"
    except:
        return "Erreur de calcul"

# Utilisation par l'agent:
# Action: calculator[2 + 2]
# → Résultat: 4
```

### Outil 2: Search (Recherche)
```python
def tool_search(query: str) -> str:
    """Cherche dans la base de connaissances"""
    # Implémenté via RAGSystem
    results = rag.retrieve(query, top_k=3)
    return format_results(results)

# Utilisation par l'agent:
# Action: search[transformer attention mechanism]
# → Résultat: Documents pertinents...
```

### Outil 3: Current Time (Horloge)
```python
def tool_current_time() -> str:
    """Retourne l'heure actuelle"""
    from datetime import datetime
    return datetime.now().isoformat()

# Utilisation par l'agent:
# Action: current_time[]
# → Résultat: 2025-01-15T14:32:00
```

### Outil 4: Summarize (Résumé)
```python
def tool_summarize(text: str) -> str:
    """Crée un résumé d'un texte"""
    lines = text.split(".")
    return ". ".join(lines[:2])  # Simplifié

# Utilisation par l'agent:
# Action: summarize[Long texte...]
# → Résultat: Résumé condensé
```

---

## 📋 Format Agent ReAct

Les agents communiquent en ce format structuré :

```
Thought: Que dois-je faire maintenant ?
Action: tool_name[param1, param2]
Observation: [Résultat de l'outil]

Thought: Prochaine étape ?
Action: tool_name[param]
Observation: [Résultat]

Thought: J'ai la réponse
Final Answer: La réponse est...
```

### Exemple complet:

```
Thought: L'utilisateur demande le capital d'un pays populeux.
Je dois d'abord identifier le pays le plus peuplé.
Action: search[most populous country]
Observation: L'Inde a 1.42 milliards d'habitants, la Chine 1.41 milliards.

Thought: L'Inde est le plus peuplé. Maintenant je dois trouver son capital.
Action: search[capital of India]
Observation: La capitale de l'Inde est New Delhi.

Thought: J'ai obtenu la réponse complète.
Final Answer: Le capital du pays le plus peuplé est New Delhi
(l'Inde avec 1.42 milliards d'habitants).
```

---

## 🔄 Patterns de Sélection d'Outils

### Pattern 1: Sélection Basée sur les Mots-clés

```python
if "calculer" in thought.lower():
    use_tool("calculator")
elif "capital" in thought.lower():
    use_tool("search")
elif "heure" in thought.lower():
    use_tool("current_time")
```

### Pattern 2: Sélection par Scoring

```python
scores = {}
for tool_name, tool_desc in available_tools.items():
    score = similarity(thought, tool_desc)
    scores[tool_name] = score

best_tool = max(scores, key=scores.get)
```

### Pattern 3: Sélection par LLM (Avancé)

```python
# Utiliser un LLM pour choisir
prompt = f"""
Pensée: {thought}
Outils disponibles: {available_tools}
Quel outil utiliserais-tu ?
"""
response = llm(prompt)
selected_tool = parse_tool(response)
```

---

## 🎓 Concepts Clés

### 1. Autonomie
L'agent décide lui-même de ce qu'il faut faire. Pas de direction humaine étape par étape.

### 2. Itération
L'agent peut faire plusieurs étapes. Il n'y a pas une seule réponse directe.

### 3. Outils
Les outils étendent les capacités de l'agent au-delà du LLM seul.

### 4. Transparence
Chaque étape est enregistrée et visible (le "trace").

### 5. Arrêt Automatique
L'agent sait quand il a terminé (détection "Final Answer").

---

## ⚠️ Limitations & Défis

### 1. Hallucination
L'agent peut inventer des réponses même avec des outils.

**Solution:** Valider les résultats de l'outil.

### 2. Boucles Infinies
L'agent peut se bloquer en boucle.

**Solution:** Limiter `max_iterations`.

### 3. Sélection Mauvaise Outil
L'agent peut choisir le mauvais outil.

**Solution:** Améliorer les descriptions d'outils.

### 4. Coût
Chaque outil = temps + argent.

**Solution:** Optimiser la sélection d'outils.

---

## 💡 Extensions Pratiques

### Ajouter un Nouvel Outil

```python
# 1. Définir la fonction
def tool_weather(city: str) -> str:
    """Récupère la météo d'une ville"""
    return f"Paris: Ensoleillé, 22°C"

# 2. L'enregistrer
agent.tools.register(
    name="weather",
    description="Récupère la météo pour une ville",
    function=tool_weather
)

# Maintenant l'agent peut l'utiliser!
# Action: weather[Paris]
```

### Améliorer la Sélection d'Outils

```python
# Ajouter des descriptions détaillées
tool_descriptions = {
    "calculator": "Pour les calculs math: additions, soustractions, etc.",
    "search": "Pour trouver des informations : concepts, définitions",
    "weather": "Pour les conditions météo d'une ville",
}

# L'agent les utilise pour meilleures décisions
```

### Ajouter la Mémoire

```python
# Enregistrer les décisions précédentes
class AgentWithMemory:
    def __init__(self):
        self.history = []
    
    def remember(self, action, result):
        self.history.append({"action": action, "result": result})
    
    def recall_similar(self, new_action):
        # Chercher une action similaire dans l'historique
        return [h for h in self.history if similar(h["action"], new_action)]
```

---

## 🎯 Cas d'Utilisation

### ✅ Parfait Pour:
- Recherche multi-étapes
- Résolution de problèmes complexes
- Questions nécessitant plusieurs sources
- Tâches avec outils spécialisés

### ❌ Pas Recommandé Pour:
- Questions simples (trop lent)
- Temps réel critique (trop coûteux)
- Décisions très critiques (risque hallucination)

---

## 🚀 Intégration Avec Vrai LLM

### Avec OpenAI

```python
from openai import OpenAI

client = OpenAI()

def agent_with_openai(query):
    context = ""
    for iteration in range(3):
        # Utiliser GPT pour réfléchir
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": f"{context}\nQ: {query}"}
            ]
        )
        thought = response.choices[0].message.content
        
        # Parser l'action
        tool_name, params = parse_action(thought)
        
        # Exécuter
        observation = execute_tool(tool_name, params)
        
        context += f"\n{thought}\nObservation: {observation}"
    
    return context
```

---

## 📚 Relation avec Script 09

Script 09 intègre tout cela :
- La **boucle réacte** complète
- Les **4 outils** disponibles
- L'**évaluation** des réponses
- La **persistence** de trace

Voir [QUICKSTART_SCRIPT_09.md](./QUICKSTART_SCRIPT_09.md) pour les détails.

---

**Prêt à créer des agents autonomes? 🤖**

Voir [Lecture Suivante](./LLAMAINDEX_GUIDE.md) pour les systèmes RAG production.
