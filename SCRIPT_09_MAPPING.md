# Script 09 : Mapping Détaillé aux Chapitres du Livre

## 📍 Comment Chaque Section du Script Correspond au Livre

Ce document montre **exactement où** chaque concept du livre est illustré dans le Script 09.

---

## Chapitre 11 : Stratégies de Génération et Prompting

### 11.1 Prompting : Zero-shot, Few-shot, Chain-of-Thought

**Où dans le script** :
```python
def _simulate_llm_reasoning(self, prompt: str, step_count: int = 1) -> str:
    """
    Simule un LLM qui fait du prompting structuré.
    
    Cette fonction implémente les techniques du Chapitre 11 :
    - Chain-of-Thought : "Thought: ... Action: ..."
    - Structuration explicite des étapes
    - Prompting zéro-shot (pas d'exemples dans la démo)
    """
```

**Code du Livre (Ch. 11)** :
```
Prompt zéro-shot :
"Question: Qu'est-ce qu'un Transformer ?"

Prompt avec CoT :
"Réfléchis étape par étape.
Pensée: ...
Action: ...
Observation: ..."
```

**Dans le Script** :
```python
# Ligne 119-125 : Détection de patterns (CoT implicite)
if "qu'est-ce" in prompt_lower:
    return f"Thought: Je dois chercher...\nAction: search(...)"

# Ligne 244-260 : Parsing du pattern Thought/Action
thought_match = re.search(r"Thought:\s*(.+?)(?:\n|$)", llm_response)
action_match = re.search(r"Action:\s*(\w+)\((.*?)\)", response)
```

**Lien** : ✅ Le script montre comment structurer un prompting en étapes claires (CoT).

---

### 11.2 Température et Sampling

**Où dans le script** :
```python
# Le script SIMULE la tempétature (pas d'implémentation réelle)
# Mais permet de voir comment différentes "stratégies" (greedy vs sampling)
# pourraient être implémentées.
```

**Code du Livre (Ch. 11)** :
```python
temperature = 0.7
logits = logits / temperature
probs = softmax(logits)
next_token = sample(probs)  # vs argmax(probs) pour greedy
```

**Extension du Script** :
Pour intégrer la température réelle :
```python
def _simulate_llm_reasoning_with_temp(self, prompt, temperature=0.7):
    # Simuler différentes réponses selon la T
    if temperature < 0.3:
        return "Réponse déterministe"
    elif temperature > 1.0:
        return "Réponse créative et variée"
```

**Lien** : ⚠️ Le script simplifie la température (concept mentionné mais pas implémenté).

---

## Chapitre 12 : Modèles de Raisonnement et Évaluation

### 12.1 Pass@k et Pass^k

**Où dans le script** :
```python
class AssistantEvaluator:
    """Évaluation de la qualité des réponses de l'assistant."""
    
    @staticmethod
    def evaluate_response(question, response, expected_answer=None):
        # Métriques clés
        evaluation = {
            "iterations": response["iterations"],
            "confidence": response["confidence"],
            "success": response["confidence"] > 0.5
        }
```

**Code du Livre (Ch. 12)** :
```
Pass@k = 1 - (1 - p)^k

Exemple :
- p_success = 0.6 (probabilité de succès d'une tentative)
- Pass@1 = 0.6 (une seule tentative)
- Pass@5 = 1 - (1-0.6)^5 = 1 - 0.01024 = 98.976%
```

**Dans le Script (Ligne 356-366)** :
```python
success_count = sum(
    1 for step in steps 
    if "❌" not in step["observation"]
)
confidence = min(1.0, success_count / len(steps))
```

**Calcul de confiance simulé** :
- Chaque étape réussie = +confiance
- Implicitement : `confidence ≈ (succès / iterations)`

**Lien** : ✅ Le script montre comment évaluer la réussite et la confiance.

---

### 12.2 Self-Consistency

**Où dans le script** :
```python
@staticmethod
def self_consistency_check(agent, question, num_samples=3) -> Dict:
    """
    Vérifier la cohérence des réponses (self-consistency).
    
    Génère plusieurs réponses et mesure leur accord.
    Concept du chapitre 12.
    """
    answers = []
    for i in range(num_samples):
        response = agent.run(question, verbose=False)
        answers.append(response["answer"])
    
    # Calculer la fréquence de chaque réponse
    from collections import Counter
    answer_counts = Counter(answers)
    most_common = answer_counts.most_common(1)[0]
    
    consistency_score = most_common[1] / num_samples
    
    return {
        "consistency_score": consistency_score,
        "unique_answers": len(answer_counts)
    }
```

**Code du Livre (Ch. 12)** :
```
Self-consistency = "Pose la question k fois, compte les réponses identiques"

Score = réponses identiques / k

Exemple :
- k=3 essais
- Réponses : [A, A, A]
- Score = 3/3 = 100% (très cohérent)

Exemple 2 :
- Réponses : [A, B, C]
- Score = 1/3 = 33% (très incohérent)
```

**Dans le Script (Ligne 379-399)** :
```python
for i in range(num_samples):
    response = agent.run(question, verbose=False)
    answers.append(response["answer"])

answer_counts = Counter(answers)
most_common = answer_counts.most_common(1)[0]
consistency_score = most_common[1] / num_samples
```

**Lien** : ✅ Le script implémente exactement self-consistency tel que décrit au Ch. 12.

---

## Chapitre 13 : Systèmes Augmentés et RAG

### 13.1 Indexation Vectorielle

**Où dans le script** :
```python
class RAGSystem:
    """
    Système RAG complet avec vectorisation TF-IDF et recherche par similarité.
    
    En production, remplacer par des embeddings denses (OpenAI, E5, etc.)
    et une base vectorielle (Pinecone, Weaviate, ChromaDB).
    """
    
    def index_documents(self):
        """Indexer tous les documents (création de l'index vectoriel)."""
        if not self.documents:
            raise ValueError("Aucun document à indexer")
        
        # Vectorisation TF-IDF
        texts = [doc.content for doc in self.documents]
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.doc_vectors = self.vectorizer.fit_transform(texts)
```

**Code du Livre (Ch. 13)** :
```
Pipeline RAG :
1. Indexation : Convertir documents en vecteurs
2. Retrieval : Chercher les K documents pertinents
3. Augmentation : Injecter contexte dans le prompt
4. Génération : LLM répond en s'appuyant sur le contexte
```

**Dans le Script (Ligne 73-100)** :
```python
# 1. Indexation
self.vectorizer = TfidfVectorizer(...)
self.doc_vectors = self.vectorizer.fit_transform(texts)

# 2. Retrieval
def retrieve(self, query, top_k=3):
    query_vec = self.vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [(self.documents[idx], similarities[idx]) for idx in top_indices]
```

**Lien** : ✅ Le script montre un RAG complet (indexation + retrieval).

---

### 13.2 Similarité Cosinus

**Où dans le script** :
```python
from sklearn.metrics.pairwise import cosine_similarity

def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
    # Vectorisation de la requête
    query_vec = self.vectorizer.transform([query])
    
    # Calcul de similarité cosinus
    similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
```

**Code du Livre (Ch. 13)** :
```
Similarité cosinus entre deux vecteurs u et v :

sim(u, v) = (u · v) / (||u|| * ||v||)

Résultat : score entre 0 et 1
- 0 : totalement différent
- 1 : identique
```

**Lien** : ✅ Le script utilise exactement la similarité cosinus.

---

## Chapitre 14 : Protocoles Standards Agentiques

### 14.1 Pattern ReAct

**Où dans le script** :
```python
class ReActAgent:
    """
    Agent autonome avec pattern ReAct (Reason + Act).
    
    Boucle : Thought → Action → Observation → ... → Final Answer
    """
    
    def run(self, question: str, max_iterations: int = 5) -> Dict:
        for iteration in range(1, max_iterations + 1):
            # 1. Pensée (Thought)
            llm_response = self._simulate_llm_reasoning(question, step_count=iteration)
            thought = re.search(r"Thought:\s*(.+?)(?:\n|$)", llm_response).group(1)
            
            # 2. Action
            action_parsed = self._parse_action(llm_response)
            tool_name, params = action_parsed
            
            # 3. Observation
            observation = self.tools.execute(tool_name, **params)
            
            # 4. Boucle ou Réponse
            if "final answer:" in llm_response.lower():
                return {"answer": answer, ...}
```

**Code du Livre (Ch. 14)** :
```
Boucle ReAct :

1. Pensée : Que dois-je faire ?
   "Thought: Je dois chercher des informations sur le Transformer"

2. Action : Quel outil utiliser ?
   "Action: search(query='Transformer')"

3. Observation : Quel résultat ?
   "Observation: [Documents pertinents...]"

4. [Continuer ou arrêter]
   "Final Answer: Les Transformers sont..."
```

**Dans le Script (Ligne 228-290)** :
```python
# Chaque itération suit le pattern ReAct :
for iteration in range(1, max_iterations + 1):
    print(f"💭 Pensée : {thought}")           # Thought
    print(f"🔧 Action : {action_str}")        # Action
    print(f"📊 Observation : {observation}")  # Observation
    # Boucle continue jusqu'à "Final Answer"
```

**Lien** : ✅ Le script implémente parfaitement la boucle ReAct.

---

### 14.2 Tool Calling et Registration

**Où dans le script** :
```python
class ToolRegistry:
    """Registre d'outils disponibles pour l'agent."""
    
    def register(self, name: str, description: str, func):
        """Enregistrer un nouvel outil."""
        self.tools[name] = {
            "name": name,
            "description": description,
            "func": func
        }
    
    def execute(self, tool_name: str, **kwargs) -> str:
        """Exécuter un outil avec les arguments fournis."""
        if tool_name not in self.tools:
            return f"❌ Outil '{tool_name}' inconnu"
        
        result = self.tools[tool_name]["func"](**kwargs)
        return str(result)
```

**Code du Livre (Ch. 14)** :
```
Tool Calling (chapitre 14) :

1. Définir les outils disponibles
   - Nom : "search"
   - Description : "Recherche dans la base de connaissances"
   - Fonction : fonction_recherche()

2. Agent appelle l'outil
   "Action: search(query='Transformer')"

3. Exécution
   result = tools.execute("search", query="Transformer")

4. Observation retournée au LLM
```

**Dans le Script (Ligne 181-197)** :
```python
# Registration (Chapitre 14)
self.tools.register(
    "calculator",
    "Évalue une expression mathématique",
    tool_calculator
)

# Tool Calling
tool_name, params = self._parse_action(llm_response)
observation = self.tools.execute(tool_name, **params)
```

**Lien** : ✅ Le script montre tool registration et tool calling.

---

### 14.3 Model Context Protocol (MCP)

**Où dans le script** :
```python
# Le script simule le MCP avec le ToolRegistry
# En production, utiliser le vrai MCP :

# from mcp.server import MCPServer
# server = MCPServer("agent")
# @server.call_tool
# def my_tool(param1: str) -> str:
#     return "résultat"
```

**Code du Livre (Ch. 14)** :
```
MCP = Standard pour définir les outils

Spécification :
{
    "name": "search",
    "description": "Recherche documents",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    }
}
```

**Lien** : ⚠️ Le script simule MCP (voir annexe pour intégration réelle).

---

## Chapitre 15 : Mise en Production

### 15.1 Gestion d'Erreurs et Robustesse

**Où dans le script** :
```python
class ToolRegistry:
    def execute(self, tool_name: str, **kwargs) -> str:
        """Exécuter un outil avec les arguments fournis."""
        if tool_name not in self.tools:
            return f"❌ Outil '{tool_name}' inconnu"  # Error handling
        
        try:
            result = self.tools[tool_name]["func"](**kwargs)
            return str(result)
        except Exception as e:
            return f"❌ Erreur lors de l'exécution: {e}"  # Exception handling
```

**Code du Livre (Ch. 15)** :
```
Production checklist :

1. ✅ Gestion d'erreurs (try/except)
2. ✅ Validation des entrées
3. ✅ Logging structuré
4. ✅ Timeouts (max_iterations ici)
5. ✅ Fallbacks et retry
6. ✅ Monitoring et métriques
```

**Lien** : ✅ Le script implémente les éléments clés pour la production.

---

### 15.2 Logging et Observation

**Où dans le script** :
```python
def run(self, question: str, max_iterations: int = 5, verbose: bool = True):
    if verbose:
        print(f"\n{'='*70}")
        print(f"🤖 Question : {question}")
        print(f"{'='*70}")
    
    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"\n{'─'*70}")
            print(f"⏳ Itération {iteration}/{max_iterations}")
            print(f"{'─'*70}")
            print(f"💭 Pensée : {thought}")
            print(f"🔧 Action : {action_str}")
            print(f"📊 Observation : {observation[:200]}...")
```

**Code du Livre (Ch. 15)** :
```
Logging production :

1. Chaque étape tracée (Thought, Action, Observation)
2. Timestamps et IDs de session
3. Métriques : latence, tokens, coût
4. Erreurs et warnings
5. Dashboard de monitoring
```

**Lien** : ✅ Le script montre le logging verbeux (production-ready).

---

### 15.3 Évaluation et Métriques

**Où dans le script** :
```python
class AssistantEvaluator:
    @staticmethod
    def evaluate_response(question, response, expected_answer=None):
        evaluation = {
            "iterations": response["iterations"],
            "confidence": response["confidence"],
            "success": response["confidence"] > 0.5
        }
        return evaluation
    
    @staticmethod
    def self_consistency_check(agent, question, num_samples=3):
        # Générer k réponses
        # Mesurer la cohérence
        # Retourner les métriques
```

**Code du Livre (Ch. 15)** :
```
Métriques en production :

1. ✅ Success rate (% de questions répondues)
2. ✅ Latency (P50, P95, P99)
3. ✅ Quality (confiance, cohérence)
4. ✅ Cost (nombre de tokens, appels API)
5. ✅ User feedback (ratings, corrections)
```

**Lien** : ✅ Le script mesure success, confiance, et cohérence.

---

## 🎯 Résumé : Couverture Complète

| Chapitre | Concept | Implémenté ? | Où dans le script ? |
|----------|---------|-------------|-------------------|
| 11 | Prompting (CoT) | ✅ | `_simulate_llm_reasoning()` |
| 11 | Température | ⚠️ Simulé | Extension possible |
| 12 | Pass@k / Confiance | ✅ | `_calculate_confidence()` |
| 12 | Self-consistency | ✅ | `self_consistency_check()` |
| 13 | RAG Pipeline | ✅ | `RAGSystem` classe complète |
| 13 | Similarité Cosinus | ✅ | `retrieve()` avec cosine_similarity |
| 14 | Pattern ReAct | ✅ | Boucle complète `run()` |
| 14 | Tool Calling | ✅ | `ToolRegistry.execute()` |
| 14 | Tool Registration | ✅ | `ToolRegistry.register()` |
| 15 | Gestion d'erreurs | ✅ | try/except dans `execute()` |
| 15 | Logging | ✅ | `verbose=True` avec print() |
| 15 | Métriques | ✅ | `AssistantEvaluator` |

---

## 🚀 Comment Étendre Chaque Concept

### Prompting (Ch. 11)
Ajouter Few-shot examples :
```python
few_shot = """
Exemple 1 : Q: "Combien ?" → A: "calculer"
Exemple 2 : Q: "Qu'est-ce ?" → A: "chercher"
Question : ...
"""
```

### Évaluation (Ch. 12)
Ajouter métriques ROUGE/BERTScore :
```python
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
scores = scorer.score(reference, hypothesis)
```

### RAG (Ch. 13)
Intégrer embeddings denses :
```python
from openai import OpenAI
client = OpenAI()
embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=text
).data[0].embedding
```

### Agents (Ch. 14)
Ajouter le vrai MCP :
```python
from mcp.server import Server
server = Server("agent")
@server.call_tool("search", query: str)
async def search_tool(query: str):
    return retrieve(query)
```

---

**Chaque ligne de code dans Script 09 correspond à un concept du livre. C'est votre pont entre théorie et pratique ! 🌉**
