# 🔗 Code ↔ Concept Mapping : Script 09

> 🌍 **English** | 📖 **[Version Française](./SCRIPT_09_MAPPING.md)**

## 📍 Navigation Rapide

- **📖 Lire d'abord:** [PEDAGOGICAL_JOURNEY.md](./PEDAGOGICAL_JOURNEY.md) - Théorie
- **⚡ Démarrage rapide:** [QUICKSTART_SCRIPT_09.md](./QUICKSTART_SCRIPT_09.md) - Exécuter
- **🏗️ Architecture:** [INDEX_SCRIPT_09.md](./INDEX_SCRIPT_09.md) - Structure

---

## 🎯 Mapping Chapitres

### Chapitre 11 : Stratégies de Génération et Prompting

**Où dans Script 09?** → Fonction `_simulate_llm_reasoning()`

**Concept du livre:**
- Zero-shot vs Few-shot vs Chain-of-Thought
- Température et sampling
- Construction du prompt

**Code du script:**
```python
def _simulate_llm_reasoning(query, context):
    """Simule le raisonnement d'un LLM"""
    
    # Construction du prompt (prompting)
    prompt = f"""
    Contexte: {context}
    Question: {query}
    
    Pensée (réfléchissez d'abord):
    """
    
    # Le système génère une réponse
    # Température: contrôle créativité
```

**Ce que vous apprenez:**
- Structure du prompting
- Chain-of-Thought en action
- Impact du contexte sur la réponse

---

### Chapitre 12 : Modèles de Raisonnement et Évaluation

**Où dans Script 09?** → Classes `AssistantEvaluator` et métriques

**Concept du livre:**
- Pass@k : probabilité d'au moins 1 succès en k essais
- Self-consistency : cohérence des réponses multiples
- Confiance et métriques de qualité

**Code du script:**
```python
class AssistantEvaluator:
    def evaluate_response(self, question, response, context):
        """Évalue la qualité de la réponse"""
        # Calcule plusieurs métriques:
        # - Longueur
        # - Couverture du contexte
        # - Pertinence
        # - Cohérence
        
        # Retourne un score 0-100
        return {"score": 78, "metrics": {...}}
    
    def self_consistency_check(self, agent, question, num_samples=3):
        """Teste si agent répond toujours pareil"""
        answers = [agent.run(question) for _ in range(num_samples)]
        # Mesure : combien de fois même réponse ?
        consistency_score = self.measure_consistency(answers)
        return {"consistency": 0.85}
```

**Ce que vous apprenez:**
- Évaluation multi-critères
- Self-consistency en pratique
- Mesure de la qualité

---

### Chapitre 13 : Systèmes Augmentés et RAG

**Où dans Script 09?** → Classe `RAGSystem`

**Concept du livre:**
- Retrieval-Augmented Generation
- Indexation vectorielle
- Top-k retrieval

**Code du script:**
```python
class RAGSystem:
    def __init__(self):
        self.documents = {}
        self.index = {}
    
    def add_document(self, text, metadata):
        """Ajoute un document à la base"""
        doc_id = f"doc_{len(self.documents)}"
        self.documents[doc_id] = {"text": text, "meta": metadata}
    
    def index_documents(self):
        """Indexe tous les documents"""
        # Créer des embeddings simples
        # Stocker pour retrieval rapide
    
    def retrieve(self, query, top_k=3):
        """Récupère les top-k docs pertinents"""
        # 1. Embedding de la requête
        # 2. Similarité cosinus avec tous les docs
        # 3. Retourner top-k
        
        return [
            {"doc_id": "doc_1", "score": 0.89, "text": "..."},
            ...
        ]
```

**Ce que vous apprenez:**
- Architecture RAG complète
- Indexation pratique
- Retrieval et ranking

---

### Chapitre 14 : Protocoles Agentiques (ReAct)

**Où dans Script 09?** → Classe `ReActAgent`

**Concept du livre:**
- Pattern ReAct : Reasoning + Acting
- Boucle autonome
- Tool calling et registration

**Code du script:**
```python
class ReActAgent:
    def __init__(self, rag_system):
        self.rag = rag_system
        self.tools = ToolRegistry()
        self.max_iterations = 3
    
    def run(self, query):
        """Exécute la boucle ReAct"""
        for iteration in range(self.max_iterations):
            # THINK: Raisonnement
            thought = self._generate_thought(query, context)
            
            # ACT: Sélection d'outil
            tool_name, params = self._parse_action(thought)
            
            # Exécuter l'outil
            if tool_name in self.tools.registry:
                observation = self.tools.execute(tool_name, params)
            
            # Vérifier si fini
            if "Final Answer" in thought:
                return {"response": thought, "iterations": iteration}
            
            # Sinon, continuer la boucle
```

**Ce que vous apprenez:**
- Boucle autonome complète
- Sélection d'outils
- Itération jusqu'à convergence

---

### Chapitre 15 : Mise en Production

**Où dans Script 09?** → Gestion d'erreurs, intégration, monitoring

**Concept du livre:**
- Gestion d'erreurs robuste
- Logging et monitoring
- Évaluation continu

**Code du script:**
```python
def main():
    """Orchestration complète"""
    try:
        # 1. Initialiser RAG
        rag = RAGSystem()
        
        # 2. Charger documents
        # 3. Indexer
        
        # 4. Créer agent
        agent = ReActAgent(rag_system=rag)
        
        # 5. Enregistrer outils
        agent.tools.register("calculator", "Calculs", tool_calculator)
        
        # 6. Traiter questions
        for question in questions:
            try:
                response = agent.run(question)
                # 7. Évaluer
                metrics = evaluator.evaluate_response(...)
                # 8. Logger résultats
                log_result(response, metrics)
            except Exception as e:
                handle_error(e)
                continue
    
    except Exception as e:
        log_error(e)
        return False
```

**Ce que vous apprenez:**
- Orchestration de système complet
- Gestion d'erreurs robuste
- Monitoring et logging

---

## 📊 Tableau de Synthèse

| Chapitre | Concept | Classe/Fonction | Ligne clé |
|----------|---------|-----------------|-----------|
| 11 | Prompting | `_simulate_llm_reasoning()` | Construction du prompt |
| 12 | Évaluation | `AssistantEvaluator` | `.evaluate_response()` |
| 13 | RAG | `RAGSystem` | `.retrieve()` |
| 14 | Agents | `ReActAgent` | `.run()` |
| 15 | Production | `main()` | Try-catch + logging |

---

## 🎯 Comment Étudier Ce Mapping

### Approche 1 : Chapitre par Chapitre
1. Lire le chapitre du livre
2. Venir voir la section correspondante ici
3. Trouver le code dans le script
4. Exécuter et observer

### Approche 2 : Code d'Abord
1. Ouvrir `09_mini_assistant_complet.py`
2. Lire une fonction
3. Chercher ici pour le contexte
4. Relire le chapitre correspondant

### Approche 3 : Question de Debug
1. Vous avez une question de débogage
2. Trouver le chapitre pertinent ici
3. Consulter le code et le concept
4. Comprendre et corriger

---

## 🔍 Index des Concepts

### A - B - C

- **Agent Autonome** → Ch. 14 (Agents)
- **Attention** → Ch. 3, Script 02
- **Base de Connaissances** → Ch. 13 (RAG)
- **Beam Search** → Ch. 11 (Génération)
- **BLEU** → Ch. 12 (Évaluation)
- **BPE** → Ch. 2, Script 01
- **Calibration** → Ch. 11 (Prompting)
- **Chain-of-Thought** → Ch. 11, Script 09

### D - E - F

- **DPO** → Ch. 6
- **Embeddings** → Ch. 2, Script 01
- **Évaluation** → Ch. 12, Script 05, 09
- **Few-shot** → Ch. 11, Script 09
- **Fine-tuning** → Ch. 9, Script 08
- **Function Calling** → Ch. 14, Script 06, 09

### G - H - I

- **Génération** → Ch. 11, Script 03, 09
- **Grounding** → Ch. 13 (RAG)
- **Hyperparamètres** → Ch. 11 (Température, Top-k)
- **Indexation** → Ch. 13 (RAG)

### J - K - L

- **KV-Cache** → Ch. 11
- **LlamaIndex** → Script 07
- **LoRA** → Ch. 9, Script 08
- **Loss** → Ch. 7

### M - N - O

- **MCP** → Ch. 14 (Model Context Protocol)
- **Multi-head Attention** → Ch. 3, Script 02
- **Observation** (ReAct) → Ch. 14, Script 06, 09
- **Optimisation** → Ch. 5-8

### P - Q - R

- **Pass@k** → Ch. 12, Script 05, 09
- **Perplexité** → Ch. 7
- **Prompting** → Ch. 11, Script 09
- **QLoRA** → Ch. 9, Script 08
- **RAG** → Ch. 13, Script 04, 07, 09
- **ReAct** → Ch. 14, Script 06, 09
- **Retrieval** → Ch. 13, Script 04, 07, 09
- **RLHF** → Ch. 6
- **ROUGE** → Ch. 12

### S - T - U

- **Self-Attention** → Ch. 3, Script 02
- **Self-Consistency** → Ch. 12, Script 05, 09
- **Softmax** → Ch. 3, Script 03
- **Température** → Ch. 11, Script 03, 09
- **Tokenization** → Ch. 2, Script 01
- **Tool Calling** → Ch. 14, Script 06, 09
- **Top-k / Top-p** → Ch. 11

### V - W - Z

- **Vectorisation** → Ch. 13 (RAG)
- **Zero-shot** → Ch. 11, Script 09

---

## 💡 Conseils de Lecture

1. **Débutant** : Lire dans l'ordre Chapitre 11 → 15
2. **Intermédiaire** : Commencer par le code, puis consulter ici
3. **Avancé** : Modifier le code et vérifier votre compréhension

---

**Bon apprentissage ! 🎓**
