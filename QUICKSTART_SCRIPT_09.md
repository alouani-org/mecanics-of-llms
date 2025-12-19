# Quick Start : Script 09 - Mini-Assistant Complet

## 🏆 C'est quoi ce script ?

Le **script 09** est le **projet intégrateur final** qui assemble **tous les concepts des chapitres 11-15** du livre en un système cohérent et fonctionnelle.

Il démontre comment un **assistant autonome moderne** fonctionne réellement :
- ✅ Recherche dans une base de connaissances (RAG)
- ✅ Réfléchit avant d'agir (Thought)
- ✅ Utilise des outils externes (Calcul, Recherche, Horloge)
- ✅ Itère jusqu'à trouver une réponse
- ✅ Évalue sa propre qualité
- ✅ Teste sa cohérence (Self-consistency)

## 🚀 Installation & Exécution

### 1. Installer les dépendances

```bash
pip install numpy scikit-learn
```

> **Note** : C'est tout ce dont vous avez besoin pour le mode standalone !
> Les scripts bonus et l'intégration LLM requièrent des packages supplémentaires.

### 2. Exécuter le script

```bash
python 09_mini_assistant_complet.py
```

### 3. Voir la démo

Le script va :
1. Créer une base de connaissances avec 5 documents
2. Générer un agent avec 4 outils disponibles
3. Poser 3 questions test
4. Évaluer les réponses
5. Tester la cohérence (self-consistency)
6. Afficher un rapport final

## 📊 Comprendre la Sortie

### Phase 1 : Base de Connaissances

```
✓ Index créé : 5 documents indexés
```

Le système a indexé 5 documents pédagogiques sur :
- Transformers
- RAG
- Agents autonomes
- Évaluation des LLMs
- LoRA et QLoRA

### Phase 2 : Agent Créé

```
✓ Agent créé avec 4 outils
```

L'agent peut utiliser 4 outils :
1. **calculator** : Calculs mathématiques
2. **search** : Recherche dans la base
3. **current_time** : Horloge système
4. **summarize** : Résumé de texte

### Phase 3 : Traitement de Questions

```
🤖 Question : Qu'est-ce qu'un Transformer ?

⏳ Itération 1/3
💭 Pensée : Je dois chercher des informations sur transformer
🔧 Action : search(query=transformer)
📊 Observation : Documents trouvés : [Architecture Transformer] (score: 0.89)

⏳ Itération 2/3
💭 Pensée : J'ai trouvé des informations pertinentes sur transformer.
✅ Réponse finale : Les documents pertinents expliquent...
```

L'agent :
1. **Pense** : formule son intention
2. **Agit** : exécute un outil
3. **Observe** : reçoit le résultat
4. **Répète** ou **Répond** : si confiance suffisante

### Phase 4 : Rapport d'Évaluation

```
Question 1 : Qu'est-ce qu'un Transformer ?...
  • Itérations : 2
  • Confiance : 100.00%
  • Succès : ✅

📈 Statistiques globales
  • Nombre de questions : 3
  • Itérations moyennes : 2.0
  • Confiance moyenne : 100.00%
  • Taux de succès : 100.00%
```

Métriques évaluées (Chapitre 12 & 15) :
- **Itérations** : Nombre de pas pour répondre (efficacité)
- **Confiance** : Score de fiabilité basé sur les outils
- **Succès** : Le système a-t-il trouvé une réponse ?

### Phase 5 : Test de Self-Consistency

```
Test de self-consistency (3 échantillons)

Résultats :
  • Réponse majoritaire : Les documents pertinents...
  • Score de cohérence : 100.00%
  • Réponses uniques : 1/3
```

**Self-consistency** (Chapitre 12) : Concept où on pose **la même question plusieurs fois** et on mesure si le modèle donne des réponses similaires.

Score = `réponses identiques / nombre d'essais`
- 100% = Très cohérent (stable)
- 50% = Ambigüité détectée
- 0% = Très incohérent (problématique)

## 🔗 Correspondance avec le Livre

| Concept | Chapitre | Démontré par | Ligne du code |
|---------|----------|-------------|--------------|
| **Prompting** | 11 | Structuration "Thought → Action" | `_simulate_llm_reasoning()` |
| **Évaluation** | 12 | `AssistantEvaluator` + metrics | `evaluate_response()`, `self_consistency_check()` |
| **RAG** | 13 | `RAGSystem` + TF-IDF + Cosine | `retrieve()`, `RAGSystem` |
| **Agents ReAct** | 14 | `ReActAgent` avec boucle complète | `run()` |
| **Production** | 15 | Intégration + Evaluation + Error handling | `main()` |

## 💡 Extensions Suggérées

### Niveau 1 : Facile (30 min)

1. **Changer les questions test**
   ```python
   test_questions = [
       "Comment fonctionne l'attention ?",
       "Quel est le coût d'un Transformer ?",
   ]
   ```

2. **Ajouter un nouvel outil**
   ```python
   def tool_weather(city: str) -> str:
       return f"Météo de {city}: Ensoleillé, 22°C"
   
   agent.tools.register("weather", "Météo d'une ville", tool_weather)
   ```

3. **Ajouter plus de documents**
   ```python
   rag.add_document("Nouveau document sur...", {"title": "Mon sujet"})
   ```

### Niveau 2 : Intermédiaire (1-2 h)

4. **Intégrer un vrai LLM**
   ```python
   # Remplacer _simulate_llm_reasoning() par :
   from openai import OpenAI
   client = OpenAI()
   response = client.chat.completions.create(
       model="gpt-4",
       messages=[{"role": "user", "content": prompt}]
   )
   return response.choices[0].message.content
   ```

5. **Persister les conversations**
   ```python
   import json
   
   def save_conversation(agent_responses):
       with open("conversations.json", "w") as f:
           json.dump(agent_responses, f, indent=2)
   ```

6. **Améliorer le RAG**
   - Remplacer TF-IDF par des embeddings denses (OpenAI, HuggingFace)
   - Utiliser une base vectorielle (Pinecone, Weaviate, ChromaDB)
   - Implémenter hybrid search (BM25 + Vector)

### Niveau 3 : Avancé (2-4 h)

7. **Créer une interface web**
   ```bash
   pip install streamlit
   # Créer app_agent.py
   ```

8. **Déployer en production**
   - Docker + FastAPI
   - Monitoring avec Prometheus
   - Logging structuré (JSON)

9. **Évaluation avancée**
   - Benchmark contre un jeu de test annoté
   - Calcul de ROUGE, BERTScore
   - Analyse des hallucinations

## 📝 Modélisation Interne

### Architecture Générale

```
                    ┌─────────────────────┐
                    │  Utilisateur        │
                    │  Question           │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   ReActAgent        │
                    │ (Boucle Autonome)   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  LLM Simulator      │
                    │  (ou OpenAI/Claude) │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   ┌────▼─────┐         ┌──────▼──────┐      ┌───────▼────┐
   │ RAGSystem │         │ ToolRegistry │      │  Evaluator │
   │ (Indexing)│         │ (4 outils)   │      │ (Métriques)│
   └──────────┘         └──────────────┘      └────────────┘
```

### Flux d'Exécution

```
1. Question de l'utilisateur
2. ReActAgent.run(question)
   a. Appel LLM → "Pensée + Action"
   b. Parse Action (tool_name, params)
   c. ToolRegistry.execute(tool_name, params)
   d. Observation retournée
   e. Boucle : répéter jusqu'à "Final Answer"
3. Retourner réponse + historique
4. Evaluator.evaluate_response() → Métriques
5. Affichage du rapport
```

## 🐛 Troubleshooting

### "ModuleNotFoundError: numpy"
```bash
pip install numpy scikit-learn
```

### L'agent boucle sans s'arrêter
→ Ajustez `max_iterations` lors de `agent.run()` :
```python
response = agent.run(question, max_iterations=3)  # 3 essais max
```

### Les réponses sont toujours "Impossible de répondre"
→ Vérifiez que les documents sont indexés :
```python
print(rag_system.documents)  # Doit afficher 5 documents
```

### Score de confiance trop bas
→ C'est normal en mode simulation ! Intégrez un vrai LLM pour de meilleurs résultats.

## 📖 Chapitre Complètement Illustré

Ce script illustre **TOUTES les concepts clés** des chapitres 11-15 :

| Concept | Où ? |
|---------|------|
| Chain-of-Thought (Ch. 11) | Pattern "Pensée → Action → Observation" |
| Température & Sampling | Simulé dans `_simulate_llm_reasoning()` |
| Pass@k & Evaluation (Ch. 12) | `_calculate_confidence()` |
| Self-Consistency | `self_consistency_check()` |
| RAG (Ch. 13) | `RAGSystem` entière + Retrieval |
| Agents (Ch. 14) | `ReActAgent` + Tool calling |
| Production (Ch. 15) | Gestion d'erreurs, monitoring, logging |

## 🎯 Objectifs d'Apprentissage

Après avoir exécuté et compris ce script, vous serez capable de :

✅ Expliquer la boucle ReAct (Reason → Act)
✅ Comprendre comment les outils s'intègrent aux agents
✅ Mesurer la qualité d'un assistant (confiance, itérations, cohérence)
✅ Implémenter un mini-RAG (indexation + retrieval)
✅ Évaluer les modèles avec Pass@k et self-consistency
✅ Adapter le code pour intégrer OpenAI, Claude, ou un modèle local
✅ Concevoir des extensions (nouveaux outils, interface web, etc.)

---

**Bon apprentissage ! 🚀**

Pour aller plus loin → Voir [README.md](./README.md) et les chapitres 11-15 du livre.
