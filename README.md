# Scripts Pratiques : Expérimenter les Concepts LLM

Ce dossier regroupe des **micro-scripts Python exécutables** permettant aux ingénieurs débutants d'expérimenter concrètement les concepts clés présentés dans le livre.

## 📋 Liste des Scripts

| # | Script | Chapitre(s) | Concepts |
|---|--------|-----------|----------|
| 1 | `01_tokenization_embeddings.py` | 2 | Tokenisation, impact sur la longueur de séquence |
| 2 | `02_multihead_attention.py` | 3 | Self-attention, multi-head, poids d'attention |
| 3 | `03_temperature_softmax.py` | 7, 11 | Température, softmax, entropie |
| 4 | `04_rag_minimal.py` | 13 | Pipeline RAG, retrieval, similarité cosinus |
| 5 | `05_pass_at_k_evaluation.py` | 12 | Pass@k, Pass^k, évaluation de modèles |
| 🎁 **BONUS 1** | `06_react_agent_bonus.py` | 13, 14 | **Agents ReAct, framework générique, tool registration** |
| 🎁 **BONUS 2** | `07_llamaindex_rag_advanced.py` | 13 | **RAG avancé, document indexing, chat persistant** |
| 🎁 **BONUS 3** | `08_lora_finetuning_example.py` | 9 | **LoRA, QLoRA, comparaison fine-tuning, cas réel SNCF** |

## 🚀 Démarrage Rapide

### 1. Créer un environnement virtuel (recommandé)

```bash
# Sur Windows
python -m venv venv
venv\Scripts\activate

# Sur macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 2. Installer les dépendances

```bash
# Installation basique (pour les 5 scripts)
pip install torch transformers numpy scikit-learn

# Installation complète (avec visualisations)
pip install torch transformers numpy scikit-learn matplotlib
```

### 3. Exécuter un script

```bash
python 01_tokenization_embeddings.py
python 02_multihead_attention.py
python 03_temperature_softmax.py
python 04_rag_minimal.py
python 05_pass_at_k_evaluation.py
python 06_react_agent_bonus.py
python 07_llamaindex_rag_advanced.py
python 08_lora_finetuning_example.py
```

## 📖 Détails par Script

### Script 1 : Tokenisation et Embeddings (Chapitre 2)

**Voir:** `02-representation-texte-modeles-sequentiels.md`

Illustre :
- Comment les tokenizers (BPE, WordPiece) fragmentent le texte.
- L'impact du nombre de tokens sur le coût computationnel.
- Les différences entre langues (français vs anglais).

```bash
python 01_tokenization_embeddings.py
```

**Exemple de sortie:**
```
Texte: L'IA est utile
  Nombre de tokens: 6
  Token IDs: [1, 2, 3, 4, 5, 6]
  Tokens (texte): ['L', "'", 'IA', 'est', 'utile']

Texte court → 2 tokens
Texte long (100x) → 198 tokens
Facteur: 99.0x

⚠️ IMPLICATIONS:
  • Plus de tokens = plus de VRAM
  • Plus de tokens = latence plus élevée
  • Coût d'inférence ∝ O(n²) pour l'attention
```

---

### Script 2 : Multi-Head Attention (Chapitre 3)

**Voir:** `03-architecture-transformer.md`

Simule une couche d'attention multi-tête minimale :
- Projections Q, K, V.
- Calcul des scores d'attention.
- Visualisation de comment chaque tête focalise différemment.

```bash
python 02_multihead_attention.py
```

**Exemple de sortie:**
```
Entrée x shape: (1, 4, 64)
  (batch=1, seq_len=4, d_model=64)

Tête 0:
  Poids d'attention (après softmax):
    [[0.25 0.35 0.25 0.15]  # Le "regarde" 35% vers "chat"
     [0.10 0.60 0.20 0.10]  # "chat" regarde 60% vers "dort"
     ...

💡 INTUITION:
  • Chaque tête capture DIFFÉRENTES dépendances.
  • Tête 0 peut se concentrer sur sujet-verbe.
  • Tête 1 peut se concentrer sur verbe-adverbe.
```

---

### Script 3 : Température et Softmax (Chapitres 7 & 11)

**Voir:** `07-preentrainement-llms.md`, `11-strategies-generation-inference.md`

Montre l'effet de la température sur la distribution softmax :
- Basse T → distribution pointue (greedy, déterministe).
- Haute T → distribution plate (diversité, créativité).
- Lien avec l'entropie.

```bash
python 03_temperature_softmax.py
```

**Exemple de sortie:**
```
Température = 0.1
  Probabilités:
    chat:    0.874  ████████████████████...
    chien:   0.099  ██
    souris:  0.019  
    oiseau:  0.008  
  Entropie: 0.347

Température = 5.0
  Probabilités:
    chat:    0.335  ███████████
    chien:   0.297  ██████████
    souris:  0.217  ███████
    oiseau:  0.151  █████
  Entropie: 1.358  (3.9x plus élevée!)

✓ À T=0.1 → Déterministe, repetitif.
✓ À T=5.0 → Créatif, mais risqué.
✓ T=0.7-0.9 → Bon compromis!
```

Génère optionnellement un graphique : `temperature_effect.png`

---

### Script 4 : RAG Minimaliste (Chapitre 13)

**Voir:** `13-systemes-augmentes-agents.md`

Simule un pipeline RAG complet :
1. **Retrieval** : chercher les 3 documents les plus pertinents.
2. **Augmentation** : injecter le contexte dans le prompt.
3. **Génération** : le LLM répond en s'appuyant sur le contexte.

```bash
python 04_rag_minimal.py
```

**Exemple de sortie:**
```
Question: "Comment fonctionne l'attention dans le Transformer?"

Top 3 documents récupérés:
1. Score: 0.892
   L'attention multi-tête permet au modèle de regarder...

2. Score: 0.756
   Le Transformer est une architecture basée sur l'attention...

Prompt augmenté envoyé au LLM:
---
Vous êtes un assistant expert.

Voici des documents pertinents:
- L'attention multi-tête permet...
- Le Transformer est une architecture...
- ...

Question: Comment fonctionne l'attention dans le Transformer?

Réponse basée sur les documents:
---

COMPARAISON:
❌ SANS RAG:
  → Hallucination possible
  → Connaissances figées
  → Pas de sources à vérifier

✅ AVEC RAG:
  → Réponses basées sur des sources
  → Accès aux données externes
  → Utilisateur peut vérifier les sources
```

---

### Script 5 : Évaluation Pass@k (Chapitre 12)

**Voir:** `12-modeles-raisonnement.md`

Évalue la fiabilité d'un modèle sur des tâches vérifiables :
- **Pass@k** : probabilité d'au moins **une** réussite en k tentatives.
- **Pass^k** : probabilité que **toutes** les k tentatives réussissent.

```bash
python 05_pass_at_k_evaluation.py
```

**Exemple de sortie:**
```
Paramètres:
  • Nombre de tentatives: 100
  • Probabilité de succès: 30%

PASS@K (Au moins UNE réussite en k tentatives):
Pass@1 = 30.0% (1 tentative)
Pass@3 = 65.7% (3 tentatives)
Pass@5 = 83.2% (5 tentatives)
Pass@10 = 97.2% (10 tentatives)

PASS^K (TOUTES les k tentatives réussissent) — STRICT:
Pass^1 = 30.0% (théorique: 0.3^1)
Pass^3 =  2.7% (théorique: 0.3^3)
Pass^5 =  0.2% (théorique: 0.3^5)

APPLICATION:
  ✓ Recherche (HumanEval): Pass@k (diversité)
  ✓ Agents critiques: Pass^k (fiabilité totale)
```

---

## 🎯 Comment Utiliser Ces Scripts

### Pour les Étudiants

1. **Lisez le chapitre pertinent du livre.**
2. **Exécutez le script associé.**
3. **Modifiez les paramètres** pour voir les effets :
   - Changez `seq_len`, `num_heads`, `temperatures`, etc.
   - Ajoutez vos propres textes/documents.
4. **Ajoutez des `print()`** pour déboguer et comprendre les dimensions.

### Pour les Ingénieurs

- Utilisez ces scripts comme **point de départ** pour vos implémentations.
- Intégrez-les dans des **pipelines de production** (RAG, évaluation, etc.).
- Adaptez le code à votre **infrastructure** (GPUs, APIs, bases de données).

## 🎁 Bonus Scripts

### BONUS 1 : ReAct Agent Framework (`06_react_agent_bonus.py`)

**Voir:** `REACT_AGENT_INTEGRATION.md`

Framework complet pour construire des **agents autonomes avec pattern ReAct**.

**Caractéristiques:**
- ✅ Classe `Agent` réutilisable et extensible
- ✅ Système de registration d'outils (tool definition)
- ✅ Boucle Thought → Action → Observation
- ✅ LLM simulation (prêt pour OpenAI, Claude, Groq, Ollama)
- ✅ Historique et gestion d'itérations
- ✅ Exemple avec 3 outils (calculator, date, knowledge base)

**Concepts couverts:**
- Agents autonomes (Chapitre 13)
- Protocoles standards agentiques (Chapitre 14)
- Pattern ReAct (Reasoning + Acting)
- Tool calling et execution

**Exécution:**
```bash
python 06_react_agent_bonus.py
```

**Intégration avec LLMs réels:**
```python
from openai import OpenAI

class OpenAIAgent(Agent):
    def _simulate_llm_reasoning(self, task, context):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
```

Voir `REACT_AGENT_INTEGRATION.md` pour intégrations complètes (OpenAI, Claude, Groq, Ollama).

---

### BONUS 2 : LlamaIndex RAG Avancé (`07_llamaindex_rag_advanced.py`)

**Voir:** `LLAMAINDEX_GUIDE.md`

Framework complet pour construire des **systèmes RAG avancés avec LlamaIndex**.

**Caractéristiques:**
- ✅ Indexation vectorielle d'documents
- ✅ Retrieval avancé (similarity search, hybrid BM25+vector)
- ✅ RAG Engine avec augmentation de contexte
- ✅ Chatbot avec mémoire conversationnelle
- ✅ Évaluation de qualité (Precision, Recall, F1)
- ✅ Export des résultats en JSON
- ✅ Fallback embeddings simulés (pas de dépendances requises)

**Concepts couverts:**
- RAG (Retrieval-Augmented Generation) - Chapitre 13
- Document parsing et indexing
- Vector similarity search
- Query augmentation avec contexte
- Conversation avec persistance

**Phases d'exécution:**
1. Chargement des documents
2. Création de l'index vectoriel
3. Initialisation du RAG Engine
4. Requêtes RAG avec retrieval
5. Chat avec mémoire
6. Évaluation de qualité
7. Export des résultats

**Exécution sans dépendances:**
```bash
python 07_llamaindex_rag_advanced.py
```
⚠️ Utilise embeddings simulés (déterministes).

**Exécution avec LlamaIndex réel:**
```bash
pip install llama-index openai
python 07_llamaindex_rag_advanced.py
```
✓ Utilise OpenAI embeddings (text-embedding-3-small).

**Production avec documents réels:**
```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Charger PDFs, Word, HTML, etc.
documents = SimpleDirectoryReader("./docs").load_data()

# Créer index
index = VectorStoreIndex.from_documents(documents)

# Query engine
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("Votre question ici")
```

Voir `LLAMAINDEX_GUIDE.md` pour :
- Installation complète (LlamaIndex, vector stores, readers)
- Intégration OpenAI, Claude, Groq
- Hybrid search (BM25 + vectoriel)
- Agents avec outils
- Évaluation de qualité
- Cas d'usage avancés

---

### Script 8 : LoRA & QLoRA Fine-tuning (Chapitre 9) 🎁 BONUS 3

**Voir:** `09-affinage-supervise-sft.md`

Démontre les techniques de fine-tuning efficace en ressources :
- LoRA (Low-Rank Adaptation) : réduction des paramètres entraînables.
- QLoRA (Quantized LoRA) : quantification + LoRA pour VRAM ultra-faible.
- Comparaison chiffrée : Full Fine-tuning vs LoRA vs QLoRA.
- Cas réel : adaptation d'un modèle LLaMA-7B pour le domaine ferroviaire (SNCF).

```bash
python 08_lora_finetuning_example.py
```

**Exemple de sortie:**
```
=== LoRA Calculations ===
LLaMA-7B (7B params total)
  LoRA Rank 64:
    Trainable params (A+B): 85,262,336 (1.22% of model)
    Reduction: 81.7×

=== Fine-tuning Method Comparison ===
Method          | VRAM Needed | Time (10K ex) | Checkpoint | Use Case
Full FT         | 28 GB       | 8h            | 26 GB      | Unlimited budget
LoRA            | 8 GB        | 2.5h          | 85 MB      | Multi-domain, quick
QLoRA           | 2 GB        | 3h            | 85 MB      | Single GPU edge

=== Real Case: SNCF Railway Adapter ===
Scenario: Adapt LLaMA-7B for railway maintenance (10K domain Q&A)
Hardware: RTX 4090 (24GB VRAM)

Full Fine-tuning:  Need 28GB → IMPOSSIBLE on RTX 4090
LoRA:              Need 8GB  → ✅ Feasible, 2.5h training
QLoRA:             Need 2GB  → ✅ Feasible, 3h training, leaves GPU RAM free
```

Concepts abordés :
- W = W₀ + BA (décomposition LoRA)
- Effet du rank (8, 16, 32, 64) sur taille vs performance
- Quantisation 8-bit et économies de mémoire
- Peft library integration (transformers + peft)
- Pseudo-code pour adapter modèles multilingues

---

## 📚 Correspondance Livre ↔ Scripts

| Chapitre | Topic | Script |
|----------|-------|--------|
| 2 | Tokenisation, Embeddings | `01_tokenization_embeddings.py` |
| 3 | Architecture Transformer, Attention | `02_multihead_attention.py` |
| 7 | Pré-entraînement, Loss | `03_temperature_softmax.py` |
| 9 | Affinage supervisé, LoRA, QLoRA | **`08_lora_finetuning_example.py`** |
| 11 | Stratégies de génération, Température | `03_temperature_softmax.py` |
| 12 | Modèles de raisonnement, Évaluation | `05_pass_at_k_evaluation.py` |
| 13 | Systèmes augmentés, RAG, Agents | `04_rag_minimal.py`, **`06_react_agent_bonus.py`**, **`07_llamaindex_rag_advanced.py`** |
| 14 | Protocoles standards agentiques | **`06_react_agent_bonus.py`** |

## 🛠️ Troubleshooting

### "ModuleNotFoundError: No module named 'transformers'"

```bash
pip install transformers
```

### "ModuleNotFoundError: No module named 'torch'"

```bash
# CPU
pip install torch

# GPU (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Matplotlib non installé

```bash
pip install matplotlib
# Script 3 continuera à fonctionner sans, mais pas de graphique.
```

### Script trop lent (transformers qui télécharge un modèle)

- Les modèles se téléchargent automatiquement à la première exécution (~3 GB pour LLaMA).
- Les prochaines exécutions seront plus rapides (cache local).
- Alternative : utiliser un modèle plus petit (`distilbert-base-multilingual-cased`).

## 📝 Notes

- **Pas de GPU requis** : tous les scripts tournent sur CPU (plus lentement).
- **Dépendances minimales** : seulement `numpy`, `torch`, `transformers`, `scikit-learn`.
- **Code éducatif** : les scripts privilégient la clarté sur l'optimisation.
- **Compatible Python 3.9+**.

## 🤝 Contribution

Si tu souhaites ajouter un script ou corriger un bug, n'hésite pas à :
1. Fork ce repository.
2. Crée une branche (`git checkout -b feature/mon-script`).
3. Commit et pousse (`git push origin feature/mon-script`).
4. Ouvre une pull request.

---

**Bon apprentissage! 🚀**
