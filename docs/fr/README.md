# Scripts Pratiques : Expérimenter les Concepts LLM

🌍 [English](../en/README.md) | 📖 **Français** | 🇪🇸 [Español](../es/README.md) | 🇧🇷 [Português](../pt/README.md)

Collection de **10 scripts Python exécutables** pour expérimenter les concepts clés du livre **"La Mécanique des LLM"**.

> 📚 **À propos** : Ces scripts accompagnent les chapitres du livre. Voir [Parcours Pédagogique](PEDAGOGICAL_JOURNEY.md) pour les correspondances détaillées.

**📕 Acheter le livre :**
- **Broché** : [Amazon](https://amzn.eu/d/3oREERI)
- **Kindle** : [Amazon](https://amzn.eu/d/b7sG5iw)

---

## 📋 Vue d'Ensemble des Scripts

| # | Script | Chapitre(s) | Concepts | Status |
|---|--------|-----------|----------|--------|
| 1 | `01_tokenization_embeddings.py` | 2 | Tokenisation, impact sur la longueur de séquence | ✅ |
| 2 | `02_multihead_attention.py` | 3 | Self-attention, multi-head, poids d'attention | ✅ |
| 3 | `03_temperature_softmax.py` | 7, 11 | Température, softmax, entropie | ✅ |
| 4 | `04_rag_minimal.py` | 13 | Pipeline RAG, retrieval, similarité cosinus | ✅ |
| 5 | `05_pass_at_k_evaluation.py` | 12 | Pass@k, Pass^k, évaluation de modèles | ✅ |
| 🎁 6 | `06_react_agent_bonus.py` | 14, 15 | **Agents ReAct, tool registration, MCP** | ✅ BONUS |
| 🎁 7 | `07_llamaindex_rag_advanced.py` | 13, 14 | **RAG avancé, indexing, chat persistant** | ✅ BONUS |
| 🎁 8 | `08_lora_finetuning_example.py` | 9, 10 | **LoRA, QLoRA, fine-tuning comparatif** | ✅ BONUS |
| 🏆 **9** | `09_mini_assistant_complet.py` | **11-15** | **🎯 Projet Final Intégrateur** | ✅ FLAGSHIP |
| 🎁 10 | `10_activation_steering_demo.py` | 10 | **Activation Steering, 3SO, vecteurs de concept** | ✅ BONUS |

---

## � Descriptions Détaillées des Scripts

### 📌 Script 01 : Tokenisation et Embeddings
**Fichier :** `01_tokenization_embeddings.py` | **Chapitre :** 2

**Ce que fait le script :**
- Charge un tokenizer (GPT-2 ou LLaMA-2) et analyse différents textes
- Compare le nombre de tokens entre français et anglais
- Démontre l'impact de la longueur de séquence sur le coût computationnel

**Ce que vous apprenez :**
- Comment le texte est découpé en tokens (BPE, WordPiece)
- Pourquoi "Bonjour" peut devenir 2-3 tokens alors que "Hello" n'en fait qu'un
- L'impact direct : plus de tokens = coût O(n²) plus élevé pour l'attention

**Sortie attendue :**
```
Text: L'IA est utile
  Token count: 5
  Tokens: ['L', "'", 'IA', 'est', 'utile']
```

---

### 📌 Script 02 : Attention Multi-Têtes
**Fichier :** `02_multihead_attention.py` | **Chapitre :** 3

**Ce que fait le script :**
- Simule une couche d'attention multi-têtes avec des tenseurs PyTorch
- Calcule les projections Q, K, V et les poids d'attention
- Affiche comment chaque tête "regarde" différemment la phrase

**Ce que vous apprenez :**
- Le mécanisme Q (Query), K (Key), V (Value)
- Pourquoi plusieurs têtes capturent des dépendances différentes
- Que les poids d'attention somment toujours à 1 (distribution de probabilité)

**Sortie attendue :**
```
Sentence: The cat sleeps well
Head 1: Attention weights from 'cat' → 'sleeps': 0.42
Head 2: Attention weights from 'cat' → 'The': 0.38
```

---

### 📌 Script 03 : Température et Softmax
**Fichier :** `03_temperature_softmax.py` | **Chapitres :** 7, 11

**Ce que fait le script :**
- Applique softmax avec différentes températures (0.1, 0.5, 1.0, 2.0)
- Calcule l'entropie de Shannon pour chaque distribution
- Génère des graphiques (si matplotlib est installé)

**Ce que vous apprenez :**
- T < 1 : distribution "pointue" → génération déterministe (greedy)
- T > 1 : distribution "plate" → génération créative/diverse
- L'entropie augmente avec la température (plus d'incertitude)

**Sortie attendue :**
```
Temperature 0.5: Token 'Paris' = 85% (sharp, deterministic)
Temperature 2.0: Token 'Paris' = 35% (flat, creative)
```

---

### 📌 Script 04 : RAG Minimal
**Fichier :** `04_rag_minimal.py` | **Chapitre :** 13

**Ce que fait le script :**
- Crée une mini base de connaissances (7 documents sur les LLM)
- Vectorise les documents avec TF-IDF
- Effectue une recherche par similarité cosinus
- Simule la génération augmentée par le contexte récupéré

**Ce que vous apprenez :**
- Le pipeline RAG complet : Retrieval → Augmentation → Generation
- Comment la similarité cosinus trouve les documents pertinents
- Pourquoi RAG permet de répondre à des questions sur des données privées

**Sortie attendue :**
```
Question: "Comment fonctionne l'attention dans le Transformer?"
→ Documents récupérés: [doc_1: 0.72, doc_4: 0.65]
→ Réponse générée avec contexte
```

---

### 📌 Script 05 : Évaluation Pass@k
**Fichier :** `05_pass_at_k_evaluation.py` | **Chapitre :** 12

**Ce que fait le script :**
- Simule 100 tentatives de génération avec un taux de succès de 30%
- Calcule Pass@k (au moins 1 succès sur k essais)
- Calcule Pass^k (tous les k essais réussissent)

**Ce que vous apprenez :**
- Pass@k = 1 - (1-p)^k : probabilité d'au moins un succès
- Pass^k = p^k : probabilité que tous réussissent (très strict)
- Pourquoi Pass@10 ≈ 97% même avec p=30% (on a 10 chances)

**Sortie attendue :**
```
Pass@1  = 30%  (chance avec 1 essai)
Pass@5  = 83%  (chance avec 5 essais)
Pass@10 = 97%  (quasi-certain avec 10 essais)
```

---

### 🎁 Script 06 : Agent ReAct (BONUS)
**Fichier :** `06_react_agent_bonus.py` | **Chapitres :** 14, 15

**Ce que fait le script :**
- Implémente un mini-framework d'agent autonome
- Démontre la boucle ReAct : Thought → Action → Observation → ...
- Inclut des outils simulés : calculatrice, recherche web, météo

**Ce que vous apprenez :**
- Le pattern ReAct (Reasoning + Acting)
- Comment un agent décide quelle action prendre
- L'auto-correction : l'agent peut réessayer si une action échoue
- La base pour comprendre les agents MCP (Model Context Protocol)

**Sortie attendue :**
```
Thought: Je dois calculer 15% de 250€
Action: calculator(250 * 0.15)
Observation: 37.5
Final Answer: Le pourboire est de 37,50€
```

---

### 🎁 Script 07 : RAG Avancé avec LlamaIndex (BONUS)
**Fichier :** `07_llamaindex_rag_advanced.py` | **Chapitres :** 13, 14

**Ce que fait le script :**
- Système RAG complet avec parsing de documents
- Indexation et embeddings (simulés ou réels avec OpenAI)
- Chat avec mémoire conversationnelle
- Évaluation de qualité (Precision, Recall, F1)

**Ce que vous apprenez :**
- Architecture RAG production : ingestion → indexation → retrieval → génération
- Comment maintenir le contexte sur plusieurs tours de conversation
- Comment évaluer la qualité d'un système RAG

**Sortie attendue :**
```
[Chat Mode]
User: Qu'est-ce qu'un Transformer?
Assistant: [Contexte: 3 documents] Un Transformer est...
User: Et l'attention multi-têtes?
Assistant: [Mémoire: question précédente + 2 docs] ...
```

---

### 🎁 Script 08 : Fine-tuning LoRA/QLoRA (BONUS)
**Fichier :** `08_lora_finetuning_example.py` | **Chapitres :** 9, 10

**Ce que fait le script :**
- Compare Full Fine-tuning vs LoRA vs QLoRA (calculs numériques)
- Affiche les économies de VRAM et de paramètres entraînables
- Cas d'usage : adapter LLaMA-7B pour un domaine métier (ferroviaire)

**Ce que vous apprenez :**
- LoRA : ajoute ~0.1% de paramètres vs fine-tuning complet
- QLoRA : quantification 4-bit + LoRA = GPU 24GB au lieu de 140GB
- Pourquoi le fine-tuning efficace démocratise les LLM

**Sortie attendue :**
```
LLaMA-7B:
  Full Fine-tuning: 28 GB VRAM, 7B params
  LoRA (rank=8):    8 GB VRAM, 4.2M params (0.06%)
  QLoRA:            6 GB VRAM, 4.2M params + 4-bit base
```

---

### � Script 10 : Activation Steering & 3SO (BONUS)
**Fichier :** `10_activation_steering_demo.py` | **Chapitre :** 10

**Ce que fait le script :**
- Démontre le pilotage par activations (steering) : injection de vecteurs de concept
- Implémente l'extraction de vecteurs par activation contrastive
- Simule un Sparse Autoencoder (SAE) pour la décomposition en concepts
- Implémente une machine à états finis pour le 3SO (sorties JSON garanties)
- Compare RLHF/DPO vs Steering avec tableau détaillé

**Ce que vous apprenez :**
- Le steering modifie les activations à l'inférence : $X_{steered} = X + (c \times V)$
- Comment extraire des vecteurs de concept (méthode contrastive, SAE)
- L'impact du coefficient de pilotage (trop faible → nul, optimal → efficace, trop fort → déraillement)
- Le 3SO garantit mathématiquement une syntaxe JSON valide
- Quand utiliser l'alignement vs le steering

**Sortie attendue :**
```
STEP 3: Analyzing Coefficient Effect
   Coeff   Direction Δ     Perturbation    Stability
   1.0     12.5°           8.2%            ✅ stable
   5.0     45.3°           35.1%           ⚠️ moderate
   15.0    78.2°           89.4%           ❌ unstable
```

---

### �🏆 Script 09 : Mini-Assistant Complet (PROJET FINAL)
**Fichier :** `09_mini_assistant_complet.py` | **Chapitres :** 11-15

**Ce que fait le script :**
- Intègre TOUS les concepts : RAG + Agents + Température + Évaluation
- Système complet avec base de connaissances, retrieval, raisonnement
- Mode interactif pour tester différentes questions

**Ce que vous apprenez :**
- Comment assembler un assistant IA complet de A à Z
- L'architecture en couches : Data → Retrieval → Reasoning → Generation
- L'évaluation de bout en bout d'un système

**Documentation dédiée :**
- [INDEX_SCRIPT_09.md](INDEX_SCRIPT_09.md) : Architecture complète
- [QUICKSTART_SCRIPT_09.md](QUICKSTART_SCRIPT_09.md) : Démarrage en 5 min
- [SCRIPT_09_MAPPING.md](SCRIPT_09_MAPPING.md) : Mapping code ↔ concepts

---

## �🚀 Démarrage Rapide

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
# Installation basique (pour les scripts 1-5)
pip install torch transformers numpy scikit-learn

# Installation complète (avec visualisations)
pip install torch transformers numpy scikit-learn matplotlib

# Pour les bonus (optionnel, scripts fonctionnent aussi sans)
pip install llama-index openai python-dotenv peft bitsandbytes
```

**Note:** Les scripts bonus (06, 07, 08) fonctionnent **sans dépendances externes** en mode démo.

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
python 09_mini_assistant_complet.py    # ← Projet intégrateur final
```

---

## 🏆 Projet Intégrateur : Mini-Assistant Complet

**LE script phare** : intègre TOUS les concepts des chapitres 11-15.

- **Script :** `09_mini_assistant_complet.py`
- **Documentation :** [INDEX_SCRIPT_09.md](INDEX_SCRIPT_09.md)
- **Démarrage rapide :** [QUICKSTART_SCRIPT_09.md](QUICKSTART_SCRIPT_09.md)
- **Architecture :** [SCRIPT_09_MAPPING.md](SCRIPT_09_MAPPING.md)

---

## 📖 Documentation Complète

- **[Parcours Pédagogique](PEDAGOGICAL_JOURNEY.md)** : Correspondance chapitre par chapitre livre ↔ scripts
- **[ReAct Agents](REACT_AGENT_INTEGRATION.md)** : Pattern ReAct et intégration
- **[LlamaIndex RAG](LLAMAINDEX_GUIDE.md)** : Framework RAG avancé

---

## 📝 Notes

- **Pas de GPU requis** : tous les scripts tournent sur CPU (plus lentement)
- **Code éducatif** : privilégient la clarté sur l'optimisation
- **Compatible Python 3.9+**

---

**Bon apprentissage! 🚀**
