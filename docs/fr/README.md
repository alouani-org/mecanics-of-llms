# Scripts Pratiques : Expérimenter les Concepts LLM

🌍 **[English Version](../en/README.md)** | 📖 **Français**

Collection de **9 scripts Python exécutables** pour expérimenter les concepts clés du livre **"La Mécanique des LLM"**.

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

---

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
