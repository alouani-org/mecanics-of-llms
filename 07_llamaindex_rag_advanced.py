"""
07_llamaindex_rag_advanced.py - Framework LlamaIndex pour RAG Avancé

Démontre comment construire un système RAG complet avec LlamaIndex :
- Chargement et parsing de documents
- Indexation et embedding
- Retrieval avancé (BM25, Hybrid Search)
- Query Engine avec résumé et raisonnement
- Chat avec persistance de contexte
- Évaluation de qualité

Installation:
    pip install llama-index openai langchain python-dotenv

Concepts couverts (Chapitre 13):
    - RAG (Retrieval-Augmented Generation)
    - Document parsing et indexing
    - Embedding et similarity search
    - Chain-of-Thought pour augmentation
"""

import os
from typing import List, Optional
from datetime import datetime
import json

# ============================================================================
# PHASE 0 : Configuration & Imports (avec fallback)
# ============================================================================

def setup_environment():
    """Configurer l'environnement avec gestion des dépendances."""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY non configurée")
        print("   Utilisez: export OPENAI_API_KEY=sk-... (Linux/Mac)")
        print("   Ou: $env:OPENAI_API_KEY='sk-...' (PowerShell Windows)")
        print("   Ou mettez-la dans .env")
    return api_key


# ============================================================================
# PHASE 1 : Données de Document (Embedded)
# ============================================================================

SAMPLE_DOCUMENTS = [
    {
        "title": "Transformers : Architecture",
        "content": """
Le Transformer est une architecture de réseau profond basée sur le mécanisme 
d'attention. Contrairement aux RNNs qui traitent les tokens séquentiellement, 
les Transformers traitent tous les tokens en parallèle.

Structure principale:
1. Embedding: Convertit les tokens en vecteurs denses
2. Positional Encoding: Ajoute l'information de position
3. Multi-Head Attention: Capture les relations complexes entre tokens
4. Feed-Forward Networks: Transformations non-linéaires
5. Layer Normalization: Stabilise l'entraînement

Avantage clé: Parallélisation complète → entraînement et inférence plus rapides.
Complexité: O(n²) en tokens, ce qui limite la longueur des contextes.
"""
    },
    {
        "title": "Attention Multi-Tête",
        "content": """
L'attention multi-tête permet au modèle d'observer différentes représentations
d'ordre supérieur du même espace d'entrée-sortie.

Formule:
Attention(Q, K, V) = softmax(Q·K^T / √d_k)·V

Où:
- Q (Query): Représentation de la question
- K (Key): Représentation des clés à interroger
- V (Value): Valeurs à récupérer
- d_k: Dimension des clés (scaling)

Avec 8 têtes (h=8), le modèle capture:
- Relation syntaxique (dans une tête)
- Relation sémantique (dans une autre tête)
- Position (dans une troisième tête)
- etc.

Avantage: Meilleure représentation qu'une seule tête.
"""
    },
    {
        "title": "Fine-tuning et Adaptation",
        "content": """
Fine-tuning = adapter un modèle pré-entraîné à une tâche spécifique.

Stratégies:
1. Full Fine-tuning: Mettre à jour tous les paramètres (coûteux en mémoire)
2. LoRA (Low-Rank Adaptation): Ajouter des matrices de petit rang
   - Paramètres supplémentaires: d·r·2 au lieu de d²
   - Où d=dimension originale, r=rang
   - Exemple: 7B model → ~8M params (0.1%)
3. QLoRA: LoRA sur modèle quantifié (4-bit)
   - Mémoire: 7B model → ~2GB au lieu de 28GB
   - Vitesse acceptable pour inférence

Best practice:
- Petit dataset (<10K examples): LoRA
- Medium dataset (10K-100K): LoRA avec learning rate 5e-4
- Gros dataset (>100K): Full fine-tuning si possible
"""
    }
]


# ============================================================================
# PHASE 2 : Constructeur de Documents (sans dépendances)
# ============================================================================

class SimpleDocument:
    """Document simplifié (fallback si LlamaIndex unavailable)."""
    
    def __init__(self, content: str, metadata: dict = None):
        self.content = content
        self.metadata = metadata or {}
        self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        import hashlib
        return hashlib.md5(
            f"{self.content[:100]}{datetime.now()}".encode()
        ).hexdigest()[:8]
    
    def __repr__(self):
        return f"Doc({self.metadata.get('title', 'Untitled')})"


# ============================================================================
# PHASE 3 : Indexation et Embedding (Simulé + Réel)
# ============================================================================

class SimpleEmbedding:
    """Embedding simulé (fallback)."""
    
    @staticmethod
    def embed_text(text: str, dimensions: int = 384) -> List[float]:
        """Créer un embedding simple basé sur hash (déterministe)."""
        import hashlib
        
        # Hash du texte → seed
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        
        # Pseudo-RNG déterministe
        import random
        random.seed(hash_val % (2**32))
        
        # Générer vecteur "plausible" (normal distribution)
        embedding = [random.gauss(0, 1) for _ in range(dimensions)]
        
        # Normaliser
        magnitude = sum(x**2 for x in embedding) ** 0.5
        return [x / magnitude for x in embedding]


class VectorIndex:
    """Index vectoriel simple pour retrieval."""
    
    def __init__(self, dimension: int = 384):
        self.vectors: dict = {}  # doc_id -> vector
        self.documents: dict = {}  # doc_id -> doc
        self.dimension = dimension
    
    def add_document(self, doc: SimpleDocument):
        """Ajouter un document et son embedding."""
        embedding = SimpleEmbedding.embed_text(doc.content, self.dimension)
        self.vectors[doc.id] = embedding
        self.documents[doc.id] = doc
    
    def similarity(self, v1: List[float], v2: List[float]) -> float:
        """Cosine similarity entre deux vecteurs."""
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(x**2 for x in v1) ** 0.5
        norm2 = sum(x**2 for x in v2) ** 0.5
        return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
    
    def retrieve(self, query: str, top_k: int = 3) -> List[SimpleDocument]:
        """Retriever les k documents les plus similaires."""
        query_vec = SimpleEmbedding.embed_text(query, self.dimension)
        
        scores = [
            (doc_id, self.similarity(query_vec, vec))
            for doc_id, vec in self.vectors.items()
        ]
        
        # Trier par score décroissant
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [self.documents[doc_id] for doc_id, _ in scores[:top_k]]


# ============================================================================
# PHASE 4 : RAG Engine (Query Processing)
# ============================================================================

class SimpleRAGEngine:
    """Moteur RAG avec augmentation de contexte."""
    
    def __init__(self, index: VectorIndex):
        self.index = index
        self.query_history: List[dict] = []
    
    def _simulate_llm_response(self, prompt: str) -> str:
        """Simuler une réponse LLM (remplacer par OpenAI API en production)."""
        # En production: appeler OpenAI, Claude, etc.
        # Pour la démo: réponse heuristique
        
        if "transformer" in prompt.lower():
            return """
D'après le contexte fourni, les Transformers sont des architectures basées
sur l'attention qui traitent tous les tokens en parallèle, contrairement aux RNNs.

Les points clés:
1. Multi-Head Attention: Capture différentes relations (syntaxe, sémantique, position)
2. Parallélisation: O(n²) en tokens mais traitement plus rapide
3. Positional Encoding: Ajoute l'information de séquence

L'architecture originale (Vaswani et al., 2017) comprend:
- Encoder (comprendre le contexte)
- Decoder (générer la réponse)
- Attention multi-tête (8 têtes par défaut)

Application: Tous les LLMs modernes (GPT, Claude, Llama) utilisent cette architecture.
"""
        elif "fine" in prompt.lower() or "lora" in prompt.lower():
            return """
Le fine-tuning adapte un modèle pré-entraîné à une tâche spécifique.

Stratégies recommandées:
- LoRA: Ajout de matrices de petit rang (économe en mémoire)
- QLoRA: LoRA sur modèles quantifiés (4-bit, ~2GB pour 7B model)
- Full fine-tuning: Si dataset très grand et ressources disponibles

Exemple: LoRA sur Llama-2 7B:
- Paramètres supplémentaires: ~8M (0.1% du modèle)
- Mémoire nécessaire: ~10GB (vs 28GB en full fine-tuning)
- Temps: 1-2h sur GPU A100

Best practice par dataset:
- <10K examples: LoRA
- 10K-100K: LoRA + lr 5e-4
- >100K: Full fine-tuning
"""
        else:
            return "Je ne suis pas certain de la réponse à cette question."
    
    def query(self, question: str, top_k: int = 3) -> dict:
        """
        Exécuter une requête RAG complète.
        
        Processus:
        1. Retriever: Chercher les documents pertinents
        2. Augmenter: Injecter le contexte dans le prompt
        3. Générer: Appeler le LLM
        4. Logger: Sauvegarder pour évaluation
        """
        # 1. Retrieval
        retrieved_docs = self.index.retrieve(question, top_k)
        
        # 2. Augmentation de contexte
        context = "\n\n".join([
            f"[{doc.metadata.get('title', 'Untitled')}]\n{doc.content}"
            for doc in retrieved_docs
        ])
        
        augmented_prompt = f"""Contexte:
{context}

Question: {question}

Réponds en utilisant le contexte fourni. Sois concis et actionnel."""
        
        # 3. Génération (simulée ou réelle)
        answer = self._simulate_llm_response(augmented_prompt)
        
        # 4. Logging
        result = {
            "question": question,
            "retrieved_docs": [
                {
                    "title": doc.metadata.get("title"),
                    "id": doc.id,
                    "snippet": doc.content[:100] + "..."
                }
                for doc in retrieved_docs
            ],
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }
        
        self.query_history.append(result)
        return result


# ============================================================================
# PHASE 5 : Chat avec Mémoire de Contexte
# ============================================================================

class RAGChatbot:
    """Chatbot avec RAG et mémoire conversationnelle."""
    
    def __init__(self, rag_engine: SimpleRAGEngine, memory_size: int = 5):
        self.rag_engine = rag_engine
        self.memory_size = memory_size
        self.conversation_history: List[dict] = []
    
    def _build_conversation_context(self) -> str:
        """Construire le contexte à partir de l'historique."""
        if not self.conversation_history:
            return "Aucun historique."
        
        history_text = "\n".join([
            f"Q: {turn['question']}\nA: {turn['answer'][:200]}..."
            for turn in self.conversation_history[-self.memory_size:]
        ])
        return history_text
    
    def chat(self, user_message: str) -> dict:
        """Traiter un message utilisateur avec contexte conversationnel."""
        
        # Enrichir la question avec l'historique
        conversation_context = self._build_conversation_context()
        enriched_question = f"""Historique:
{conversation_context}

Nouvelle question: {user_message}"""
        
        # Utiliser le RAG engine
        rag_result = self.rag_engine.query(enriched_question)
        
        # Ajouter à l'historique
        self.conversation_history.append({
            "question": user_message,
            "answer": rag_result["answer"],
            "timestamp": rag_result["timestamp"]
        })
        
        return {
            "user_message": user_message,
            "retrieved_docs": rag_result["retrieved_docs"],
            "response": rag_result["answer"],
            "turn_number": len(self.conversation_history)
        }


# ============================================================================
# PHASE 6 : Évaluation de Qualité
# ============================================================================

class RAGEvaluator:
    """Évaluer la qualité d'un système RAG."""
    
    @staticmethod
    def evaluate_retrieval(
        query: str,
        retrieved_docs: List[SimpleDocument],
        expected_doc_ids: List[str]
    ) -> dict:
        """
        Évaluer la pertinence du retrieval.
        
        Métriques:
        - Precision@k: % des docs retrievés qui sont pertinents
        - Recall@k: % des docs pertinents qui ont été retrievés
        - MRR: Mean Reciprocal Rank (position du 1er bon doc)
        """
        retrieved_ids = [doc.id for doc in retrieved_docs]
        
        # Precision
        if retrieved_ids:
            precision = len(
                set(retrieved_ids) & set(expected_doc_ids)
            ) / len(retrieved_ids)
        else:
            precision = 0
        
        # Recall
        recall = len(
            set(retrieved_ids) & set(expected_doc_ids)
        ) / max(1, len(expected_doc_ids))
        
        # F1
        f1 = 2 * (precision * recall) / max(1, precision + recall)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "retrieved_count": len(retrieved_ids),
            "expected_count": len(expected_doc_ids)
        }
    
    @staticmethod
    def evaluate_generation(
        generated_answer: str,
        reference_answer: str
    ) -> dict:
        """
        Évaluer la qualité de génération avec BLEU / ROUGE simplifiés.
        """
        gen_tokens = set(generated_answer.lower().split())
        ref_tokens = set(reference_answer.lower().split())
        
        # Jaccard similarity
        if gen_tokens | ref_tokens:
            jaccard = len(gen_tokens & ref_tokens) / len(gen_tokens | ref_tokens)
        else:
            jaccard = 0
        
        return {
            "jaccard_similarity": jaccard,
            "generated_length": len(generated_answer),
            "reference_length": len(reference_answer)
        }


# ============================================================================
# MAIN : Démonstration Complète
# ============================================================================

def main():
    """Exécuter une démonstration complète du RAG avec LlamaIndex."""
    
    print("=" * 80)
    print("🦙 LlamaIndex RAG Advanced Demo")
    print("=" * 80)
    print()
    
    # 1. Charger les documents
    print("📚 Phase 1: Chargement des documents")
    print("-" * 80)
    documents = [
        SimpleDocument(doc["content"], {"title": doc["title"]})
        for doc in SAMPLE_DOCUMENTS
    ]
    for doc in documents:
        print(f"  ✓ {doc.metadata['title']} ({len(doc.content)} chars)")
    print()
    
    # 2. Créer l'index
    print("🔍 Phase 2: Création de l'index vectoriel")
    print("-" * 80)
    index = VectorIndex(dimension=384)
    for doc in documents:
        index.add_document(doc)
    print(f"  ✓ Index créé avec {len(documents)} documents")
    print(f"  ✓ Dimension embedding: 384")
    print()
    
    # 3. Créer le RAG engine
    print("⚙️  Phase 3: Initialisation du RAG Engine")
    print("-" * 80)
    rag_engine = SimpleRAGEngine(index)
    print("  ✓ RAG Engine prêt")
    print()
    
    # 4. Exécuter des requêtes
    print("💬 Phase 4: Requêtes RAG")
    print("-" * 80)
    
    questions = [
        "Qu'est-ce qu'un Transformer?",
        "Comment fonctionne l'attention multi-tête?",
        "Quelle est la différence entre LoRA et fine-tuning complet?"
    ]
    
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\nQ{i}: {question}")
        result = rag_engine.query(question, top_k=2)
        results.append(result)
        
        print(f"📄 Documents retrievés:")
        for doc in result["retrieved_docs"]:
            print(f"   - {doc['title']} ({doc['id']})")
        
        print(f"🤖 Réponse:\n{result['answer'][:300]}...")
    
    print()
    
    # 5. Chat conversationnel
    print("💬 Phase 5: Chat avec Mémoire")
    print("-" * 80)
    
    chatbot = RAGChatbot(rag_engine, memory_size=3)
    
    chat_messages = [
        "Parle-moi des Transformers",
        "Et comment ça marche en pratique?",
        "Peux-tu comparer avec les RNNs?"
    ]
    
    for msg in chat_messages:
        print(f"\n👤 Utilisateur: {msg}")
        chat_result = chatbot.chat(msg)
        print(f"🤖 Bot: {chat_result['response'][:250]}...")
        print(f"   📄 Tour {chat_result['turn_number']}")
    
    print()
    
    # 6. Évaluation
    print("📊 Phase 6: Évaluation de Qualité")
    print("-" * 80)
    
    evaluator = RAGEvaluator()
    
    # Évaluer le retrieval de la première requête
    first_result = results[0]
    retrieved_doc_ids = [d["id"] for d in first_result["retrieved_docs"]]
    
    # Supposer que les docs "pertinents" sont tous ceux contenant "Transformer"
    expected_docs = [
        d.id for d in documents
        if "transformer" in d.content.lower()
    ]
    
    # Créer des objets SimpleDocument avec les ID corrects
    retrieved_docs_objects = [
        documents[i] for i, d in enumerate(documents) 
        if d.id in retrieved_doc_ids
    ]
    
    retrieval_eval = evaluator.evaluate_retrieval(
        questions[0],
        retrieved_docs_objects,
        expected_docs
    )
    
    print("Évaluation du Retrieval (Q1):")
    print(f"  - Precision@2: {retrieval_eval['precision']:.2%}")
    print(f"  - Recall@2:    {retrieval_eval['recall']:.2%}")
    print(f"  - F1:          {retrieval_eval['f1']:.2%}")
    
    print()
    
    # 7. Résumé statistique
    print("📈 Statistiques Finales")
    print("-" * 80)
    print(f"  - Requêtes traitées: {len(results)}")
    print(f"  - Tours conversationnels: {len(chatbot.conversation_history)}")
    print(f"  - Documents dans l'index: {len(index.documents)}")
    print(f"  - Historique de requêtes sauvegardé: {len(rag_engine.query_history)} entrées")
    
    print()
    print("=" * 80)
    print("✅ Démo complétée!")
    print("=" * 80)
    
    # 8. Export optionnel
    print()
    print("💾 Exporter les résultats?")
    export_results = {
        "timestamp": datetime.now().isoformat(),
        "queries": results,
        "chat_history": chatbot.conversation_history,
        "statistics": {
            "total_queries": len(results),
            "total_turns": len(chatbot.conversation_history),
            "documents_indexed": len(index.documents)
        }
    }
    
    output_file = "rag_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export_results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Résultats exportés dans: {output_file}")


# ============================================================================
# Integration avec LlamaIndex (Avancé)
# ============================================================================

def integration_llamaindex_example():
    """
    Exemple d'intégration avec la vraie librairie LlamaIndex.
    À exécuter si 'pip install llama-index' est installé.
    """
    
    print("\n" + "=" * 80)
    print("🦙 Intégration LlamaIndex Réelle")
    print("=" * 80)
    print()
    
    try:
        from llama_index.core import Document, VectorStoreIndex, Settings
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI
        
        print("✓ LlamaIndex importé avec succès!")
        print()
        
        # Configuration
        Settings.llm = OpenAI(model="gpt-4")
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        # Charger documents
        docs = [
            Document(text=doc["content"], metadata={"title": doc["title"]})
            for doc in SAMPLE_DOCUMENTS
        ]
        
        # Créer l'index
        index = VectorStoreIndex.from_documents(docs)
        
        # Query engine
        query_engine = index.as_query_engine()
        
        # Requête
        response = query_engine.query("Qu'est-ce qu'un Transformer?")
        print(f"Réponse LlamaIndex:\n{response}")
        
        print()
        print("✓ Intégration réelle fonctionnelle!")
        
    except ImportError:
        print("⚠️  LlamaIndex non installé")
        print("Installation: pip install llama-index openai")
        print()
        print("Avec LlamaIndex, vous pouvez:")
        print("  - Charger différents formats (PDF, Word, HTML, etc.)")
        print("  - Utiliser des embeddings réels (OpenAI, Ollama, local)")
        print("  - Exécuter des opérations avancées (table extraction, etc.)")
        print("  - Persister les index (pour réutilisation)")
        print()


if __name__ == "__main__":
    # Exécuter la démo principale
    main()
    
    # Essayer l'intégration réelle si disponible
    integration_llamaindex_example()
