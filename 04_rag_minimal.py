#!/usr/bin/env python
"""
Script 4 : RAG Minimaliste (Chapitre 13).

Ce script simule un pipeline RAG complet et simple :
1. Retrieval : chercher les documents les plus similaires.
2. Augmentation : injecter ces documents dans le contexte.
3. Génération : le LLM utilise le contexte pour répondre.

Dépendances :
    pip install scikit-learn numpy

Utilisation :
    python 04_rag_minimal.py
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def main():
    # Mini-base de connaissances (knowledge base)
    documents = [
        "Le Transformer est une architecture basée sur l'attention multi-tête.",
        "La tokenisation convertit le texte en jetons (tokens).",
        "Le Transformer a été introduit en 2017 par Vaswani et ses collègues.",
        "L'attention multi-tête permet au modèle de regarder différentes représentations.",
        "Le BERT utilise des embeddings positionnels pour coder l'ordre des mots.",
        "La perplexité mesure à quel point un modèle est 'surpris' par un texte.",
        "Les LLMs modernes utilisent le décodage par paquets (beam search) ou par échantillonnage.",
    ]

    # Question de l'utilisateur
    question = "Comment fonctionne l'attention dans le Transformer?"

    print("=" * 70)
    print("RAG MINIMALISTE")
    print("=" * 70 + "\n")

    print(f"Question utilisateur: '{question}'\n")

    # 1. Vectorisation (TF-IDF pour la simplicité)
    print("=" * 70)
    print("ÉTAPE 1 : RETRIEVAL (Recherche des documents pertinents)")
    print("=" * 70 + "\n")

    # Combiner documents et question pour vectoriser
    all_texts = documents + [question]

    # TF-IDF : transformer les textes en vecteurs
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words=["le", "la", "de", "du", "et", "les"]
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Vecteur de la question (dernier indice)
    question_vector = tfidf_matrix[-1]
    doc_vectors = tfidf_matrix[:-1]

    # 2. Calcul de similarité (cosine similarity)
    similarities = cosine_similarity(question_vector, doc_vectors)[0]

    # 3. Récupérer les top-k documents pertinents
    k = 3
    top_k_indices = np.argsort(similarities)[::-1][:k]

    print(f"Top {k} documents récupérés (par score de similarité cosinus):\n")

    for rank, idx in enumerate(top_k_indices, 1):
        score = similarities[idx]
        doc = documents[idx]
        print(f"{rank}. Score: {score:.3f}")
        print(f"   {doc}\n")

    # 4. Construction du prompt augmenté
    print("\n" + "=" * 70)
    print("ÉTAPE 2 : AUGMENTATION (Injection du contexte)")
    print("=" * 70 + "\n")

    context = "\n".join(
        [f"- {documents[i]}" for i in top_k_indices]
    )

    augmented_prompt = f"""Vous êtes un assistant expert en Machine Learning.

Voici des documents pertinents:
{context}

Question: {question}

Réponse basée sur les documents:"""

    print("Prompt augmenté envoyé au LLM:")
    print("-" * 70)
    print(augmented_prompt)
    print("-" * 70)

    # 5. Génération (simulée)
    print("\n" + "=" * 70)
    print("ÉTAPE 3 : GÉNÉRATION (Le LLM répond)")
    print("=" * 70 + "\n")

    simulated_response = """L'attention dans le Transformer fonctionne en plusieurs étapes :

1. **Projections Q, K, V** : Chaque jeton est projeté en trois représentations
   (Query, Key, Value).

2. **Calcul des scores** : On calcule Q @ K^T pour voir "qui regarde qui".

3. **Normalisation** : On applique softmax pour obtenir des poids entre 0 et 1.

4. **Agrégation** : On utilise ces poids pour combiner les valeurs (V).

5. **Multi-tête** : On répète ce processus avec plusieurs "têtes" en parallèle
   pour capturer différentes dépendances."""

    print("Réponse du modèle:")
    print("-" * 70)
    print(simulated_response)
    print("-" * 70)

    # Comparaison : sans RAG vs avec RAG
    print("\n" + "=" * 70)
    print("COMPARAISON")
    print("=" * 70 + "\n")

    print("❌ SANS RAG:")
    print("   Le modèle répond UNIQUEMENT de mémoire.")
    print("   → Risque d'hallucination")
    print("   → Connaissances figées (pas d'accès aux données récentes)")
    print("   → Pas de sources à vérifier\n")

    print("✅ AVEC RAG:")
    print("   Le modèle reçoit les documents DIRECTEMENT dans le contexte.")
    print("   → Réponses basées sur des sources verifiables")
    print("   → Peut accéder à des données externes (fichiers, APIs, BDD)")
    print("   → Utilisateur peut vérifier les sources citées")

    print("\n" + "=" * 70)
    print("NOTES PRATIQUES")
    print("=" * 70 + "\n")

    print("• TF-IDF (utilisé ici) est simple mais basique.")
    print("  → En production, on utilise des embeddings denses")
    print("    (BERT, E5, OpenAI embeddings, etc.)\n")

    print("• Chunking : découper les documents de manière intelligente.")
    print("  → Chunk trop gros → perte de précision")
    print("  → Chunk trop petit → fragmentation\n")

    print("• Re-ranking : après retrieval, filtrer/classer les résultats.")
    print("  → Utiliser un modèle spécialisé (CrossEncoder)")
    print("  → Réduit le 'bruit' injecté au LLM\n")

    print("• En-context learning : plus le contexte est long, plus il est")
    print("  important de bien le structurer et de le prioriser.")


if __name__ == "__main__":
    main()
