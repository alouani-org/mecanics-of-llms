#!/usr/bin/env python
"""
Script 4: Minimal RAG (Chapter 13).

This script simulates a complete and simple RAG pipeline:
1. Retrieval: search for the most similar documents.
2. Augmentation: inject these documents into the context.
3. Generation: the LLM uses the context to respond.

Dependencies:
    pip install scikit-learn numpy

Usage:
    python 04_rag_minimal.py
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def main():
    # Mini knowledge base
    documents = [
        "The Transformer is an architecture based on multi-head attention.",
        "Tokenization converts text into tokens.",
        "The Transformer was introduced in 2017 by Vaswani and colleagues.",
        "Multi-head attention allows the model to look at different representations.",
        "BERT uses positional embeddings to encode word order.",
        "Perplexity measures how 'surprised' a model is by a text.",
        "Modern LLMs use beam search decoding or sampling.",
    ]

    # User question
    question = "How does attention work in the Transformer?"

    print("=" * 70)
    print("MINIMAL RAG")
    print("=" * 70 + "\n")

    print(f"User question: '{question}'\n")

    # 1. Vectorization (TF-IDF for simplicity)
    print("=" * 70)
    print("STEP 1: RETRIEVAL (Search for relevant documents)")
    print("=" * 70 + "\n")

    # Combine documents and question for vectorization
    all_texts = documents + [question]

    # TF-IDF: transform texts into vectors
    # Common English stop words (improves quality)
    english_stop_words = [
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall",
        "can", "need", "dare", "to", "of", "in", "for", "on", "with",
        "at", "by", "from", "as", "into", "through", "during", "before",
        "after", "above", "below", "between", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how",
        "all", "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "just", "and", "but", "if", "or", "because", "until",
        "while", "although", "this", "that", "these", "those"
    ]
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words=english_stop_words
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Question vector (last index)
    question_vector = tfidf_matrix[-1]
    doc_vectors = tfidf_matrix[:-1]

    # 2. Similarity calculation (cosine similarity)
    similarities = cosine_similarity(question_vector, doc_vectors)[0]

    # 3. Retrieve top-k relevant documents
    k = 3
    top_k_indices = np.argsort(similarities)[::-1][:k]

    print(f"Top {k} documents retrieved (by cosine similarity score):\n")

    for rank, idx in enumerate(top_k_indices, 1):
        score = similarities[idx]
        doc = documents[idx]
        print(f"{rank}. Score: {score:.3f}")
        print(f"   {doc}\n")

    # 4. Build augmented prompt
    print("\n" + "=" * 70)
    print("STEP 2: AUGMENTATION (Context injection)")
    print("=" * 70 + "\n")

    context = "\n".join(
        [f"- {documents[i]}" for i in top_k_indices]
    )

    augmented_prompt = f"""You are an expert Machine Learning assistant.

Here are relevant documents:
{context}

Question: {question}

Answer based on the documents:"""

    print("Augmented prompt sent to LLM:")
    print("-" * 70)
    print(augmented_prompt)
    print("-" * 70)

    # 5. Generation (simulated)
    print("\n" + "=" * 70)
    print("STEP 3: GENERATION (The LLM responds)")
    print("=" * 70 + "\n")

    simulated_response = """Attention in the Transformer works in several steps:

1. **Q, K, V Projections**: Each token is projected into three representations
   (Query, Key, Value).

2. **Score calculation**: We compute Q @ K^T to see "who looks at whom".

3. **Normalization**: We apply softmax to get weights between 0 and 1.

4. **Aggregation**: We use these weights to combine the values (V).

5. **Multi-head**: We repeat this process with several "heads" in parallel
   to capture different dependencies."""

    print("Model response:")
    print("-" * 70)
    print(simulated_response)
    print("-" * 70)

    # Comparison: without RAG vs with RAG
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70 + "\n")

    print("❌ WITHOUT RAG:")
    print("   The model responds ONLY from memory.")
    print("   → Risk of hallucination")
    print("   → Frozen knowledge (no access to recent data)")
    print("   → No sources to verify\n")

    print("✅ WITH RAG:")
    print("   The model receives documents DIRECTLY in the context.")
    print("   → Answers based on verifiable sources")
    print("   → Can access external data (files, APIs, databases)")
    print("   → User can verify cited sources")

    print("\n" + "=" * 70)
    print("PRACTICAL NOTES")
    print("=" * 70 + "\n")

    print("• TF-IDF (used here) is simple but basic.")
    print("  → In production, use dense embeddings")
    print("    (BERT, E5, OpenAI embeddings, etc.)\n")

    print("• Chunking: split documents intelligently.")
    print("  → Chunk too large → loss of precision")
    print("  → Chunk too small → fragmentation\n")

    print("• Re-ranking: after retrieval, filter/rank results.")
    print("  → Use a specialized model (CrossEncoder)")
    print("  → Reduces 'noise' injected to the LLM\n")

    print("• In-context learning: the longer the context, the more")
    print("  important it is to structure and prioritize it well.")


if __name__ == "__main__":
    main()
