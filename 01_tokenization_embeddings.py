#!/usr/bin/env python
"""
Script 1 : Tokenisation et impact sur la longueur des séquences (Chapitre 2).

Ce script illustre :
- Comment les tokenizers (BPE, WordPiece) fragmentent le texte.
- L'impact du nombre de tokens sur le coût computationnel (O(n²) pour l'attention).
- Comment des langues différentes ont des ratios différents.

Note : Utilise GPT-2 comme fallback si LLaMA n'est pas accessible.

Dépendances :
    pip install transformers torch

Utilisation :
    python 01_tokenization_embeddings.py
"""

from transformers import AutoTokenizer
import sys


def main():
    try:
        # Charger un tokenizer (LLaMA 2 est petit et facile à charger)
        print("Chargement du tokenizer LLaMA 2...")
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Erreur : {e}")
        print("\nAlternative : utiliser un tokenizer ouvert (GPT-2)")
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Textes de test (français et anglais)
    textes = [
        "L'IA est utile",
        "Bonjour, comment allez-vous?",
        "The quick brown fox jumps over the lazy dog.",
        "Python est un langage de programmation.",
    ]

    print("\n" + "=" * 60)
    print("ANALYSE DE TOKENISATION")
    print("=" * 60 + "\n")

    for texte in textes:
        tokens = tokenizer.encode(texte)
        token_strings = tokenizer.convert_ids_to_tokens(tokens)

        print(f"Texte: {texte}")
        print(f"  Nombre de tokens: {len(tokens)}")
        print(f"  Token IDs: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        print(f"  Tokens (texte): {token_strings[:10]}{'...' if len(tokens) > 10 else ''}")
        print()

    # Démonstration : impact de la longueur
    print("\n" + "=" * 60)
    print("IMPACT SUR LE COÛT DE CALCUL")
    print("=" * 60 + "\n")

    texte_court = "Bonjour"
    texte_long = texte_court * 100  # Répétition artificielle

    n_court = len(tokenizer.encode(texte_court))
    n_long = len(tokenizer.encode(texte_long))

    print(f"Texte court ({len(texte_court)} caractères) → {n_court} tokens")
    print(f"Texte long ({len(texte_long)} caractères) → {n_long} tokens")
    print(f"Facteur d'augmentation: {n_long / n_court:.1f}x")

    print("\n⚠️ IMPLICATIONS:")
    print("  • Plus de tokens = plus de VRAM consommée")
    print("  • Plus de tokens = latence plus élevée")
    print("  • Plus de tokens = coût API plus cher")
    print("  • Coût d'inférence ∝ O(n²) pour l'attention multi-tête")


if __name__ == "__main__":
    main()
