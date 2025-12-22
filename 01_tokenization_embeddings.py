#!/usr/bin/env python
"""
Script 1: Tokenization and Sequence Length Impact (Chapter 2).

This script illustrates:
- How tokenizers (BPE, WordPiece) fragment text.
- The impact of token count on computational cost (O(n²) for attention).
- How different languages have different ratios.

Note: Uses GPT-2 as fallback if LLaMA is not accessible.

Dependencies:
    pip install transformers torch

Usage:
    python 01_tokenization_embeddings.py
"""

from transformers import AutoTokenizer
import sys


def main():
    try:
        # Load a tokenizer (LLaMA 2 is small and easy to load)
        print("Loading LLaMA 2 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Error: {e}")
        print("\nAlternative: using an open tokenizer (GPT-2)")
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Test texts (French and English)
    textes = [
        "L'IA est utile",
        "Bonjour, comment allez-vous?",
        "The quick brown fox jumps over the lazy dog.",
        "Python est un langage de programmation.",
    ]

    print("\n" + "=" * 60)
    print("TOKENIZATION ANALYSIS")
    print("=" * 60 + "\n")

    for texte in textes:
        tokens = tokenizer.encode(texte)
        token_strings = tokenizer.convert_ids_to_tokens(tokens)

        print(f"Text: {texte}")
        print(f"  Token count: {len(tokens)}")
        print(f"  Token IDs: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        print(f"  Tokens (text): {token_strings[:10]}{'...' if len(tokens) > 10 else ''}")
        print()

    # Demonstration: length impact
    print("\n" + "=" * 60)
    print("IMPACT ON COMPUTATIONAL COST")
    print("=" * 60 + "\n")

    texte_court = "Bonjour"
    texte_long = texte_court * 100  # Artificial repetition

    n_court = len(tokenizer.encode(texte_court))
    n_long = len(tokenizer.encode(texte_long))

    print(f"Short text ({len(texte_court)} characters) → {n_court} tokens")
    print(f"Long text ({len(texte_long)} characters) → {n_long} tokens")
    print(f"Increase factor: {n_long / n_court:.1f}x")

    print("\n⚠️ IMPLICATIONS:")
    print("  • More tokens = more VRAM consumed")
    print("  • More tokens = higher latency")
    print("  • More tokens = higher API cost")
    print("  • Inference cost ∝ O(n²) for multi-head attention")


if __name__ == "__main__":
    main()
