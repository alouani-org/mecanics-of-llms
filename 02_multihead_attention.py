#!/usr/bin/env python
"""
Script 2: Self-Attention and Multi-Head (Chapter 3).

This script simulates a minimal multi-head attention layer, allowing you to:
- Understand Q, K, V projections.
- See how each head focuses on different dependencies.
- Verify that attention weights sum to 1 (probability distribution).

Dependencies:
    pip install torch numpy

Usage:
    python 02_multihead_attention.py
"""

import torch
import torch.nn.functional as F
import numpy as np


def main():
    # Parameters
    batch_size = 1
    seq_len = 4        # Sequence length ("The cat sleeps well")
    d_model = 64       # Model dimension
    num_heads = 2      # Number of attention heads
    d_head = d_model // num_heads

    print("=" * 60)
    print("MULTI-HEAD ATTENTION (SIMPLIFIED SIMULATION)")
    print("=" * 60 + "\n")

    # Concrete example with token names
    token_names = ["The", "cat", "sleeps", "well"]
    print(f"Example sentence: {' '.join(token_names)}\n")

    # Create a sequence of embeddings (simulated)
    # In practice, these are outputs from previous layers
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input x shape: {x.shape}")
    print(f"  (batch={batch_size}, seq_len={seq_len}, d_model={d_model})\n")

    # Linear projections for Q, K, V
    W_q = torch.randn(d_model, d_model)
    W_k = torch.randn(d_model, d_model)
    W_v = torch.randn(d_model, d_model)

    Q = x @ W_q  # [batch, seq_len, d_model]
    K = x @ W_k
    V = x @ W_v

    print(f"Q, K, V shapes: {Q.shape}\n")

    # Compute attention per head
    print("=" * 60)
    print("ATTENTION COMPUTATION PER HEAD")
    print("=" * 60 + "\n")

    attention_outputs = []

    for head_idx in range(num_heads):
        start, end = head_idx * d_head, (head_idx + 1) * d_head
        Q_h = Q[:, :, start:end]  # [batch, seq_len, d_head]
        K_h = K[:, :, start:end]
        V_h = V[:, :, start:end]

        # Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_head)) @ V
        scores = Q_h @ K_h.transpose(-2, -1) / np.sqrt(d_head)  # [batch, seq_len, seq_len]
        attention_weights = F.softmax(scores, dim=-1)            # [batch, seq_len, seq_len]
        output_h = attention_weights @ V_h                       # [batch, seq_len, d_head]

        attention_outputs.append(output_h)

        print(f"Head {head_idx}:")
        print(f"  Scores (raw): shape {scores.shape}")
        print(f"  Attention weights (after softmax):")
        print(f"    {attention_weights[0].detach().numpy()}")
        print(f"  Sum of weights for each token:")
        print(f"    {attention_weights[0].sum(dim=1).detach().numpy()}")
        print(f"  (Verification: each row should sum to ~1.0)")
        print()

    # Concatenate all heads
    output = torch.cat(attention_outputs, dim=-1)  # [batch, seq_len, d_model]

    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60 + "\n")
    print(f"Concatenated output: {output.shape}")
    print(f"(The {num_heads} heads are combined into a final vector per token)")

    print("\n💡 INTUITION:")
    print("  • Each head captures DIFFERENT dependencies in the sentence.")
    print(f"  • With our tokens {token_names}:")
    print("    - Head 0 may focus on 'The → cat' (subject-verb).")
    print("    - Head 1 may focus on 'cat → sleeps' (verb-adverb).")
    print("  • The fusion allows the model to combine these perspectives.")
    print(f"\n  Observation: Each head assigns different attention weights.")
    print(f"  Example with {token_names[1]} (token 1):")
    print(f"    - Tête 0: 'chat' regarde surtout vers 'Le' et 'dort'")
    print(f"    - Tête 1: 'chat' regarde plus vers 'bien'")
    print(f"  → Perspectives complémentaires = représentation riche!")


if __name__ == "__main__":
    main()
