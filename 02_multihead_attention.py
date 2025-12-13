#!/usr/bin/env python
"""
Script 2 : Self-Attention et Multi-Head (Chapitre 3).

Ce script simule une couche d'attention multi-tête minimale, permettant de :
- Comprendre les projections Q, K, V.
- Voir comment chaque tête focalise sur différentes dépendances.
- Vérifier que les poids d'attention somment à 1 (distribution de probabilité).

Dépendances :
    pip install torch numpy

Utilisation :
    python 02_multihead_attention.py
"""

import torch
import torch.nn.functional as F
import numpy as np


def main():
    # Paramètres
    batch_size = 1
    seq_len = 4        # Longueur de séquence ("Le chat dort bien")
    d_model = 64       # Dimension du modèle
    num_heads = 2      # Nombre de têtes d'attention
    d_head = d_model // num_heads

    print("=" * 60)
    print("MULTI-HEAD ATTENTION (SIMULATION SIMPLIFIÉE)")
    print("=" * 60 + "\n")

    # Exemple concret avec noms de tokens
    token_names = ["Le", "chat", "dort", "bien"]
    print(f"Phrase d'exemple: {' '.join(token_names)}\n")

    # Créer une séquence d'embeddings (simulée)
    # En pratique, ce sont les sorties des couches précédentes
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Entrée x shape: {x.shape}")
    print(f"  (batch={batch_size}, seq_len={seq_len}, d_model={d_model})\n")

    # Projections linéaires pour Q, K, V
    W_q = torch.randn(d_model, d_model)
    W_k = torch.randn(d_model, d_model)
    W_v = torch.randn(d_model, d_model)

    Q = x @ W_q  # [batch, seq_len, d_model]
    K = x @ W_k
    V = x @ W_v

    print(f"Q, K, V shapes: {Q.shape}\n")

    # Calcul de l'attention par tête
    print("=" * 60)
    print("CALCUL DE L'ATTENTION PAR TÊTE")
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

        print(f"Tête {head_idx}:")
        print(f"  Scores (bruts): shape {scores.shape}")
        print(f"  Poids d'attention (après softmax):")
        print(f"    {attention_weights[0].detach().numpy()}")
        print(f"  Somme des poids pour chaque token:")
        print(f"    {attention_weights[0].sum(dim=1).detach().numpy()}")
        print(f"  (Vérification: chaque ligne doit sommer à ~1.0)")
        print()

    # Concaténer toutes les têtes
    output = torch.cat(attention_outputs, dim=-1)  # [batch, seq_len, d_model]

    print("=" * 60)
    print("RÉSULTAT FINAL")
    print("=" * 60 + "\n")
    print(f"Sortie concaténée: {output.shape}")
    print(f"(Les {num_heads} têtes sont réunies pour un vecteur final par token)")

    print("\n💡 INTUITION:")
    print("  • Chaque tête capture DIFFÉRENTES dépendances dans la phrase.")
    print(f"  • Avec nos tokens {token_names}:")
    print("    - Tête 0 peut se concentrer sur 'Le → chat' (sujet-verbe).")
    print("    - Tête 1 peut se concentrer sur 'chat → dort' (verbe-adverbe).")
    print("  • La fusion permet au modèle de combiner ces perspectives.")
    print(f"\n  Observation: Chaque tête assigne des poids d'attention différents.")
    print(f"  Exemple avec {token_names[1]} (token 1):")
    print(f"    - Tête 0: 'chat' regarde surtout vers 'Le' et 'dort'")
    print(f"    - Tête 1: 'chat' regarde plus vers 'bien'")
    print(f"  → Perspectives complémentaires = représentation riche!")


if __name__ == "__main__":
    main()
