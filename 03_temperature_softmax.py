#!/usr/bin/env python
"""
Script 3 : Softmax et Température (Chapitres 7 & 11).

Ce script montre comment la température modifie la distribution softmax :
- À basse température (T < 1), la distribution devient pointue → greedy.
- À haute température (T > 1), la distribution s'aplatit → diversité.
- L'effet sur l'entropie (mesure de dispersion).

Dépendances :
    pip install torch numpy
    pip install matplotlib  # optionnel, pour les graphiques

Utilisation :
    python 03_temperature_softmax.py
"""

import torch
import torch.nn.functional as F
import numpy as np


def plot_temperature(logits, temperatures):
    """Visualise l'effet de la température (nécessite matplotlib)."""
    try:
        import matplotlib.pyplot as plt

        probabilities_list = []
        entropies = []

        for T in temperatures:
            probs = F.softmax(logits / T, dim=0).numpy()
            probabilities_list.append(probs)

            # Entropie Shannon
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Subplot 1 : Probabilités par température
        x = np.arange(len(logits))
        width = 0.15

        for i, T in enumerate(temperatures):
            ax1.bar(x + i * width, probabilities_list[i], width, label=f"T={T}")

        ax1.set_xlabel("Token ID")
        ax1.set_ylabel("Probabilité")
        ax1.set_title("Effet de la température sur softmax")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        # Subplot 2 : Entropie vs Température
        ax2.plot(temperatures, entropies, "o-", linewidth=2, markersize=8)
        ax2.set_xlabel("Température")
        ax2.set_ylabel("Entropie Shannon")
        ax2.set_title("Entropie augmente avec la température")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("temperature_effect.png", dpi=100)
        print("\n✅ Graphique sauvegardé: temperature_effect.png\n")
        return True
    except ImportError:
        print("\n⚠️  matplotlib non installé. Graphique ignoré.")
        print("   Pour voir le graphique: pip install matplotlib\n")
        return False


def main():
    # Logits simplifiés (comme la sortie du modèle avant softmax)
    # Imagine que ce sont les scores pour [chat, chien, souris, oiseau]
    logits = torch.tensor([2.0, 1.0, 0.5, 0.1])
    token_names = ["chat", "chien", "souris", "oiseau"]
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]

    print("=" * 70)
    print("EFFET DE LA TEMPÉRATURE SUR LA DISTRIBUTION SOFTMAX")
    print("=" * 70 + "\n")

    print(f"Logits bruts: {logits.numpy()}")
    print(f"Tokens: {token_names}\n")

    probabilities_list = []
    entropies = []

    print("=" * 70)
    print("RÉSULTATS PAR TEMPÉRATURE")
    print("=" * 70 + "\n")

    for T in temperatures:
        probs = F.softmax(logits / T, dim=0)
        probabilities_list.append(probs.numpy())

        # Entropie Shannon : -sum(p * log(p))
        entropy = -np.sum(probs.numpy() * np.log(probs.numpy() + 1e-10))
        entropies.append(entropy)

        print(f"Température = {T}")
        print(f"  Probabilités (normalisées):")
        for name, prob in zip(token_names, probs.numpy()):
            bar = "█" * int(prob * 50)  # Barre simple
            print(f"    {name:8s}: {prob:.3f}  {bar}")
        print(f"  Entropie: {entropy:.3f}")
        print()

    # Visualisation optionnelle
    print("=" * 70)
    print("VISUALISATION")
    print("=" * 70)

    has_plot = plot_temperature(logits, temperatures)

    # Interprétation
    print("=" * 70)
    print("INTERPRÉTATION")
    print("=" * 70 + "\n")

    print("✓ À T=0.1 (basse température):")
    print("  → 'chat' domine largement (greedy decoding).")
    print("  → Faible entropie → Sortie déterministe et répétitive.\n")

    print("✓ À T=1.0 (température neutre):")
    print("  → Distribution 'naturelle' selon les logits.")
    print("  → C'est la température par défaut.\n")

    print("✓ À T=5.0 (haute température):")
    print("  → Distribution quasi-uniforme (tous les tokens presque égaux).")
    print("  → Haute entropie → Diversité, mais aussi incohérence.\n")

    print("=" * 70)
    print("APPLICATIONS PRATIQUES")
    print("=" * 70 + "\n")

    print("📌 Récapitulatif rapidement:")
    print(f"  Entropie min (T={temperatures[0]}): {entropies[0]:.3f}")
    print(f"  Entropie max (T={temperatures[-1]}): {entropies[-1]:.3f}")
    print()
    print("  → Augmenter T de 0.1 à 5.0 multiplie l'entropie par")
    print(f"    {entropies[-1] / (entropies[0] + 1e-10):.1f}x")
    print()
    print("  Pièges courants:")
    print("    🔴 T=0 sur GPU n'est pas parfaitement déterministe (arrondis float).")
    print("    🔴 T trop élevé → hallucinations et incohérence.")
    print("    🟢 T=0.7-0.9 : bon compromis créativité/stabilité.")


if __name__ == "__main__":
    main()
