#!/usr/bin/env python
"""
Script 3: Softmax and Temperature (Chapters 7 & 11).

This script shows how temperature modifies the softmax distribution:
- At low temperature (T < 1), the distribution becomes sharp → greedy.
- At high temperature (T > 1), the distribution flattens → diversity.
- The effect on entropy (dispersion measure).

Dependencies:
    pip install torch numpy
    pip install matplotlib  # optional, for graphs

Usage:
    python 03_temperature_softmax.py
"""

import torch
import torch.nn.functional as F
import numpy as np


def plot_temperature(logits, temperatures):
    """Visualize the temperature effect (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt

        probabilities_list = []
        entropies = []

        for T in temperatures:
            probs = F.softmax(logits / T, dim=0).numpy()
            probabilities_list.append(probs)

            # Shannon entropy
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Subplot 1: Probabilities by temperature
        x = np.arange(len(logits))
        width = 0.15

        for i, T in enumerate(temperatures):
            ax1.bar(x + i * width, probabilities_list[i], width, label=f"T={T}")

        ax1.set_xlabel("Token ID")
        ax1.set_ylabel("Probability")
        ax1.set_title("Temperature effect on softmax")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        # Subplot 2: Entropy vs Temperature
        ax2.plot(temperatures, entropies, "o-", linewidth=2, markersize=8)
        ax2.set_xlabel("Temperature")
        ax2.set_ylabel("Shannon Entropy")
        ax2.set_title("Entropy increases with temperature")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("temperature_effect.png", dpi=100)
        print("\n✅ Graph saved: temperature_effect.png\n")
        return True
    except ImportError:
        print("\n⚠️  matplotlib not installed. Graph skipped.")
        print("   To see the graph: pip install matplotlib\n")
        return False


def main():
    # Simplified logits (like model output before softmax)
    # Imagine these are scores for [cat, dog, mouse, bird]
    logits = torch.tensor([2.0, 1.0, 0.5, 0.1])
    token_names = ["cat", "dog", "mouse", "bird"]
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]

    print("=" * 70)
    print("TEMPERATURE EFFECT ON SOFTMAX DISTRIBUTION")
    print("=" * 70 + "\n")

    print(f"Raw logits: {logits.numpy()}")
    print(f"Tokens: {token_names}\n")

    probabilities_list = []
    entropies = []

    print("=" * 70)
    print("RESULTS BY TEMPERATURE")
    print("=" * 70 + "\n")

    for T in temperatures:
        probs = F.softmax(logits / T, dim=0)
        probabilities_list.append(probs.numpy())

        # Shannon entropy: -sum(p * log(p))
        entropy = -np.sum(probs.numpy() * np.log(probs.numpy() + 1e-10))
        entropies.append(entropy)

        print(f"Temperature = {T}")
        print(f"  Probabilities (normalized):")
        for name, prob in zip(token_names, probs.numpy()):
            bar = "█" * int(prob * 50)  # Simple bar
            print(f"    {name:8s}: {prob:.3f}  {bar}")
        print(f"  Entropy: {entropy:.3f}")
        print()

    # Optional visualization
    print("=" * 70)
    print("VISUALIZATION")
    print("=" * 70)

    has_plot = plot_temperature(logits, temperatures)

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70 + "\n")

    print("✓ At T=0.1 (low temperature):")
    print("  → 'cat' dominates largely (greedy decoding).")
    print("  → Low entropy → Deterministic and repetitive output.\n")

    print("✓ At T=1.0 (neutral temperature):")
    print("  → 'Natural' distribution according to logits.")
    print("  → This is the default temperature.\n")

    print("✓ At T=5.0 (high temperature):")
    print("  → Near-uniform distribution (all tokens almost equal).")
    print("  → High entropy → Diversity, but also incoherence.\n")

    print("=" * 70)
    print("PRACTICAL APPLICATIONS")
    print("=" * 70 + "\n")

    print("📌 Quick summary:")
    print(f"  Min entropy (T={temperatures[0]}): {entropies[0]:.3f}")
    print(f"  Max entropy (T={temperatures[-1]}): {entropies[-1]:.3f}")
    print()
    print("  → Increasing T from 0.1 to 5.0 multiplies entropy by")
    print(f"    {entropies[-1] / (entropies[0] + 1e-10):.1f}x")
    print()
    print("  Common pitfalls:")
    print("    🔴 T=0 on GPU is not perfectly deterministic (float rounding).")
    print("    🔴 T too high → hallucinations and incoherence.")
    print("    🟢 T=0.7-0.9: good creativity/stability compromise.")


if __name__ == "__main__":
    main()
