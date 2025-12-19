#!/usr/bin/env python
"""
Script 5 : Evaluation Pass@k (Chapitre 12).

Ce script simule une évaluation Pass@k :
- Pass@k : probabilité d'au moins UNE réussite en k tentatives.
- Pass^k : probabilité que TOUTES les k tentatives réussissent.

Dépendances :
    pip install numpy

Utilisation :
    python 05_pass_at_k_evaluation.py
"""

import numpy as np


def main():
    print("=" * 70)
    print("ÉVALUATION PASS@K")
    print("=" * 70 + "\n")

    # Simuler les résultats de plusieurs générations pour une même question
    # (par ex: "Écris une fonction Python pour calculer la factorielle")
    p_success = 0.3  # Le modèle réussit 30% du temps
    np.random.seed(42)
    n_attempts = 100

    # True = succès, False = échec
    results = np.random.rand(n_attempts) < p_success

    print(f"Paramètres de simulation:")
    print(f"  • Nombre de tentatives: {n_attempts}")
    print(f"  • Probabilité de succès par tentative: {p_success:.0%}\n")

    print(f"Résultats bruts:")
    print(f"  • Réussites: {results.sum()} / {n_attempts}")
    print(f"  • Taux de succès brut: {results.mean():.1%}\n")

    # === Pass@k ===
    print("=" * 70)
    print("PASS@K (Au moins UNE réussite en k tentatives)")
    print("=" * 70 + "\n")

    print("Formule: Pass@k = 1 - (1 - p)^k")
    print("         où p = taux de succès unitaire\n")

    for k in [1, 3, 5, 10, 20]:
        # Probabilité : au moins une réussite sur k tentatives
        pass_at_k = 1 - (1 - results.mean()) ** k

        print(f"Pass@{k:2d} = {pass_at_k:.1%}")
        print(f"          (chance d'obtenir ≥1 succès si je tente {k} fois)")
        print(f"          (1 - 0.7^{k} = 1 - {(1-results.mean())**k:.4f})")
        print()

    # === Pass^k (strict) ===
    print("\n" + "=" * 70)
    print("PASS^K (TOUTES les k tentatives réussissent) — STRICT")
    print("=" * 70 + "\n")

    print("Formule: Pass^k = p^k")
    print("         où p = taux de succès unitaire")
    print("\n⚠️  CLARIFICATION:")
    print("  Pass^k est PLUS DIFFICILE que Pass@k (courbe descendante)")
    print("  • Pass@k: 'j'ai besoin que AU MOINS 1 réussisse'")
    print("  • Pass^k:  'j'ai besoin que TOUS les k réussissent'")
    print("\n  Cas d'usage: Systèmes critiques où AUCUNE erreur n'est acceptable.\n")

    # Diviser les 100 tentatives en groupes de k
    for k in [1, 3, 5, 10]:
        groups = n_attempts // k

        # Compter combien de groupes sont "tout succès"
        success_all_k = sum(
            all(results[i * k : (i + 1) * k])
            for i in range(groups)
        )

        # Probabilité empirique
        pass_strict_k = success_all_k / groups if groups > 0 else 0

        # Probabilité théorique
        theoretical = p_success ** k

        print(f"Pass^{k} = {pass_strict_k:.1%} (empirique)")
        print(f"          {theoretical:.1%} (théorique : 0.3^{k})")
        print(f"          (tous les {k} essais DOIVENT réussir)")
        print()

    # === Application pratique ===
    print("\n" + "=" * 70)
    print("APPLICATIONS PRATIQUES")
    print("=" * 70 + "\n")

    print("1️⃣ RECHERCHE (Coding competitions):")
    print("   • Problème : générer du code correct pour HumanEval")
    print("   • Métrique: Pass@k avec k=1, 5, 10, 100")
    print("   • Utilisation: échantillonner plusieurs réponses et prendre la meilleure\n")

    print("2️⃣ SYSTÈMES À HAUTE FIABILITÉ (Agents en production):")
    print("   • Problème: exécuter une action complexe (réserver un vol, passer une")
    print("     commande, etc.)")
    print("   • Métrique: Pass^k — les TOUTES les exécutions doivent réussir")
    print("   • Besoin: p_success très élevé (90%+) sinon le système échoue\n")

    print("3️⃣ CHAT GÉNÉRALISTE:")
    print("   • Métrique: souvent Pass@1 (une seule tentative)")
    print("   • Limitation: mauvaises réponses ne sont vues qu'une fois\n")

    # === Graphique textuel ===
    print("\n" + "=" * 70)
    print("VISUALISATION SIMPLE (Pass@k vs Pass^k)")
    print("=" * 70 + "\n")

    ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pass_at_ks = [1 - (1 - p_success) ** k for k in ks]
    pass_strict_ks = [p_success ** k for k in ks]

    print("Pass@k (au moins une réussite) — MONTE rapidement:")
    for k, p_at_k in zip(ks[:5], pass_at_ks[:5]):
        bar = "█" * int(p_at_k * 40)
        print(f"  k={k:2d}:  {p_at_k:.1%}  {bar}")

    print("\nPass^k (toutes les k réussissent) — DESCEND rapidement:")
    for k, p_str_k in zip(ks[:5], pass_strict_ks[:5]):
        bar = "█" * int(p_str_k * 40)
        print(f"  k={k:2d}:  {p_str_k:.1%}  {bar}")

    # === Pièges courants ===
    print("\n" + "=" * 70)
    print("PIÈGES COURANTS")
    print("=" * 70 + "\n")

    print("🔴 Confondre Pass@1 et Pass@k:")
    print(f"   • Pass@1 = {pass_at_ks[0]:.1%}  (une seule tentative)")
    print(f"   • Pass@10 = {pass_at_ks[9]:.1%}  (10 tentatives)")
    print(f"   → Ne pas comparer directement!\n")

    print("🔴 Oublier que Pass^k diminue exponentiellement:")
    print(f"   • Même un modèle à 90% d'accuracy:")
    print(f"     - Pass^1 = 90%")
    print(f"     - Pass^5 = 59%")
    print(f"     - Pass^10 = 35%")
    print(f"   → En systèmes critiques, besoin de p TRÈS élevé\n")

    print("🔴 Mémoriser les benchmarks sans comprendre Pass@k:")
    print("   • Quand on dit 'GPT-4 = 92% sur HumanEval'")
    print("   • C'est souvent Pass@1 (une seule tentative par problème)")
    print("   • Pas Pass@100 (100 tentatives et prendre la meilleure)")


if __name__ == "__main__":
    main()
