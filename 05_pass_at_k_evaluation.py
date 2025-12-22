#!/usr/bin/env python
"""
Script 5: Pass@k Evaluation (Chapter 12).

This script simulates a Pass@k evaluation:
- Pass@k: probability of at least ONE success in k attempts.
- Pass^k: probability that ALL k attempts succeed.

Dependencies:
    pip install numpy

Usage:
    python 05_pass_at_k_evaluation.py
"""

import numpy as np


def main():
    print("=" * 70)
    print("PASS@K EVALUATION")
    print("=" * 70 + "\n")

    # Simulate results of multiple generations for the same question
    # (e.g., "Write a Python function to calculate factorial")
    p_success = 0.3  # The model succeeds 30% of the time
    np.random.seed(42)
    n_attempts = 100

    # True = success, False = failure
    results = np.random.rand(n_attempts) < p_success

    print(f"Simulation parameters:")
    print(f"  • Number of attempts: {n_attempts}")
    print(f"  • Success probability per attempt: {p_success:.0%}\n")

    print(f"Raw results:")
    print(f"  • Successes: {results.sum()} / {n_attempts}")
    print(f"  • Raw success rate: {results.mean():.1%}\n")

    # === Pass@k ===
    print("=" * 70)
    print("PASS@K (At least ONE success in k attempts)")
    print("=" * 70 + "\n")

    print("Formula: Pass@k = 1 - (1 - p)^k")
    print("         where p = unit success rate\n")

    for k in [1, 3, 5, 10, 20]:
        # Probability: at least one success in k attempts
        pass_at_k = 1 - (1 - results.mean()) ** k

        print(f"Pass@{k:2d} = {pass_at_k:.1%}")
        print(f"          (chance of getting ≥1 success if I try {k} times)")
        print(f"          (1 - 0.7^{k} = 1 - {(1-results.mean())**k:.4f})")
        print()

    # === Pass^k (strict) ===
    print("\n" + "=" * 70)
    print("PASS^K (ALL k attempts succeed) — STRICT")
    print("=" * 70 + "\n")

    print("Formula: Pass^k = p^k")
    print("         where p = unit success rate")
    print("\n⚠️  CLARIFICATION:")
    print("  Pass^k is HARDER than Pass@k (descending curve)")
    print("  • Pass@k: 'I need AT LEAST 1 to succeed'")
    print("  • Pass^k: 'I need ALL k to succeed'")
    print("\n  Use case: Critical systems where NO errors are acceptable.\n")

    # Divide the 100 attempts into groups of k
    for k in [1, 3, 5, 10]:
        groups = n_attempts // k

        # Count how many groups are "all success"
        success_all_k = sum(
            all(results[i * k : (i + 1) * k])
            for i in range(groups)
        )

        # Empirical probability
        pass_strict_k = success_all_k / groups if groups > 0 else 0

        # Theoretical probability
        theoretical = p_success ** k

        print(f"Pass^{k} = {pass_strict_k:.1%} (empirical)")
        print(f"          {theoretical:.1%} (theoretical: 0.3^{k})")
        print(f"          (all {k} attempts MUST succeed)")
        print()

    # === Practical applications ===
    print("\n" + "=" * 70)
    print("PRACTICAL APPLICATIONS")
    print("=" * 70 + "\n")

    print("1️⃣ RESEARCH (Coding competitions):")
    print("   • Problem: generate correct code for HumanEval")
    print("   • Metric: Pass@k with k=1, 5, 10, 100")
    print("   • Usage: sample multiple responses and take the best one\n")

    print("2️⃣ HIGH-RELIABILITY SYSTEMS (Production agents):")
    print("   • Problem: execute a complex action (book a flight, place an")
    print("     order, etc.)")
    print("   • Metric: Pass^k — ALL executions must succeed")
    print("   • Requirement: very high p_success (90%+) otherwise the system fails\n")

    print("3️⃣ GENERAL CHAT:")
    print("   • Metric: often Pass@1 (single attempt)")
    print("   • Limitation: bad responses are only seen once\n")

    # === Text-based graph ===
    print("\n" + "=" * 70)
    print("SIMPLE VISUALIZATION (Pass@k vs Pass^k)")
    print("=" * 70 + "\n")

    ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pass_at_ks = [1 - (1 - p_success) ** k for k in ks]
    pass_strict_ks = [p_success ** k for k in ks]

    print("Pass@k (at least one success) — RISES quickly:")
    for k, p_at_k in zip(ks[:5], pass_at_ks[:5]):
        bar = "█" * int(p_at_k * 40)
        print(f"  k={k:2d}:  {p_at_k:.1%}  {bar}")

    print("\nPass^k (all k succeed) — FALLS quickly:")
    for k, p_str_k in zip(ks[:5], pass_strict_ks[:5]):
        bar = "█" * int(p_str_k * 40)
        print(f"  k={k:2d}:  {p_str_k:.1%}  {bar}")

    # === Common pitfalls ===
    print("\n" + "=" * 70)
    print("COMMON PITFALLS")
    print("=" * 70 + "\n")

    print("🔴 Confusing Pass@1 and Pass@k:")
    print(f"   • Pass@1 = {pass_at_ks[0]:.1%}  (single attempt)")
    print(f"   • Pass@10 = {pass_at_ks[9]:.1%}  (10 attempts)")
    print(f"   → Do not compare directly!\n")

    print("🔴 Forgetting that Pass^k decreases exponentially:")
    print(f"   • Even a model with 90% accuracy:")
    print(f"     - Pass^1 = 90%")
    print(f"     - Pass^5 = 59%")
    print(f"     - Pass^10 = 35%")
    print(f"   → In critical systems, need VERY high p\n")

    print("🔴 Memorizing benchmarks without understanding Pass@k:")
    print("   • When they say 'GPT-4 = 92% on HumanEval'")
    print("   • It's often Pass@1 (single attempt per problem)")
    print("   • Not Pass@100 (100 attempts and take the best one)")


if __name__ == "__main__":
    main()
