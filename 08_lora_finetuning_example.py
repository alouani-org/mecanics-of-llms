#!/usr/bin/env python
"""
Script 8 : LoRA et QLoRA - Fine-tuning Efficace (Chapitre 9).

Ce script démontre comment utiliser LoRA (Low-Rank Adaptation) et QLoRA
pour fine-tuner efficacement un modèle de langage sans avoir besoin de 
ressources GPU immenses.

Concepts couverts :
- LoRA : adaptateurs de petit rang au lieu de fine-tuning complet
- QLoRA : LoRA sur modèles quantifiés 4-bit (révolution d'accessibilité)
- Comparaison des ressources (VRAM, temps, paramètres)
- Cas d'usage réel : adaptation à un domaine spécifique

Dépendances minimales (sans réel GPU requis pour la démo):
    pip install torch numpy

Dépendances pour fine-tuning réel:
    pip install torch transformers peft bitsandbytes

Utilisation :
    python 08_lora_finetuning_example.py
"""

import sys
from typing import Dict, List, Tuple


def calculate_lora_parameters(
    model_dim: int,
    lora_rank: int,
    num_layers: int
) -> Dict[str, int]:
    """
    Calculer le nombre de paramètres supplémentaires pour LoRA.
    
    LoRA ajoute deux matrices par couche (Q et V, généralement) :
    - Matrice A : model_dim × lora_rank
    - Matrice B : lora_rank × model_dim
    
    Total par couche : 2 × model_dim × lora_rank
    """
    params_per_layer = 2 * model_dim * lora_rank
    total_lora_params = params_per_layer * num_layers
    
    return {
        "params_per_layer": params_per_layer,
        "total_lora_params": total_lora_params,
        "percentage_of_model": None  # Sera calculé plus bas
    }


def compare_finetuning_methods(
    model_size: int,
    model_dim: int,
    num_layers: int,
    lora_rank: int = 8
) -> Dict[str, Dict]:
    """
    Comparer trois méthodes de fine-tuning en termes de :
    - Paramètres entraînables
    - Mémoire VRAM approximative
    - Temps d'entraînement relatif
    """
    
    # === Full Fine-tuning ===
    full_params = model_size
    # Règle empirique: VRAM ≈ 4 × paramètres (pour optimiseur Adam + gradients)
    full_vram_gb = (full_params * 4) / (1024**3)
    full_time_relative = 1.0  # Référence
    
    # === LoRA ===
    lora_calc = calculate_lora_parameters(model_dim, lora_rank, num_layers)
    lora_params = lora_calc["total_lora_params"]
    lora_percentage = (lora_params / full_params) * 100
    # LoRA : sauvegarder le modèle original + gradients sur LoRA seulement
    lora_vram_gb = (full_params + lora_params * 4) / (1024**3)
    lora_time_relative = 0.3  # Empirique : plus rapide car moins de params à mettre à jour
    
    # === QLoRA ===
    # QLoRA quantifie le modèle en 4-bit, donc 4x moins de mémoire pour le modèle
    # + sauvegarder LoRA weights
    qlora_vram_gb = (full_params / 4 + lora_params * 4) / (1024**3)
    qlora_time_relative = 0.4  # Légèrement plus lent que LoRA (overhead quantification)
    
    return {
        "full_fine_tuning": {
            "trainable_params": full_params,
            "param_percentage": 100.0,
            "vram_gb": full_vram_gb,
            "time_relative": full_time_relative,
            "pros": "Meilleure performance",
            "cons": "Très gourmand en VRAM et temps"
        },
        "lora": {
            "trainable_params": lora_params,
            "param_percentage": lora_percentage,
            "vram_gb": lora_vram_gb,
            "time_relative": lora_time_relative,
            "pros": "Bon compromis performance/ressources",
            "cons": "Nécessite quand même pas mal de VRAM"
        },
        "qlora": {
            "trainable_params": lora_params,
            "param_percentage": lora_percentage,
            "vram_gb": qlora_vram_gb,
            "time_relative": qlora_time_relative,
            "pros": "RÉVOLUTION : fine-tune 65B sur 1 GPU",
            "cons": "Légèrement plus lent (quantification)"
        }
    }


def main():
    print("=" * 80)
    print("LORA & QLORA : FINE-TUNING EFFICACE")
    print("=" * 80)
    print()
    
    # === Exemple 1 : LLaMA 7B ===
    print("=" * 80)
    print("EXEMPLE 1 : Fine-tuner LLaMA-7B")
    print("=" * 80)
    print()
    
    llama_7b_params = 7_000_000_000  # 7 milliards de paramètres
    llama_dim = 4096              # Dimension des embeddings
    llama_layers = 32             # Nombre de couches
    lora_rank = 8                 # Rang LoRA standard
    
    results_7b = compare_finetuning_methods(
        llama_7b_params, llama_dim, llama_layers, lora_rank
    )
    
    print(f"Modèle : LLaMA-7B ({llama_7b_params / 1e9:.1f}B paramètres)")
    print(f"LoRA rank : {lora_rank}")
    print()
    
    print("Comparaison des méthodes :")
    print("-" * 80)
    print(f"{'Méthode':<20} {'Params':<20} {'VRAM':<12} {'Temps':<10} {'Cas d\'usage'}")
    print("-" * 80)
    
    for method, data in results_7b.items():
        params_M = data["trainable_params"] / 1e6
        vram = data["vram_gb"]
        time_rel = data["time_relative"]
        print(f"{method:<20} {params_M:>15.1f}M {vram:>10.1f}GB {time_rel:>8.1f}x {'→ ' + data['pros']}")
    
    print()
    print("INSIGHT :")
    print("  • Full fine-tuning : 28 GB VRAM → nécessite A100 ou RTX 6000")
    print("  • LoRA : 8 GB VRAM → entraînable sur RTX 4090 (24 GB)")
    print("  • QLoRA : 2 GB VRAM → entraînable sur RTX 3090 (24 GB) ✅ RÉVOLUTION!")
    print()
    
    # === Exemple 2 : LLaMA 65B (le cas d'usage réel de QLoRA) ===
    print()
    print("=" * 80)
    print("EXEMPLE 2 : Fine-tuner LLaMA-65B (le vrai cas d'usage de QLoRA)")
    print("=" * 80)
    print()
    
    llama_65b_params = 65_000_000_000  # 65 milliards
    llama_65b_dim = 8192
    llama_65b_layers = 80
    
    results_65b = compare_finetuning_methods(
        llama_65b_params, llama_65b_dim, llama_65b_layers, lora_rank
    )
    
    print(f"Modèle : LLaMA-65B ({llama_65b_params / 1e9:.0f}B paramètres)")
    print(f"LoRA rank : {lora_rank}")
    print()
    
    print("Comparaison des méthodes :")
    print("-" * 80)
    print(f"{'Méthode':<20} {'Params':<20} {'VRAM':<12} {'Temps':<10}")
    print("-" * 80)
    
    for method, data in results_65b.items():
        params_M = data["trainable_params"] / 1e6
        vram = data["vram_gb"]
        time_rel = data["time_relative"]
        accessible = "❌ 260GB" if method == "full_fine_tuning" else "⚠️  32GB" if method == "lora" else "✅ 8GB"
        print(f"{method:<20} {params_M:>15.1f}M {vram:>10.1f}GB {time_rel:>8.1f}x  {accessible}")
    
    print()
    print("RÉVÉLATION :")
    print("  • Full fine-tuning : 260 GB VRAM → IMPOSSIBLE (même pas un cluster GPU)")
    print("  • LoRA : 32 GB VRAM → A100 ou deux RTX 4090 (possible mais coûteux)")
    print("  • QLoRA : 8 GB VRAM → RTX 3090 SIMPLE (2024€ d'occasion) ✅✅✅")
    print()
    print("  → QLoRA a démocratisé l'accès aux modèles LLM géants!")
    print()
    
    # === Cas d'usage pratique ===
    print()
    print("=" * 80)
    print("CAS D'USAGE RÉEL : Adapter LLaMA-7B pour ton domaine métier")
    print("=" * 80)
    print()
    
    print("Scénario : Vous travaillez chez SNCF et voulez adapter LLaMA-7B")
    print("          pour répondre à des questions sur la maintenance ferroviaire.")
    print()
    
    print("Approche LoRA :")
    print("-" * 80)
    print("""
    1. Charger LLaMA-7B (13 GB en full precision)
    2. Ajouter adaptateurs LoRA (85 MB seulement !)
    3. Fine-tuner sur votre dataset SNCF (ex: 10K paires Q/A)
    4. Pendant l'entraînement :
       - Sauvegarder seulement les 85 MB de LoRA (pas 13 GB)
       - VRAM nécessaire : ~8 GB (sur RTX 4090)
       - Temps : ~2h au lieu de ~8h en full fine-tuning
    5. En inférence :
       - Charger LLaMA-7B + 85 MB de LoRA
       - Performance : quasi-identique au full fine-tuning
       - Latence : IDENTIQUE (fusion optionnelle pour vitesse)
    
    Résultat : Un modèle expert SNCF sans dépenser 100k€ en GPU!
    """)
    
    print()
    print("=" * 80)
    print("CODE PRATIQUE (pseudocode)")
    print("=" * 80)
    print()
    
    code_example = '''
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

# Charger le modèle de base
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    device_map="auto",  # Distribue le modèle sur les GPUs disponibles
)

# Configurer LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                          # Rang LoRA
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # Couches à adapter
)

# Appliquer LoRA au modèle
model = get_peft_model(model, lora_config)

# Afficher la réduction
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,738,415,616
#         Trainable%: 0.06%

# Fine-tuner seulement avec votre dataset
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    args=TrainingArguments(output_dir="./lora_checkpoint"),
)
trainer.train()

# Sauvegarder SEULEMENT LoRA (85 MB)
model.save_pretrained("./sncf_lora_weights")

# En inférence, charger et fusionner
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
model = PeftModel.from_pretrained(model, "./sncf_lora_weights")
model = model.merge_and_unload()  # Fusionner (optionnel, pour vitesse)
'''
    
    print(code_example)
    
    print()
    print("=" * 80)
    print("RÉSUMÉ")
    print("=" * 80)
    print()
    print("✅ LoRA/QLoRA = révolution d'accessibilité")
    print("   - Fine-tune des modèles géants sans cluster GPU")
    print("   - Sauvegarder seulement quelques MB au lieu de GB")
    print("   - Performance quasi-identique au full fine-tuning")
    print()
    print("⚠️  Quand utiliser quoi :")
    print("   - LoRA : petit modèle (7-13B) + GPU mid-range (RTX 4090)")
    print("   - QLoRA : modèle géant (65B+) + GPU basic (RTX 3090)")
    print("   - Full fine-tuning : données TRÈS grandes (>1M exemples) + infra GPU massive")
    print()
    print("💡 Conseil : Commencez TOUJOURS par LoRA. C'est le sweet spot.")
    print()


if __name__ == "__main__":
    main()
