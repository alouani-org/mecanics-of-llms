#!/usr/bin/env python
"""
Script 10: Activation Steering & Structured Output (3SO) - Chapter 10

This script demonstrates dynamic behavior steering techniques for LLMs:
- Activation Steering: injecting concept vectors into hidden states
- Contrastive Activation Extraction: computing steering vectors
- 3SO (Schema-Steered Structured Output): guaranteed JSON compliance
- Comparison: RLHF/DPO alignment vs. inference-time steering

This script is COMPUTATIONAL only (no GPU required):
- It demonstrates steering mechanics with simulated activations
- It shows the math behind vector injection and filtering
- It includes a working JSON state machine implementation

Minimal dependencies (demo mode):
    None (uses only Python standard library)

Dependencies for real steering:
    pip install torch transformers

Usage:
    python 10_activation_steering_demo.py
"""

import math
import random
import json
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum, auto


# =============================================================================
# PART 1: ACTIVATION STEERING MECHANICS
# =============================================================================

class ActivationSteering:
    """
    Simulates activation steering in a Transformer model.
    
    In real models, hidden states are high-dimensional vectors (e.g., 4096 dims).
    Here we use smaller dimensions for visualization and understanding.
    """
    
    def __init__(self, hidden_dim: int = 64, num_layers: int = 32):
        """
        Initialize the steering simulator.
        
        Args:
            hidden_dim: Dimension of hidden states (real models: 4096+)
            num_layers: Number of transformer layers (real models: 32-80)
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.intervention_layer = num_layers // 2  # Middle layer is optimal
        
    def generate_random_activation(self, seed: Optional[int] = None) -> List[float]:
        """
        Generate a random activation vector (simulating a hidden state).
        
        In real models, this comes from the forward pass through layers.
        """
        if seed:
            random.seed(seed)
        return [random.gauss(0, 1) for _ in range(self.hidden_dim)]
    
    def compute_contrastive_vector(
        self,
        positive_examples: List[List[float]],
        negative_examples: List[List[float]]
    ) -> List[float]:
        """
        Compute a steering vector using contrastive activation.
        
        Formula: V_concept = mean(positive_activations) - mean(negative_activations)
        
        This vector points in the direction that separates the desired behavior
        from the undesired behavior in activation space.
        
        Args:
            positive_examples: Activations from "good" examples (e.g., polite responses)
            negative_examples: Activations from "bad" examples (e.g., rude responses)
        
        Returns:
            The steering vector V
        """
        # Compute mean of positive examples
        n_pos = len(positive_examples)
        mean_positive = [
            sum(ex[i] for ex in positive_examples) / n_pos 
            for i in range(self.hidden_dim)
        ]
        
        # Compute mean of negative examples
        n_neg = len(negative_examples)
        mean_negative = [
            sum(ex[i] for ex in negative_examples) / n_neg 
            for i in range(self.hidden_dim)
        ]
        
        # Steering vector = difference
        steering_vector = [
            mean_positive[i] - mean_negative[i] 
            for i in range(self.hidden_dim)
        ]
        
        return steering_vector
    
    def apply_steering(
        self,
        activation: List[float],
        steering_vector: List[float],
        coefficient: float = 1.0
    ) -> Tuple[List[float], Dict[str, float]]:
        """
        Apply steering to an activation vector.
        
        Formula: X_steered = X + (c × V)
        
        Args:
            activation: Original hidden state X
            steering_vector: Concept vector V
            coefficient: Steering intensity c (1.0 = subtle, 5.0 = strong)
        
        Returns:
            Tuple of (steered_activation, metrics_dict)
        """
        # Apply the steering formula
        steered = [
            activation[i] + (coefficient * steering_vector[i])
            for i in range(self.hidden_dim)
        ]
        
        # Compute metrics for analysis
        original_norm = math.sqrt(sum(x**2 for x in activation))
        steered_norm = math.sqrt(sum(x**2 for x in steered))
        steering_norm = math.sqrt(sum(x**2 for x in steering_vector))
        
        # Cosine similarity between original and steered
        dot_product = sum(activation[i] * steered[i] for i in range(self.hidden_dim))
        cosine_sim = dot_product / (original_norm * steered_norm) if original_norm * steered_norm > 0 else 0
        
        # Relative change in direction
        direction_change = math.acos(max(-1, min(1, cosine_sim))) * 180 / math.pi
        
        metrics = {
            "original_norm": original_norm,
            "steered_norm": steered_norm,
            "steering_magnitude": coefficient * steering_norm,
            "cosine_similarity": cosine_sim,
            "direction_change_degrees": direction_change,
            "relative_perturbation": (coefficient * steering_norm) / original_norm
        }
        
        return steered, metrics
    
    def analyze_coefficient_effect(
        self,
        activation: List[float],
        steering_vector: List[float],
        coefficients: List[float]
    ) -> List[Dict]:
        """
        Analyze how different steering coefficients affect the output.
        
        This helps understand the "sweet spot" for coefficient selection:
        - Too low: no effect
        - Optimal: controlled steering
        - Too high: derailment (incoherent output)
        """
        results = []
        
        for c in coefficients:
            _, metrics = self.apply_steering(activation, steering_vector, c)
            
            # Heuristic: relative perturbation > 50% often causes derailment
            stability = "stable" if metrics["relative_perturbation"] < 0.3 else \
                       "moderate" if metrics["relative_perturbation"] < 0.5 else \
                       "unstable"
            
            results.append({
                "coefficient": c,
                "direction_change": metrics["direction_change_degrees"],
                "relative_perturbation": metrics["relative_perturbation"],
                "stability_prediction": stability
            })
        
        return results


# =============================================================================
# PART 2: SPARSE AUTOENCODER (SAE) CONCEPT EXTRACTION
# =============================================================================

class SimpleSparseAutoencoder:
    """
    Simplified demonstration of Sparse Autoencoder for concept extraction.
    
    Real SAEs are trained on millions of activations to discover interpretable
    features. This simulation shows the principle: decomposing activations
    into sparse combinations of concept vectors.
    """
    
    def __init__(self, input_dim: int = 64, num_concepts: int = 512):
        """
        Initialize the SAE simulator.
        
        Args:
            input_dim: Dimension of input activations
            num_concepts: Number of concept vectors to learn (real SAEs: 10k-100k)
        """
        self.input_dim = input_dim
        self.num_concepts = num_concepts
        
        # Simulated concept dictionary (in reality, learned through training)
        # Each concept is a normalized direction in activation space
        self.concept_vectors = self._initialize_concepts()
        self.concept_labels = self._generate_concept_labels()
    
    def _initialize_concepts(self) -> List[List[float]]:
        """Initialize random concept directions (normally learned via training)."""
        concepts = []
        for i in range(self.num_concepts):
            random.seed(42 + i)  # Reproducible
            vec = [random.gauss(0, 1) for _ in range(self.input_dim)]
            # Normalize to unit length
            norm = math.sqrt(sum(x**2 for x in vec))
            vec = [x / norm for x in vec]
            concepts.append(vec)
        return concepts
    
    def _generate_concept_labels(self) -> List[str]:
        """Generate example concept labels for demonstration."""
        base_concepts = [
            "formal_tone", "casual_tone", "technical_jargon", "simple_language",
            "positive_sentiment", "negative_sentiment", "uncertainty", "confidence",
            "medical_domain", "legal_domain", "programming", "mathematics",
            "creativity", "factual", "humor", "seriousness",
            "politeness", "directness", "empathy", "neutrality",
            "question_asking", "explaining", "summarizing", "elaborating",
            "english_language", "french_language", "code_generation", "analysis",
            "safety_awareness", "risk_tolerance", "ethical_reasoning", "pragmatism"
        ]
        # Repeat and number for all concepts
        labels = []
        for i in range(self.num_concepts):
            base_idx = i % len(base_concepts)
            variant = i // len(base_concepts)
            label = f"{base_concepts[base_idx]}_{variant}" if variant > 0 else base_concepts[base_idx]
            labels.append(label)
        return labels
    
    def decompose_activation(
        self,
        activation: List[float],
        top_k: int = 5,
        sparsity_threshold: float = 0.1
    ) -> List[Tuple[str, float]]:
        """
        Decompose an activation into its top contributing concepts.
        
        This simulates what a trained SAE does: finding which interpretable
        concepts are most active in a given hidden state.
        
        Args:
            activation: Hidden state to decompose
            top_k: Number of top concepts to return
            sparsity_threshold: Minimum activation to consider
        
        Returns:
            List of (concept_name, activation_strength) tuples
        """
        # Compute dot product with each concept (projection)
        projections = []
        for i, concept_vec in enumerate(self.concept_vectors):
            projection = sum(activation[j] * concept_vec[j] for j in range(self.input_dim))
            if abs(projection) > sparsity_threshold:
                projections.append((self.concept_labels[i], projection))
        
        # Sort by absolute strength and return top_k
        projections.sort(key=lambda x: abs(x[1]), reverse=True)
        return projections[:top_k]
    
    def get_concept_vector(self, concept_name: str) -> Optional[List[float]]:
        """Retrieve a concept vector by name for use in steering."""
        try:
            idx = self.concept_labels.index(concept_name)
            return self.concept_vectors[idx]
        except ValueError:
            return None


# =============================================================================
# PART 3: SCHEMA-STEERED STRUCTURED OUTPUT (3SO)
# =============================================================================

class JSONState(Enum):
    """States for the JSON parsing state machine."""
    START = auto()
    OBJECT_START = auto()
    KEY_START = auto()
    KEY_CONTENT = auto()
    KEY_END = auto()
    COLON = auto()
    VALUE_START = auto()
    STRING_VALUE = auto()
    NUMBER_VALUE = auto()
    BOOLEAN_VALUE = auto()
    NULL_VALUE = auto()
    VALUE_END = auto()
    COMMA_OR_END = auto()
    OBJECT_END = auto()
    DONE = auto()


@dataclass
class JSONSchema:
    """Simple JSON schema representation."""
    properties: Dict[str, str]  # property_name -> type ("string", "number", "boolean")
    required: List[str]


class SchemaSteeringSampler:
    """
    Demonstrates Schema-Steered Structured Output (3SO).
    
    Instead of "begging" the model to output valid JSON through prompting,
    3SO uses a finite state machine to filter token probabilities at each
    generation step, making invalid tokens impossible.
    """
    
    def __init__(self, schema: JSONSchema):
        """
        Initialize the 3SO sampler with a JSON schema.
        
        Args:
            schema: The JSON schema that outputs must conform to
        """
        self.schema = schema
        self.state = JSONState.START
        self.current_key: Optional[str] = None
        self.expected_keys = list(schema.properties.keys())
        self.seen_keys: Set[str] = set()
        self.buffer = ""
    
    def get_valid_tokens(self) -> Tuple[Set[str], str]:
        """
        Get the set of valid tokens for the current state.
        
        This is the core of 3SO: at each position, we know exactly
        which tokens are syntactically valid.
        
        Returns:
            Tuple of (valid_tokens_set, explanation_string)
        """
        valid = set()
        explanation = ""
        
        if self.state == JSONState.START:
            valid = {"{"}
            explanation = "JSON must start with opening brace"
            
        elif self.state == JSONState.OBJECT_START:
            remaining_keys = set(self.expected_keys) - self.seen_keys
            if remaining_keys:
                valid = {'"'}
                explanation = f"Expecting key, remaining: {remaining_keys}"
            else:
                valid = {"}"}
                explanation = "All keys provided, must close object"
                
        elif self.state == JSONState.KEY_START:
            remaining_keys = set(self.expected_keys) - self.seen_keys
            # In real 3SO, we'd filter to only valid key characters
            valid = set("abcdefghijklmnopqrstuvwxyz_")
            explanation = f"Key characters for: {remaining_keys}"
            
        elif self.state == JSONState.KEY_CONTENT:
            valid = set("abcdefghijklmnopqrstuvwxyz_0123456789") | {'"'}
            explanation = "Continue key or close with quote"
            
        elif self.state == JSONState.COLON:
            valid = {":"}
            explanation = "Colon must follow key"
            
        elif self.state == JSONState.VALUE_START:
            if self.current_key and self.current_key in self.schema.properties:
                expected_type = self.schema.properties[self.current_key]
                if expected_type == "string":
                    valid = {'"'}
                    explanation = f"String value expected for '{self.current_key}'"
                elif expected_type == "number":
                    valid = set("0123456789-")
                    explanation = f"Number value expected for '{self.current_key}'"
                elif expected_type == "boolean":
                    valid = {"t", "f"}  # true or false
                    explanation = f"Boolean value expected for '{self.current_key}'"
            else:
                valid = {'"', "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-", "t", "f", "n"}
                explanation = "Any value type"
                
        elif self.state == JSONState.STRING_VALUE:
            # Simplified: allow printable ASCII except unescaped quotes
            valid = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-_") | {'"'}
            explanation = "String content or closing quote"
            
        elif self.state == JSONState.NUMBER_VALUE:
            valid = set("0123456789.") | {",", "}"}
            explanation = "Continue number or end value"
            
        elif self.state == JSONState.COMMA_OR_END:
            remaining_keys = set(self.expected_keys) - self.seen_keys
            if remaining_keys:
                valid = {","}
                explanation = f"More keys required: {remaining_keys}"
            elif self.seen_keys == set(self.expected_keys):
                valid = {"}", ","}
                explanation = "Can close object or add optional keys"
            else:
                valid = {",", "}"}
                explanation = "Comma for more keys or close object"
                
        elif self.state == JSONState.OBJECT_END:
            valid = {"}"}
            explanation = "Must close object"
            
        return valid, explanation
    
    def filter_probabilities(
        self,
        token_probs: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Filter token probabilities to only valid tokens.
        
        This is the key mechanism: invalid tokens get probability 0,
        making it mathematically impossible to generate invalid syntax.
        
        Args:
            token_probs: Original probability distribution over tokens
        
        Returns:
            Filtered distribution (invalid tokens removed, renormalized)
        """
        valid_tokens, _ = self.get_valid_tokens()
        
        # Zero out invalid tokens
        filtered = {
            token: prob for token, prob in token_probs.items()
            if any(token.startswith(v) for v in valid_tokens) or token in valid_tokens
        }
        
        # Renormalize
        total = sum(filtered.values())
        if total > 0:
            filtered = {t: p / total for t, p in filtered.items()}
        
        return filtered
    
    def transition(self, token: str):
        """
        Transition the state machine based on the generated token.
        
        Args:
            token: The token that was generated
        """
        self.buffer += token
        
        if self.state == JSONState.START and token == "{":
            self.state = JSONState.OBJECT_START
            
        elif self.state == JSONState.OBJECT_START and token == '"':
            self.state = JSONState.KEY_START
            self.current_key = ""
            
        elif self.state == JSONState.KEY_START:
            if token == '"':
                self.state = JSONState.COLON
            else:
                self.current_key = token
                self.state = JSONState.KEY_CONTENT
                
        elif self.state == JSONState.KEY_CONTENT:
            if token == '"':
                self.seen_keys.add(self.current_key)
                self.state = JSONState.COLON
            else:
                self.current_key += token
                
        elif self.state == JSONState.COLON and token == ":":
            self.state = JSONState.VALUE_START
            
        elif self.state == JSONState.VALUE_START:
            if token == '"':
                self.state = JSONState.STRING_VALUE
            elif token in "0123456789-":
                self.state = JSONState.NUMBER_VALUE
            elif token in "tf":
                self.state = JSONState.BOOLEAN_VALUE
                
        elif self.state == JSONState.STRING_VALUE and token == '"':
            self.state = JSONState.COMMA_OR_END
            
        elif self.state == JSONState.NUMBER_VALUE:
            if token in ",}":
                self.state = JSONState.OBJECT_START if token == "," else JSONState.DONE
                
        elif self.state == JSONState.COMMA_OR_END:
            if token == ",":
                self.state = JSONState.OBJECT_START
            elif token == "}":
                self.state = JSONState.DONE


def demonstrate_3so_filtering():
    """
    Demonstrate how 3SO filters invalid tokens at each generation step.
    """
    print("\n" + "=" * 80)
    print("3SO DEMONSTRATION: GUARANTEED JSON STRUCTURE")
    print("=" * 80)
    
    # Define a simple schema
    schema = JSONSchema(
        properties={"name": "string", "age": "number", "active": "boolean"},
        required=["name", "age"]
    )
    
    print(f"\nTarget Schema:")
    print(f'  {{"name": string, "age": number, "active": boolean}}')
    print(f"  Required: {schema.required}")
    
    # Simulate token generation
    sampler = SchemaSteeringSampler(schema)
    
    # Simulated token probabilities from a language model
    # In reality, these come from the model's softmax output
    simulated_token_probs = {
        "{": 0.3, "Hello": 0.2, '"': 0.15, "[": 0.1, 
        "The": 0.1, "null": 0.05, "42": 0.05, "true": 0.05
    }
    
    print("\n" + "-" * 60)
    print("Step-by-step token filtering:")
    print("-" * 60)
    
    # Step 1: Start
    print("\n📍 State: START")
    print(f"   Original probs: { {k: f'{v:.2f}' for k, v in simulated_token_probs.items()} }")
    valid, explanation = sampler.get_valid_tokens()
    print(f"   Valid tokens: {valid}")
    print(f"   Reason: {explanation}")
    filtered = sampler.filter_probabilities(simulated_token_probs)
    print(f"   Filtered probs: { {k: f'{v:.2f}' for k, v in filtered.items()} }")
    print(f"   ✅ ONLY '{{' can be generated - guaranteed valid start!")
    
    # Simulate choosing "{"
    sampler.transition("{")
    
    # Step 2: After opening brace
    print("\n📍 State: OBJECT_START (after '{')")
    simulated_token_probs_2 = {
        '"': 0.4, "name": 0.2, "}": 0.15, ",": 0.1,
        "Hello": 0.1, "[": 0.05
    }
    print(f"   Original probs: { {k: f'{v:.2f}' for k, v in simulated_token_probs_2.items()} }")
    valid, explanation = sampler.get_valid_tokens()
    print(f"   Valid tokens: {valid}")
    print(f"   Reason: {explanation}")
    filtered = sampler.filter_probabilities(simulated_token_probs_2)
    print(f"   Filtered probs: { {k: f'{v:.2f}' for k, v in filtered.items()} }")
    print(f"   ✅ Must start a key with '\"' - no invalid syntax possible!")
    
    print("\n" + "-" * 60)
    print("KEY INSIGHT: 3SO vs Traditional Prompting")
    print("-" * 60)
    print("""
    Traditional approach (prompting/begging):
    ❌ "Please output ONLY valid JSON, no introduction text..."
    ❌ Model can still fail ~5-20% of the time
    ❌ Requires large models (70B+) for reliability
    ❌ Wastes tokens on formatting instructions
    
    3SO approach (mathematical guarantee):
    ✅ Invalid tokens get probability = 0
    ✅ 100% valid syntax by construction
    ✅ Works with small models (7B, 12B)
    ✅ Model focuses on CONTENT, not syntax
    
    Result: 12B model + 3SO > 70B model without 3SO
            on structured extraction tasks!
    """)


# =============================================================================
# PART 4: COMPARISON RLHF/DPO vs STEERING
# =============================================================================

def compare_alignment_methods():
    """
    Compare traditional alignment (RLHF/DPO) with inference-time steering.
    """
    print("\n" + "=" * 80)
    print("COMPARISON: RLHF/DPO ALIGNMENT vs STEERING")
    print("=" * 80)
    
    comparison = [
        {
            "dimension": "Cost",
            "rlhf_dpo": "Very High (GPU clusters, weeks of training)",
            "steering": "Near Zero (inference-time intervention)"
        },
        {
            "dimension": "Permanence",
            "rlhf_dpo": "Permanent (weights modified)",
            "steering": "Temporary (activations modified)"
        },
        {
            "dimension": "Reversibility",
            "rlhf_dpo": "Requires retraining",
            "steering": "Instant (remove hook)"
        },
        {
            "dimension": "Granularity",
            "rlhf_dpo": "Global (entire personality)",
            "steering": "Surgical (specific concept)"
        },
        {
            "dimension": "Reliability",
            "rlhf_dpo": "Statistical (model can disobey)",
            "steering": "Mathematical for 3SO"
        },
        {
            "dimension": "Interpretability",
            "rlhf_dpo": "Opaque (black box)",
            "steering": "Transparent (identifiable vectors)"
        },
        {
            "dimension": "Combinability",
            "rlhf_dpo": "Difficult (personality conflicts)",
            "steering": "Easy (vector addition)"
        },
        {
            "dimension": "Use Case",
            "rlhf_dpo": "Base model alignment, safety",
            "steering": "Dynamic adaptation, specialization"
        }
    ]
    
    # Print table
    print(f"\n{'Dimension':<18} {'RLHF/DPO':<38} {'Steering':<35}")
    print("-" * 91)
    for row in comparison:
        print(f"{row['dimension']:<18} {row['rlhf_dpo']:<38} {row['steering']:<35}")
    
    print("\n" + "-" * 60)
    print("ANALOGY: The Ship and the Autopilot")
    print("-" * 60)
    print("""
    🚢 Imagine your LLM is a SHIP:
    
    RLHF/DPO = Modifying the HULL shape and ballast position
    ├── The ship naturally tends to sail North
    ├── Permanent change requiring dry dock (expensive!)
    └── If you want to go East, rebuild the ship
    
    STEERING = Installing a precision AUTOPILOT
    ├── Ship structure unchanged
    ├── Real-time course corrections, wave by wave
    ├── Change destination instantly
    └── Remove autopilot = original behavior
    
    Best approach: RLHF for base alignment + Steering for fine control
    """)


# =============================================================================
# PART 5: PRACTICAL DEMONSTRATION
# =============================================================================

def run_steering_demonstration():
    """
    Complete demonstration of activation steering mechanics.
    """
    print("=" * 80)
    print("ACTIVATION STEERING: COMPLETE DEMONSTRATION")
    print("=" * 80)
    
    # Initialize steering system
    steerer = ActivationSteering(hidden_dim=64, num_layers=32)
    
    print(f"\n📊 Model Configuration:")
    print(f"   Hidden dimension: {steerer.hidden_dim}")
    print(f"   Number of layers: {steerer.num_layers}")
    print(f"   Intervention layer: {steerer.intervention_layer} (middle)")
    
    # === Part A: Contrastive Vector Extraction ===
    print("\n" + "-" * 60)
    print("STEP 1: Extracting Steering Vector via Contrastive Activation")
    print("-" * 60)
    
    # Simulate activations from positive examples (e.g., medical responses)
    print("\n   Generating activations from 'medical' examples...")
    positive_activations = [
        steerer.generate_random_activation(seed=100 + i) 
        for i in range(10)
    ]
    # Add a bias to simulate medical domain encoding
    for act in positive_activations:
        for i in range(len(act)):
            if i < 20:  # First 20 dims encode "medical" concept
                act[i] += 0.5
    
    # Simulate activations from negative examples (e.g., general responses)
    print("   Generating activations from 'general' examples...")
    negative_activations = [
        steerer.generate_random_activation(seed=200 + i) 
        for i in range(10)
    ]
    
    # Compute steering vector
    medical_vector = steerer.compute_contrastive_vector(
        positive_activations, negative_activations
    )
    
    print(f"\n   ✅ Medical steering vector computed!")
    print(f"   Vector norm: {math.sqrt(sum(x**2 for x in medical_vector)):.4f}")
    print(f"   First 5 components: {[f'{x:.3f}' for x in medical_vector[:5]]}")
    
    # === Part B: Applying Steering ===
    print("\n" + "-" * 60)
    print("STEP 2: Applying Steering to a Neutral Activation")
    print("-" * 60)
    
    # Generate a "neutral" activation (simulating response to general question)
    neutral_activation = steerer.generate_random_activation(seed=999)
    
    print("\n   Original activation (neutral topic):")
    print(f"   Norm: {math.sqrt(sum(x**2 for x in neutral_activation)):.4f}")
    
    # Apply steering with coefficient 3.0
    steered_activation, metrics = steerer.apply_steering(
        neutral_activation, medical_vector, coefficient=3.0
    )
    
    print(f"\n   After steering (coefficient=3.0):")
    print(f"   Steered norm: {metrics['steered_norm']:.4f}")
    print(f"   Direction change: {metrics['direction_change_degrees']:.1f}°")
    print(f"   Relative perturbation: {metrics['relative_perturbation']*100:.1f}%")
    print(f"   Cosine similarity with original: {metrics['cosine_similarity']:.4f}")
    
    # === Part C: Coefficient Analysis ===
    print("\n" + "-" * 60)
    print("STEP 3: Analyzing Coefficient Effect")
    print("-" * 60)
    
    coefficients = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
    analysis = steerer.analyze_coefficient_effect(
        neutral_activation, medical_vector, coefficients
    )
    
    print(f"\n   {'Coeff':<8} {'Direction Δ':<15} {'Perturbation':<15} {'Stability':<12}")
    print("   " + "-" * 50)
    for result in analysis:
        stability_icon = "✅" if result['stability_prediction'] == "stable" else \
                        "⚠️" if result['stability_prediction'] == "moderate" else "❌"
        print(f"   {result['coefficient']:<8.1f} "
              f"{result['direction_change']:<15.1f}° "
              f"{result['relative_perturbation']*100:<15.1f}% "
              f"{stability_icon} {result['stability_prediction']}")
    
    print("""
    📝 Interpretation:
    • Coefficient 0.5-2.0: Subtle effect, high stability
    • Coefficient 3.0-5.0: Optimal zone for most use cases
    • Coefficient 8.0+: Risk of incoherent outputs (derailment)
    """)
    
    # === Part D: SAE Concept Decomposition ===
    print("\n" + "-" * 60)
    print("STEP 4: Sparse Autoencoder Concept Decomposition")
    print("-" * 60)
    
    sae = SimpleSparseAutoencoder(input_dim=64, num_concepts=512)
    
    print("\n   Decomposing the STEERED activation into concepts...")
    concepts = sae.decompose_activation(steered_activation, top_k=8)
    
    print(f"\n   Top activated concepts:")
    for concept_name, strength in concepts:
        bar = "█" * int(abs(strength) * 10)
        sign = "+" if strength > 0 else "-"
        print(f"   {sign}{abs(strength):.3f} | {bar:<15} | {concept_name}")
    
    print("""
    📝 Note: In real SAEs (like Anthropic's), these concepts are
    truly interpretable features discovered through training on
    millions of activations. Famous example: the "Golden Gate Bridge"
    feature that makes Claude talk about the bridge in every response!
    """)


def main():
    """Main entry point for the demonstration."""
    print("\n" + "🧠" * 40)
    print("\n  ACTIVATION STEERING & 3SO - Chapter 10 Demonstration")
    print("  Dynamic Behavior Control at Inference Time")
    print("\n" + "🧠" * 40)
    
    # Part 1: Activation Steering mechanics
    run_steering_demonstration()
    
    # Part 2: 3SO for structured output
    demonstrate_3so_filtering()
    
    # Part 3: Method comparison
    compare_alignment_methods()
    
    # Summary
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. ACTIVATION STEERING modifies hidden states during inference:
       X_steered = X + (coefficient × V_concept)
    
    2. STEERING VECTORS can be extracted via:
       • Contrastive activation (positive - negative examples)
       • Sparse Autoencoders (SAE) for interpretable features
    
    3. 3SO (Schema-Steered Structured Output) guarantees valid syntax
       by filtering token probabilities through a finite state machine.
    
    4. STEERING complements RLHF/DPO alignment:
       • RLHF/DPO: permanent base personality
       • Steering: dynamic, reversible fine-tuning
    
    5. COEFFICIENT SELECTION is critical:
       • Too low → no effect
       • Optimal (2-5) → controlled steering
       • Too high (>10) → incoherent outputs
    """)
    
    print("\n" + "=" * 80)
    print("For real implementations, see:")
    print("  • TransformerLens: https://github.com/neelnanda-io/TransformerLens")
    print("  • Steering Vectors: https://github.com/steering-vectors/steering-vectors")
    print("  • Outlines (3SO): https://github.com/outlines-dev/outlines")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
