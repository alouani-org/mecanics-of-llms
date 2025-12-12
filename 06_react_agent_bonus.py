#!/usr/bin/env python
"""
Script BONUS : Agent Autonome (Pattern ReAct)

Ce script implémente un mini-framework générique pour construire des agents
autonomes capables de :
1. Raisonner sur la tâche (Thought)
2. Décider d'une action (Action) 
3. Observer le résultat (Observation)
4. Boucler jusqu'à la résolution

Un agent ReAct est plus sophistiqué qu'un simple appel de fonction :
- Il peut utiliser des outils (calculatrice, web search, APIs)
- Il peut corriger ses erreurs
- Il peut itérer et affiner sa réponse

Dépendances :
    pip install pydantic

Utilisation :
    python 06_react_agent_bonus.py
"""

from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json


class ActionType(Enum):
    """Types d'action qu'un agent peut prendre."""
    THINK = "Thought"          # Réfléchir, analyser
    ACTION = "Action"          # Appeler un outil
    OBSERVATION = "Observation"  # Recevoir le résultat
    FINAL_ANSWER = "Final Answer"  # Donner la réponse finale


@dataclass
class ToolDefinition:
    """Définition d'un outil disponible pour l'agent."""
    name: str
    description: str
    parameters: Dict[str, str]  # {"param_name": "type description"}
    func: Callable


class Agent:
    """
    Un agent autonome capable de raisonner et d'agir.
    
    Implémente le pattern ReAct :
    Thought → Action → Observation → Thought → ... → Final Answer
    """

    def __init__(self, name: str = "BasicAgent", max_iterations: int = 10):
        self.name = name
        self.tools: Dict[str, ToolDefinition] = {}
        self.max_iterations = max_iterations
        self.history: list = []

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, str],
        func: Callable,
    ) -> None:
        """Enregistrer un nouvel outil disponible pour l'agent."""
        self.tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
        )
        print(f"✅ Outil enregistré: {name}")

    def _format_tools_description(self) -> str:
        """Générer une description des outils disponibles."""
        if not self.tools:
            return "Aucun outil disponible."

        tools_desc = "Outils disponibles:\n"
        for tool in self.tools.values():
            params_str = ", ".join(
                [f"{k}: {v}" for k, v in tool.parameters.items()]
            )
            tools_desc += f"\n  • {tool.name}({params_str})"
            tools_desc += f"\n    Description: {tool.description}"
        return tools_desc

    def _simulate_llm_reasoning(self, task: str, context: str) -> str:
        """
        Simuler un appel LLM pour générer du raisonnement.
        
        En pratique, ce serait un appel à OpenAI, Anthropic, etc.
        Ici, on utilise une simple heuristique pour la démo.
        """
        # Prompt simplifié
        prompt = f"""Tu es un agent autonome efficace.

Tâche: {task}

Contexte actuel:
{context}

{self._format_tools_description()}

Réponds au format suivant:
Thought: [Ton analyse de la situation]
Action: [nom_outil](param1=val1, param2=val2) OU "Final Answer: [réponse finale]"

Sois concis et actionnel."""

        print(f"\n{'='*70}")
        print("💭 PROMPT ENVOYÉ AU LLM (simulé):")
        print(f"{'='*70}")
        print(prompt)
        print(f"{'='*70}\n")

        # Simulation : générer une réponse heuristique
        return self._generate_simulated_response(task, context)

    def _generate_simulated_response(self, task: str, context: str) -> str:
        """Générer une réponse simulée (sans appel API)."""
        # Heuristiques simples pour la démo
        if "calculer" in task.lower() and "+" in context:
            return "Thought: Je vois deux nombres à additionner.\nAction: calculatrice(operation=addition, a=5, b=3)"
        elif "calculer" in task.lower() and "*" in context:
            return "Thought: Je dois multiplier deux nombres.\nAction: calculatrice(operation=multiplication, a=4, b=6)"
        elif "jour" in task.lower():
            return "Thought: Je dois récupérer la date d'aujourd'hui.\nAction: get_current_date()"
        else:
            return f"Thought: Je dois répondre à: {task}\nFinal Answer: {task}"

    def _parse_action(self, response: str) -> tuple[str, Optional[str]]:
        """Parser la réponse du LLM pour extraire l'action."""
        lines = response.strip().split("\n")

        thought = None
        action = None

        for line in lines:
            if line.startswith("Thought:"):
                thought = line.replace("Thought:", "").strip()
            elif line.startswith("Action:"):
                action = line.replace("Action:", "").strip()

        return action, thought

    def _execute_action(self, action: str) -> str:
        """Exécuter une action (appeler un outil)."""
        # Parser le format: tool_name(param1=val1, param2=val2)
        if action.startswith("Final Answer:"):
            answer = action.replace("Final Answer:", "").strip()
            return f"FINAL_ANSWER:{answer}"

        # Extraire le nom de l'outil et les paramètres
        try:
            tool_name = action.split("(")[0].strip()
            params_str = action.split("(")[1].rstrip(")")

            if tool_name not in self.tools:
                return f"❌ Outil inconnu: {tool_name}"

            # Parser les paramètres (format: key=value, key=value)
            params = {}
            for param in params_str.split(","):
                if "=" in param:
                    key, val = param.split("=", 1)
                    params[key.strip()] = val.strip().strip("'\"")

            # Exécuter l'outil
            tool = self.tools[tool_name]
            result = tool.func(**params)
            return f"✅ {tool_name}({params_str}) → {result}"

        except Exception as e:
            return f"❌ Erreur lors de l'exécution: {e}"

    def run(self, task: str, verbose: bool = True) -> str:
        """
        Exécuter l'agent sur une tâche donnée.
        
        Implémente la boucle ReAct jusqu'à résolution ou max_iterations.
        """
        print(f"\n{'='*70}")
        print(f"🤖 AGENT: {self.name}")
        print(f"📌 TÂCHE: {task}")
        print(f"{'='*70}\n")

        context = ""
        final_answer = None

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'─'*70}")
            print(f"⏳ ITÉRATION {iteration}/{self.max_iterations}")
            print(f"{'─'*70}")

            # 1. THOUGHT : Demander à l'LLM de réfléchir
            llm_response = self._simulate_llm_reasoning(task, context)

            # 2. Parser la réponse
            action, thought = self._parse_action(llm_response)

            if thought and verbose:
                print(f"💭 Pensée: {thought}")

            if not action:
                print("⚠️ Pas d'action générée, arrêt.")
                break

            # 3. Vérifier si c'est la réponse finale
            if action.startswith("Final Answer:"):
                final_answer = action.replace("Final Answer:", "").strip()
                print(f"\n✅ RÉPONSE FINALE: {final_answer}")
                break

            # 4. OBSERVATION : Exécuter l'action
            observation = self._execute_action(action)
            print(f"🔧 Action: {action}")
            print(f"📊 Résultat: {observation}")

            # 5. Mettre à jour le contexte
            context += f"\nItération {iteration}:\n"
            context += f"  Pensée: {thought}\n"
            context += f"  Action: {action}\n"
            context += f"  Observation: {observation}"

            # Sauvegarder dans l'historique
            self.history.append({
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "observation": observation,
            })

        if not final_answer:
            final_answer = "Nombre maximum d'itérations atteint sans réponse finale."

        return final_answer

    def get_history(self) -> list:
        """Retourner l'historique des itérations."""
        return self.history


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("AGENT AUTONOME - PATTERN REACT")
    print("=" * 70)

    # Créer un agent
    agent = Agent(name="MonAgent", max_iterations=5)

    # ===== Enregistrer des outils =====

    def calculatrice(operation: str = "addition", a: float = 0, b: float = 0) -> str:
        """Effectuer une opération arithmétique."""
        try:
            a_val = float(a)
            b_val = float(b)

            if operation.lower() == "addition":
                result = a_val + b_val
            elif operation.lower() == "multiplication":
                result = a_val * b_val
            elif operation.lower() == "soustraction":
                result = a_val - b_val
            elif operation.lower() == "division":
                if b_val == 0:
                    return "❌ Division par zéro"
                result = a_val / b_val
            else:
                return f"❌ Opération inconnue: {operation}"

            return f"{a_val} {operation} {b_val} = {result}"
        except Exception as e:
            return f"❌ Erreur: {e}"

    def get_current_date() -> str:
        """Obtenir la date actuelle."""
        from datetime import date
        return f"Date d'aujourd'hui: {date.today().strftime('%d/%m/%Y')}"

    def search_knowledge_base(query: str = "") -> str:
        """Rechercher dans une base de connaissances."""
        kb = {
            "transformer": "Architecture basée sur l'attention multi-tête",
            "llm": "Large Language Model — modèle de langage de grande taille",
            "bert": "Modèle encodeur bidirectionnel pré-entraîné",
            "rag": "Retrieval-Augmented Generation — génération augmentée",
        }
        key = query.lower().strip()
        if key in kb:
            return f"✅ {key}: {kb[key]}"
        else:
            return f"❌ Concept '{key}' non trouvé dans la base de connaissances"

    # Enregistrer les outils
    agent.register_tool(
        name="calculatrice",
        description="Effectuer des opérations arithmétiques (+, -, *, /)",
        parameters={"operation": "str", "a": "float", "b": "float"},
        func=calculatrice,
    )

    agent.register_tool(
        name="get_current_date",
        description="Récupérer la date actuelle",
        parameters={},
        func=get_current_date,
    )

    agent.register_tool(
        name="search_knowledge_base",
        description="Rechercher des informations dans la base de connaissances",
        parameters={"query": "str"},
        func=search_knowledge_base,
    )

    # ===== Exécuter des tâches =====

    tasks = [
        "Calcule 5 + 3 et dis-moi le résultat",
        "Multiplie 4 par 6, puis additionne 2",
        "Quel est le jour aujourd'hui?",
    ]

    all_results = []

    for i, task in enumerate(tasks, 1):
        print(f"\n\n{'#' * 70}")
        print(f"TÂCHE {i}/{len(tasks)}")
        print(f"{'#' * 70}")

        agent.history = []  # Reset history
        result = agent.run(task, verbose=True)
        all_results.append({"task": task, "result": result})

    # ===== Résumé =====
    print(f"\n\n{'='*70}")
    print("RÉSUMÉ DES RÉSULTATS")
    print(f"{'='*70}\n")

    for i, item in enumerate(all_results, 1):
        print(f"{i}. Tâche: {item['task']}")
        print(f"   Résultat: {item['result']}\n")

    # ===== Analyse =====
    print(f"\n{'='*70}")
    print("ANALYSE")
    print(f"{'='*70}\n")

    print("✅ AVANTAGES DU PATTERN REACT:")
    print("  • Transparence : chaque étape est explicitée")
    print("  • Flexibilité : l'agent peut utiliser n'importe quel outil")
    print("  • Correction : peut revenir en arrière et corriger ses erreurs")
    print("  • Extensibilité : facile d'ajouter de nouveaux outils\n")

    print("⚠️ LIMITATIONS (VERSION SIMULÉE):")
    print("  • LLM simulé : utilise des heuristiques, pas un vrai modèle")
    print("  • Pas de vrai LLM : résultats prévisibles et limités")
    print("  • Token limit : un vrai agent est limité par la fenêtre de contexte\n")

    print("🔧 POUR UTILISER AVEC UN VRAI LLM:")
    print("  1. Remplacer _simulate_llm_reasoning() par un appel API")
    print("  2. Utiliser OpenAI, Anthropic, ou tout autre provider")
    print("  3. Gérer rate limits et timeouts\n")

    print("💡 CAS D'USAGE RÉELS:")
    print("  • Assistants de support client (ticketing, FAQ)")
    print("  • Agents de recherche autonomes (web scraping, APIs)")
    print("  • Systèmes de planification (calendrier, logistics)")
    print("  • Code debugging et code generation")
    print("  • Analyse de données et reporting")


if __name__ == "__main__":
    main()
