#!/usr/bin/env python
"""
BONUS Script 1: Autonomous Agent - ReAct Pattern (Chapters 13 & 14)

This script implements a generic mini-framework for building autonomous agents
capable of:
1. Reasoning about the task (Thought)
2. Deciding on an action (Action)
3. Observing the result (Observation)
4. Looping until resolution

A ReAct agent is more sophisticated than a simple function call:
- It can use tools (calculator, web search, APIs)
- It can correct its errors
- It can iterate and refine its response

Dependencies:
    None (uses only the Python standard library)

Usage:
    python 06_react_agent_bonus.py

Note: This script uses a SIMULATED LLM (basic heuristics).
      To integrate a real LLM, see REACT_AGENT_INTEGRATION.md
"""

from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json


class ActionType(Enum):
    """Types of actions an agent can take."""
    THINK = "Thought"          # Think, analyze
    ACTION = "Action"          # Call a tool
    OBSERVATION = "Observation"  # Receive the result
    FINAL_ANSWER = "Final Answer"  # Give the final answer


@dataclass
class ToolDefinition:
    """Definition of a tool available to the agent."""
    name: str
    description: str
    parameters: Dict[str, str]  # {"param_name": "type description"}
    func: Callable


class Agent:
    """
    An autonomous agent capable of reasoning and acting.
    
    Implements the ReAct pattern:
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
        """Register a new tool available to the agent."""
        self.tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
        )
        print(f"✅ Tool registered: {name}")

    def _format_tools_description(self) -> str:
        """Generate a description of available tools."""
        if not self.tools:
            return "No tools available."

        tools_desc = "Available tools:\n"
        for tool in self.tools.values():
            params_str = ", ".join(
                [f"{k}: {v}" for k, v in tool.parameters.items()]
            )
            tools_desc += f"\n  • {tool.name}({params_str})"
            tools_desc += f"\n    Description: {tool.description}"
        return tools_desc

    def _simulate_llm_reasoning(self, task: str, context: str) -> str:
        """
        Simulate an LLM call to generate reasoning.
        
        In practice, this would be a call to OpenAI, Anthropic, etc.
        Here, we use a simple heuristic for the demo.
        """
        # Simplified prompt
        prompt = f"""You are an efficient autonomous agent.

Task: {task}

Current context:
{context}

{self._format_tools_description()}

Respond in the following format:
Thought: [Your analysis of the situation]
Action: [tool_name](param1=val1, param2=val2) OR "Final Answer: [final answer]"

Be concise and action-oriented."""

        print(f"\n{'='*70}")
        print("💭 PROMPT SENT TO LLM (simulated):")
        print(f"{'='*70}")
        print(prompt)
        print(f"{'='*70}\n")

        # Simulation: generate a heuristic response
        return self._generate_simulated_response(task, context)

    def _generate_simulated_response(self, task: str, context: str) -> str:
        """Generate a simulated response (without API call)."""
        # Simple heuristics for the demo
        if "calculate" in task.lower() and "+" in context:
            return "Thought: I see two numbers to add.\nAction: calculator(operation=addition, a=5, b=3)"
        elif "calculate" in task.lower() and "*" in context:
            return "Thought: I need to multiply two numbers.\nAction: calculator(operation=multiplication, a=4, b=6)"
        elif "day" in task.lower() or "today" in task.lower():
            return "Thought: I need to get today's date.\nAction: get_current_date()"
        else:
            return f"Thought: I need to answer: {task}\nFinal Answer: {task}"

    def _parse_action(self, response: str) -> tuple[str, Optional[str]]:
        """Parse the LLM response to extract the action."""
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
        """Execute an action (call a tool)."""
        # Parse the format: tool_name(param1=val1, param2=val2)
        if action.startswith("Final Answer:"):
            answer = action.replace("Final Answer:", "").strip()
            return f"FINAL_ANSWER:{answer}"

        # Extract the tool name and parameters
        try:
            # Robust parentheses handling
            if "(" not in action or ")" not in action:
                return f"❌ Invalid action format: {action}"
            
            tool_name = action.split("(")[0].strip()
            params_str = action.split("(")[1].rsplit(")", 1)[0]

            if tool_name not in self.tools:
                return f"❌ Unknown tool: {tool_name}"

            # Parse the parameters (format: key=value, key=value)
            # Improved handling of quotes and escapes
            params = {}
            for param in params_str.split(","):
                if "=" in param:
                    key, val = param.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    # Remove single AND double quotes
                    if (val.startswith('"') and val.endswith('"')) or \
                       (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    params[key] = val

            # Execute the tool
            tool = self.tools[tool_name]
            result = tool.func(**params)
            return f"✅ {tool_name}({params_str}) → {result}"

        except Exception as e:
            return f"❌ Error during execution: {e}"

    def run(self, task: str, verbose: bool = True) -> str:
        """
        Execute the agent on a given task.
        
        Implements the ReAct loop until resolution or max_iterations.
        """
        print(f"\n{'='*70}")
        print(f"🤖 AGENT: {self.name}")
        print(f"📌 TASK: {task}")
        print(f"{'='*70}\n")

        context = ""
        final_answer = None

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'─'*70}")
            print(f"⏳ ITERATION {iteration}/{self.max_iterations}")
            print(f"{'─'*70}")

            # 1. THOUGHT: Ask the LLM to think
            llm_response = self._simulate_llm_reasoning(task, context)

            # 2. Parse the response
            action, thought = self._parse_action(llm_response)

            if thought and verbose:
                print(f"💭 Thought: {thought}")

            if not action:
                print("⚠️ No action generated, stopping.")
                break

            # 3. Check if it's the final answer
            if action.startswith("Final Answer:"):
                final_answer = action.replace("Final Answer:", "").strip()
                print(f"\n✅ FINAL ANSWER: {final_answer}")
                break

            # 4. OBSERVATION: Execute the action
            observation = self._execute_action(action)
            print(f"🔧 Action: {action}")
            print(f"📊 Result: {observation}")

            # 5. Update the context
            context += f"\nIteration {iteration}:\n"
            context += f"  Thought: {thought}\n"
            context += f"  Action: {action}\n"
            context += f"  Observation: {observation}"

            # Save in history
            self.history.append({
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "observation": observation,
            })

        if not final_answer:
            final_answer = "Maximum number of iterations reached without final answer."

        return final_answer

    def get_history(self) -> list:
        """Return the iteration history."""
        return self.history


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("AUTONOMOUS AGENT - REACT PATTERN")
    print("=" * 70)

    # Create an agent
    agent = Agent(name="MyAgent", max_iterations=5)

    # ===== Register tools =====

    def calculator(operation: str = "addition", a: float = 0, b: float = 0) -> str:
        """Perform an arithmetic operation."""
        try:
            a_val = float(a)
            b_val = float(b)

            if operation.lower() == "addition":
                result = a_val + b_val
            elif operation.lower() == "multiplication":
                result = a_val * b_val
            elif operation.lower() == "subtraction":
                result = a_val - b_val
            elif operation.lower() == "division":
                if b_val == 0:
                    return "❌ Division by zero"
                result = a_val / b_val
            else:
                return f"❌ Unknown operation: {operation}"

            return f"{a_val} {operation} {b_val} = {result}"
        except Exception as e:
            return f"❌ Error: {e}"

    def get_current_date() -> str:
        """Get the current date."""
        from datetime import date
        return f"Today's date: {date.today().strftime('%m/%d/%Y')}"

    def search_knowledge_base(query: str = "") -> str:
        """Search in a knowledge base."""
        kb = {
            "transformer": "Architecture based on multi-head attention",
            "llm": "Large Language Model — large-scale language model",
            "bert": "Bidirectional pre-trained encoder model",
            "rag": "Retrieval-Augmented Generation — augmented generation",
        }
        key = query.lower().strip()
        if key in kb:
            return f"✅ {key}: {kb[key]}"
        else:
            return f"❌ Concept '{key}' not found in the knowledge base"

    # Register the tools
    agent.register_tool(
        name="calculator",
        description="Perform arithmetic operations (+, -, *, /)",
        parameters={"operation": "str", "a": "float", "b": "float"},
        func=calculator,
    )

    agent.register_tool(
        name="get_current_date",
        description="Get the current date",
        parameters={},
        func=get_current_date,
    )

    agent.register_tool(
        name="search_knowledge_base",
        description="Search for information in the knowledge base",
        parameters={"query": "str"},
        func=search_knowledge_base,
    )

    # ===== Execute tasks =====

    tasks = [
        "Calculate 5 + 3 and tell me the result",
        "Multiply 4 by 6, then add 2",
        "What day is it today?",
    ]

    all_results = []

    for i, task in enumerate(tasks, 1):
        print(f"\n\n{'#' * 70}")
        print(f"TASK {i}/{len(tasks)}")
        print(f"{'#' * 70}")

        agent.history = []  # Reset history
        result = agent.run(task, verbose=True)
        all_results.append({"task": task, "result": result})

    # ===== Summary =====
    print(f"\n\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}\n")

    for i, item in enumerate(all_results, 1):
        print(f"{i}. Task: {item['task']}")
        print(f"   Result: {item['result']}\n")

    # ===== Analysis =====
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}\n")

    print("✅ ADVANTAGES OF THE REACT PATTERN:")
    print("  • Transparency: each step is made explicit")
    print("  • Flexibility: the agent can use any tool")
    print("  • Correction: can go back and correct its errors")
    print("  • Extensibility: easy to add new tools\n")

    print("⚠️ LIMITATIONS (SIMULATED VERSION):")
    print("  • Simulated LLM: uses heuristics, not a real model")
    print("  • No real LLM: predictable and limited results")
    print("  • Token limit: a real agent is limited by the context window\n")

    print("🔧 TO USE WITH A REAL LLM:")
    print("  1. Replace _simulate_llm_reasoning() with an API call")
    print("  2. Use OpenAI, Anthropic, or any other provider")
    print("  3. Handle rate limits and timeouts\n")

    print("💡 REAL-WORLD USE CASES:")
    print("  • Customer support assistants (ticketing, FAQ)")
    print("  • Autonomous research agents (web scraping, APIs)")
    print("  • Planning systems (calendar, logistics)")
    print("  • Code debugging and code generation")
    print("  • Data analysis and reporting")


if __name__ == "__main__":
    main()
