#!/usr/bin/env python
"""
Script BONUS 4: Complete Mini-Assistant - Integration Project (Chapters 11-15)

This final project combines RAG, Agents, Prompting and Evaluation into a coherent system.
It demonstrates how to assemble all the concepts from the book into a real application.

Architecture:
    1. RAG: Document indexing and retrieval (Ch. 13)
    2. ReAct Agent: Thought→Action→Observation loop (Ch. 14)
    3. Prompting: Zero-shot, Few-shot, Chain-of-Thought (Ch. 11)
    4. Evaluation: Self-consistency, confidence scoring (Ch. 12, 15)
    5. Tools: Calculator, search, summary (Ch. 14)

Execution modes:
    - STANDALONE mode: Works without external API (simulated LLM)
    - PRODUCTION mode: OpenAI/Claude integration (uncomment the code)

Minimal dependencies (standalone mode):
    pip install numpy scikit-learn

Production dependencies (optional):
    pip install openai anthropic

Usage:
    python 09_mini_assistant_complet.py

Extension points for students:
    - Add new tools (weather, news, etc.)
    - Integrate a real LLM (OpenAI, Ollama, etc.)
    - Persist conversations (SQLite, JSON)
    - Add a web interface (Streamlit, Gradio)
    - Implement more advanced evaluation metrics
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib

# Imports for RAG (basic vectorization)
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================================
# PART 1: RAG SYSTEM (Chapter 13)
# ============================================================================

@dataclass
class Document:
    """Representation of a document in the knowledge base."""
    id: str
    content: str
    metadata: Dict[str, Any]
    
    def __repr__(self):
        return f"Doc({self.metadata.get('title', 'Untitled')})"


class RAGSystem:
    """
    Complete RAG system with TF-IDF vectorization and similarity search.
    
    In production, replace with dense embeddings (OpenAI, E5, etc.)
    and a vector database (Pinecone, Weaviate, ChromaDB).
    """
    
    def __init__(self):
        self.documents: List[Document] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_vectors: Optional[np.ndarray] = None
    
    def add_document(self, content: str, metadata: Dict[str, Any]) -> str:
        """Add a document to the knowledge base."""
        doc_id = hashlib.md5(
            f"{content[:100]}{datetime.now()}".encode()
        ).hexdigest()[:8]
        
        doc = Document(id=doc_id, content=content, metadata=metadata)
        self.documents.append(doc)
        return doc_id
    
    def index_documents(self):
        """Index all documents (create the vector index)."""
        if not self.documents:
            raise ValueError("No documents to index")
        
        # TF-IDF vectorization
        texts = [doc.content for doc in self.documents]
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',  # Use 'french' for French text
            ngram_range=(1, 2)
        )
        self.doc_vectors = self.vectorizer.fit_transform(texts)
        
        print(f"✓ Index created: {len(self.documents)} documents indexed")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """
        Search for the most relevant documents.
        
        Returns:
            List of tuples (Document, similarity score)
        """
        if self.vectorizer is None or self.doc_vectors is None:
            raise ValueError("Index not created. Call index_documents() first.")
        
        # Query vectorization
        query_vec = self.vectorizer.transform([query])
        
        # Cosine similarity calculation
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
        
        # Top-K documents
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = [
            (self.documents[idx], similarities[idx])
            for idx in top_indices
        ]
        
        return results


# ============================================================================
# PART 2: TOOLS (Chapter 14)
# ============================================================================

class ToolRegistry:
    """Registry of tools available to the agent."""
    
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str, description: str, func):
        """Register a new tool."""
        self.tools[name] = {
            "name": name,
            "description": description,
            "func": func
        }
    
    def get_tools_description(self) -> str:
        """Generate a text description of all tools."""
        if not self.tools:
            return "No tools available."
        
        desc = "Available tools:\n"
        for tool in self.tools.values():
            desc += f"  - {tool['name']}: {tool['description']}\n"
        return desc
    
    def execute(self, tool_name: str, **kwargs) -> str:
        """Execute a tool with the provided arguments."""
        if tool_name not in self.tools:
            return f"❌ Unknown tool '{tool_name}'"
        
        try:
            result = self.tools[tool_name]["func"](**kwargs)
            return str(result)
        except Exception as e:
            return f"❌ Error executing '{tool_name}': {e}"


# Predefined tools

def tool_calculator(expression: str) -> str:
    """Evaluate a simple mathematical expression."""
    try:
        # Security: whitelist of allowed operations
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Invalid expression (forbidden characters)"
        
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"


def tool_search_knowledge(query: str, rag_system: RAGSystem) -> str:
    """Search in the RAG knowledge base."""
    try:
        results = rag_system.retrieve(query, top_k=2)
        
        if not results:
            return "No relevant documents found."
        
        response = "Documents found:\n"
        for doc, score in results:
            title = doc.metadata.get('title', 'Untitled')
            snippet = doc.content[:200] + "..."
            response += f"\n[{title}] (score: {score:.2f})\n{snippet}\n"
        
        return response
    except Exception as e:
        return f"Search error: {e}"


def tool_current_time() -> str:
    """Returns the current time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def tool_summarize(text: str) -> str:
    """Summarize a text (simplified version)."""
    sentences = text.split('.')
    # Take the first 2 sentences
    summary = '. '.join(sentences[:2]) + '.'
    return f"Summary: {summary}"


# ============================================================================
# PART 3: REACT AGENT (Chapter 14)
# ============================================================================

class ReActAgent:
    """
    Autonomous agent with ReAct pattern (Reason + Act).
    
    Loop: Thought → Action → Observation → ... → Final Answer
    """
    
    def __init__(self, rag_system: RAGSystem, use_real_llm: bool = False):
        self.rag_system = rag_system
        self.tools = ToolRegistry()
        self.history: List[Dict[str, str]] = []
        self.use_real_llm = use_real_llm
        
        # Register tools
        self._register_tools()
    
    def _register_tools(self):
        """Register all available tools."""
        self.tools.register(
            "calculator",
            "Evaluates a mathematical expression (e.g.: 2+2, 5*3)",
            tool_calculator
        )
        
        self.tools.register(
            "search",
            "Search in the knowledge base",
            lambda query: tool_search_knowledge(query, self.rag_system)
        )
        
        self.tools.register(
            "current_time",
            "Returns the current date and time",
            tool_current_time
        )
        
        self.tools.register(
            "summarize",
            "Summarizes a long text",
            tool_summarize
        )
    
    def _simulate_llm_reasoning(self, prompt: str, step_count: int = 1) -> str:
        """
        Simulate an LLM (standalone mode).
        
        In production, replace with a call to OpenAI, Claude, etc.
        After 1-2 iterations, return a final answer to avoid infinite loops.
        """
        # Simple pattern detection for demo
        prompt_lower = prompt.lower()
        
        # Pattern: math question
        if any(op in prompt_lower for op in ['calcul', 'combien', '+', '*', '/', '-']):
            # Extract the math expression
            match = re.search(r'(\d+\s*[+\-*/]\s*\d+)', prompt)
            if match:
                expr = match.group(1)
                # First iteration: call the calculator
                if step_count <= 1:
                    return f"Thought: I need to calculate {expr}\nAction: calculator(expression='{expr}')"
                else:
                    # Next iteration: give the final answer
                    try:
                        result = eval(expr)
                        return f"Thought: I got the result of the calculation.\nFinal Answer: The result of {expr} is {result}."
                    except:
                        return "Thought: I have the answer.\nFinal Answer: The calculation was performed successfully."
        
        # Pattern: information search
        if any(word in prompt_lower for word in ['qu\'est-ce', 'définition', 'explique', 'parle-moi', 'qu est ce']):
            # Extract the subject
            for keyword in ['transformer', 'attention', 'llm', 'rag', 'agent', 'lora']:
                if keyword in prompt_lower:
                    if step_count <= 1:
                        return f"Thought: I need to search for information about {keyword}\nAction: search(query='{keyword}')"
                    else:
                        return f"Thought: I found relevant information about {keyword}.\nFinal Answer: The relevant documents explain the key concepts on the topic. According to the knowledge base, {keyword.upper()} is an important concept covered in detail."
        
        # Pattern: time/date
        if any(word in prompt_lower for word in ['heure', 'date', 'aujourd\'hui', 'maintenant', 'quelle heure']):
            if step_count <= 1:
                return "Thought: I need to get the current time\nAction: current_time()"
            else:
                return "Thought: I have the current time.\nFinal Answer: The time was retrieved successfully."
        
        # Default: generic search
        if step_count <= 1:
            return f"Thought: I will search for information about this question\nAction: search(query='{prompt[:50]}')"
        else:
            return "Thought: I have explored the knowledge base.\nFinal Answer: Based on the documents found, here is what I could determine about your question."
    
    def _call_real_llm(self, prompt: str) -> str:
        """
        Call a real LLM (OpenAI, Claude, etc.).
        
        Uncomment and configure in production.
        """
        # Example with OpenAI:
        # from openai import OpenAI
        # client = OpenAI()
        # response = client.chat.completions.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=0.7
        # )
        # return response.choices[0].message.content
        
        raise NotImplementedError("LLM configuration required")
    
    def _parse_action(self, response: str) -> Optional[Tuple[str, Dict[str, str]]]:
        """
        Parse the LLM response to extract the action.
        
        Expected format: Action: tool_name(param1='val1', param2='val2')
        """
        action_match = re.search(r"Action:\s*(\w+)\((.*?)\)", response)
        
        if not action_match:
            return None
        
        tool_name = action_match.group(1)
        params_str = action_match.group(2)
        
        # Parser les paramètres
        params = {}
        if params_str.strip():
            for param in params_str.split(','):
                if '=' in param:
                    key, val = param.split('=', 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    params[key] = val
        
        return tool_name, params
    
    def run(self, question: str, max_iterations: int = 5, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute the agent on a question.
        
        Returns:
            Dict with 'answer', 'steps', 'confidence'
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"🤖 Question: {question}")
            print(f"{'='*70}")
        
        context = f"User question: {question}\n\n{self.tools.get_tools_description()}\n"
        steps = []
        
        for iteration in range(1, max_iterations + 1):
            if verbose:
                print(f"\n{'─'*70}")
                print(f"⏳ Iteration {iteration}/{max_iterations}")
                print(f"{'─'*70}")
            
            # 1. Thought - LLM call
            prompt = context + "\n".join([
                f"Step {s['iteration']}: {s['thought']}\nAction: {s['action']}\nObservation: {s['observation']}"
                for s in steps
            ])
            
            if self.use_real_llm:
                llm_response = self._call_real_llm(prompt)
            else:
                llm_response = self._simulate_llm_reasoning(question, step_count=iteration)
            
            # Extract the thought
            thought_match = re.search(r"Thought:\s*(.+?)(?:\n|$)", llm_response)
            thought = thought_match.group(1) if thought_match else "Analysis in progress..."
            
            if verbose:
                print(f"💭 Thought: {thought}")
            
            # 2. Check if final answer
            if "final answer:" in llm_response.lower():
                answer = llm_response.split("Final Answer:", 1)[1].strip() if "Final Answer:" in llm_response else llm_response.split("final answer:", 1)[1].strip()
                if verbose:
                    print(f"\n✅ Final Answer: {answer}")
                
                return {
                    "answer": answer,
                    "steps": steps,
                    "iterations": iteration,
                    "confidence": self._calculate_confidence(steps)
                }
            
            # 3. Parse and execute the action
            action_parsed = self._parse_action(llm_response)
            
            if not action_parsed:
                if verbose:
                    print("⚠️ No action detected")
                continue
            
            tool_name, params = action_parsed
            action_str = f"{tool_name}({', '.join(f'{k}={v}' for k, v in params.items())})"
            
            if verbose:
                print(f"🔧 Action: {action_str}")
            
            # 4. Observation - Tool execution
            observation = self.tools.execute(tool_name, **params)
            
            if verbose:
                print(f"📊 Observation: {observation[:200]}...")
            
            # Save the step
            steps.append({
                "iteration": iteration,
                "thought": thought,
                "action": action_str,
                "observation": observation
            })
            
            # Update the context
            context += f"\nStep {iteration}:\nThought: {thought}\nAction: {action_str}\nObservation: {observation}\n"
        
        # Max iterations reached
        final_answer = "Unable to answer within the allowed number of iterations."
        
        return {
            "answer": final_answer,
            "steps": steps,
            "iterations": max_iterations,
            "confidence": 0.0
        }
    
    def _calculate_confidence(self, steps: List[Dict]) -> float:
        """
        Calculate a confidence score based on the steps.
        
        Simple heuristic: more successful steps = more confidence.
        In production, use more sophisticated metrics.
        """
        if not steps:
            return 0.0
        
        success_count = sum(
            1 for step in steps 
            if "❌" not in step["observation"]
        )
        
        return min(1.0, success_count / len(steps))


# ============================================================================
# PART 4: EVALUATION (Chapters 12, 15)
# ============================================================================

class AssistantEvaluator:
    """Evaluation of the quality of assistant responses."""
    
    @staticmethod
    def evaluate_response(
        question: str,
        response: Dict[str, Any],
        expected_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate an assistant response.
        
        Metrics:
            - Iterations used
            - Confidence score
            - Latency (simulated)
            - Coherence (if expected answer provided)
        """
        evaluation = {
            "question": question,
            "iterations": response["iterations"],
            "confidence": response["confidence"],
            "steps_count": len(response["steps"]),
            "success": response["confidence"] > 0.5
        }
        
        # Coherence evaluation (if expected answer provided)
        if expected_answer:
            # Simple similarity based on common words
            answer_words = set(response["answer"].lower().split())
            expected_words = set(expected_answer.lower().split())
            
            if answer_words and expected_words:
                overlap = len(answer_words & expected_words)
                total = len(answer_words | expected_words)
                evaluation["coherence_score"] = overlap / total
            else:
                evaluation["coherence_score"] = 0.0
        
        return evaluation
    
    @staticmethod
    def self_consistency_check(
        agent: ReActAgent,
        question: str,
        num_samples: int = 3
    ) -> Dict[str, Any]:
        """
        Check response consistency (self-consistency).
        
        Generates multiple responses and measures their agreement.
        Concept from chapter 12.
        """
        print(f"\n{'='*70}")
        print(f"🔬 Self-consistency test ({num_samples} samples)")
        print(f"{'='*70}")
        
        answers = []
        
        for i in range(num_samples):
            print(f"\nSample {i+1}/{num_samples}...")
            response = agent.run(question, verbose=False)
            answers.append(response["answer"])
        
        # Calculer la fréquence de chaque réponse
        from collections import Counter
        answer_counts = Counter(answers)
        most_common = answer_counts.most_common(1)[0]
        
        consistency_score = most_common[1] / num_samples
        
        return {
            "question": question,
            "num_samples": num_samples,
            "unique_answers": len(answer_counts),
            "most_common_answer": most_common[0],
            "consistency_score": consistency_score,
            "all_answers": answers
        }


# ============================================================================
# PART 5: DEMONSTRATION & MAIN
# ============================================================================

def initialize_knowledge_base() -> RAGSystem:
    """Initialize a knowledge base with demo documents."""
    rag = RAGSystem()
    
    # Documents about LLMs and AI
    documents = [
        {
            "content": """
Transformers are a neural network architecture introduced in 2017.
They use an attention mechanism that allows processing all tokens in parallel,
unlike RNNs which process sequentially. Transformers are the foundation
of all modern LLMs such as GPT, BERT, LLaMA, Claude, and Mistral.
The architecture includes an encoder and a decoder, although modern LLMs
often use only the decoder part.
            """,
            "metadata": {"title": "Transformer Architecture", "chapter": 3}
        },
        {
            "content": """
RAG (Retrieval-Augmented Generation) is a technique that combines information
retrieval with text generation. Before answering a question, the system
first searches for relevant documents in a knowledge base, then uses
these documents as context to generate a more accurate and factual response.
RAG helps reduce hallucinations and update knowledge
without retraining the model.
            """,
            "metadata": {"title": "RAG and Augmented Systems", "chapter": 13}
        },
        {
            "content": """
Autonomous agents use the ReAct (Reason + Act) pattern to solve complex
problems. The agent enters an iterative loop: it thinks (Thought), decides on
an action to perform (Action), observes the result (Observation), then repeats until
finding the answer. Agents can use external tools such as calculators,
APIs, or databases. The Model Context Protocol (MCP) standardizes how
these tools are integrated.
            """,
            "metadata": {"title": "Autonomous Agents", "chapter": 14}
        },
        {
            "content": """
LLM evaluation uses several metrics. Pass@k measures the probability of success
in k attempts. Self-consistency checks if the model gives the same answer multiple times.
Perplexity measures the quality of next token prediction. For agents,
we evaluate success rate, number of iterations, and robustness to errors.
Benchmarks like HumanEval (code) and MMLU (general knowledge) are standard.
            """,
            "metadata": {"title": "LLM Evaluation", "chapter": 12}
        },
        {
            "content": """
LoRA (Low-Rank Adaptation) is an efficient fine-tuning technique that freezes the base
model and adds small trainable matrices. This drastically reduces the number
of parameters to train (often less than 1% of the total model). QLoRA combines LoRA
with 4-bit quantization, enabling fine-tuning of 65B parameter models
on a single consumer GPU. These techniques have democratized access to fine-tuning.
            """,
            "metadata": {"title": "LoRA and QLoRA", "chapter": 9}
        }
    ]
    
    for doc in documents:
        rag.add_document(doc["content"], doc["metadata"])
    
    rag.index_documents()
    
    return rag


def run_demo():
    """Complete demonstration of the mini-assistant."""
    print("\n" + "="*70)
    print("🚀 COMPLETE MINI-ASSISTANT - INTEGRATIVE PROJECT")
    print("="*70)
    print("\nThis project combines:")
    print("  • RAG (Chapter 13): Document retrieval")
    print("  • ReAct Agents (Chapter 14): Autonomous loop")
    print("  • Prompting (Chapter 11): Structured generation")
    print("  • Evaluation (Chapters 12, 15): Quality metrics")
    
    # Phase 1: Initialization
    print("\n" + "="*70)
    print("📚 Phase 1: Initializing the knowledge base")
    print("="*70)
    
    rag_system = initialize_knowledge_base()
    
    # Phase 2: Creating the agent
    print("\n" + "="*70)
    print("🤖 Phase 2: Creating the agent")
    print("="*70)
    
    agent = ReActAgent(rag_system, use_real_llm=False)
    print(f"✓ Agent created with {len(agent.tools.tools)} tools")
    
    # Phase 3: Test questions
    print("\n" + "="*70)
    print("💬 Phase 3: Test questions")
    print("="*70)
    
    test_questions = [
        "What is a Transformer?",
        "What is 15 * 8?",
        "Explain RAG to me",
    ]
    
    results = []
    evaluator = AssistantEvaluator()
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'#'*70}")
        print(f"Question {i}/{len(test_questions)}")
        print(f"{'#'*70}")
        
        response = agent.run(question, max_iterations=3, verbose=True)
        
        # Évaluation
        evaluation = evaluator.evaluate_response(question, response)
        results.append({
            "question": question,
            "response": response,
            "evaluation": evaluation
        })
    
    # Phase 4: Evaluation report
    print("\n" + "="*70)
    print("📊 Phase 4: Evaluation report")
    print("="*70)
    
    print("\nPerformance summary:")
    print("-" * 70)
    
    for i, result in enumerate(results, 1):
        eval_data = result["evaluation"]
        print(f"\nQuestion {i}: {result['question'][:50]}...")
        print(f"  • Iterations: {eval_data['iterations']}")
        print(f"  • Confidence: {eval_data['confidence']:.2%}")
        print(f"  • Success: {'✅' if eval_data['success'] else '❌'}")
    
    # Global statistics
    avg_iterations = np.mean([r["evaluation"]["iterations"] for r in results])
    avg_confidence = np.mean([r["evaluation"]["confidence"] for r in results])
    success_rate = np.mean([r["evaluation"]["success"] for r in results])
    
    print("\n" + "="*70)
    print("📈 Global statistics")
    print("="*70)
    print(f"  • Number of questions: {len(results)}")
    print(f"  • Average iterations: {avg_iterations:.1f}")
    print(f"  • Average confidence: {avg_confidence:.2%}")
    print(f"  • Success rate: {success_rate:.2%}")
    
    # Phase 5: Self-consistency test (optional)
    print("\n" + "="*70)
    print("🔬 Phase 5: Self-consistency test (BONUS)")
    print("="*70)
    print("\nThis test generates multiple responses for the same question")
    print("and measures their consistency (concept from chapter 12).")
    
    consistency_test = evaluator.self_consistency_check(
        agent,
        "What is RAG?",
        num_samples=3
    )
    
    print(f"\nResults:")
    print(f"  • Majority answer: {consistency_test['most_common_answer'][:80]}...")
    print(f"  • Consistency score: {consistency_test['consistency_score']:.2%}")
    print(f"  • Unique answers: {consistency_test['unique_answers']}/{consistency_test['num_samples']}")
    
    # Conclusion
    print("\n" + "="*70)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*70)
    print("\n💡 Extension points for students:")
    print("  1. Integrate a real LLM (OpenAI, Claude, Ollama)")
    print("  2. Add new tools (weather, news, etc.)")
    print("  3. Persist conversations in a database")
    print("  4. Create a web interface with Streamlit or Gradio")
    print("  5. Implement more advanced evaluation metrics")
    print("  6. Add logging and monitoring for production")
    print("  7. Handle errors and timeouts more robustly")
    print("\n📖 Reference: See chapters 11-15 of the book for detailed concepts.")
    print()


if __name__ == "__main__":
    run_demo()
