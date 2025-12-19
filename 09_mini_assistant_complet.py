#!/usr/bin/env python
"""
Script BONUS 4 : Mini-Assistant Complet - Projet Intégrateur (Chapitres 11-15)

Ce projet final combine RAG, Agents, Prompting et Évaluation en un système cohérent.
Il démontre comment assembler tous les concepts du livre dans une application réelle.

Architecture :
    1. RAG : Indexation et recherche de documents (Ch. 13)
    2. Agent ReAct : Boucle Thought→Action→Observation (Ch. 14)
    3. Prompting : Zero-shot, Few-shot, Chain-of-Thought (Ch. 11)
    4. Évaluation : Self-consistency, confidence scoring (Ch. 12, 15)
    5. Outils : Calculatrice, recherche, résumé (Ch. 14)

Modes d'exécution :
    - Mode STANDALONE : Fonctionne sans API externe (LLM simulé)
    - Mode PRODUCTION : Intégration OpenAI/Claude (décommenter le code)

Dépendances minimales (mode standalone) :
    pip install numpy scikit-learn

Dépendances production (optionnel) :
    pip install openai anthropic

Utilisation :
    python 09_mini_assistant_complet.py

Points d'extension pour étudiants :
    - Ajouter de nouveaux outils (météo, actualités, etc.)
    - Intégrer un vrai LLM (OpenAI, Ollama, etc.)
    - Persister les conversations (SQLite, JSON)
    - Ajouter une interface web (Streamlit, Gradio)
    - Implémenter des métriques d'évaluation plus avancées
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib

# Imports pour le RAG (vectorisation basique)
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================================
# PARTIE 1 : SYSTÈME RAG (Chapitre 13)
# ============================================================================

@dataclass
class Document:
    """Représentation d'un document dans la base de connaissances."""
    id: str
    content: str
    metadata: Dict[str, Any]
    
    def __repr__(self):
        return f"Doc({self.metadata.get('title', 'Untitled')})"


class RAGSystem:
    """
    Système RAG complet avec vectorisation TF-IDF et recherche par similarité.
    
    En production, remplacer par des embeddings denses (OpenAI, E5, etc.)
    et une base vectorielle (Pinecone, Weaviate, ChromaDB).
    """
    
    def __init__(self):
        self.documents: List[Document] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_vectors: Optional[np.ndarray] = None
    
    def add_document(self, content: str, metadata: Dict[str, Any]) -> str:
        """Ajouter un document à la base de connaissances."""
        doc_id = hashlib.md5(
            f"{content[:100]}{datetime.now()}".encode()
        ).hexdigest()[:8]
        
        doc = Document(id=doc_id, content=content, metadata=metadata)
        self.documents.append(doc)
        return doc_id
    
    def index_documents(self):
        """Indexer tous les documents (création de l'index vectoriel)."""
        if not self.documents:
            raise ValueError("Aucun document à indexer")
        
        # Vectorisation TF-IDF
        texts = [doc.content for doc in self.documents]
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',  # Utilisez 'french' pour du français
            ngram_range=(1, 2)
        )
        self.doc_vectors = self.vectorizer.fit_transform(texts)
        
        print(f"✓ Index créé : {len(self.documents)} documents indexés")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """
        Rechercher les documents les plus pertinents.
        
        Returns:
            Liste de tuples (Document, score de similarité)
        """
        if self.vectorizer is None or self.doc_vectors is None:
            raise ValueError("Index non créé. Appelez index_documents() d'abord.")
        
        # Vectorisation de la requête
        query_vec = self.vectorizer.transform([query])
        
        # Calcul de similarité cosinus
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
        
        # Top-K documents
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = [
            (self.documents[idx], similarities[idx])
            for idx in top_indices
        ]
        
        return results


# ============================================================================
# PARTIE 2 : OUTILS (Chapitre 14)
# ============================================================================

class ToolRegistry:
    """Registre d'outils disponibles pour l'agent."""
    
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str, description: str, func):
        """Enregistrer un nouvel outil."""
        self.tools[name] = {
            "name": name,
            "description": description,
            "func": func
        }
    
    def get_tools_description(self) -> str:
        """Générer une description texte de tous les outils."""
        if not self.tools:
            return "Aucun outil disponible."
        
        desc = "Outils disponibles :\n"
        for tool in self.tools.values():
            desc += f"  - {tool['name']}: {tool['description']}\n"
        return desc
    
    def execute(self, tool_name: str, **kwargs) -> str:
        """Exécuter un outil avec les arguments fournis."""
        if tool_name not in self.tools:
            return f"❌ Outil '{tool_name}' inconnu"
        
        try:
            result = self.tools[tool_name]["func"](**kwargs)
            return str(result)
        except Exception as e:
            return f"❌ Erreur lors de l'exécution de '{tool_name}': {e}"


# Outils prédéfinis

def tool_calculator(expression: str) -> str:
    """Évalue une expression mathématique simple."""
    try:
        # Sécurité : whitelist des opérations autorisées
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Expression invalide (caractères interdits)"
        
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Erreur de calcul : {e}"


def tool_search_knowledge(query: str, rag_system: RAGSystem) -> str:
    """Recherche dans la base de connaissances RAG."""
    try:
        results = rag_system.retrieve(query, top_k=2)
        
        if not results:
            return "Aucun document pertinent trouvé."
        
        response = "Documents trouvés :\n"
        for doc, score in results:
            title = doc.metadata.get('title', 'Sans titre')
            snippet = doc.content[:200] + "..."
            response += f"\n[{title}] (score: {score:.2f})\n{snippet}\n"
        
        return response
    except Exception as e:
        return f"Erreur de recherche : {e}"


def tool_current_time() -> str:
    """Retourne l'heure actuelle."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def tool_summarize(text: str) -> str:
    """Résume un texte (version simplifiée)."""
    sentences = text.split('.')
    # Prendre les 2 premières phrases
    summary = '. '.join(sentences[:2]) + '.'
    return f"Résumé : {summary}"


# ============================================================================
# PARTIE 3 : AGENT REACT (Chapitre 14)
# ============================================================================

class ReActAgent:
    """
    Agent autonome avec pattern ReAct (Reason + Act).
    
    Boucle : Thought → Action → Observation → ... → Final Answer
    """
    
    def __init__(self, rag_system: RAGSystem, use_real_llm: bool = False):
        self.rag_system = rag_system
        self.tools = ToolRegistry()
        self.history: List[Dict[str, str]] = []
        self.use_real_llm = use_real_llm
        
        # Enregistrement des outils
        self._register_tools()
    
    def _register_tools(self):
        """Enregistrer tous les outils disponibles."""
        self.tools.register(
            "calculator",
            "Évalue une expression mathématique (ex: 2+2, 5*3)",
            tool_calculator
        )
        
        self.tools.register(
            "search",
            "Recherche dans la base de connaissances",
            lambda query: tool_search_knowledge(query, self.rag_system)
        )
        
        self.tools.register(
            "current_time",
            "Retourne la date et l'heure actuelles",
            tool_current_time
        )
        
        self.tools.register(
            "summarize",
            "Résume un texte long",
            tool_summarize
        )
    
    def _simulate_llm_reasoning(self, prompt: str, step_count: int = 1) -> str:
        """
        Simuler un LLM (mode standalone).
        
        En production, remplacer par un appel à OpenAI, Claude, etc.
        Après 1-2 itérations, retourner une réponse finale pour éviter les boucles infinies.
        """
        # Détection de patterns simples pour la démo
        prompt_lower = prompt.lower()
        
        # Pattern : question mathématique
        if any(op in prompt_lower for op in ['calcul', 'combien', '+', '*', '/', '-']):
            # Extraire l'expression mathématique
            match = re.search(r'(\d+\s*[+\-*/]\s*\d+)', prompt)
            if match:
                expr = match.group(1)
                # Première itération : appeler la calculatrice
                if step_count <= 1:
                    return f"Thought: Je dois calculer {expr}\nAction: calculator(expression='{expr}')"
                else:
                    # Itération suivante : donner la réponse finale
                    try:
                        result = eval(expr)
                        return f"Thought: J'ai obtenu le résultat du calcul.\nFinal Answer: Le résultat de {expr} est {result}."
                    except:
                        return "Thought: J'ai la réponse.\nFinal Answer: Le calcul a été effectué avec succès."
        
        # Pattern : recherche d'information
        if any(word in prompt_lower for word in ['qu\'est-ce', 'définition', 'explique', 'parle-moi', 'qu est ce']):
            # Extraire le sujet
            for keyword in ['transformer', 'attention', 'llm', 'rag', 'agent', 'lora']:
                if keyword in prompt_lower:
                    if step_count <= 1:
                        return f"Thought: Je dois chercher des informations sur {keyword}\nAction: search(query='{keyword}')"
                    else:
                        return f"Thought: J'ai trouvé des informations pertinentes sur {keyword}.\nFinal Answer: Les documents pertinents expliquent les concepts clés sur le sujet. Selon la base de connaissances, {keyword.upper()} est un concept important couvert en détail."
        
        # Pattern : heure/date
        if any(word in prompt_lower for word in ['heure', 'date', 'aujourd\'hui', 'maintenant', 'quelle heure']):
            if step_count <= 1:
                return "Thought: Je dois obtenir l'heure actuelle\nAction: current_time()"
            else:
                return "Thought: J'ai l'heure actuelle.\nFinal Answer: L'heure a été obtenue avec succès."
        
        # Par défaut : recherche générique
        if step_count <= 1:
            return f"Thought: Je vais chercher des informations sur cette question\nAction: search(query='{prompt[:50]}')"
        else:
            return "Thought: J'ai exploré la base de connaissances.\nFinal Answer: Basé sur les documents trouvés, voici ce que j'ai pu déterminer sur votre question."
    
    def _call_real_llm(self, prompt: str) -> str:
        """
        Appeler un vrai LLM (OpenAI, Claude, etc.).
        
        À décommenter et configurer en production.
        """
        # Exemple avec OpenAI :
        # from openai import OpenAI
        # client = OpenAI()
        # response = client.chat.completions.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=0.7
        # )
        # return response.choices[0].message.content
        
        raise NotImplementedError("Configuration LLM requise")
    
    def _parse_action(self, response: str) -> Optional[Tuple[str, Dict[str, str]]]:
        """
        Parser la réponse du LLM pour extraire l'action.
        
        Format attendu : Action: tool_name(param1='val1', param2='val2')
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
        Exécuter l'agent sur une question.
        
        Returns:
            Dict avec 'answer', 'steps', 'confidence'
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"🤖 Question : {question}")
            print(f"{'='*70}")
        
        context = f"Question utilisateur : {question}\n\n{self.tools.get_tools_description()}\n"
        steps = []
        
        for iteration in range(1, max_iterations + 1):
            if verbose:
                print(f"\n{'─'*70}")
                print(f"⏳ Itération {iteration}/{max_iterations}")
                print(f"{'─'*70}")
            
            # 1. Pensée (Thought) - Appel au LLM
            prompt = context + "\n".join([
                f"Étape {s['iteration']}: {s['thought']}\nAction: {s['action']}\nObservation: {s['observation']}"
                for s in steps
            ])
            
            if self.use_real_llm:
                llm_response = self._call_real_llm(prompt)
            else:
                llm_response = self._simulate_llm_reasoning(question, step_count=iteration)
            
            # Extraire la pensée
            thought_match = re.search(r"Thought:\s*(.+?)(?:\n|$)", llm_response)
            thought = thought_match.group(1) if thought_match else "Analyse en cours..."
            
            if verbose:
                print(f"💭 Pensée : {thought}")
            
            # 2. Vérifier si réponse finale
            if "final answer:" in llm_response.lower():
                answer = llm_response.split("Final Answer:", 1)[1].strip() if "Final Answer:" in llm_response else llm_response.split("final answer:", 1)[1].strip()
                if verbose:
                    print(f"\n✅ Réponse finale : {answer}")
                
                return {
                    "answer": answer,
                    "steps": steps,
                    "iterations": iteration,
                    "confidence": self._calculate_confidence(steps)
                }
            
            # 3. Parser et exécuter l'action
            action_parsed = self._parse_action(llm_response)
            
            if not action_parsed:
                if verbose:
                    print("⚠️ Pas d'action détectée")
                continue
            
            tool_name, params = action_parsed
            action_str = f"{tool_name}({', '.join(f'{k}={v}' for k, v in params.items())})"
            
            if verbose:
                print(f"🔧 Action : {action_str}")
            
            # 4. Observation - Exécution de l'outil
            observation = self.tools.execute(tool_name, **params)
            
            if verbose:
                print(f"📊 Observation : {observation[:200]}...")
            
            # Sauvegarder l'étape
            steps.append({
                "iteration": iteration,
                "thought": thought,
                "action": action_str,
                "observation": observation
            })
            
            # Mettre à jour le contexte
            context += f"\nÉtape {iteration}:\nPensée: {thought}\nAction: {action_str}\nObservation: {observation}\n"
        
        # Max iterations atteint
        final_answer = "Impossible de répondre dans le nombre d'itérations autorisé."
        
        return {
            "answer": final_answer,
            "steps": steps,
            "iterations": max_iterations,
            "confidence": 0.0
        }
    
    def _calculate_confidence(self, steps: List[Dict]) -> float:
        """
        Calculer un score de confiance basé sur les étapes.
        
        Heuristique simple : plus d'étapes réussies = plus de confiance.
        En production, utiliser des métriques plus sophistiquées.
        """
        if not steps:
            return 0.0
        
        success_count = sum(
            1 for step in steps 
            if "❌" not in step["observation"]
        )
        
        return min(1.0, success_count / len(steps))


# ============================================================================
# PARTIE 4 : ÉVALUATION (Chapitres 12, 15)
# ============================================================================

class AssistantEvaluator:
    """Évaluation de la qualité des réponses de l'assistant."""
    
    @staticmethod
    def evaluate_response(
        question: str,
        response: Dict[str, Any],
        expected_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Évaluer une réponse de l'assistant.
        
        Métriques :
            - Iterations utilisées
            - Confidence score
            - Latence (simulée)
            - Cohérence (si réponse attendue fournie)
        """
        evaluation = {
            "question": question,
            "iterations": response["iterations"],
            "confidence": response["confidence"],
            "steps_count": len(response["steps"]),
            "success": response["confidence"] > 0.5
        }
        
        # Évaluation de cohérence (si réponse attendue)
        if expected_answer:
            # Similarité simple basée sur mots communs
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
        Vérifier la cohérence des réponses (self-consistency).
        
        Génère plusieurs réponses et mesure leur accord.
        Concept du chapitre 12.
        """
        print(f"\n{'='*70}")
        print(f"🔬 Test de self-consistency ({num_samples} échantillons)")
        print(f"{'='*70}")
        
        answers = []
        
        for i in range(num_samples):
            print(f"\nÉchantillon {i+1}/{num_samples}...")
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
# PARTIE 5 : DÉMONSTRATION & MAIN
# ============================================================================

def initialize_knowledge_base() -> RAGSystem:
    """Initialiser une base de connaissances avec des documents de démo."""
    rag = RAGSystem()
    
    # Documents sur les LLMs et l'IA
    documents = [
        {
            "content": """
Les Transformers sont une architecture de réseaux de neurones introduite en 2017.
Ils utilisent un mécanisme d'attention qui permet de traiter tous les tokens en parallèle,
contrairement aux RNN qui traitent séquentiellement. Les Transformers sont la base
de tous les LLMs modernes comme GPT, BERT, LLaMA, Claude et Mistral.
L'architecture comprend un encodeur et un décodeur, bien que les LLMs modernes
utilisent souvent seulement la partie décodeur.
            """,
            "metadata": {"title": "Architecture Transformer", "chapter": 3}
        },
        {
            "content": """
Le RAG (Retrieval-Augmented Generation) est une technique qui combine la recherche
d'information avec la génération de texte. Avant de répondre à une question, le système
recherche d'abord des documents pertinents dans une base de connaissances, puis utilise
ces documents comme contexte pour générer une réponse plus précise et factuelle.
Le RAG permet de réduire les hallucinations et de mettre à jour les connaissances
sans réentraîner le modèle.
            """,
            "metadata": {"title": "RAG et systèmes augmentés", "chapter": 13}
        },
        {
            "content": """
Les agents autonomes utilisent le pattern ReAct (Reason + Act) pour résoudre des problèmes
complexes. L'agent entre dans une boucle itérative : il réfléchit (Thought), décide d'une
action à effectuer (Action), observe le résultat (Observation), puis recommence jusqu'à
trouver la réponse. Les agents peuvent utiliser des outils externes comme des calculatrices,
des API ou des bases de données. Le Model Context Protocol (MCP) standardise la manière
dont ces outils sont intégrés.
            """,
            "metadata": {"title": "Agents autonomes", "chapter": 14}
        },
        {
            "content": """
L'évaluation des LLMs utilise plusieurs métriques. Pass@k mesure la probabilité de succès
en k tentatives. La self-consistency vérifie si le modèle donne la même réponse plusieurs fois.
La perplexité mesure la qualité de la prédiction du prochain token. Pour les agents,
on évalue le taux de succès, le nombre d'itérations et la robustesse aux erreurs.
Les benchmarks comme HumanEval (code) et MMLU (connaissances générales) sont standards.
            """,
            "metadata": {"title": "Évaluation des LLMs", "chapter": 12}
        },
        {
            "content": """
LoRA (Low-Rank Adaptation) est une technique de fine-tuning efficace qui gèle le modèle
de base et ajoute de petites matrices entraînables. Cela réduit drastiquement le nombre
de paramètres à entraîner (souvent moins de 1% du modèle total). QLoRA combine LoRA
avec la quantification 4-bit, permettant de fine-tuner des modèles de 65B paramètres
sur une seule carte GPU grand public. Ces techniques ont démocratisé l'accès au fine-tuning.
            """,
            "metadata": {"title": "LoRA et QLoRA", "chapter": 9}
        }
    ]
    
    for doc in documents:
        rag.add_document(doc["content"], doc["metadata"])
    
    rag.index_documents()
    
    return rag


def run_demo():
    """Démonstration complète du mini-assistant."""
    print("\n" + "="*70)
    print("🚀 MINI-ASSISTANT COMPLET - PROJET INTÉGRATEUR")
    print("="*70)
    print("\nCe projet combine :")
    print("  • RAG (Chapitre 13) : Recherche de documents")
    print("  • Agents ReAct (Chapitre 14) : Boucle autonome")
    print("  • Prompting (Chapitre 11) : Génération structurée")
    print("  • Évaluation (Chapitres 12, 15) : Métriques de qualité")
    
    # Phase 1 : Initialisation
    print("\n" + "="*70)
    print("📚 Phase 1 : Initialisation de la base de connaissances")
    print("="*70)
    
    rag_system = initialize_knowledge_base()
    
    # Phase 2 : Création de l'agent
    print("\n" + "="*70)
    print("🤖 Phase 2 : Création de l'agent")
    print("="*70)
    
    agent = ReActAgent(rag_system, use_real_llm=False)
    print(f"✓ Agent créé avec {len(agent.tools.tools)} outils")
    
    # Phase 3 : Questions de test
    print("\n" + "="*70)
    print("💬 Phase 3 : Questions de test")
    print("="*70)
    
    test_questions = [
        "Qu'est-ce qu'un Transformer ?",
        "Combien font 15 * 8 ?",
        "Explique-moi le RAG",
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
    
    # Phase 4 : Rapport d'évaluation
    print("\n" + "="*70)
    print("📊 Phase 4 : Rapport d'évaluation")
    print("="*70)
    
    print("\nRésumé des performances :")
    print("-" * 70)
    
    for i, result in enumerate(results, 1):
        eval_data = result["evaluation"]
        print(f"\nQuestion {i} : {result['question'][:50]}...")
        print(f"  • Itérations : {eval_data['iterations']}")
        print(f"  • Confiance : {eval_data['confidence']:.2%}")
        print(f"  • Succès : {'✅' if eval_data['success'] else '❌'}")
    
    # Statistiques globales
    avg_iterations = np.mean([r["evaluation"]["iterations"] for r in results])
    avg_confidence = np.mean([r["evaluation"]["confidence"] for r in results])
    success_rate = np.mean([r["evaluation"]["success"] for r in results])
    
    print("\n" + "="*70)
    print("📈 Statistiques globales")
    print("="*70)
    print(f"  • Nombre de questions : {len(results)}")
    print(f"  • Itérations moyennes : {avg_iterations:.1f}")
    print(f"  • Confiance moyenne : {avg_confidence:.2%}")
    print(f"  • Taux de succès : {success_rate:.2%}")
    
    # Phase 5 : Test de self-consistency (optionnel)
    print("\n" + "="*70)
    print("🔬 Phase 5 : Test de self-consistency (BONUS)")
    print("="*70)
    print("\nCe test génère plusieurs réponses pour la même question")
    print("et mesure leur cohérence (concept du chapitre 12).")
    
    consistency_test = evaluator.self_consistency_check(
        agent,
        "Qu'est-ce que le RAG ?",
        num_samples=3
    )
    
    print(f"\nRésultats :")
    print(f"  • Réponse majoritaire : {consistency_test['most_common_answer'][:80]}...")
    print(f"  • Score de cohérence : {consistency_test['consistency_score']:.2%}")
    print(f"  • Réponses uniques : {consistency_test['unique_answers']}/{consistency_test['num_samples']}")
    
    # Conclusion
    print("\n" + "="*70)
    print("✅ DÉMONSTRATION TERMINÉE")
    print("="*70)
    print("\n💡 Points d'extension pour les étudiants :")
    print("  1. Intégrer un vrai LLM (OpenAI, Claude, Ollama)")
    print("  2. Ajouter de nouveaux outils (météo, actualités, etc.)")
    print("  3. Persister les conversations dans une base de données")
    print("  4. Créer une interface web avec Streamlit ou Gradio")
    print("  5. Implémenter des métriques d'évaluation plus avancées")
    print("  6. Ajouter du logging et du monitoring en production")
    print("  7. Gérer les erreurs et timeouts plus robustement")
    print("\n📖 Référence : Voir chapitres 11-15 du livre pour les concepts détaillés.")
    print()


if __name__ == "__main__":
    run_demo()
