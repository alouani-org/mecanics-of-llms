# Intégration Avancée : Agent ReAct avec des LLMs Réels

Ce document montre comment adapter le script `06_react_agent_bonus.py` pour utiliser des **vrais LLMs** (OpenAI, Anthropic, Ollama, etc.) au lieu de la simulation incluse.

## 📌 Table des Matières

1. [OpenAI (GPT-4, GPT-3.5)](#openai)
2. [Anthropic (Claude)](#anthropic)
3. [Groq (inférence ultra-rapide)](#groq)
4. [Ollama (LLMs locaux)](#ollama)
5. [Gestion des Erreurs et Timeouts](#erreurs)
6. [Architecture Robuste](#architecture)

---

## <a name="openai"></a>1️⃣ OpenAI (GPT-4, GPT-3.5)

### Installation

```bash
pip install openai
```

### Configuration

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class OpenAIAgent(Agent):
    """Agent avec intégration OpenAI."""

    def __init__(self, model: str = "gpt-4", **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.client = client

    def _simulate_llm_reasoning(self, task: str, context: str) -> str:
        """Utiliser l'API OpenAI au lieu de la simulation."""
        
        tools_desc = self._format_tools_description()
        
        system_prompt = f"""Tu es un agent autonome capable de raisonner et d'agir.

Tu as accès aux outils suivants:
{tools_desc}

Réponds au format suivant:
Thought: [Ton analyse de la situation]
Action: nom_outil(param1=val1, param2=val2) OU Final Answer: [réponse finale]

Sois concis et actionnel."""

        user_message = f"""Tâche: {task}

Contexte actuel:
{context if context else '[Aucun contexte précédent]'}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.7,
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Erreur OpenAI: {e}")
            return "Thought: Erreur d'appel API.\nFinal Answer: Impossible de traiter la tâche."
```

### Utilisation

```python
# Créer un agent OpenAI
agent = OpenAIAgent(name="OpenAI-Agent", model="gpt-4")

# Enregistrer les outils (comme avant)
agent.register_tool(...)

# Exécuter
result = agent.run("Calcule 5 + 3")
```

### Coûts Estimés

- **GPT-4** : ~$0.03 / 1K input tokens, ~$0.06 / 1K output tokens
- **GPT-3.5-turbo** : ~$0.0005 / 1K input tokens, ~$0.0015 / 1K output tokens

---

## <a name="anthropic"></a>2️⃣ Anthropic (Claude)

### Installation

```bash
pip install anthropic
```

### Configuration

```python
import os
from anthropic import Anthropic

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class ClaudeAgent(Agent):
    """Agent avec intégration Anthropic Claude."""

    def __init__(self, model: str = "claude-3-opus-20240229", **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.client = client

    def _simulate_llm_reasoning(self, task: str, context: str) -> str:
        """Utiliser l'API Anthropic au lieu de la simulation."""
        
        tools_desc = self._format_tools_description()
        
        system_prompt = f"""Tu es un agent autonome expert en raisonnement et en planification.

Tu as accès aux outils suivants:
{tools_desc}

Format de réponse:
Thought: [Ton analyse]
Action: nom_outil(param1=val1) OU Final Answer: [réponse]"""

        messages = [
            {
                "role": "user",
                "content": f"""Tâche: {task}

Contexte:
{context if context else '[Nouveau contexte]'}""",
            }
        ]

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
            )
            return response.content[0].text
        except Exception as e:
            print(f"❌ Erreur Anthropic: {e}")
            return "Thought: Erreur API.\nFinal Answer: Service indisponible."
```

### Utilisation

```python
# Claude 3 Opus (le plus puissant)
agent = ClaudeAgent(model="claude-3-opus-20240229")

# Ou Claude 3 Sonnet (plus rapide, moins cher)
agent = ClaudeAgent(model="claude-3-sonnet-20240229")
```

### Avantages de Claude

- ✅ **Contexte long** : jusqu'à 200K tokens
- ✅ **Raisonnement supérieur** : particulièrement bon pour les agents complexes
- ✅ **Moins de hallucinations** : généralement plus fiable
- ✅ **Vision** : Claude 3 supporte les images

---

## <a name="groq"></a>3️⃣ Groq (Inférence Ultra-Rapide)

### Installation

```bash
pip install groq
```

### Configuration

```python
import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class GroqAgent(Agent):
    """Agent avec inférence Groq (très rapide)."""

    def __init__(self, model: str = "mixtral-8x7b-32768", **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.client = client

    def _simulate_llm_reasoning(self, task: str, context: str) -> str:
        """Groq : inférence extrêmement rapide."""
        
        tools_desc = self._format_tools_description()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"Agent autonome. Outils:\n{tools_desc}",
                    },
                    {"role": "user", "content": f"Tâche: {task}\n\nContexte: {context}"},
                ],
                temperature=0.7,
                max_tokens=512,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Erreur Groq: {e}")
            return "Thought: Erreur.\nFinal Answer: Réessayez."
```

### Utilisation & Avantages

```python
agent = GroqAgent(model="mixtral-8x7b-32768")  # ~500ms latence!
```

**Avantages:**
- ⚡ **Ultra-rapide** : 100-200 tokens/sec
- 💰 **Gratuit** : jusqu'à certains quotas
- 📊 **Bon pour les agents** : latence basse = meilleure réactivité

---

## <a name="ollama"></a>4️⃣ Ollama (LLMs Locaux)

### Installation

```bash
# Télécharger Ollama depuis https://ollama.ai
ollama pull mistral        # ou llama2, neural-chat, etc.
ollama serve               # Démarre le serveur sur localhost:11434
```

### Configuration

```python
import requests

class OllamaAgent(Agent):
    """Agent avec Ollama (LLM local)."""

    def __init__(self, model: str = "mistral", host: str = "localhost:11434", **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.host = host

    def _simulate_llm_reasoning(self, task: str, context: str) -> str:
        """Utiliser Ollama en local."""
        
        tools_desc = self._format_tools_description()
        prompt = f"""Tu es un agent autonome.

Outils:
{tools_desc}

Tâche: {task}
Contexte: {context}

Réponds au format:
Thought: ...
Action: ..."""

        try:
            response = requests.post(
                f"http://{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=30,
            )
            return response.json()["response"]
        except Exception as e:
            print(f"❌ Erreur Ollama: {e}")
            return "Thought: Erreur locale.\nFinal Answer: Serveur indisponible."
```

### Utilisation & Avantages

```python
agent = OllamaAgent(model="mistral")  # Mistral 7B en local
```

**Avantages:**
- 🔒 **Privé** : tout s'exécute en local
- 💰 **Gratuit** : une fois téléchargé
- 🚀 **Rapide** : GPU accélération possible
- ⚠️ **Limitation** : moins performant que GPT-4

---

## <a name="erreurs"></a>5️⃣ Gestion Robuste des Erreurs et Timeouts

### Exemple Avancé

```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

class RobustAgent(Agent):
    """Agent avec gestion d'erreurs robuste."""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _call_llm_with_retry(self, task: str, context: str) -> str:
        """Appeler l'LLM avec retry automatique."""
        return self._simulate_llm_reasoning(task, context)

    def run(self, task: str, verbose: bool = True) -> str:
        """Exécution avec gestion d'erreurs."""
        try:
            context = ""
            for iteration in range(1, self.max_iterations + 1):
                try:
                    # Appel avec retry
                    response = self._call_llm_with_retry(task, context)
                    
                    action, thought = self._parse_action(response)
                    if not action:
                        print("⚠️ Pas d'action générée")
                        break

                    if action.startswith("Final Answer:"):
                        return action.replace("Final Answer:", "").strip()

                    observation = self._execute_action(action)
                    context += f"\nItération {iteration}: {observation}"

                except TimeoutError:
                    print(f"❌ Timeout à l'itération {iteration}, réessai...")
                    continue
                except Exception as e:
                    print(f"❌ Erreur à l'itération {iteration}: {e}")
                    if iteration == self.max_iterations:
                        raise
                    continue

            return "Itérations maximales atteintes sans réponse."

        except Exception as e:
            print(f"❌ Erreur fatale: {e}")
            return f"Erreur: {e}"
```

### Installation pour les Retries

```bash
pip install tenacity
```

---

## <a name="architecture"></a>6️⃣ Architecture Robuste pour la Production

### Structure Recommandée

```python
from dataclasses import dataclass
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration centralisée de l'agent."""
    name: str
    model: str
    provider: str  # "openai", "anthropic", "groq", "ollama"
    max_iterations: int = 10
    timeout: int = 30
    temperature: float = 0.7
    api_key: Optional[str] = None


class ProductionAgent(Agent):
    """Agent optimisé pour la production."""

    def __init__(self, config: AgentConfig):
        super().__init__(name=config.name, max_iterations=config.max_iterations)
        self.config = config
        self._init_llm_client()

    def _init_llm_client(self):
        """Initialiser le client LLM basé sur le provider."""
        if self.config.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=self.config.api_key)
        elif self.config.provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.config.api_key)
        # ... autres providers

    def _call_llm(self, prompt: str) -> str:
        """Appel unifié avec logging."""
        logger.info(f"Appel LLM ({self.config.provider}), tokens: ~{len(prompt)//4}")
        # Implémentation
        pass

    def run(self, task: str, verbose: bool = True) -> str:
        """Exécution avec logging structuré."""
        logger.info(f"Démarrage agent: {self.config.name}, tâche: {task}")
        result = super().run(task, verbose)
        logger.info(f"Fin agent: {result[:100]}...")
        return result


# Utilisation
config = AgentConfig(
    name="Production-Agent",
    model="gpt-4",
    provider="openai",
    api_key="sk-...",
)
agent = ProductionAgent(config)
```

---

## 📊 Comparaison des Providers

| Provider | Latence | Coût | Qualité | Contexte | Local |
|----------|---------|------|---------|----------|-------|
| **OpenAI GPT-4** | 2-5s | $$ | ⭐⭐⭐⭐⭐ | 128K | ❌ |
| **Claude 3 Opus** | 2-4s | $$ | ⭐⭐⭐⭐⭐ | 200K | ❌ |
| **Groq Mixtral** | 0.5s | Free | ⭐⭐⭐⭐ | 32K | ❌ |
| **Mistral 7B (Ollama)** | 1-3s | Free | ⭐⭐⭐ | 4K | ✅ |
| **GPT-3.5-turbo** | 1-2s | $ | ⭐⭐⭐⭐ | 128K | ❌ |

---

## 🎯 Recommandations

- **Développement / Testing** → Groq (rapide, gratuit)
- **Agents complexes** → Claude (meilleur raisonnement)
- **Production scalable** → OpenAI (fiable, API robuste)
- **Privacy critique** → Ollama (local)
- **Coût sensible** → GPT-3.5-turbo ou Ollama

---

**Bon développement !** 🚀
