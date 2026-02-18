# 🤖 A2A Multi-Agent System

> Three AI agents, three frameworks, one protocol — working together seamlessly.

A multi-agent system demonstrating **Google's Agent-to-Agent (A2A) Protocol** with heterogeneous AI frameworks communicating over a standardized HTTP-based protocol.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![A2A Protocol](https://img.shields.io/badge/protocol-A2A-green.svg)](https://google.github.io/A2A)
[![Azure OpenAI](https://img.shields.io/badge/LLM-Azure%20OpenAI-purple.svg)](https://azure.microsoft.com/en-us/products/ai-services/openai-service)

---

## ✨ What Is This?

This project demonstrates that AI agents built with **completely different frameworks** can discover and communicate with each other using a shared protocol. The orchestrator doesn't know (or care) how the other agents are built internally — it talks to them the same way.

| Agent                | Framework  | Responsibility                                    | Port    |
| -------------------- | ---------- | ------------------------------------------------- | ------- |
| **Orchestrator**     | LangGraph  | Classifies queries & routes to the right agent    | —       |
| **Arithmetic Agent** | CrewAI     | Math calculations (add, subtract, multiply, etc.) | `10001` |
| **Logic Agent**      | Google ADK | Boolean logic & reasoning (AND, OR, NOT, etc.)    | `10002` |
| **Streamlit UI**     | Streamlit  | Chat interface with routing visualization         | `8501`  |

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────┐
│               Streamlit UI (:8501)                  │
│            Chat Interface + Agent Status            │
└───────────────────────┬────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────┐
│           Orchestrator (LangGraph)                   │
│                                                      │
│  classify_query ──► route ──► call agent(s)          │
│                       │                              │
│         ┌─────────────┼──────────────┐               │
│         │             │              │               │
│    "arithmetic"   "logical"       "both"             │
└─────────┼─────────────┼──────────────┼───────────────┘
          │             │              │
     A2A Protocol  A2A Protocol   A2A Protocol
          │             │         (parallel)
┌─────────▼────┐  ┌─────▼────────┐
│  Arithmetic  │  │   Logical    │
│  Agent       │  │   Reasoning  │
│  (CrewAI)    │  │   Agent      │
│  :10001      │  │   (ADK)      │
│              │  │   :10002     │
└──────────────┘  └──────────────┘
```

### Routing Modes

| Classification | Behavior                                               | Badge |
| -------------- | ------------------------------------------------------ | ----- |
| `arithmetic`   | Routes to CrewAI agent only                            | 🧮    |
| `logical`      | Routes to ADK agent only                               | 🧠    |
| `both`         | Calls both agents **in parallel**, synthesizes results | 🧮🧠  |
| `general`      | Orchestrator handles directly                          | 💬    |

---

## 🚀 Quick Start

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- An [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) resource with a deployed model

### 1. Clone & Setup

```bash
git clone <repo-url>
cd A2A

# Create the conda environment
conda env create -f environment.yml
conda activate a2a-agents
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` with your Azure OpenAI credentials:

```bash
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-08-01-preview
```

### 3. Run

```bash
chmod +x run_all.sh
./run_all.sh
```

This starts all three services. Open **http://localhost:8501** to use the UI.

> **Manual start** (if you prefer separate terminals):
>
> ```bash
> python -m agents.arithmetic           # Terminal 1
> python -m agents.logical_reasoning    # Terminal 2
> streamlit run ui/app.py               # Terminal 3
> ```

---

## 🧪 Try It Out

### Arithmetic queries → CrewAI agent

- _"What is 25 _ 4 + 10?"\*
- _"Calculate 2 to the power of 10"_
- _"What is 17 modulo 5?"_

### Logic queries → Google ADK agent

- _"What is True AND False?"_
- _"If P→Q and P is True, what is Q?"_
- _"Evaluate: NOT (True OR False)"_

### Mixed queries → Both agents in parallel

- _"What is 5 _ 3, and is the result greater than 10?"\*
- _"Calculate 2^3 and check if the result equals 8 using AND with True"_

---

## 🧪 Test Suite

The project includes **97 offline tests** — no LLM calls or running servers needed.

```bash
# Run all tests
conda run -n a2a-agents python -m pytest tests/ -v

# Run a specific test file
conda run -n a2a-agents python -m pytest tests/test_arithmetic_tools.py -v
```

| Test File                  | Tests | What It Covers                                                                 |
| -------------------------- | ----- | ------------------------------------------------------------------------------ |
| `test_arithmetic_tools.py` | 32    | All 6 math tools — edge cases, zero, negatives, floats, division-by-zero       |
| `test_logical_tools.py`    | 33    | Full truth tables for all 7 logic ops + expression evaluator + De Morgan's law |
| `test_orchestrator.py`     | 17    | Routing logic, LLM classification (mocked), graph construction                 |
| `test_a2a_integration.py`  | 15    | Agent Card validation, A2A error types, response parsing                       |

---

## 🔧 Technology Stack

| Technology                                                                      | Role                      | Why                                                 |
| ------------------------------------------------------------------------------- | ------------------------- | --------------------------------------------------- |
| **[A2A Protocol](https://google.github.io/A2A)**                                | Inter-agent communication | Framework-agnostic agent interoperability           |
| **[LangGraph](https://langchain-ai.github.io/langgraph)**                       | Orchestrator              | Stateful graph-based routing with conditional edges |
| **[CrewAI](https://docs.crewai.com)**                                           | Arithmetic agent          | Role-based agents with tool calling                 |
| **[Google ADK](https://google.github.io/adk-docs)**                             | Logic agent               | Google's agent framework with LiteLLM adapter       |
| **[Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/)** | LLM backend               | GPT-4o powers all three agents                      |
| **[Streamlit](https://streamlit.io)**                                           | UI                        | Chat interface with real-time agent status          |
| **[Conda](https://docs.conda.io)**                                              | Environment               | Reproducible Python environment                     |

### LLM Integration per Agent

| Agent        | Integration          | Connection                     |
| ------------ | -------------------- | ------------------------------ |
| Orchestrator | `langchain-openai`   | `AzureChatOpenAI` direct       |
| Arithmetic   | `azure-ai-inference` | CrewAI's native Azure provider |
| Logic        | `litellm`            | ADK's LiteLLM adapter          |

---

## 📁 Project Structure

```
A2A/
├── agents/
│   ├── arithmetic/
│   │   ├── __main__.py          # A2A server (port 10001)
│   │   ├── agent.py             # CrewAI agent + 6 math tools
│   │   └── agent_executor.py    # A2A ↔ CrewAI bridge
│   ├── logical_reasoning/
│   │   ├── __main__.py          # A2A server (port 10002)
│   │   ├── agent.py             # ADK agent + 7 logic tools
│   │   └── agent_executor.py    # A2A ↔ ADK bridge
│   └── orchestrator/
│       ├── orchestrator.py      # LangGraph state machine
│       └── a2a_tools.py         # A2A client utilities
├── tests/
│   ├── test_arithmetic_tools.py # 32 unit tests for math tools
│   ├── test_logical_tools.py    # 33 unit tests for logic tools
│   ├── test_orchestrator.py     # 17 routing & classification tests
│   └── test_a2a_integration.py  # 15 A2A protocol layer tests
├── ui/
│   └── app.py                   # Streamlit chat interface
├── .env.example                 # Credential template
├── environment.yml              # Conda environment
├── run_all.sh                   # One-command launcher
├── GUIDE.md                     # Detailed tutorial & tech deep dive
└── README.md                    # This file
```

---

## 🔍 Verify Agents are Running

```bash
# Check Agent Cards
curl http://localhost:10001/.well-known/agent-card.json | python -m json.tool
curl http://localhost:10002/.well-known/agent-card.json | python -m json.tool
```

---

## 📖 Learn More

For a comprehensive tutorial explaining every technology and design decision in detail, see **[GUIDE.md](GUIDE.md)**.

---

## 📜 License

This project is for educational and demonstration purposes.

---

_Built with the A2A Protocol · LangGraph · CrewAI · Google ADK · Azure OpenAI_
