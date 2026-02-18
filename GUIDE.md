# Building a Multi-Agent System with the A2A Protocol

### A Complete Tutorial: LangGraph + CrewAI + Google ADK Working Together

---

## 📌 Introduction

What if your AI agents could talk to each other — regardless of which framework built them? That's the promise of Google's **Agent-to-Agent (A2A) protocol**, and in this tutorial, we'll build a working multi-agent system that proves it.

We'll create three specialized agents, each built with a **different** framework, and wire them together using A2A so they can discover and communicate with each other over HTTP — like microservices, but for AI agents.

**What we'll build:**

| Agent                | Framework  | Responsibility                                   |
| -------------------- | ---------- | ------------------------------------------------ |
| **Orchestrator**     | LangGraph  | Classifies queries and routes to the right agent |
| **Arithmetic Agent** | CrewAI     | Performs math calculations using tools           |
| **Logic Agent**      | Google ADK | Evaluates boolean logic and reasoning            |

**The big idea:** The orchestrator doesn't know _how_ the other agents work internally — it only communicates via the A2A protocol. You could swap the CrewAI agent for a LangChain agent tomorrow, and the orchestrator wouldn't notice. That's the power of protocol-driven interoperability.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI (:8501)                  │
│                  Chat Interface + Status                │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────┐
│              Orchestrator (LangGraph)                    │
│                                                         │
│   ┌──────────┐    ┌───────────┐    ┌──────────────┐    │
│   │ Classify │───►│   Route   │───►│ Call Agent(s)│    │
│   └──────────┘    └───────────┘    └──────┬───────┘    │
│                                           │             │
└───────────────────────────────────────────┼─────────────┘
                    A2A Protocol            │
              ┌─────────────┬───────────────┤
              │             │               │
    ┌─────────▼──────┐ ┌───▼────────────┐ ┌▼──────────┐
    │  Arithmetic    │ │    Logical     │ │   Both    │
    │  Agent (:10001)│ │  Agent (:10002)│ │ (parallel)│
    │   (CrewAI)     │ │  (Google ADK)  │ │           │
    └────────────────┘ └────────────────┘ └───────────┘
```

### How Data Flows

1. User types a question in Streamlit
2. Streamlit calls the **Orchestrator**
3. Orchestrator uses Azure OpenAI to **classify** the query
4. Based on classification, it **routes** via A2A protocol:
   - `"arithmetic"` → CrewAI agent
   - `"logical"` → Google ADK agent
   - `"both"` → Both agents in parallel, then combines results
   - `"general"` → Handles directly
5. Response flows back through the chain to the UI

---

## 🔧 Technology Deep Dive

### 1. The A2A Protocol

**What is it?** The Agent-to-Agent (A2A) protocol, created by Google, is an open standard for agent interoperability. It lets agents discover each other and exchange tasks over HTTP using JSON-RPC 2.0.

**Core concepts:**

| Concept         | Description                                                                                                |
| --------------- | ---------------------------------------------------------------------------------------------------------- |
| **Agent Card**  | A JSON manifest at `/.well-known/agent-card.json` describing an agent's capabilities, skills, and endpoint |
| **SendMessage** | The primary JSON-RPC method for sending tasks to an agent                                                  |
| **Task**        | A unit of work containing messages with text/file/data parts                                               |
| **Parts**       | The content units within messages (TextPart, FilePart, DataPart)                                           |

**Why it matters:** Without A2A, each framework (CrewAI, ADK, LangGraph) has its own API. A2A gives them a common language. Think of it like REST for AI agents.

**In our code**, the A2A integration happens in three places:

**a) Agent servers** — Each agent runs an A2A-compliant HTTP server:

```python
# agents/arithmetic/__main__.py

# Define what this agent can do (Agent Card)
agent_card = AgentCard(
    name="Arithmetic Agent",
    description="A CrewAI-powered agent that performs arithmetic...",
    url=f"http://{host}:{port}/",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=False),
    skills=[skill],
)

# Wire the agent executor to the A2A request handler
request_handler = DefaultRequestHandler(
    agent_executor=ArithmeticAgentExecutor(),
    task_store=InMemoryTaskStore(),
)

# Build the Starlette ASGI application
server = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=request_handler,
)

# Start the HTTP server
uvicorn.run(server.build(), host=host, port=port)
```

**b) Agent executors** — Bridge framework-specific agents to A2A:

```python
# agents/arithmetic/agent_executor.py

class ArithmeticAgentExecutor(AgentExecutor):
    """Bridges CrewAI to the A2A protocol."""

    def __init__(self):
        self.agent = ArithmeticAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        query = context.get_user_input()       # A2A → plain text
        result = self.agent.invoke(query)       # Run CrewAI
        await event_queue.enqueue_event(        # plain text → A2A
            new_agent_text_message(result)
        )
```

**c) Client-side discovery and task sending:**

```python
# agents/orchestrator/a2a_tools.py

async def discover_agent(base_url: str) -> AgentCard | None:
    """Fetch the AgentCard from a remote A2A agent."""
    resolver = A2ACardResolver(httpx_client=client, base_url=base_url)
    agent_card = await resolver.get_agent_card()
    return agent_card

async def send_task_to_agent(base_url: str, message_text: str) -> str:
    """Send a task to a remote A2A agent and return the response."""
    # 1. Discover the agent
    agent_card = await resolver.get_agent_card()

    # 2. Create an A2A client
    a2a_client = A2AClient(httpx_client=client, agent_card=agent_card)

    # 3. Build and send the request
    request = SendMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(
            message={
                "role": "user",
                "parts": [{"kind": "text", "text": message_text}],
                "messageId": uuid4().hex,
            }
        ),
    )
    response = await a2a_client.send_message(request)

    # 4. Extract the text from the response
    return _extract_response_text(response)
```

---

### 2. CrewAI (Arithmetic Agent)

**What is CrewAI?** A framework for building multi-agent systems with a "crew" metaphor — you define Agents with roles, give them Tasks, and organize them into a Crew that executes sequentially or in parallel.

**Key components we use:**

| Component | Purpose                                                       |
| --------- | ------------------------------------------------------------- |
| `@tool`   | Decorates Python functions as callable tools                  |
| `Agent`   | Defines personality (role/goal/backstory) and available tools |
| `Task`    | Describes what to do and what output to expect                |
| `Crew`    | Orchestrates agent(s) executing task(s)                       |
| `LLM`     | Configures the language model backend                         |

**Our arithmetic tools:**

```python
@tool("AdditionTool")
def add(a: float, b: float) -> float:
    """Add two numbers together. Returns a + b."""
    return a + b

@tool("MultiplicationTool")
def multiply(a: float, b: float) -> float:
    """Multiply two numbers. Returns a * b."""
    return a * b

# Also: subtract, divide, modulo, power
```

The `@tool` decorator is key — it registers the function with CrewAI's tool system, including the docstring as the tool description that the LLM reads to decide when to use it.

**Agent configuration:**

```python
self.calculator_agent = Agent(
    role="Arithmetic Calculator",
    goal="Perform accurate arithmetic calculations...",
    backstory="You are an expert mathematician...",
    verbose=False,
    allow_delegation=False,     # Don't pass tasks to other agents
    tools=[add, subtract, multiply, divide, modulo, power],
    llm=self.model,             # Azure OpenAI via native provider
)
```

**How it works at runtime:**

1. User asks: _"What is 25 _ 4 + 10?"\*
2. CrewAI formats the task prompt with the user's question
3. The LLM (Azure OpenAI) decides which tools to call and in what order
4. It calls `multiply(25, 4)` → `100`, then `add(100, 10)` → `110`
5. The LLM returns a natural language answer: _"The result is 110"_

**Azure OpenAI integration:** CrewAI uses `azure-ai-inference` as its native Azure provider. We set `AZURE_API_KEY`, `AZURE_API_BASE`, and `AZURE_API_VERSION` as environment variables:

```python
self.model = LLM(model=f"azure/{deployment}")
```

---

### 3. Google ADK (Logical Reasoning Agent)

**What is Google ADK?** The Agent Development Kit from Google — a framework for building AI agents with built-in support for Google's Gemini models and third-party LLMs via LiteLLM.

**Key components we use:**

| Component                      | Purpose                                              |
| ------------------------------ | ---------------------------------------------------- |
| `LlmAgent`                     | Core agent class with instructions, tools, and model |
| `LiteLlm`                      | Adapter to use any LiteLLM-supported model           |
| `InMemoryRunner`               | Runs agents with in-memory session management        |
| `types.Content` / `types.Part` | Message structure for agent communication            |

**Our logical tools:**

```python
def logical_and(a: bool, b: bool) -> bool:
    """Perform logical AND. Returns True only if both are True."""
    return a and b

def logical_implies(p: bool, q: bool) -> bool:
    """Perform logical implication (p → q).
    Returns False only if p is True and q is False."""
    return (not p) or q

def evaluate_expression(expression: str) -> str:
    """Evaluate a logical expression string safely."""
    allowed_names = {"True": True, "False": False}
    result = eval(expression, {"__builtins__": {}}, allowed_names)
    return str(bool(result))

# Also: logical_or, logical_not, logical_xor, logical_biconditional
```

Notice ADK tools are **plain Python functions** — no decorator needed. ADK inspects the function signature and docstring automatically.

**Agent configuration:**

```python
self.agent = LlmAgent(
    name="logical_reasoning_agent",
    model=LiteLlm(model=f"azure/{deployment}"),
    instruction="You are a logical reasoning expert...",
    description="Agent for logical reasoning and boolean operations",
    tools=[
        logical_and, logical_or, logical_not,
        logical_xor, logical_implies,
        logical_biconditional, evaluate_expression,
    ],
)

self.runner = InMemoryRunner(
    agent=self.agent,
    app_name="logical_reasoning_app",
)
self.runner.auto_create_session = True
```

**Session management:** ADK's `InMemoryRunner` manages sessions internally. We set `auto_create_session = True` so it creates new sessions on the fly for incoming A2A requests (which carry unique request IDs).

**Invocation pattern:**

```python
async def invoke(self, query: str, session_id: str = "default") -> str:
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=query)],
    )

    final_response = ""
    async for event in self.runner.run_async(
        user_id="a2a_user",
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        final_response += part.text

    return final_response
```

ADK uses an **async event stream** — you iterate over events and collect the final response. This supports streaming use cases too.

---

### 4. LangGraph (Orchestrator)

**What is LangGraph?** A library from LangChain for building stateful, graph-based agent workflows. Unlike simple chains, LangGraph lets you define nodes (steps) and edges (transitions) with conditional routing.

**Key concepts:**

| Concept        | Purpose                                        |
| -------------- | ---------------------------------------------- |
| `StateGraph`   | Defines the workflow graph with typed state    |
| Nodes          | Async functions that transform state           |
| Edges          | Connections between nodes (can be conditional) |
| `add_messages` | LangGraph's message accumulator reducer        |

**Our graph structure:**

```python
class OrchestratorState(TypedDict):
    messages: Annotated[list, add_messages]
    classification: str
    agent_response: str
    agent_used: str

graph = StateGraph(OrchestratorState)

# Nodes
graph.add_node("classify_query", classify_query)
graph.add_node("call_arithmetic_agent", call_arithmetic_agent)
graph.add_node("call_logic_agent", call_logic_agent)
graph.add_node("call_both_agents", call_both_agents)
graph.add_node("handle_general", handle_general)

# Edges
graph.add_edge(START, "classify_query")
graph.add_conditional_edges("classify_query", route_query, {
    "call_arithmetic_agent": "call_arithmetic_agent",
    "call_logic_agent": "call_logic_agent",
    "call_both_agents": "call_both_agents",
    "handle_general": "handle_general",
})
graph.add_edge("call_arithmetic_agent", END)
graph.add_edge("call_logic_agent", END)
graph.add_edge("call_both_agents", END)
graph.add_edge("handle_general", END)
```

**Visually:**

```
START → classify_query → route:
    "arithmetic" → call_arithmetic_agent → END
    "logical"    → call_logic_agent      → END
    "both"       → call_both_agents      → END
    "general"    → handle_general         → END
```

**The classifier node** uses Azure OpenAI to categorize user queries:

```python
async def classify_query(state: OrchestratorState) -> dict:
    classification_prompt = SystemMessage(content=(
        "Classify the query into: 'arithmetic', 'logical', 'both', or 'general'. "
        "Respond with ONLY the category name."
    ))
    response = await llm.ainvoke([
        classification_prompt,
        HumanMessage(content=user_input)
    ])
    classification = response.content.strip().lower()
    return {"classification": classification}
```

**The routing function** maps classifications to node names:

```python
def route_query(state: OrchestratorState) -> str:
    classification = state.get("classification", "general")
    if classification == "arithmetic":
        return "call_arithmetic_agent"
    elif classification == "logical":
        return "call_logic_agent"
    elif classification == "both":
        return "call_both_agents"
    else:
        return "handle_general"
```

**Parallel fan-out** for queries needing both agents:

```python
async def call_both_agents(state: OrchestratorState) -> dict:
    # Fan-out: call both agents simultaneously
    arithmetic_response, logic_response = await asyncio.gather(
        send_task_to_agent(ARITHMETIC_AGENT_URL, user_input),
        send_task_to_agent(LOGIC_AGENT_URL, user_input),
        return_exceptions=True
    )

    # Combine: use LLM to synthesize results
    combined = await llm.ainvoke([
        SystemMessage(content="Combine these two agent responses..."),
        HumanMessage(content=f"Arithmetic: {arithmetic_response}\n"
                             f"Logic: {logic_response}")
    ])
    return {"agent_response": combined.content, "agent_used": "Both"}
```

---

### 5. Azure OpenAI

All three agents use **Azure OpenAI** as their LLM, but each connects differently:

| Agent            | Integration Method   | How It Connects                                                  |
| ---------------- | -------------------- | ---------------------------------------------------------------- |
| **Orchestrator** | `langchain-openai`   | `AzureChatOpenAI(azure_deployment=..., azure_endpoint=...)`      |
| **Arithmetic**   | `azure-ai-inference` | `LLM(model="azure/gpt-4o")` via CrewAI's native provider         |
| **Logic**        | `litellm`            | `LiteLlm(model="azure/gpt-4o")` via Google ADK's LiteLLM adapter |

All read from the same environment variables:

```bash
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-08-01-preview
```

**Why different integrations?** Each framework has its own preferred way to connect to LLMs. The orchestrator uses LangChain's native Azure integration (most ergonomic). CrewAI has a built-in Azure provider backed by `azure-ai-inference`. ADK uses LiteLLM as its model adapter layer.

---

### 6. Streamlit UI

The UI provides a chat interface with rich visual feedback:

**Key features:**

- **Real-time agent status** — sidebar shows which agents are connected via Agent Card discovery
- **Routing badges** — each response shows which agent handled it with a color-coded badge
- **Example prompts** — clickable examples for each agent type
- **Architecture diagram** — visual representation of the system

**Badge system:**

```python
def get_badge_class(classification: str) -> str:
    if classification == "arithmetic":
        return "badge-arithmetic"    # amber gradient
    elif classification == "logical":
        return "badge-logical"       # purple gradient
    elif classification == "both":
        return "badge-both"          # amber→purple gradient
    return "badge-general"           # gray gradient
```

---

## 📁 Project Structure

```
A2A/
├── agents/
│   ├── __init__.py
│   ├── arithmetic/
│   │   ├── __init__.py
│   │   ├── __main__.py          # A2A server entry point (port 10001)
│   │   ├── agent.py             # CrewAI agent + arithmetic tools
│   │   └── agent_executor.py    # A2A ↔ CrewAI bridge
│   ├── logical_reasoning/
│   │   ├── __init__.py
│   │   ├── __main__.py          # A2A server entry point (port 10002)
│   │   ├── agent.py             # ADK agent + logic tools
│   │   └── agent_executor.py    # A2A ↔ ADK bridge
│   └── orchestrator/
│       ├── __init__.py
│       ├── orchestrator.py      # LangGraph state machine
│       └── a2a_tools.py         # A2A client (discover + send)
├── tests/
│   ├── test_arithmetic_tools.py # Unit tests for all 6 math tools
│   ├── test_logical_tools.py    # Truth-table tests for all 7 logic tools
│   ├── test_orchestrator.py     # Routing, classification, graph tests
│   └── test_a2a_integration.py  # Agent Card, error handling, parsing tests
├── ui/
│   └── app.py                   # Streamlit chat interface
├── .env                         # Azure OpenAI credentials
├── .env.example                 # Template for credentials
├── environment.yml              # Conda environment (all dependencies)
├── run_all.sh                   # Starts everything with one command
├── README.md                    # Project overview + setup
└── GUIDE.md                     # This file!
```

---

## 🧪 Testing

The project includes a comprehensive test suite with **97 tests** that run entirely offline — no LLM API calls or running servers needed.

### Running Tests

```bash
# Run the full suite
conda run -n a2a-agents python -m pytest tests/ -v

# Run a specific file
conda run -n a2a-agents python -m pytest tests/test_arithmetic_tools.py -v

# Run with short summary
conda run -n a2a-agents python -m pytest tests/ --tb=short
```

### Test Structure

| File                       | Tests | Coverage                                                                                                                      |
| -------------------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------- |
| `test_arithmetic_tools.py` | 32    | All 6 math tools — positives, negatives, zero, floats, division-by-zero, large numbers                                        |
| `test_logical_tools.py`    | 33    | Complete truth tables for AND, OR, NOT, XOR, IMPLIES, BICONDITIONAL + expression evaluator with De Morgan's law               |
| `test_orchestrator.py`     | 17    | `route_query` for all 5 paths, `classify_query` with mocked LLM (normalization, whitespace), graph node verification          |
| `test_a2a_integration.py`  | 15    | Agent Card construction, `InternalError`/`ServerError` types, `_extract_part_text` from RootModel wrappers, import validation |

### Testing Strategy

**Tool tests** call the Python functions directly (e.g., `add.func(3, 5)` for CrewAI tools) — no LLM involved:

```python
def test_add_positive_numbers(self):
    assert add.func(3, 5) == 8

def test_divide_by_zero_raises(self):
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide.func(10, 0)
```

**Logic tests** verify complete truth tables:

```python
def test_true_implies_false(self):
    # Only case where implication is False
    assert logical_implies(True, False) is False

def test_de_morgans_law(self):
    # NOT (A AND B) == (NOT A) OR (NOT B)
    result1 = evaluate_expression("not (True and False)")
    result2 = evaluate_expression("(not True) or (not False)")
    assert result1 == result2
```

**Orchestrator tests** use `unittest.mock` to mock the Azure OpenAI LLM:

```python
@pytest.mark.asyncio
async def test_classify_arithmetic(self):
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = AIMessage(content="arithmetic")

    state = {
        "messages": [HumanMessage(content="What is 2 + 2?")],
        "classification": "", "agent_response": "", "agent_used": "",
    }

    with patch("agents.orchestrator.orchestrator.get_llm", return_value=mock_llm):
        result = await classify_query(state)

    assert result["classification"] == "arithmetic"
```

**A2A integration tests** validate protocol types without network calls:

```python
def test_server_error_wraps_internal_error(self):
    internal = InternalError(message="test error")
    server_err = ServerError(error=internal)
    assert server_err.error.message == "test error"
```

---

## 🚀 Setup & Running

### Prerequisites

- **Conda** (Miniconda or Anaconda)
- **Azure OpenAI** resource with a deployed model (e.g., `gpt-4o`)

### Step 1: Create the environment

```bash
conda env create -f environment.yml
conda activate a2a-agents
```

### Step 2: Configure credentials

```bash
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

### Step 3: Run everything

```bash
chmod +x run_all.sh
./run_all.sh
```

This starts:

- 🧮 Arithmetic Agent on `http://localhost:10001`
- 🧠 Logic Agent on `http://localhost:10002`
- 🖥️ Streamlit UI on `http://localhost:8501`

### Step 4: Open the UI

Navigate to `http://localhost:8501` and start asking questions!

---

## 🧪 Example Queries

### Arithmetic (routes to CrewAI agent)

- _"What is 25 _ 4 + 10?"\*
- _"Calculate 2 to the power of 10"_
- _"What is 144 / 12?"_

### Logical Reasoning (routes to ADK agent)

- _"What is True AND False?"_
- _"If P→Q and P is True, what is Q?"_
- _"Evaluate: NOT (True OR False)"_

### Both Agents (parallel fan-out)

- _"What is 5 _ 3, and is the result greater than 10?"\*
- _"Calculate 2^3 and check if the result equals 8 using logical AND with True"_

### General (handled by orchestrator directly)

- _"Tell me a joke"_
- _"What's the weather like?"_

---

## 🧩 Key Design Decisions

### 1. Why A2A over direct function calls?

Direct calls would be simpler, but A2A gives us:

- **Framework independence** — swap agents without changing the orchestrator
- **Network distribution** — agents can run on different machines
- **Discovery** — agents self-describe via Agent Cards
- **Standardization** — any A2A-compatible client can call any agent

### 2. Why three different frameworks?

This is deliberate — it demonstrates that A2A works across heterogeneous systems. In production, you'd pick the best framework for each use case:

- **CrewAI** excels at role-based agents with tool use
- **Google ADK** integrates deeply with Google's ecosystem
- **LangGraph** is ideal for complex, stateful workflows

### 3. Why LangGraph for orchestration?

LangGraph's `StateGraph` is perfect for routing because:

- Conditional edges model the classification → routing pattern naturally
- State management tracks messages, classification, and responses
- It's async-native, enabling parallel agent calls

### 4. Why Azure OpenAI?

Any OpenAI-compatible model would work. Azure was chosen for reliability and enterprise features. The architecture is LLM-agnostic — swap to regular OpenAI, Anthropic, or local models by changing the LLM configuration.

---

## 🔍 Debugging Tips

### Check if agents are running

```bash
# Should return the Agent Card JSON
curl http://localhost:10001/.well-known/agent-card.json
curl http://localhost:10002/.well-known/agent-card.json
```

### Test an agent directly

```python
import asyncio
from agents.orchestrator.a2a_tools import send_task_to_agent

result = asyncio.run(
    send_task_to_agent("http://localhost:10001", "What is 2 + 2?")
)
print(result)
```

### Common issues

| Issue                       | Cause                                         | Fix                                                                   |
| --------------------------- | --------------------------------------------- | --------------------------------------------------------------------- |
| `Session not found`         | ADK runner doesn't have `auto_create_session` | Set `runner.auto_create_session = True`                               |
| `is_litellm` in API request | CrewAI leaks parameter to Azure               | Use native Azure provider, install `azure-ai-inference`               |
| Stale behavior after edits  | Python bytecode cache                         | Delete `__pycache__` dirs or use `run_all.sh` (does it automatically) |
| `Part.from_text()` error    | ADK API change                                | Use keyword arg: `Part.from_text(text=query)`                         |

---

## 📚 Further Reading

- **A2A Protocol:** [google.github.io/A2A](https://google.github.io/A2A)
- **CrewAI Docs:** [docs.crewai.com](https://docs.crewai.com)
- **Google ADK:** [google.github.io/adk-docs](https://google.github.io/adk-docs)
- **LangGraph:** [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph)
- **a2a-sdk (Python):** [pypi.org/project/a2a-sdk](https://pypi.org/project/a2a-sdk)

---

## 🎯 What's Next?

Ideas for extending this project:

- **Add streaming** — enable real-time token streaming from agents
- **Add more agents** — code generation, web search, database queries
- **Persistent sessions** — replace `InMemoryTaskStore` with a database
- **Authentication** — add API key / OAuth to agent endpoints
- **Docker deployment** — containerize each agent as a microservice
- **Observability** — add OpenTelemetry tracing across agent calls

---

_Built with ❤️ using the A2A Protocol, LangGraph, CrewAI, and Google ADK_
