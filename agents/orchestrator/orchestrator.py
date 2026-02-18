"""LangGraph-based Orchestrator Agent.

Routes user queries to the appropriate remote A2A agent (Arithmetic or
Logical Reasoning) based on query classification using Azure OpenAI.
Supports parallel routing when a query requires both agents.
"""

import asyncio
import logging
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agents.orchestrator.a2a_tools import discover_agent, send_task_to_agent

load_dotenv()

logger = logging.getLogger(__name__)

# Agent URLs
ARITHMETIC_AGENT_URL = os.getenv("ARITHMETIC_AGENT_URL", "http://localhost:10001")
LOGIC_AGENT_URL = os.getenv("LOGIC_AGENT_URL", "http://localhost:10002")


# ─── State Definition ───────────────────────────────────────────────


class OrchestratorState(TypedDict):
    """State for the orchestrator graph."""
    messages: Annotated[list, add_messages]
    classification: str
    agent_response: str
    agent_used: str


# ─── LLM Setup ──────────────────────────────────────────────────────


def get_llm() -> AzureChatOpenAI:
    """Create Azure OpenAI LLM instance."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        temperature=0,
    )


# ─── Graph Nodes ─────────────────────────────────────────────────────


async def classify_query(state: OrchestratorState) -> dict:
    """Classify the user's query as arithmetic, logical, both, or general."""
    llm = get_llm()

    # Get the last user message
    last_message = state["messages"][-1]
    user_input = last_message.content if hasattr(last_message, "content") else str(last_message)

    classification_prompt = SystemMessage(content=(
        "You are a query classifier. Classify the following user query into EXACTLY "
        "one of these categories:\n\n"
        "- 'arithmetic': For queries that ONLY need mathematical calculations, "
        "  number operations, math expressions (addition, subtraction, multiplication, "
        "  division, modulo, exponentiation, percentages, etc.)\n"
        "- 'logical': For queries that ONLY need boolean logic, logical operations "
        "  (AND, OR, NOT, XOR, IMPLIES), truth tables, logical reasoning, "
        "  propositional logic, logical deductions, syllogisms\n"
        "- 'both': For queries that need BOTH arithmetic calculations AND logical "
        "  reasoning. Examples: 'Calculate 5*3 and check if the result > 10', "
        "  'If 2+2=4, is True AND False equal to True?', "
        "  'What is 8/2 and is the result > 3 OR < 1?'\n"
        "- 'general': For anything that doesn't fit the above categories\n\n"
        "Respond with ONLY the category name, nothing else. Just one word."
    ))

    response = await llm.ainvoke([classification_prompt, HumanMessage(content=user_input)])
    classification = response.content.strip().lower()

    # Normalize
    if "both" in classification:
        classification = "both"
    elif "arithmetic" in classification or "math" in classification:
        classification = "arithmetic"
    elif "logical" in classification or "logic" in classification:
        classification = "logical"
    else:
        classification = "general"

    logger.info(f"Query classified as: {classification}")
    return {"classification": classification}


async def call_arithmetic_agent(state: OrchestratorState) -> dict:
    """Send the query to the Arithmetic Agent via A2A."""
    last_message = state["messages"][-1]
    user_input = last_message.content if hasattr(last_message, "content") else str(last_message)

    logger.info(f"Routing to Arithmetic Agent at {ARITHMETIC_AGENT_URL}")
    response = await send_task_to_agent(ARITHMETIC_AGENT_URL, user_input)

    return {
        "agent_response": response,
        "agent_used": "Arithmetic Agent (CrewAI)",
        "messages": [AIMessage(content=response)],
    }


async def call_logic_agent(state: OrchestratorState) -> dict:
    """Send the query to the Logical Reasoning Agent via A2A."""
    last_message = state["messages"][-1]
    user_input = last_message.content if hasattr(last_message, "content") else str(last_message)

    logger.info(f"Routing to Logical Reasoning Agent at {LOGIC_AGENT_URL}")
    response = await send_task_to_agent(LOGIC_AGENT_URL, user_input)

    return {
        "agent_response": response,
        "agent_used": "Logical Reasoning Agent (ADK)",
        "messages": [AIMessage(content=response)],
    }


async def call_both_agents(state: OrchestratorState) -> dict:
    """Send the query to BOTH agents in parallel and combine results."""
    last_message = state["messages"][-1]
    user_input = last_message.content if hasattr(last_message, "content") else str(last_message)

    logger.info("Routing to BOTH Arithmetic and Logical Reasoning agents")

    # Fan-out: call both agents in parallel
    arithmetic_task = send_task_to_agent(ARITHMETIC_AGENT_URL, user_input)
    logic_task = send_task_to_agent(LOGIC_AGENT_URL, user_input)

    arithmetic_response, logic_response = await asyncio.gather(
        arithmetic_task, logic_task, return_exceptions=True
    )

    # Handle exceptions from either agent
    if isinstance(arithmetic_response, Exception):
        arithmetic_response = f"Arithmetic Agent error: {arithmetic_response}"
    if isinstance(logic_response, Exception):
        logic_response = f"Logical Reasoning Agent error: {logic_response}"

    # Combine: use LLM to synthesize both results into a coherent answer
    llm = get_llm()

    combine_prompt = SystemMessage(content=(
        "You received answers from two specialized agents for the user's query. "
        "Combine their responses into a single, coherent, well-structured answer.\n\n"
        "Guidelines:\n"
        "- Integrate both results naturally, don't just paste them together\n"
        "- If one agent's answer builds on the other's, show that connection\n"
        "- Keep the final answer clear and concise\n"
        "- Credit which part came from which domain (arithmetic vs logic)"
    ))

    user_context = HumanMessage(content=(
        f"Original question: {user_input}\n\n"
        f"--- Arithmetic Agent Response ---\n{arithmetic_response}\n\n"
        f"--- Logical Reasoning Agent Response ---\n{logic_response}"
    ))

    combined = await llm.ainvoke([combine_prompt, user_context])

    return {
        "agent_response": combined.content,
        "agent_used": "Both (Arithmetic + Logical Reasoning)",
        "messages": [AIMessage(content=combined.content)],
    }


async def handle_general(state: OrchestratorState) -> dict:
    """Handle general queries directly with the LLM."""
    llm = get_llm()
    last_message = state["messages"][-1]
    user_input = last_message.content if hasattr(last_message, "content") else str(last_message)

    system_msg = SystemMessage(content=(
        "You are a helpful assistant. The user's query doesn't fit into "
        "arithmetic calculations or logical reasoning categories. "
        "Provide a helpful response and let them know you specialize in "
        "arithmetic and logical reasoning tasks. Suggest how they might "
        "rephrase their question if it could be related to these areas."
    ))

    response = await llm.ainvoke([system_msg, HumanMessage(content=user_input)])

    return {
        "agent_response": response.content,
        "agent_used": "Orchestrator (General)",
        "messages": [AIMessage(content=response.content)],
    }


def route_query(state: OrchestratorState) -> str:
    """Route based on classification."""
    classification = state.get("classification", "general")
    if classification == "arithmetic":
        return "call_arithmetic_agent"
    elif classification == "logical":
        return "call_logic_agent"
    elif classification == "both":
        return "call_both_agents"
    else:
        return "handle_general"


# ─── Build Graph ─────────────────────────────────────────────────────


def build_orchestrator_graph() -> StateGraph:
    """Build and compile the orchestrator LangGraph.

    Graph flow:
        START → classify_query → route:
            "arithmetic" → call_arithmetic_agent → END
            "logical"    → call_logic_agent      → END
            "both"       → call_both_agents      → END
            "general"    → handle_general         → END
    """
    graph = StateGraph(OrchestratorState)

    # Add nodes
    graph.add_node("classify_query", classify_query)
    graph.add_node("call_arithmetic_agent", call_arithmetic_agent)
    graph.add_node("call_logic_agent", call_logic_agent)
    graph.add_node("call_both_agents", call_both_agents)
    graph.add_node("handle_general", handle_general)

    # Add edges
    graph.add_edge(START, "classify_query")
    graph.add_conditional_edges(
        "classify_query",
        route_query,
        {
            "call_arithmetic_agent": "call_arithmetic_agent",
            "call_logic_agent": "call_logic_agent",
            "call_both_agents": "call_both_agents",
            "handle_general": "handle_general",
        },
    )
    graph.add_edge("call_arithmetic_agent", END)
    graph.add_edge("call_logic_agent", END)
    graph.add_edge("call_both_agents", END)
    graph.add_edge("handle_general", END)

    return graph.compile()


# ─── Main Interface ──────────────────────────────────────────────────


class Orchestrator:
    """High-level orchestrator interface."""

    def __init__(self):
        self.graph = build_orchestrator_graph()

    async def run(self, query: str) -> dict:
        """Run a query through the orchestrator.

        Returns:
            dict with 'response', 'agent_used', and 'classification' keys.
        """
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "classification": "",
            "agent_response": "",
            "agent_used": "",
        }

        result = await self.graph.ainvoke(initial_state)

        return {
            "response": result["agent_response"],
            "agent_used": result["agent_used"],
            "classification": result["classification"],
        }

    async def check_agent_status(self) -> dict:
        """Check which remote agents are available."""
        arithmetic_card = await discover_agent(ARITHMETIC_AGENT_URL)
        logic_card = await discover_agent(LOGIC_AGENT_URL)

        return {
            "arithmetic": {
                "connected": arithmetic_card is not None,
                "name": arithmetic_card.name if arithmetic_card else None,
                "url": ARITHMETIC_AGENT_URL,
            },
            "logical": {
                "connected": logic_card is not None,
                "name": logic_card.name if logic_card else None,
                "url": LOGIC_AGENT_URL,
            },
        }
