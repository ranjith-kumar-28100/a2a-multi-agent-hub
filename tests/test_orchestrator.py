"""Tests for the Orchestrator's routing and classification logic.

Uses mocked LLM responses to test the routing logic without
making actual API calls to Azure OpenAI.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from agents.orchestrator.orchestrator import (
    OrchestratorState,
    classify_query,
    route_query,
    build_orchestrator_graph,
)


# ─── Route Query (Deterministic) ────────────────────────────────────


class TestRouteQuery:
    """Tests for the route_query function — no LLM needed."""

    def test_route_arithmetic(self):
        state: OrchestratorState = {
            "messages": [],
            "classification": "arithmetic",
            "agent_response": "",
            "agent_used": "",
        }
        assert route_query(state) == "call_arithmetic_agent"

    def test_route_logical(self):
        state: OrchestratorState = {
            "messages": [],
            "classification": "logical",
            "agent_response": "",
            "agent_used": "",
        }
        assert route_query(state) == "call_logic_agent"

    def test_route_both(self):
        state: OrchestratorState = {
            "messages": [],
            "classification": "both",
            "agent_response": "",
            "agent_used": "",
        }
        assert route_query(state) == "call_both_agents"

    def test_route_general(self):
        state: OrchestratorState = {
            "messages": [],
            "classification": "general",
            "agent_response": "",
            "agent_used": "",
        }
        assert route_query(state) == "handle_general"

    def test_route_unknown_defaults_to_general(self):
        state: OrchestratorState = {
            "messages": [],
            "classification": "unknown_category",
            "agent_response": "",
            "agent_used": "",
        }
        assert route_query(state) == "handle_general"

    def test_route_empty_defaults_to_general(self):
        state: OrchestratorState = {
            "messages": [],
            "classification": "",
            "agent_response": "",
            "agent_used": "",
        }
        assert route_query(state) == "handle_general"

    def test_route_missing_classification_defaults_to_general(self):
        # Simulate missing key with dict.get() fallback
        state = {
            "messages": [],
            "agent_response": "",
            "agent_used": "",
        }
        assert route_query(state) == "handle_general"


# ─── Classify Query (Mocked LLM) ────────────────────────────────────


class TestClassifyQuery:
    """Tests for classify_query with mocked LLM."""

    @pytest.mark.asyncio
    async def test_classify_arithmetic(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="arithmetic")

        state: OrchestratorState = {
            "messages": [HumanMessage(content="What is 2 + 2?")],
            "classification": "",
            "agent_response": "",
            "agent_used": "",
        }

        with patch("agents.orchestrator.orchestrator.get_llm", return_value=mock_llm):
            result = await classify_query(state)

        assert result["classification"] == "arithmetic"

    @pytest.mark.asyncio
    async def test_classify_logical(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="logical")

        state: OrchestratorState = {
            "messages": [HumanMessage(content="What is True AND False?")],
            "classification": "",
            "agent_response": "",
            "agent_used": "",
        }

        with patch("agents.orchestrator.orchestrator.get_llm", return_value=mock_llm):
            result = await classify_query(state)

        assert result["classification"] == "logical"

    @pytest.mark.asyncio
    async def test_classify_both(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="both")

        state: OrchestratorState = {
            "messages": [HumanMessage(content="Calculate 5*3 and check if result > 10")],
            "classification": "",
            "agent_response": "",
            "agent_used": "",
        }

        with patch("agents.orchestrator.orchestrator.get_llm", return_value=mock_llm):
            result = await classify_query(state)

        assert result["classification"] == "both"

    @pytest.mark.asyncio
    async def test_classify_general(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="general")

        state: OrchestratorState = {
            "messages": [HumanMessage(content="Tell me a joke")],
            "classification": "",
            "agent_response": "",
            "agent_used": "",
        }

        with patch("agents.orchestrator.orchestrator.get_llm", return_value=mock_llm):
            result = await classify_query(state)

        assert result["classification"] == "general"

    @pytest.mark.asyncio
    async def test_classify_normalizes_math_to_arithmetic(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="math")

        state: OrchestratorState = {
            "messages": [HumanMessage(content="some query")],
            "classification": "",
            "agent_response": "",
            "agent_used": "",
        }

        with patch("agents.orchestrator.orchestrator.get_llm", return_value=mock_llm):
            result = await classify_query(state)

        assert result["classification"] == "arithmetic"

    @pytest.mark.asyncio
    async def test_classify_normalizes_logic_to_logical(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="logic")

        state: OrchestratorState = {
            "messages": [HumanMessage(content="some query")],
            "classification": "",
            "agent_response": "",
            "agent_used": "",
        }

        with patch("agents.orchestrator.orchestrator.get_llm", return_value=mock_llm):
            result = await classify_query(state)

        assert result["classification"] == "logical"

    @pytest.mark.asyncio
    async def test_classify_strips_whitespace(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="  Arithmetic  \n")

        state: OrchestratorState = {
            "messages": [HumanMessage(content="What is 5+5?")],
            "classification": "",
            "agent_response": "",
            "agent_used": "",
        }

        with patch("agents.orchestrator.orchestrator.get_llm", return_value=mock_llm):
            result = await classify_query(state)

        assert result["classification"] == "arithmetic"


# ─── Graph Construction ──────────────────────────────────────────────


class TestGraphConstruction:
    """Tests that the graph builds correctly."""

    def test_graph_builds_without_error(self):
        graph = build_orchestrator_graph()
        assert graph is not None

    def test_graph_has_all_nodes(self):
        graph = build_orchestrator_graph()
        node_names = set(graph.nodes.keys())
        expected = {
            "__start__",
            "classify_query",
            "call_arithmetic_agent",
            "call_logic_agent",
            "call_both_agents",
            "handle_general",
        }
        assert expected.issubset(node_names)
