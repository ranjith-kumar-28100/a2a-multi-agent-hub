"""Logical Reasoning Agent built with Google ADK.

Provides logical reasoning tools (AND, OR, NOT, XOR, IMPLIES,
evaluate_expression) exposed via a Google ADK LlmAgent.
Uses Azure OpenAI via LiteLLM as the LLM backend.
"""

import asyncio
import logging
import os

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

load_dotenv()

logger = logging.getLogger(__name__)


# ─── Logical Reasoning Tools ────────────────────────────────────────


def logical_and(a: bool, b: bool) -> bool:
    """Perform logical AND operation. Returns True only if both a AND b are True."""
    return a and b


def logical_or(a: bool, b: bool) -> bool:
    """Perform logical OR operation. Returns True if either a OR b is True."""
    return a or b


def logical_not(a: bool) -> bool:
    """Perform logical NOT operation. Returns the opposite of a."""
    return not a


def logical_xor(a: bool, b: bool) -> bool:
    """Perform logical XOR (exclusive or) operation. Returns True if exactly one of a, b is True."""
    return a ^ b


def logical_implies(p: bool, q: bool) -> bool:
    """Perform logical implication (p → q). Returns False only if p is True and q is False."""
    return (not p) or q


def logical_biconditional(p: bool, q: bool) -> bool:
    """Perform logical biconditional (p ↔ q). Returns True if both have the same truth value."""
    return p == q


def evaluate_expression(expression: str) -> str:
    """Evaluate a logical expression string containing True, False, and, or, not, and parentheses.

    Args:
        expression: A string logical expression like 'True and (False or True)'

    Returns:
        The string result of evaluating the expression ('True' or 'False')
    """
    allowed_names = {"True": True, "False": False}
    allowed_ops = {"and", "or", "not"}

    try:
        # Validate expression contains only safe tokens
        tokens = expression.replace("(", " ").replace(")", " ").split()
        for token in tokens:
            if token not in allowed_names and token not in allowed_ops:
                try:
                    float(token)  # allow numbers
                except ValueError:
                    return f"Error: Invalid token '{token}'. Only True, False, and, or, not, and parentheses are allowed."

        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(bool(result))
    except Exception as e:
        return f"Error evaluating expression: {e}"


# ─── Google ADK Agent ────────────────────────────────────────────────


class LogicalReasoningAgent:
    """Google ADK-based agent that performs logical reasoning."""

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        api_base = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

        # Set env vars for LiteLLM Azure OpenAI
        os.environ["AZURE_API_KEY"] = api_key
        os.environ["AZURE_API_BASE"] = api_base
        os.environ["AZURE_API_VERSION"] = api_version

        self.agent = LlmAgent(
            name="logical_reasoning_agent",
            model=LiteLlm(model=f"azure/{deployment}"),
            instruction=(
                "You are a logical reasoning expert. You help users evaluate "
                "logical expressions and reason about boolean logic. You have "
                "tools for AND, OR, NOT, XOR, IMPLIES, BICONDITIONAL operations "
                "and can evaluate logical expressions.\n\n"
                "When a user asks a logical question:\n"
                "1. Identify the logical operation(s) needed\n"
                "2. Use the appropriate tool(s) to compute the result\n"
                "3. Explain the reasoning step by step\n"
                "4. Provide the final answer clearly\n\n"
                "For complex reasoning, break it down into steps using "
                "individual logical operations."
            ),
            description="Agent for logical reasoning and boolean operations",
            tools=[
                logical_and,
                logical_or,
                logical_not,
                logical_xor,
                logical_implies,
                logical_biconditional,
                evaluate_expression,
            ],
        )

        # InMemoryRunner manages sessions internally
        self.runner = InMemoryRunner(
            agent=self.agent,
            app_name="logical_reasoning_app",
        )
        # Enable auto session creation for incoming A2A requests
        self.runner.auto_create_session = True

    async def invoke(self, query: str, session_id: str = "default") -> str:
        """Run the ADK agent and return the result."""
        logger.info(f"Logical Reasoning Agent invoked with: {query}")

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

        logger.info(f"Logical Reasoning Agent result: {final_response}")
        return final_response if final_response else "No response generated."

