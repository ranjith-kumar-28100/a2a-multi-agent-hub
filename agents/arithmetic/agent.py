"""Arithmetic Agent built with CrewAI.

Provides arithmetic calculation tools (add, subtract, multiply, divide,
modulo, power) exposed via a CrewAI Agent and Crew.
Uses Azure OpenAI via LiteLLM as the LLM backend.
"""

import logging
import os

from crewai import LLM, Agent, Crew, Task
from crewai.process import Process
from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ─── Arithmetic Tools ───────────────────────────────────────────────


@tool("AdditionTool")
def add(a: float, b: float) -> float:
    """Add two numbers together. Returns a + b."""
    return a + b


@tool("SubtractionTool")
def subtract(a: float, b: float) -> float:
    """Subtract b from a. Returns a - b."""
    return a - b


@tool("MultiplicationTool")
def multiply(a: float, b: float) -> float:
    """Multiply two numbers. Returns a * b."""
    return a * b


@tool("DivisionTool")
def divide(a: float, b: float) -> float:
    """Divide a by b. Returns a / b. Raises error if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@tool("ModuloTool")
def modulo(a: float, b: float) -> float:
    """Calculate the remainder of a divided by b. Returns a % b."""
    if b == 0:
        raise ValueError("Cannot modulo by zero")
    return a % b


@tool("PowerTool")
def power(a: float, b: float) -> float:
    """Raise a to the power of b. Returns a ** b."""
    return a ** b


# ─── CrewAI Agent ────────────────────────────────────────────────────


class ArithmeticAgent:
    """CrewAI-based agent that performs arithmetic calculations."""

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        api_base = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

        # Set env vars that CrewAI's native Azure provider reads
        os.environ["AZURE_API_KEY"] = api_key
        os.environ["AZURE_API_BASE"] = api_base
        os.environ["AZURE_API_VERSION"] = api_version

        # Use CrewAI's native Azure provider (backed by azure-ai-inference)
        self.model = LLM(
            model=f"azure/{deployment}",
        )

        self.calculator_agent = Agent(
            role="Arithmetic Calculator",
            goal=(
                "Perform accurate arithmetic calculations based on the user's "
                "request. Break down complex expressions into individual operations "
                "and use the appropriate tool for each step. Always return the "
                "final numerical result."
            ),
            backstory=(
                "You are an expert mathematician who specializes in arithmetic "
                "calculations. You have access to tools for addition, subtraction, "
                "multiplication, division, modulo, and exponentiation. You always "
                "show your work step by step and provide accurate results."
            ),
            verbose=False,
            allow_delegation=False,
            tools=[add, subtract, multiply, divide, modulo, power],
            llm=self.model,
        )

        self.calculation_task = Task(
            description=(
                "Perform the following arithmetic calculation: '{user_prompt}'. "
                "Break down the expression into individual operations if needed. "
                "Use the appropriate arithmetic tools (AdditionTool, SubtractionTool, "
                "MultiplicationTool, DivisionTool, ModuloTool, PowerTool) to compute "
                "the result step by step. Return the final numerical answer with "
                "a brief explanation of the steps taken."
            ),
            expected_output=(
                "The numerical result of the calculation with a brief explanation "
                "of the steps taken to arrive at the answer."
            ),
            agent=self.calculator_agent,
        )

        self.crew = Crew(
            agents=[self.calculator_agent],
            tasks=[self.calculation_task],
            process=Process.sequential,
            verbose=False,
        )

    def invoke(self, query: str) -> str:
        """Run the arithmetic crew and return the result."""
        inputs = {"user_prompt": query}
        logger.info(f"Arithmetic Agent invoked with: {query}")
        response = self.crew.kickoff(inputs)
        logger.info(f"Arithmetic Agent result: {response.raw}")
        return response.raw
