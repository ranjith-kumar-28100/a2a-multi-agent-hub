"""Entry point for the Arithmetic Agent A2A server."""

import logging

import click
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from dotenv import load_dotenv

from agents.arithmetic.agent_executor import ArithmeticAgentExecutor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", default="localhost", help="Host to bind to")
@click.option("--port", default=10001, help="Port to bind to")
def main(host: str, port: int):
    """Start the Arithmetic Agent A2A server (CrewAI)."""

    skill = AgentSkill(
        id="arithmetic_calculation",
        name="Arithmetic Calculator",
        description=(
            "Performs arithmetic calculations including addition, subtraction, "
            "multiplication, division, modulo, and exponentiation. Can handle "
            "complex multi-step arithmetic expressions."
        ),
        tags=["add", "subtract", "multiply", "divide", "modulo", "power", "math", "arithmetic"],
        examples=[
            "What is 25 * 4 + 10?",
            "Calculate 144 / 12",
            "What is 2 to the power of 10?",
            "What is 17 modulo 5?",
            "Compute (3 + 5) * (10 - 2)",
        ],
    )

    agent_card = AgentCard(
        name="Arithmetic Agent",
        description=(
            "A CrewAI-powered agent that performs arithmetic calculations. "
            "Supports addition, subtraction, multiplication, division, "
            "modulo, and exponentiation operations."
        ),
        url=f"http://{host}:{port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=ArithmeticAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    logger.info(f"🧮 Arithmetic Agent (CrewAI) starting on http://{host}:{port}")
    uvicorn.run(server.build(), host=host, port=port)


if __name__ == "__main__":
    main()
