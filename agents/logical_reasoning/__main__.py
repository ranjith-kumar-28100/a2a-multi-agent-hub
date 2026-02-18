"""Entry point for the Logical Reasoning Agent A2A server."""

import logging

import click
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from dotenv import load_dotenv

from agents.logical_reasoning.agent_executor import LogicalReasoningAgentExecutor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", default="localhost", help="Host to bind to")
@click.option("--port", default=10002, help="Port to bind to")
def main(host: str, port: int):
    """Start the Logical Reasoning Agent A2A server (Google ADK)."""

    skill = AgentSkill(
        id="logical_reasoning",
        name="Logical Reasoning Engine",
        description=(
            "Performs logical reasoning operations including AND, OR, NOT, "
            "XOR, IMPLIES, and BICONDITIONAL. Can evaluate complex logical "
            "expressions and explain reasoning step by step."
        ),
        tags=["and", "or", "not", "xor", "implies", "logic", "boolean", "reasoning"],
        examples=[
            "What is True AND False?",
            "Evaluate: True OR (False AND True)",
            "If P implies Q, and P is True, what is Q?",
            "What is the XOR of True and True?",
            "Is NOT (True AND False) equal to (NOT True) OR (NOT False)?",
        ],
    )

    agent_card = AgentCard(
        name="Logical Reasoning Agent",
        description=(
            "A Google ADK-powered agent that performs logical reasoning. "
            "Supports AND, OR, NOT, XOR, IMPLIES, and BICONDITIONAL operations. "
            "Can evaluate complex boolean expressions and explain reasoning."
        ),
        url=f"http://{host}:{port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=LogicalReasoningAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    logger.info(f"🧠 Logical Reasoning Agent (ADK) starting on http://{host}:{port}")
    uvicorn.run(server.build(), host=host, port=port)


if __name__ == "__main__":
    main()
