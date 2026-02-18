"""A2A AgentExecutor for the Arithmetic Agent."""

import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import InternalError, UnsupportedOperationError
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError

from agents.arithmetic.agent import ArithmeticAgent

logger = logging.getLogger(__name__)


class ArithmeticAgentExecutor(AgentExecutor):
    """A2A executor that wraps the CrewAI ArithmeticAgent."""

    def __init__(self) -> None:
        self.agent = ArithmeticAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        query = context.get_user_input()
        logger.info(f"ArithmeticAgentExecutor received: {query}")

        try:
            result = self.agent.invoke(query)
        except Exception as e:
            logger.error(f"Error invoking arithmetic agent: {e}")
            raise ServerError(error=InternalError(message=str(e))) from e

        await event_queue.enqueue_event(new_agent_text_message(result))

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        raise ServerError(error=UnsupportedOperationError())
