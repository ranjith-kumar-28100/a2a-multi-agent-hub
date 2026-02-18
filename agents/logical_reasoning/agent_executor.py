"""A2A AgentExecutor for the Logical Reasoning Agent."""

import asyncio
import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import InternalError, UnsupportedOperationError
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError

from agents.logical_reasoning.agent import LogicalReasoningAgent

logger = logging.getLogger(__name__)


class LogicalReasoningAgentExecutor(AgentExecutor):
    """A2A executor that wraps the Google ADK LogicalReasoningAgent."""

    def __init__(self) -> None:
        self.agent = LogicalReasoningAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        query = context.get_user_input()
        logger.info(f"LogicalReasoningAgentExecutor received: {query}")

        try:
            result = await self.agent.invoke(query, session_id=context.context_id)
        except Exception as e:
            logger.error(f"Error invoking logical reasoning agent: {e}")
            raise ServerError(error=InternalError(message=str(e))) from e

        await event_queue.enqueue_event(new_agent_text_message(result))

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        raise ServerError(error=UnsupportedOperationError())
