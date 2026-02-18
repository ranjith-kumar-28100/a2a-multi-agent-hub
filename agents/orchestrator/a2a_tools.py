"""A2A client utilities for communicating with remote agents.

Provides helper functions to discover remote agents via Agent Cards
and send tasks to them using the A2A protocol.
"""

import logging
from uuid import uuid4

import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    Message,
    MessageSendParams,
    SendMessageRequest,
    SendMessageSuccessResponse,
    Task,
)

logger = logging.getLogger(__name__)


async def discover_agent(base_url: str) -> AgentCard | None:
    """Fetch the AgentCard from a remote A2A agent.

    Args:
        base_url: The base URL of the remote agent (e.g., http://localhost:10001)

    Returns:
        The AgentCard if successfully fetched, None otherwise.
    """
    try:
        async with httpx.AsyncClient() as client:
            resolver = A2ACardResolver(
                httpx_client=client,
                base_url=base_url,
            )
            agent_card = await resolver.get_agent_card()
            logger.info(f"Discovered agent: {agent_card.name} at {base_url}")
            return agent_card
    except Exception as e:
        logger.error(f"Failed to discover agent at {base_url}: {e}")
        return None


async def send_task_to_agent(base_url: str, message_text: str) -> str:
    """Send a task to a remote A2A agent and return the response text.

    Args:
        base_url: The base URL of the remote agent.
        message_text: The text message to send.

    Returns:
        The response text from the agent, or an error message.
    """
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # First discover the agent
            resolver = A2ACardResolver(
                httpx_client=client,
                base_url=base_url,
            )
            agent_card = await resolver.get_agent_card()

            # Create A2A client
            a2a_client = A2AClient(
                httpx_client=client,
                agent_card=agent_card,
            )

            # Build the request
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

            # Send and get response
            response = await a2a_client.send_message(request)

            # Extract text from response
            return _extract_response_text(response)

    except Exception as e:
        logger.error(f"Error communicating with agent at {base_url}: {e}")
        return f"Error communicating with remote agent: {e}"


def _extract_response_text(response) -> str:
    """Extract text content from an A2A SendMessageResponse.

    SendMessageResponse is a RootModel[JSONRPCErrorResponse | SendMessageSuccessResponse].
    SendMessageSuccessResponse.result is Task | Message.
    """
    try:
        # SendMessageResponse is a RootModel — unwrap via .root
        inner = response.root
        logger.info(f"Response type: {type(inner).__name__}")

        # Check if it's an error response
        if not isinstance(inner, SendMessageSuccessResponse):
            logger.error(f"Got error response: {inner}")
            error_dict = inner.model_dump(mode="json", exclude_none=True)
            return f"Agent returned an error: {error_dict}"

        result = inner.result

        # Handle Task response
        if isinstance(result, Task):
            # Check artifacts first
            if result.artifacts:
                texts = []
                for artifact in result.artifacts:
                    if artifact.parts:
                        for part in artifact.parts:
                            text = _extract_part_text(part)
                            if text:
                                texts.append(text)
                if texts:
                    return "\n".join(texts)

            # Check history
            if result.history:
                for msg in reversed(result.history):
                    if msg.role == "agent":
                        for part in msg.parts:
                            text = _extract_part_text(part)
                            if text:
                                return text

        # Handle Message response
        if isinstance(result, Message):
            for part in result.parts:
                text = _extract_part_text(part)
                if text:
                    return text

        # Fallback: dump the whole response
        resp_dict = response.model_dump(mode="json", exclude_none=True)
        return str(resp_dict)

    except Exception as e:
        logger.error(f"Error extracting response text: {e}")
        return f"Error parsing agent response: {e}"


def _extract_part_text(part) -> str | None:
    """Extract text from a Part object (handles RootModel wrapping)."""
    # Part could be a RootModel wrapping TextPart, FilePart, DataPart
    inner = part.root if hasattr(part, 'root') else part
    if hasattr(inner, 'text'):
        return inner.text
    return None
