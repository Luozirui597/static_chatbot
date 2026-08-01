"""Chat service — orchestrates validation, LLM calls, and replies."""

from backend.llm_client import LLMClient


class ChatService:
    """Receives a validated user message, calls the LLM client, and
    returns the reply.

    The LLM client is injected so it can be swapped without touching
    the service or route code.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def handle_message(self, message: str) -> str:
        """Return the assistant reply for *message*."""
        return await self._llm.generate(message)
