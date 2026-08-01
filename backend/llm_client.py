"""LLM client protocol and fake implementation."""

from typing import Protocol


class LLMClient(Protocol):
    """Protocol for replaceable LLM backends.

    Every concrete client must implement ``generate``.  The method may
    be synchronous or asynchronous — callers should ``await`` it.
    """

    async def generate(self, message: str) -> str: ...


class FakeLLMClient:
    """Returns a fixed test reply without touching the network.

    >>> client = FakeLLMClient()
    >>> await client.generate("你好")
    '测试回复：你好'
    """

    async def generate(self, message: str) -> str:
        return f"测试回复：{message}"
