"""Reusable stdio MCP client used for the server lifetime."""
import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self, script: Path | None = None):
        self.script = script or Path(__file__).parent.parent / "mcp_server.py"
        self.stack: AsyncExitStack | None = None
        self.session: ClientSession | None = None
        self.lock = asyncio.Lock()

    async def start(self) -> None:
        self.stack = AsyncExitStack()
        params = StdioServerParameters(command=sys.executable, args=[str(self.script)],
                                       env={**os.environ, "FASTMCP_LOG_LEVEL": "ERROR"})
        read, write = await self.stack.enter_async_context(stdio_client(params))
        self.session = await self.stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()

    async def stop(self) -> None:
        if self.stack:
            await self.stack.aclose()
        self.stack = None
        self.session = None

    async def call_tool(self, name: str, arguments: dict) -> dict:
        if self.session is None:
            raise RuntimeError("MCP client is not started")
        async with self.lock:
            result = await self.session.call_tool(name, arguments)
        return json.loads(result.content[0].text)

    async def read_rule(self, rule_type: str) -> str:
        if self.session is None:
            raise RuntimeError("MCP client is not started")
        async with self.lock:
            result = await self.session.read_resource(
                f"quality-rule://requirements/{rule_type}")
        return result.contents[0].text

    async def get_prompt(self, document_id: str, domain: str) -> str:
        if self.session is None:
            raise RuntimeError("MCP client is not started")
        async with self.lock:
            result = await self.session.get_prompt(
                "requirement_review_prompt", {"document_id": document_id, "domain": domain})
        return result.messages[0].content.text
