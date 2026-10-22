import asyncio

import httpx
import pytest

from agent.runtime import ToolError, with_retry


def test_transient_error_is_retried():
    calls = 0

    async def flaky():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ReadTimeout("temporary")
        return "ok"

    assert asyncio.run(with_retry(flaky, tool_name="test", initial_delay=0)) == "ok"
    assert calls == 3


def test_permanent_error_is_not_retried():
    calls = 0

    async def broken():
        nonlocal calls
        calls += 1
        raise ValueError("permanent")

    with pytest.raises(ToolError) as error:
        asyncio.run(with_retry(broken, tool_name="test", initial_delay=0))
    assert error.value.code == "TOOL_EXECUTION_ERROR"
    assert calls == 1
