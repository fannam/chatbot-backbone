from __future__ import annotations

import pytest

from chatbot_api.settings import Settings
from chatbot_api.workflow_runtime import ChatWorkflowRuntime


@pytest.mark.anyio
async def test_get_workflow_caches_instance() -> None:
    runtime = ChatWorkflowRuntime(Settings(langgraph_checkpoint_database_url=None))

    first = await runtime.get_workflow()
    second = await runtime.get_workflow()

    assert first is second


@pytest.mark.anyio
async def test_close_clears_cached_workflow_so_it_rebuilds() -> None:
    runtime = ChatWorkflowRuntime(Settings(langgraph_checkpoint_database_url=None))

    first = await runtime.get_workflow()
    await runtime.close()
    second = await runtime.get_workflow()

    assert second is not first
