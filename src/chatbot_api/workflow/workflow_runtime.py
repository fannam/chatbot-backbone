from __future__ import annotations

import asyncio
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from chatbot_api.settings import Settings
from chatbot_api.workflow.graph import ChatWorkflow, build_chat_workflow


class ChatWorkflowRuntime:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = asyncio.Lock()
        self._workflow: ChatWorkflow | None = None
        self._checkpointer_cm: Any | None = None

    async def get_workflow(self) -> ChatWorkflow:
        if self._workflow is not None:
            return self._workflow

        async with self._lock:
            if self._workflow is not None:
                return self._workflow

            workflow = await self._build_workflow()
            self._workflow = workflow
            return workflow

    async def close(self) -> None:
        self._workflow = None
        if self._checkpointer_cm is None:
            return

        await self._checkpointer_cm.__aexit__(None, None, None)
        self._checkpointer_cm = None

    async def _build_workflow(self) -> ChatWorkflow:
        checkpoint_database_url = self._settings.langgraph_checkpoint_database_url
        if checkpoint_database_url is None:
            return build_chat_workflow(checkpointer=InMemorySaver())

        checkpointer_cm = AsyncPostgresSaver.from_conn_string(checkpoint_database_url)
        checkpointer = await checkpointer_cm.__aenter__()

        try:
            await checkpointer.setup()
        except Exception:
            await checkpointer_cm.__aexit__(None, None, None)
            raise

        self._checkpointer_cm = checkpointer_cm
        return build_chat_workflow(checkpointer=checkpointer)
