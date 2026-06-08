from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from openai import APIError, APITimeoutError, AsyncOpenAI

from chatbot_api.settings import Settings


class EmbeddingProviderError(Exception):
    """Base embedding provider error."""


class EmbeddingProviderConfigurationError(EmbeddingProviderError):
    """Raised when the embedding provider is not configured correctly."""


class EmbeddingProviderTimeoutError(EmbeddingProviderError):
    """Raised when the upstream embedding request times out."""


class EmbeddingProvider(Protocol):
    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAIEmbeddingProvider:
    provider_name = "openai"

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise EmbeddingProviderConfigurationError(
                "OPENAI_API_KEY is required to use the OpenAI embedding provider"
            )

        self._model = settings.openai_embedding_model
        self._dimensions = settings.document_embedding_dimensions
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds,
        )

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        normalized_texts = [text.strip() for text in texts if text.strip()]
        if not normalized_texts:
            return []

        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=normalized_texts,
                dimensions=self._dimensions,
            )
        except APITimeoutError as exc:
            raise EmbeddingProviderTimeoutError("Embedding request timed out") from exc
        except APIError as exc:
            raise EmbeddingProviderError("Embedding provider request failed") from exc

        embeddings = [item.embedding for item in response.data]
        if len(embeddings) != len(normalized_texts):
            raise EmbeddingProviderError("Embedding provider returned an unexpected payload")

        return embeddings
