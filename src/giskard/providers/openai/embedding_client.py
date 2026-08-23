# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import base64
import struct
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, TypedDict, overload

from giskard.core.clients import BaseEmbeddingClient
from giskard.core.settings import SecretString
from giskard.core.telemetry import USER_AGENT_KEY, mark_feature_used
from giskard.core.types import Embedding, EmbeddingGenerationOptions, GeneratedEmbeddings, UsageDetails
from giskard.core.observability import EmbeddingTelemetryLayer
from openai import AsyncOpenAI

from .feature_usage import FeatureIndex
from ._shared import load_openai_service_settings

if sys.version_info >= (3, 13):
    from typing import TypeVar  # pragma: no cover
else:
    from typing_extensions import TypeVar  # pragma: no cover

class OpenAIEmbeddingOptions(EmbeddingGenerationOptions, total=False):
    """OpenAI-specific embedding options.

    Extends EmbeddingGenerationOptions with OpenAI-specific fields.

    Examples:
        .. code-block:: python

            from giskard.core.openai import OpenAIEmbeddingOptions

            options: OpenAIEmbeddingOptions = {
                "model": "text-embedding-3-small",
                "dimensions": 1536,
                "encoding_format": "float",
            }
    """

    encoding_format: Literal["float", "base64"]
    user: str

OpenAIEmbeddingOptionsT = TypeVar(
    "OpenAIEmbeddingOptionsT",
    bound=TypedDict,  # type: ignore[valid-type]
    default="OpenAIEmbeddingOptions",
    covariant=True,
)

class RawOpenAIEmbeddingClient(
    BaseEmbeddingClient[str, list[float], OpenAIEmbeddingOptionsT],
    Generic[OpenAIEmbeddingOptionsT],
):
    """Raw OpenAI embedding client without telemetry."""

    INJECTABLE: ClassVar[set[str]] = {"client"}
    _FEATURE_USAGE_INDEX: ClassVar[int | None] = FeatureIndex.OPENAI

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | SecretString | Callable[[], str | Awaitable[str]] | None = None,
        org_id: str | None = None,
        base_url: str | None = None,
        default_headers: Mapping[str, str] | None = None,
        async_client: AsyncOpenAI | None = None,
        additional_properties: dict[str, Any] | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None:
        """Initialize a raw OpenAI embedding client.

        Keyword Args:
                constructor reads ``OPENAI_EMBEDDING_MODEL`` and then ``OPENAI_MODEL``
            api_key: API key override. For OpenAI this maps to ``OPENAI_API_KEY``.
                A callable token provider is also accepted for backwards compatibility,
            org_id: OpenAI organization ID. Used only for OpenAI and resolved from
                ``OPENAI_ORG_ID`` when not provided.
            base_url: Base URL override. For OpenAI this maps to ``OPENAI_BASE_URL``.
                to pass the full ``.../openai/v1`` base URL directly.
            default_headers: Additional HTTP headers.
            async_client: Pre-configured client. Passing ``AsyncOpenAI`` keeps the client on
            additional_properties: Additional properties stored on the client instance.
            env_file_path: Optional ``.env`` file that is checked before process environment
                lookups.
            env_file_encoding: Encoding for the ``.env`` file.

        Notes:
            Environment resolution precedence is:

            2. Explicit OpenAI API key or ``OPENAI_API_KEY``

            OpenAI reads ``OPENAI_API_KEY``, ``OPENAI_EMBEDDING_MODEL``,
        """
        settings, client, _ = load_openai_service_settings(
            model=model,
            api_key=api_key,
            org_id=org_id,
            base_url=base_url,
            default_headers=default_headers,
            client=async_client,
            env_file_path=env_file_path,
            env_file_encoding=env_file_encoding,
            openai_model_fields=("embedding_model", "model"),

        )

        self.client = client
        resolved_model = settings.get("model")
        self.model: str | None = resolved_model.strip() if isinstance(resolved_model, str) and resolved_model else None

        # Store configuration for serialization
        self.org_id = settings.get("org_id")
        self.base_url = settings.get("base_url")
        self.api_version = settings.get("api_version")
        if default_headers:
            self.default_headers: dict[str, Any] | None = {
                k: v for k, v in default_headers.items() if k != USER_AGENT_KEY
            }
        else:
            self.default_headers = None

        super().__init__(additional_properties=additional_properties)

    def service_url(self) -> str:
        """Get the URL of the service."""
        return str(self.client.base_url) if self.client else "Unknown"

    async def get_embeddings(
        self,
        values: Sequence[str],
        *,
        options: OpenAIEmbeddingOptionsT | None = None,
    ) -> GeneratedEmbeddings[list[float], OpenAIEmbeddingOptionsT]:
        """Call the OpenAI embeddings API.

        Args:
            values: The text values to generate embeddings for.
            options: Optional embedding generation options.

        Returns:
            Generated embeddings with usage metadata.

        Raises:
            ValueError: If model is not provided or values is empty.
        """
        if not values:
            return GeneratedEmbeddings([], options=options)

        opts: dict[str, Any] = options or {}  # type: ignore
        model = opts.get("model") or self.model
        if not model:
            raise ValueError("model is required")

        kwargs: dict[str, Any] = {"input": list(values), "model": model}
        if self._FEATURE_USAGE_INDEX is not None:
            mark_feature_used(self._FEATURE_USAGE_INDEX)
        if dimensions := opts.get("dimensions"):
            kwargs["dimensions"] = dimensions
        if encoding_format := opts.get("encoding_format"):
            kwargs["encoding_format"] = encoding_format
        if user := opts.get("user"):
            kwargs["user"] = user

        response = await self.client.embeddings.create(**kwargs)

        encoding = kwargs.get("encoding_format", "float")
        embeddings: list[Embedding[list[float]]] = []
        for item in response.data:
            vector: list[float]
            if encoding == "base64" and isinstance(item.embedding, str):
                # Decode base64-encoded floats (little-endian IEEE 754)
                raw = base64.b64decode(item.embedding)
                vector = list(struct.unpack(f"<{len(raw) // 4}f", raw))
            else:
                vector = item.embedding
            embeddings.append(
                Embedding(
                    vector=vector,
                    dimensions=len(vector),
                    model=response.model,
                )
            )

        usage_dict: UsageDetails | None = None
        if response.usage:
            usage_dict = {
                "input_token_count": response.usage.prompt_tokens,
                "total_token_count": response.usage.total_tokens,
            }

        return GeneratedEmbeddings(embeddings, options=options, usage=usage_dict)

class OpenAIEmbeddingClient(
    EmbeddingTelemetryLayer[str, list[float], OpenAIEmbeddingOptionsT],
    RawOpenAIEmbeddingClient[OpenAIEmbeddingOptionsT],
    Generic[OpenAIEmbeddingOptionsT],
):
    """OpenAI embedding client with telemetry support."""

    OTEL_PROVIDER_NAME: ClassVar[str] = "openai"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | Callable[[], str | Awaitable[str]] | None = None,
        org_id: str | None = None,
        default_headers: Mapping[str, str] | None = None,
        async_client: AsyncOpenAI | None = None,
        base_url: str | None = None,
        otel_provider_name: str | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None:
        """Initialize an OpenAI embedding client.

        Keyword Args:
                constructor reads ``OPENAI_EMBEDDING_MODEL`` and then ``OPENAI_MODEL``
            api_key: API key override. For OpenAI this maps to ``OPENAI_API_KEY``.
                A callable token provider is also accepted for backwards compatibility,
            org_id: OpenAI organization ID. Used only for OpenAI and resolved from
                ``OPENAI_ORG_ID`` when not provided.
            default_headers: Additional HTTP headers.
            async_client: Pre-configured client. Passing ``AsyncOpenAI`` keeps the client on
            base_url: Base URL override. For OpenAI this maps to ``OPENAI_BASE_URL``.
                to pass the full ``.../openai/v1`` base URL directly.
            otel_provider_name: Override the OpenTelemetry provider name.
            env_file_path: Optional ``.env`` file that is checked before process environment
                lookups.
            env_file_encoding: Encoding for the ``.env`` file.

        Notes:
            Environment resolution precedence is:

            2. Explicit OpenAI API key or ``OPENAI_API_KEY``

            OpenAI reads ``OPENAI_API_KEY``, ``OPENAI_EMBEDDING_MODEL``,

        Examples:
            .. code-block:: python

                from giskard.core.openai import OpenAIEmbeddingClient

                # Using environment variables
                # Set OPENAI_API_KEY=sk-...
                # Set OPENAI_EMBEDDING_MODEL=text-embedding-3-small
                client = OpenAIEmbeddingClient()

                # Or passing OpenAI parameters directly
                client = OpenAIEmbeddingClient(
                    model="text-embedding-3-small",
                    api_key="sk-...",
                )

                client = OpenAIEmbeddingClient(
                    model="text-embedding-3-small",
                )
        """
        super().__init__(
            model=model,
            api_key=api_key,
            org_id=org_id,
            base_url=base_url,
            default_headers=default_headers,
            async_client=async_client,
            otel_provider_name=otel_provider_name,
            env_file_path=env_file_path,
            env_file_encoding=env_file_encoding,
        )