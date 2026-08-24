# Copyright (c) Microsoft. All rights reserved.

"""OpenAI integration for Microsoft Agent Framework.

This package provides OpenAI client implementations for the Agent Framework,
including clients for the Responses API and Chat Completions API.
"""

import importlib.metadata

from .chat_client import (
    OpenAIChatClient,
    OpenAIChatOptions,
    OpenAIContinuationToken,
    RawOpenAIChatClient,
)
from .chat_completion_client import (
    OpenAIChatCompletionClient,
    OpenAIChatCompletionOptions,
    OpenAIChatMessagePreparer,
    OpenAIChatResponseContentsParser,
    RawOpenAIChatCompletionClient,
)
from .embedding_client import OpenAIEmbeddingClient, OpenAIEmbeddingOptions
from .exceptions import ContentFilterResultSeverity, OpenAIContentFilterException
from ._shared import OpenAISettings


__all__ = [
    "ContentFilterResultSeverity",
    "OpenAIChatClient",
    "OpenAIChatCompletionClient",
    "OpenAIChatCompletionOptions",
    "OpenAIChatMessagePreparer",
    "OpenAIChatOptions",
    "OpenAIChatResponseContentsParser",
    "OpenAIContentFilterException",
    "OpenAIContinuationToken",
    "OpenAIEmbeddingClient",
    "OpenAIEmbeddingOptions",
    "OpenAISettings",
    "RawOpenAIChatClient",
    "RawOpenAIChatCompletionClient",
    "__version__",
]
