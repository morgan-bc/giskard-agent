import importlib.metadata
from typing import Final

try:
    _version = importlib.metadata.version("giskard-agent")
except importlib.metadata.PackageNotFoundError:
    try:
        _version = importlib.metadata.version("agent-framework-core")
    except importlib.metadata.PackageNotFoundError:
        _version = "0.0.0"  # Fallback for development mode
__version__: Final[str] = _version

from .core.agents import Agent
from .core.harness import create_harness_agent
from .core import workflows
from .core import harness
from .providers.openai import OpenAIChatClient, OpenAIChatCompletionClient