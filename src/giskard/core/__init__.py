"""Public API surface for Giskard core (mirrors gikard)."""

import importlib.metadata
from typing import Final

try:
    _version = importlib.metadata.version("giskard-agent")
except importlib.metadata.PackageNotFoundError:
    try:
        _version = importlib.metadata.version("agent-framework-core")
    except importlib.metadata.PackageNotFoundError:
        _version = "0.0.0"
__version__: Final[str] = _version
