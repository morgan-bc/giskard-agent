# Copyright (c) Microsoft. All rights reserved.

"""Cross-platform local shell tool for the Microsoft Agent Framework."""

from __future__ import annotations

from .docker import (
    DEFAULT_IMAGE as DOCKER_DEFAULT_IMAGE,
)
from .docker import (
    DockerNotAvailableError,
    DockerShellTool,
    is_docker_available,
)
from .environment import (
    ShellEnvironmentProvider,
    ShellEnvironmentProviderOptions,
    ShellEnvironmentSnapshot,
    ShellFamily,
    default_instructions_formatter,
)
from .executor_base import ShellExecutor
from .policy import ShellDecision, ShellPolicy, ShellRequest
from .tool import LocalShellTool
from .types import (
    ShellCommandError,
    ShellExecutionError,
    ShellMode,
    ShellResult,
    ShellTimeoutError,
)

__all__ = [
    "DOCKER_DEFAULT_IMAGE",
    "DockerNotAvailableError",
    "DockerShellTool",
    "LocalShellTool",
    "ShellCommandError",
    "ShellDecision",
    "ShellEnvironmentProvider",
    "ShellEnvironmentProviderOptions",
    "ShellEnvironmentSnapshot",
    "ShellExecutionError",
    "ShellExecutor",
    "ShellFamily",
    "ShellMode",
    "ShellPolicy",
    "ShellRequest",
    "ShellResult",
    "ShellTimeoutError",
    "default_instructions_formatter",
    "is_docker_available",
]
