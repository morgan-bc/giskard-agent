# Copyright (c) Microsoft. All rights reserved.

"""Agent Skills provider, models, and discovery utilities.

Defines the core data model classes for the agent skills system:

- **Skills:** :class:`Skill` (abstract base) and :class:`FileSkill` (filesystem-backed).
- **Resources:** :class:`SkillResource` (abstract base).
- **Sources:** :class:`SkillsSource` (abstract base for custom skill origins).
- **Provider:** :class:`SkillsProvider` which implements the
  progressive-disclosure pattern from the
  `Agent Skills specification <https://agentskills.io/>`_:

1. **Advertise** — skill names and descriptions are injected into the system prompt.
2. **Load** — skill location, resource list, script list and instructions are returned via the ``load_skill`` tool.

Skills can come from different sources:

- **File-based** — discovered by scanning configured directories for ``SKILL.md`` files.
  Represented as :class:`FileSkill` instances.
- **Custom sources** — any :class:`SkillsSource` implementation that provides
  skills from arbitrary origins (REST APIs, databases, etc.).

Multiple sources can be composed if needed, but the built-in decorators have been removed; use `FileSkillsSource` directly.

**Security:** file-based skill metadata is XML-escaped before prompt injection, and
file-based resource reads are guarded against path traversal and symlink escape.
Only use skills from trusted sources.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from html import escape as xml_escape
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar, Final, Protocol, TypeAlias, TypeVar, cast, runtime_checkable

from ._filesystem import is_link_or_reparse_point
from .sessions import ContextProvider
from .telemetry import FeatureIndex, mark_feature_used
from .tools import ApprovalMode, FunctionTool

if TYPE_CHECKING:
    from .agents import SupportsAgentRun
    from .sessions import AgentSession, SessionContext
    from .types import Content

logger = logging.getLogger(__name__)

# region Models


"""Callable that converts raw script arguments before an inline script runs.

The parser receives the raw ``args`` value supplied by the agent/LLM (a
``dict`` of named arguments, a ``list[str]`` of positional arguments, a
``str`` for backends that send arguments as an unparsed JSON string, or
``None``) and returns the named keyword arguments to pass to the inline
script callable: a ``dict`` (or ``None`` for no arguments).  Inline scripts
bind arguments by keyword name, so the parser must normalize whatever shape
it receives into a ``dict`` or ``None``.

When no parser is configured, inline scripts use the raw value unchanged.
This hook lets callers plug in their own argument conversion logic to support
backends (for example, vLLM and some OpenAI-compatible servers) that encode
tool-call arguments as a JSON string instead of a JSON object.
"""


class SkillResource(ABC):
    """Abstract base class for supplementary content attached to a skill.

    A resource provides data that an agent can retrieve on demand.
    The concrete implementation is :class:`FileSkillResource`, which reads
    file-backed content from disk.

    Attributes:
        name: Resource identifier.
        description: Optional human-readable summary, or ``None``.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str | None = None,
    ) -> None:
        """Initialize a SkillResource.

        Args:
            name: Identifier for this resource (e.g. ``"reference"``, ``"get-schema"``).
            description: Optional human-readable summary shown when advertising the resource.
        """
        if not name or not name.strip():
            raise ValueError("Resource name cannot be empty.")

        self.name = name
        self.description = description

    @abstractmethod
    async def read(self, **kwargs: Any) -> Any:
        """Read the resource content.

        Args:
            **kwargs: Runtime keyword arguments forwarded to resource
                functions that accept ``**kwargs``.

        Returns:
            The resource content (any type).
        """


class FileSkillResource(SkillResource):
    """A file-path-backed skill resource that reads content from disk.

    Stores a pre-resolved absolute file path and reads content directly,
    file-backed resource.

    Attributes:
        name: Resource identifier (relative path within the skill directory).
        description: Optional human-readable summary, or ``None``.
        full_path: Absolute path to the resource file.
    """

    def __init__(
        self,
        *,
        name: str,
        full_path: str,
        description: str | None = None,
    ) -> None:
        """Initialize a FileSkillResource.

        Args:
            name: Relative path of the resource within the skill directory.
            full_path: Absolute path to the resource file.
            description: Optional human-readable summary.

        Raises:
            ValueError: If ``full_path`` is empty.
        """
        super().__init__(name=name, description=description)

        if not full_path or not full_path.strip():
            raise ValueError("full_path cannot be empty.")

        self.full_path = full_path

    async def read(self, **kwargs: Any) -> Any:
        """Read the resource content from disk.

        Args:
            **kwargs: Unused.

        Returns:
            The UTF-8 text content of the resource file.

        Raises:
            ValueError: If the resource file does not exist.
        """
        if not await asyncio.to_thread(Path(self.full_path).is_file):
            raise ValueError(f"Resource file '{self.name}' not found at '{self.full_path}'.")

        logger.info("Reading resource '%s' from '%s'", self.name, self.full_path)
        return await asyncio.to_thread(Path(self.full_path).read_text, encoding="utf-8")


class Skill(ABC):
    """Abstract base class for all agent skills.

    A skill represents a domain-specific capability with instructions
    and resources.  Concrete implementation is :class:`FileSkill`
    (filesystem-backed).

    Skill spec metadata (name, description, license, compatibility,
    allowed_tools, metadata) is exposed via the :attr:`frontmatter`
    property, which returns a :class:`SkillFrontmatter` instance.
    """

    @property
    @abstractmethod
    def frontmatter(self) -> SkillFrontmatter:
        """The L1 discovery metadata for this skill.

        Contains the name, description, and other spec fields as defined by
        the `Agent Skills specification <https://agentskills.io/specification>`_.
        """
        ...

    @abstractmethod
    async def get_content(self) -> str:
        """Get the full skill content.

        For file-based skills this is the raw SKILL.md file content,
        optionally augmented with a synthesized scripts block when scripts
        are present.  For code-defined skills this is a synthesized XML
        document containing name, description, and body (instructions,
        resources, scripts).

        Returns:
            The full skill content string.
        """
        ...

    async def get_resource(self, name: str) -> SkillResource | None:
        """Get a resource owned by this skill by name.

        Args:
            name: The resource name (e.g. an identifier or a relative path
                referenced inside the skill content).

        Returns:
            The :class:`SkillResource`, or ``None`` when no resource with the
            given name exists.
        """
        return None

class SkillFrontmatter:
    """L1 discovery metadata for a :class:`Skill`.

    Encapsulates all `Agent Skills specification <https://agentskills.io/specification>`_
    frontmatter fields in a single object. All fields are mutable plain
    attributes; callers may freely reassign them after construction.

    The constructor validates ``name``, ``description``, and ``compatibility``
    against specification rules and raises :class:`ValueError` on invalid
    input. Assignments made after construction are **not** re-validated;
    callers are expected to honor the spec.

    Attributes:
        name: Skill name (lowercase letters, numbers, hyphens only).
        description: Human-readable description of the skill.
        license: Optional license name or reference.
        compatibility: Optional compatibility information (≤500 characters).
        allowed_tools: Optional space-delimited pre-approved tool names.
        metadata: Optional arbitrary key-value pairs (shallow-copied on
            construction to avoid caller-owned dict aliasing).
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        license: str | None = None,
        compatibility: str | None = None,
        allowed_tools: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Initialize a SkillFrontmatter.

        Args:
            name: Skill name (lowercase letters, numbers, hyphens only;
                max 64 characters; no leading/trailing/consecutive hyphens).
            description: Human-readable description of the skill
                (≤1024 characters).
            license: Optional license name or reference.
            compatibility: Optional compatibility information
                (≤500 characters).
            allowed_tools: Optional space-delimited pre-approved tool names.
            metadata: Optional arbitrary key-value pairs.

        Raises:
            ValueError: If the name, description, or compatibility is invalid.
        """
        _validate_skill_name(name)
        _validate_skill_description(name, description)
        _validate_compatibility(compatibility)

        self.name = name
        self.description = description
        self.compatibility = compatibility
        self.license = license
        self.allowed_tools = allowed_tools
        # Shallow-copy to avoid aliasing with caller-owned dict.
        self.metadata: dict[str, str] | None = dict(metadata) if metadata is not None else None


def _validate_skill_name(name: str) -> None:
    """Validate a skill name against specification rules.

    Args:
        name: The skill name to validate.

    Raises:
        ValueError: If the name is empty, too long, or does not match
            the required pattern.
    """
    if not name or not name.strip():
        raise ValueError("Skill name cannot be empty.")
    if len(name) > MAX_NAME_LENGTH or not VALID_NAME_RE.match(name):
        raise ValueError(
            f"Invalid skill name '{name}': Must be {MAX_NAME_LENGTH} characters or fewer, "
            "using only lowercase letters, numbers, and hyphens, and must not start or end with a hyphen "
            "or contain consecutive hyphens."
        )


def _validate_skill_description(name: str, description: str) -> None:
    """Validate a skill description against specification rules.

    Args:
        name: The skill name (used in error messages).
        description: The description to validate.

    Raises:
        ValueError: If the description is empty or too long.
    """
    if not description or not description.strip():
        raise ValueError("Skill description cannot be empty.")
    if len(description) > MAX_DESCRIPTION_LENGTH:
        raise ValueError(
            f"Skill '{name}' has an invalid description: Must be {MAX_DESCRIPTION_LENGTH} characters or fewer."
        )


def _validate_compatibility(compatibility: str | None) -> None:
    """Validate an optional compatibility value against specification rules.

    Args:
        compatibility: The optional compatibility value to validate.

    Raises:
        ValueError: If the value exceeds the maximum allowed length.
    """
    if compatibility is not None and len(compatibility) > MAX_COMPATIBILITY_LENGTH:
        raise ValueError(f"Skill compatibility must be {MAX_COMPATIBILITY_LENGTH} characters or fewer.")


def _build_skill_content(
    name: str,
    description: str,
    instructions: str,
    resources: Sequence[SkillResource] | None = None,
) -> str:
    """Build XML-structured content for code-defined skills."""


    result = (
        f"<name>{xml_escape(name, quote=False)}</name>\n"
        f"<description>{xml_escape(description, quote=False)}</description>\n"
        "\n"
        "<instructions>\n"
        f"{instructions}\n"
        "</instructions>"
    )

    result += f"\n\n{_build_available_resources_block(resources)}"

    return result


def _create_resource_element(resource: SkillResource) -> str:
    """Create a self-closing ``<resource …/>`` XML element from a :class:`SkillResource`.

    Args:
        resource: The resource to create the element from.

    Returns:
        A single indented XML element string with ``name`` and optional
        ``description`` attributes.
    """
    attrs = f'name="{xml_escape(resource.name, quote=True)}"'
    if resource.description:
        attrs += f' description="{xml_escape(resource.description, quote=True)}"'
    return f"  <resource {attrs}/>"


def _build_available_resources_block(resources: Sequence[SkillResource] | None) -> str:
    """Build an ``<available_resources>`` XML block for the given resources.

    Each resource is emitted as a ``<resource name="…"/>`` element (with an
    optional ``description`` attribute).  When there are no resources, a
    self-closing ``<available_resources />`` element is returned so the model
    knows none are available and does not hallucinate resource names.

    Args:
        resources: The resources to include in the block, if any.

    Returns:
        The ``<available_resources>`` XML block, or ``<available_resources />``
        when *resources* is empty or ``None``.
    """
    if not resources:
        return "<available_resources />"
    resource_lines = "\n".join(_create_resource_element(r) for r in resources)
    return f"<available_resources>\n{resource_lines}\n</available_resources>"



class FileSkill(Skill):
    """A :class:`Skill` discovered from a filesystem directory backed by a SKILL.md file.

    Attributes:
        path: Absolute path to the directory containing this skill.
    """

    def __init__(
        self,
        *,
        frontmatter: SkillFrontmatter,
        content: str,
        path: str,
        resources: Sequence[SkillResource] | None = None,
        scripts: Sequence[str] | None = None,
    ) -> None:
        """Initialize a FileSkill.

        Args:
            frontmatter: Skill specification metadata parsed from the
                SKILL.md file's YAML frontmatter (name, description,
                and optional spec fields).
            content: The full raw SKILL.md file content including YAML frontmatter.
            path: Absolute path to the skill directory on disk.
            resources: Resources discovered for this skill.
            scripts: Scripts discovered for this skill (relative paths).
        """
        self._frontmatter = frontmatter

        self._content = content
        self.path = path
        self._resources: list[SkillResource] = list(resources) if resources is not None else []
        self._scripts: list[str] = list(scripts) if scripts is not None else []
        self._cached_content: str | None = None

    @property
    def frontmatter(self) -> SkillFrontmatter:
        """The L1 discovery metadata for this skill."""
        return self._frontmatter

    async def get_content(self) -> str:
        """The skill content without frontmatter and without resource/script blocks."""
        if self._cached_content is not None:
            return self._cached_content
        body = self._content
        m = FRONTMATTER_RE.search(body)
        if m:
            body = body[m.end():].lstrip("\r\n")
        self._cached_content = body
        return self._cached_content

    async def get_resource(self, name: str) -> SkillResource | None:
        """Get a resource by name.

        Args:
            name: The resource name to look up (case-insensitive).

        Returns:
            The :class:`SkillResource`, or ``None`` when no resource with the
            given name exists.
        """
        name_lower = name.lower()
        return next((r for r in self._resources if r.name.lower() == name_lower), None)


SKILL_FILE_NAME: Final[str] = "SKILL.md"
# How deep to search for SKILL.md files within the top-level skill_paths directories.
# This is separate from DEFAULT_SEARCH_DEPTH which controls per-skill resource/script scanning.
MAX_SEARCH_DEPTH: Final[int] = 2
MAX_NAME_LENGTH: Final[int] = 64
MAX_DESCRIPTION_LENGTH: Final[int] = 1024
MAX_COMPATIBILITY_LENGTH: Final[int] = 500
DEFAULT_RESOURCE_EXTENSIONS: Final[tuple[str, ...]] = (
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".xml",
    ".txt",
)
DEFAULT_SCRIPT_EXTENSIONS: Final[tuple[str, ...]] = (".py",)
# How deep to scan for resource/script files within each individual skill directory.
# This is separate from MAX_SEARCH_DEPTH which controls SKILL.md discovery.
DEFAULT_SEARCH_DEPTH: Final[int] = 2


# region Patterns and prompt template

# Matches YAML frontmatter delimited by "---" lines.
# The \uFEFF? prefix allows an optional UTF-8 BOM.
FRONTMATTER_RE = re.compile(
    r"\A\uFEFF?---\s*$(.+?)^---\s*$",
    re.MULTILINE | re.DOTALL,
)

# Matches top-level YAML "key: value" lines (unindented). Group 1 = key,
# Group 2 = quoted value, Group 3 = unquoted value. Only matches keys at
# column 0 so that indented children (e.g. under "metadata:") are not
# mistakenly captured as top-level fields.
YAML_KV_RE = re.compile(
    r"^([\w-]+)\s*:\s*(?:[\"'](.+?)[\"']|(.+?))\s*$",
    re.MULTILINE,
)

# Matches a YAML "metadata:" block followed by indented key-value pairs.
YAML_METADATA_BLOCK_RE = re.compile(
    r"^metadata\s*:\s*$\n((?:[ \t]+\S.*\n?)+)",
    re.MULTILINE,
)

# Matches indented "key: value" lines within a metadata block.
YAML_INDENTED_KV_RE = re.compile(
    r"^\s+([\w-]+)\s*:\s*(?:[\"'](.+?)[\"']|(.+?))\s*$",
    re.MULTILINE,
)

# Validates skill names: lowercase letters, numbers, hyphens only;
# must not start or end with a hyphen, and must not contain consecutive hyphens.
VALID_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9]*-[a-z0-9])*[a-z0-9]*$")

# Block scalar indicator characters recognised by the lightweight YAML parser.
_BLOCK_SCALAR_INDICATORS = ("|", ">")


def _parse_yaml_scalar_value(yaml_content: str, kv_match: re.Match[str]) -> str:
    """Resolve the scalar value for an unquoted YAML key-value match.

    If the captured value starts with a YAML block scalar indicator (``|`` or
    ``>``), the function reads subsequent indented continuation lines, strips
    the common leading indentation, and joins them according to the scalar
    style (literal preserves newlines, folded replaces them with spaces).

    Chomping indicators are respected per YAML 1.2 §8.1.1.2:

    * ``-`` (strip) — final line break and trailing empty lines excluded
    * ``+`` (keep) — final line break and any trailing empty lines preserved
    * default (clip) — final line break preserved, trailing empty lines excluded

    For plain (non-block-scalar) values the captured text is returned as-is.
    Note: explicit indentation indicators (e.g. ``|2``) are not supported;
    indentation is auto-detected from the common leading whitespace.
    """
    value: str = kv_match.group(3)

    if not value or value[0] not in _BLOCK_SCALAR_INDICATORS:
        return value

    scalar_style = value[0]
    keep_trailing_newline = len(value) > 1 and value[1] == "+"
    strip_trailing_newline = len(value) > 1 and value[1] == "-"

    # Find the start of the next line after this key-value match.
    next_line_start = yaml_content.find("\n", kv_match.end())
    if next_line_start < 0:
        return value
    next_line_start += 1  # skip the newline character itself

    # Collect indented continuation lines (or blank lines within the block).
    block_lines: list[str] = []
    pos = next_line_start
    while pos < len(yaml_content):
        line_end = yaml_content.find("\n", pos)
        if line_end < 0:
            line = yaml_content[pos:]
            line_end = len(yaml_content)
        else:
            line = yaml_content[pos:line_end]

        if not line or line.isspace():
            # Blank / whitespace-only lines are part of the block.
            block_lines.append("")
            pos = line_end + 1 if line_end < len(yaml_content) else line_end
            continue

        if line[0] not in (" ", "\t"):
            # Non-indented, non-blank line — end of the block.
            break

        block_lines.append(line)
        pos = line_end + 1 if line_end < len(yaml_content) else line_end

    # Strip trailing blank lines collected from the block.
    while block_lines and block_lines[-1] == "":
        block_lines.pop()

    if not block_lines:
        return ""

    # Determine the common leading indentation across non-empty lines.
    # Only space/tab characters count as indentation (matches YAML semantics).
    def _indent_width(s: str) -> int:
        i = 0
        while i < len(s) and s[i] in (" ", "\t"):
            i += 1
        return i

    common_indent = min(_indent_width(line) for line in block_lines if line)
    normalized = [line[common_indent:] if line else "" for line in block_lines]

    # Literal preserves newlines; folded joins non-empty lines with spaces.
    parsed = "\n".join(normalized) if scalar_style == "|" else " ".join(line for line in normalized if line)

    if keep_trailing_newline:
        return parsed + "\n"
    if strip_trailing_newline:
        return parsed
    # Clip (default): literal gets a trailing newline, folded does not.
    if scalar_style == "|":
        return parsed + "\n"
    return parsed


# Default system prompt template for advertising available skills to the model.
# Use {skills} as the placeholder for the generated skills XML list.
DEFAULT_SKILLS_INSTRUCTION_PROMPT = """\
You have access to skills containing domain-specific knowledge and capabilities.
Each skill provides specialized instructions, reference documents, and assets for specific tasks.

<available_skills>
{skills}
</available_skills>

When a task aligns with a skill's domain, follow these steps in exact order:
- Use `load_skill` to retrieve the skill's location, resource list, script list and instructions.
- Follow the provided guidance.
{resource_instructions}
{runner_instructions}
Only load what is needed, when it is needed."""

RESOURCE_INSTRUCTIONS: Final[str] = (
    "- Use `read_file` to read any referenced resources, using the name exactly as listed\n"
    '   (e.g. `"style-guide"` not `"style-guide.md"`, `"references/FAQ.md"` not `"FAQ.md"`).\n'
)

SCRIPT_RUNNER_INSTRUCTIONS: Final[str] = (
    "- Use `shell tool` to run referenced scripts, using the name exactly as listed.\n"
)


_TSkillsProvider = TypeVar("_TSkillsProvider", bound="SkillsProvider")


class SkillsProvider(ContextProvider):
    """Context provider that advertises skills and exposes the load_skill tool.

    Accepts a :class:`SkillsSource`, a single :class:`Skill`, or a
    sequence of :class:`Skill` instances. For file-based skills, use
    :meth:`from_paths`.

    Follows the progressive-disclosure pattern from the
    `Agent Skills specification <https://agentskills.io/>`_:

    1. **Advertise** — injects skill names and descriptions into the system
       prompt (~100 tokens per skill).
    2. **Load** — returns skill location, resource list, script list and
       instructions (SKILL.md body) via ``load_skill``. The agent then uses
       ``read_file`` for resources and ``shell tool`` for scripts.

    **Security:** file-based metadata is XML-escaped before prompt injection,
    and file-based resource reads are guarded against path traversal and
    symlink escape.  Only use skills from trusted sources.

    **Security considerations (external skill sources):** which skills are
    available, and how much trust to place in them, is entirely determined by
    the :class:`SkillsSource` instances this provider is configured with — see
    :class:`SkillsSource` for source-level trust-boundary guidance. Skill content
    (names, descriptions, and location/resources/scripts/instructions loaded via ``load_skill``)
    is injected into the agent's context as-is, so a compromised or adversarial
    source can attempt indirect prompt injection. Only enable sources you trust.

    **Tool approval:** by default the ``load_skill`` tool is registered with
    ``approval_mode="always_require"``, so each skill load needs approval.
    To run unattended, pass :meth:`all_tools_auto_approval_rule` to
    :class:`~agent_framework.ToolApprovalMiddleware` (via ``auto_approval_rules``)
    or set ``disable_load_skill_approval=True`` for trusted skills.

    Examples:
        File-based factory (recommended for single-source file skills):

        .. code-block:: python

            provider = SkillsProvider.from_paths("./skills")

        Multiple skills via in-memory source:

        .. code-block:: python

            provider = SkillsProvider([skill_a, skill_b])

        Composing multiple sources:

        .. code-block:: python

            provider = SkillsProvider(FileSkillsSource("./skills"))

    .. note::

        By default, skills are cached after first load.  Set
        ``disable_caching=True`` to re-query the source on every agent
        run, so that updates to file-based skills or code-defined skill
        lists are always picked up while filtering and deduplication
        remain in effect.

    Attributes:
        DEFAULT_SOURCE_ID: Default value for the ``source_id`` used by this provider.
        LOAD_SKILL_TOOL_NAME: Name of the tool that loads a skill.
    """

    DEFAULT_SOURCE_ID: ClassVar[str] = "agent_skills"

    #: Name of the tool that loads skill metadata.
    LOAD_SKILL_TOOL_NAME: ClassVar[str] = "load_skill"

    @staticmethod
    def _is_local_tool_call(function_call: Content) -> bool:
        """Return whether a function call targets this provider's local tools.

        Hosted-tool calls carry a ``server_label`` in their
        ``additional_properties`` and are a separate server-scoped approval
        boundary that must be passed through untouched (see
        :func:`gikard._tools._is_hosted_tool_approval`). These rules
        only ever auto-approve the provider's own local tools, so any call that
        carries a ``server_label`` is rejected even if its name collides with a
        skill tool name.
        """
        return not function_call.additional_properties.get("server_label")

    @staticmethod
    def read_only_tools_auto_approval_rule(function_call: Content) -> bool:
        """Auto-approval rule that approves the read-only skill tool (load_skill)."""
        return (
            SkillsProvider._is_local_tool_call(function_call)
            and function_call.name == SkillsProvider.LOAD_SKILL_TOOL_NAME
        )

    @staticmethod
    def all_tools_auto_approval_rule(function_call: Content) -> bool:
        """Auto-approval rule that approves the skill tool."""
        return (
            SkillsProvider._is_local_tool_call(function_call) and function_call.name == SkillsProvider.LOAD_SKILL_TOOL_NAME
        )

    def __init__(
        self,
        source: SkillsSource | Sequence[Skill] | Skill,
        *,
        instruction_template: str | None = None,
        disable_caching: bool = False,
        cache_refresh_interval: timedelta | None = None,
        disable_load_skill_approval: bool = False,
        source_id: str | None = None,
    ) -> None:
        """Initialize a SkillsProvider.

        Accepts a :class:`SkillsSource`, a single :class:`Skill`, or a
        sequence of :class:`Skill` instances.  When skills are passed
        directly, they are automatically deduplicated and cached.

        A caller-supplied :class:`SkillsSource` is used as-is: it is **not**
        automatically deduplicated or wrapped in a
        :class:`CachingSkillsSource`. This keeps context-aware sources safe —
        auto-caching a caller source in a single shared cache could replay one
        agent's/tenant's skills for another. Compose
        :class:`DeduplicatingSkillsSource` / :class:`CachingSkillsSource`
        (optionally with a ``cache_isolation_key_selector``) yourself when you
        need them.

        For file-based skills, use :meth:`from_paths` or compose sources
        directly using :class:`FileSkillsSource` and other source classes.

        Args:
            source: A :class:`SkillsSource`, a single :class:`Skill`,
                or a sequence of :class:`Skill` instances.

        Keyword Args:
            instruction_template: Custom system-prompt template for
                advertising skills. Must contain a ``{skills}`` placeholder for the
                generated skills list. May optionally contain
                ``{runner_instructions}`` and/or ``{resource_instructions}``
                placeholders; when present, they are filled with built-in
                guidance for script execution and resource reading respectively.
                When omitted, those instructions are simply not included in the
                rendered prompt (the corresponding tools are still registered).
                Uses a built-in template when ``None``.
            disable_caching: When ``True``, the built-in file/in-memory source
                is not wrapped in a :class:`CachingSkillsSource`, so skills are
                rebuilt on every invocation. This only affects the sources the
                provider builds internally (from a :class:`Skill`, a sequence of
                skills, or :meth:`from_paths`); a caller-supplied
                :class:`SkillsSource` is never auto-cached regardless of this
                flag. Defaults to ``False``.
            cache_refresh_interval: Optional duration after which the built-in
                cache is considered stale and skills are re-discovered on the
                next invocation. Like ``disable_caching``, this only affects the
                :class:`CachingSkillsSource` the provider builds internally
                (from a :class:`Skill` or a sequence of skills); it has no
                effect on a caller-supplied :class:`SkillsSource` (compose your
                own :class:`CachingSkillsSource` with a ``refresh_interval`` for
                those) and is ignored when ``disable_caching=True``. When
                ``None`` (the default), the built-in cache never expires.
            disable_load_skill_approval: When ``True``, the ``load_skill`` tool
                is registered with ``approval_mode="never_require"`` so it runs
                without approval.  Defaults to ``False`` (approval required).
                Only enable this for skills from a trusted source.
            disable_read_skill_resource_approval: When ``True``, the
                ``read_skill_resource`` tool is registered with
                ``approval_mode="never_require"`` so it runs without approval.
                Defaults to ``False`` (approval required).  Only enable this for
                skills from a trusted source.
            disable_run_skill_script_approval: When ``True``, the
                ``run_skill_script`` tool is registered with
                ``approval_mode="never_require"`` so it runs without approval.
                Defaults to ``False`` (approval required).  Only enable this for
                skills and scripts from a trusted source.
            source_id: Unique identifier for this provider instance.

        .. note::

            By default every skill tool requires approval. To approve them
            automatically, pass :meth:`read_only_tools_auto_approval_rule` or
            :meth:`all_tools_auto_approval_rule` to
            :class:`~gikard.ToolApprovalMiddleware`. Alternatively, for
            trusted skills, set one or more of
            ``disable_load_skill_approval``, ``disable_read_skill_resource_approval``,
            and ``disable_run_skill_script_approval`` to opt individual tools out
            of approval entirely (the auto-approval rules only apply to tools
            that still require approval). See
            ``samples/02-agents/skills/skills_auto_approval/skills_auto_approval.py``
            for the auto-approval pattern and
            ``samples/02-agents/skills/script_approval/script_approval.py`` for
            the manual approval loop.
        """
        super().__init__(source_id or self.DEFAULT_SOURCE_ID)

        if isinstance(source, (str, Path)):
            raise TypeError(
                f"SkillsProvider does not accept path strings directly. "
                f"Use SkillsProvider.from_paths({source!r}) for file-based skills."
            )

        if isinstance(source, Skill):
            # Single skill - wrap in simple list source without decorators
            class _SimpleListSource(SkillsSource):
                def __init__(self, skills: list[Skill]):
                    self._skills = skills
                async def get_skills(self, context: SkillsSourceContext) -> list[Skill]:
                    return self._skills
            source = _SimpleListSource([source])
        elif isinstance(source, SkillsSource):
            pass
        else:
            class _SimpleListSource2(SkillsSource):
                def __init__(self, skills: list[Skill]):
                    self._skills = skills
                async def get_skills(self, context: SkillsSourceContext) -> list[Skill]:
                    return self._skills
            source = _SimpleListSource2(list(source))
        self._source = source
        self._instruction_template = instruction_template
        self._disable_caching = disable_caching
        self._cache_refresh_interval = cache_refresh_interval
        self._disable_load_skill_approval = disable_load_skill_approval


    @classmethod
    def from_paths(
        cls: type[_TSkillsProvider],
        skill_paths: str | Path | Sequence[str | Path],
        *,
        resource_extensions: tuple[str, ...] | None = DEFAULT_RESOURCE_EXTENSIONS,
        script_extensions: tuple[str, ...] | None = DEFAULT_SCRIPT_EXTENSIONS,
        search_depth: int = DEFAULT_SEARCH_DEPTH,
        script_filter: Callable[[str, str], bool] | None = None,
        resource_filter: Callable[[str, str], bool] | None = None,
        instruction_template: str | None = None,
        disable_caching: bool = False,
        cache_refresh_interval: timedelta | None = None,
        disable_load_skill_approval: bool = False,
        source_id: str | None = None,
    ) -> _TSkillsProvider:
        """Create a provider from one or more file-based skill directories.

        Discovers skills from ``SKILL.md`` files in the given directories,
        deduplicates them, and creates the provider.

        Args:
            skill_paths: One or more directory paths to search for
                file-based skills.

        Keyword Args:

            resource_extensions: File extensions recognized as discoverable
                resources.  Defaults to
                ``(".md", ".json", ".yaml", ".yml", ".csv", ".xml", ".txt")``.
                ``None`` is treated the same as the default; pass an empty
                tuple to discover no resources.
            script_extensions: File extensions recognized as discoverable
                scripts.  Defaults to ``(".py",)``.  ``None`` is treated the
                same as the default; pass an empty tuple to disable script
                discovery entirely.
            search_depth: Maximum depth to search for script and resource
                files within each skill directory.  A value of ``1`` searches
                only the skill root; ``2`` (the default) searches the root
                plus one level of subdirectories.  Must be >= 1.
            script_filter: Optional predicate ``(skill_name, relative_file_path) -> bool``
                that filters discovered script files.  Returns ``True`` to
                include or ``False`` to exclude.  When ``None``, all scripts
                matching allowed extensions are included.
            resource_filter: Optional predicate ``(skill_name, relative_file_path) -> bool``
                that filters discovered resource files.  Returns ``True`` to
                include or ``False`` to exclude.  When ``None``, all resources
                matching allowed extensions are included.
            instruction_template: Custom system-prompt template for
                advertising skills.  Must contain a ``{skills}`` placeholder.
                Uses a built-in template when ``None``.
            disable_caching: When ``True``, the file-discovery source is not
                wrapped in a :class:`CachingSkillsSource`, so skills are
                re-discovered on every invocation. Defaults to ``False``.
            cache_refresh_interval: Optional duration after which the file-
                discovery cache is considered stale and skills are re-discovered
                on the next invocation. When ``None`` (the default) the cache
                never expires; ignored when ``disable_caching=True``.
            disable_load_skill_approval: When ``True``, the ``load_skill`` tool
                runs without approval.  Defaults to ``False``.  Only enable this
                for skills from a trusted source.
            disable_read_skill_resource_approval: When ``True``, the
                ``read_skill_resource`` tool runs without approval.  Defaults to
                ``False``.  Only enable this for skills from a trusted source.
            disable_run_skill_script_approval: When ``True``, the
                ``run_skill_script`` tool runs without approval.  Defaults to
                ``False``.  Only enable this for skills and scripts from a
                trusted source.
            source_id: Unique identifier for this provider instance.

        Returns:
            A configured :class:`SkillsProvider`.

        .. note::

            By default every skill tool requires approval. To approve them
            automatically, pass :meth:`read_only_tools_auto_approval_rule` or
            :meth:`all_tools_auto_approval_rule` to
            :class:`~gikard.ToolApprovalMiddleware`. Alternatively, for
            trusted skills, set one or more of ``disable_load_skill_approval``,
            ``disable_read_skill_resource_approval``, and
            ``disable_run_skill_script_approval`` to opt individual tools out of
            approval entirely.
        """
        file_source: SkillsSource = FileSkillsSource(
            skill_paths,
            resource_extensions=resource_extensions,
            script_extensions=script_extensions,
            search_depth=search_depth,
            script_filter=script_filter,
            resource_filter=resource_filter,
        )
        # File discovery source used directly (decorators removed)
        source = file_source
        forwarded_kwargs: dict[str, Any] = {}
        if disable_load_skill_approval:
            forwarded_kwargs["disable_load_skill_approval"] = True
        return cls(
            source,
            instruction_template=instruction_template,
            disable_caching=disable_caching,
            source_id=source_id,
            **forwarded_kwargs,
        )

    @staticmethod
    def _create_instructions(
        prompt_template: str | None,
        skills: Sequence[Skill],
    ) -> str | None:
        """Create the system-prompt text that advertises available skills.

        Generates an XML list of ``<skill>`` elements (sorted by name) and
        inserts it into *prompt_template* at the ``{skills}`` placeholder.
        Script-runner instructions are inserted at the
        ``{runner_instructions}`` placeholder and resource-reading
        instructions at the ``{resource_instructions}`` placeholder.

        Args:
            prompt_template: Custom template string with ``{skills}`` and
                optional ``{runner_instructions}`` and ``{resource_instructions}``
                placeholders, or ``None`` to use the built-in default.
            skills: Registered skills.

        Returns:
            The formatted instruction string, or ``None`` when *skills* is empty.

        Raises:
            ValueError: If *prompt_template* is not a valid format string
                (e.g. missing ``{skills}`` placeholder).
        """
        runner_instructions = SCRIPT_RUNNER_INSTRUCTIONS
        resource_instructions = RESOURCE_INSTRUCTIONS
        template = DEFAULT_SKILLS_INSTRUCTION_PROMPT

        if prompt_template is not None:
            # Validate that the custom template contains a valid {skills} placeholder
            try:
                result = prompt_template.format(
                    skills="__PROBE__",
                    runner_instructions="__EXEC_PROBE__",
                    resource_instructions="__RES_PROBE__",
                )
            except (KeyError, IndexError, ValueError) as exc:
                raise ValueError(
                    "The provided instruction_template is not a valid format string. "
                    "It must contain a '{skills}' placeholder and escape any literal"  # ruff:ignore[missing-f-string-syntax]
                    " '{' or '}' "
                    "by doubling them ('{{' or '}}')."
                ) from exc
            if "__PROBE__" not in result:
                raise ValueError(
                    "The provided instruction_template must contain a '{skills}' placeholder."  # ruff:ignore[missing-f-string-syntax]
                )
            template = prompt_template

        if not skills:
            return None

        lines: list[str] = []
        # Sort by name for deterministic output
        for skill in sorted(skills, key=lambda s: s.frontmatter.name):
            lines.append("  <skill>")
            lines.append(f"    <name>{xml_escape(skill.frontmatter.name, quote=False)}</name>")
            lines.append(f"    <description>{xml_escape(skill.frontmatter.description, quote=False)}</description>")
            lines.append("  </skill>")

        return template.format(
            skills="\n".join(lines),
            runner_instructions=runner_instructions or "",
            resource_instructions=resource_instructions or "",
        )

    async def _create_context(
        self, source_context: SkillsSourceContext
    ) -> tuple[Sequence[Skill], str | None, list[FunctionTool]]:
        """Build skills, instructions, and tools from the source.

        Queries the source for skills and constructs the instruction prompt
        and tool definitions.  Caching of the skills list is handled by the
        source pipeline (see :class:`CachingSkillsSource`), so this method
        rebuilds instructions and tools from the (possibly cached) skills on
        every call.

        Args:
            source_context: Contextual information about the agent and session
                requesting skills, forwarded to the source pipeline.

        Returns:
            A tuple of ``(skills, instructions, tools)``.
        """
        skills = await self._source.get_skills(source_context)

        if not skills:
            return skills, None, []

        instructions = self._create_instructions(
            prompt_template=self._instruction_template,
            skills=skills,
        )

        tools = self._create_tools(skills=skills)

        return skills, instructions, tools

    async def before_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Inject skill instructions and tools into the session context.

        Called by the framework before the agent runs.  Loads skills from the
        configured source (built-in file/in-memory sources are cached by the
        source pipeline unless ``disable_caching=True``; a caller-supplied
        source is queried as its own pipeline dictates) and builds the
        instruction prompt and tool definitions.  When at least one skill is
        registered, appends the skill-list system prompt and the ``load_skill``
        / ``read_skill_resource`` tools to *context*.

        When any registered skill defines one or more scripts (file-based or
        code-based), the system prompt also includes script-runner
        instructions (embedded via the ``{runner_instructions}`` placeholder),
        and the ``run_skill_script`` tool is included alongside the base tools.

        Args:
            agent: The agent instance about to run.
            session: The current agent session.
            context: Session context to extend with instructions and tools.
            state: Mutable per-run state dictionary (unused by this provider).
        """
        mark_feature_used(FeatureIndex.CORE_SKILLS_PROVIDER)
        source_context = SkillsSourceContext(agent=agent, session=session)
        skills, instructions, tools = await self._create_context(source_context)

        if not skills:
            return

        context.extend_instructions(self.source_id, instructions)  # type: ignore[arg-type]
        context.extend_tools(self.source_id, tools)

    @staticmethod
    def _approval_mode(approval_disabled: bool) -> ApprovalMode:
        """Return the ``approval_mode`` for a tool given its disable flag.

        Args:
            approval_disabled: When ``True``, the tool runs without approval.

        Returns:
            ``"never_require"`` when approval is disabled, otherwise
            ``"always_require"``.
        """
        return "never_require" if approval_disabled else "always_require"

    def _create_tools(
        self,
        skills: Sequence[Skill],
    ) -> list[FunctionTool]:
        """Create the tool definitions for skill interaction. Only load_skill."""

        async def _load(skill_name: str) -> dict[str, Any] | str:
            return await self._load_skill(skills, skill_name)

        return [
            FunctionTool(
                name=self.LOAD_SKILL_TOOL_NAME,
                description="Loads skill metadata: location, resource list, script list and instructions (SKILL.md body).",
                func=_load,
                approval_mode=self._approval_mode(self._disable_load_skill_approval),
                input_model={
                    "type": "object",
                    "properties": {
                        "skill_name": {"type": "string", "description": "The name of the skill to load."},
                    },
                    "required": ["skill_name"],
                },
            ),
        ]

    @staticmethod
    def _find_skill(skills: Sequence[Skill], name: str) -> Skill | None:
        """Find a skill by name (case-insensitive linear scan)."""
        name_lower = name.lower()
        return next((s for s in skills if s.frontmatter.name.lower() == name_lower), None)

    async def _load_skill(self, skills: Sequence[Skill], skill_name: str) -> dict[str, Any] | str:
        """Return skill location, resource list, script list and instructions.

        Returns a dict with ``location``, ``resources``, ``scripts`` and
        ``instructions`` (the SKILL.md body) for the named skill, or a
        user-facing error message if not found.
        """
        if not skill_name or not skill_name.strip():
            return "Error: Skill name cannot be empty."

        skill = self._find_skill(skills, skill_name)
        if skill is None:
            return f"Error: Skill '{skill_name}' not found."

        logger.info("Loading skill: %s", skill_name)

        location = getattr(skill, "path", "")
        if not location:
            location = getattr(skill, "_skill_md_uri", "") or getattr(skill, "_skill_root_uri", "") or ""
        location = Path(location).resolve()

        resources: list[str] = []
        scripts: list[str] = []
        if hasattr(skill, "_resources"):
            try:
                res = getattr(skill, "_resources")
                resources = [r.name for r in res] if res else []
            except Exception:
                resources = []
        if hasattr(skill, "_scripts"):
            try:
                scr = getattr(skill, "_scripts")
                if scr and len(scr) > 0 and isinstance(scr[0], str):
                    scripts = list(scr)
                else:
                    scripts = [getattr(s, "name", str(s)) for s in scr] if scr else []
            except Exception:
                scripts = []

        try:
            instructions = await skill.get_content()
        except Exception:
            logger.exception("Failed to load content for skill '%s'", skill_name)
            instructions = ""

        result: dict[str, Any] = {
            "name": skill.frontmatter.name,
            "description": skill.frontmatter.description,
            "location": location,
            "resources": sorted(resources),
            "scripts": sorted(scripts),
            "instructions": instructions,
        }
        return result


@dataclass(frozen=True)
class SkillsSourceContext:
    """Contextual information passed to a :class:`SkillsSource` when retrieving skills.

    Exposes the invoking *agent* and, when available, the current *session* so
    that skill sources and decorators can make context-aware decisions such as
    per-agent filtering (see :class:`FilteringSkillsSource`) or per-key cache
    isolation (see :class:`CachingSkillsSource`).

    The context is constructed by :class:`SkillsProvider` from the invoking
    agent run and flows through every source and decorator in the pipeline.

    Attributes:
        agent: The agent requesting skills.
        session: The session associated with the agent invocation, if any.
    """

    agent: SupportsAgentRun
    session: AgentSession | None = None


class SkillsSource(ABC):
    """Abstract base class for skill sources.

    A skill source discovers and returns :class:`Skill` instances from a
    particular origin.  The framework calls :meth:`get_skills` to obtain
    the available skills; implementations decide *where* and *how* skills
    are discovered (filesystem, memory, network, etc.).

    Subclass this to create custom skill sources.

    Security considerations:
        A skill source is a trust boundary. The skills it returns — their
        names, descriptions, instructions, and any scripts or resources — are
        injected into the agent's context and tool surface, and may be
        executed (for sources that support script execution). Skills only
        reach the agent when a source is explicitly registered, so this is
        opt-in. Sources that read from a remote or third-party origin (e.g. a
        remote server, a shared filesystem, or
        a database) can be compromised or adversarial, and may return skill
        content designed to manipulate the agent (indirect prompt injection) or
        to exfiltrate data through instructions or scripts the agent is induced
        to run. Only register skill sources for origins you trust, and evaluate
        the content they can return before enabling them in production.
    """

    @abstractmethod
    async def get_skills(self, context: SkillsSourceContext) -> list[Skill]:
        """Discover and return all skills from this source.

        Args:
            context: Contextual information about the agent and session
                requesting skills.

        Returns:
            A list of :class:`Skill` instances discovered by this source.
        """
        ...


class FileSkillsSource(SkillsSource):
    """Skill source that discovers skills from filesystem ``SKILL.md`` files.

    Recursively scans the configured *skill_paths* directories for
    ``SKILL.md`` files (up to 2 levels deep), parses their YAML frontmatter,
    and discovers associated resource and script files by recursively scanning
    each skill directory up to the configured *search_depth*.

    By default, the scan depth is 2 (root + one level of subdirectories).
    Use *script_filter* and *resource_filter* predicates to control which
    discovered files are included.

    Security: file-based metadata is XML-escaped before prompt injection,
    and resource reads are guarded against path traversal and symlink escape.
    Only use skills from trusted sources.

    Examples:
        Basic usage:

        .. code-block:: python

            source = FileSkillsSource(skill_paths="./skills")
            # `context` is normally supplied by SkillsProvider at runtime.
            context = SkillsSourceContext(agent=agent)
            skills = await source.get_skills(context)

        With a script runner and filter predicates:

        .. code-block:: python

            source = FileSkillsSource(
                skill_paths=["./skills", "./more-skills"],
                script_runner=my_runner,
                search_depth=3,
                script_filter=lambda name, path: not path.startswith("tests/"),
            )
    """

    def __init__(
        self,
        skill_paths: str | Path | Sequence[str | Path],
        *,
        resource_extensions: tuple[str, ...] | None = DEFAULT_RESOURCE_EXTENSIONS,
        script_extensions: tuple[str, ...] | None = DEFAULT_SCRIPT_EXTENSIONS,
        search_depth: int = DEFAULT_SEARCH_DEPTH,
        script_filter: Callable[[str, str], bool] | None = None,
        resource_filter: Callable[[str, str], bool] | None = None,
    ) -> None:
        """Initialize a FileSkillsSource.

        Args:
            skill_paths: One or more directory paths to search for file-based
                skills.  Each path may point to an individual skill directory
                (containing ``SKILL.md``) or to a parent that contains skill
                subdirectories.

        Keyword Args:
            script_runner: Strategy for running file-based skill scripts.
                When ``None``, discovered scripts are included but not
                executable (the provider will raise an error if execution
                is attempted without a runner).
            resource_extensions: File extensions recognized as discoverable
                resources.  Defaults to
                ``(".md", ".json", ".yaml", ".yml", ".csv", ".xml", ".txt")``.
                ``None`` is treated the same as the default; pass an empty
                tuple to discover no resources.
            script_extensions: File extensions recognized as discoverable
                scripts.  Defaults to ``(".py",)``.  ``None`` is treated the
                same as the default; pass an empty tuple to disable script
                discovery entirely (files that would otherwise be scripts are
                then only discoverable as resources); this is used to serve
                untrusted skills whose bundled scripts must never be exposed as
                runnable.
            search_depth: Maximum depth to search for script and resource
                files within each skill directory.  A value of ``1`` searches
                only the skill root; ``2`` (the default) searches the root
                plus one level of subdirectories.  Must be >= 1.
            script_filter: Optional predicate ``(skill_name, relative_file_path) -> bool``
                that filters discovered script files.  Returns ``True`` to
                include or ``False`` to exclude.  When ``None``, all scripts
                matching allowed extensions are included.
            resource_filter: Optional predicate ``(skill_name, relative_file_path) -> bool``
                that filters discovered resource files.  Returns ``True`` to
                include or ``False`` to exclude.  When ``None``, all resources
                matching allowed extensions are included.

        Raises:
            ValueError: If *search_depth* is less than 1.
        """
        if isinstance(skill_paths, (str, Path)):
            self._skill_paths: list[str] = [str(skill_paths)]
        else:
            self._skill_paths = [str(p) for p in skill_paths]

        self._resource_extensions = (
            resource_extensions if resource_extensions is not None else DEFAULT_RESOURCE_EXTENSIONS
        )
        self._script_extensions = script_extensions if script_extensions is not None else DEFAULT_SCRIPT_EXTENSIONS

        if search_depth < 1:
            raise ValueError(f"search_depth must be >= 1, got {search_depth}")
        self._search_depth: int = search_depth
        self._script_filter = script_filter
        self._resource_filter = resource_filter

    async def get_skills(self, context: SkillsSourceContext) -> list[Skill]:
        """Discover and return all file-based skills from configured paths.

        Scans directories for ``SKILL.md`` files, parses their frontmatter,
        discovers resource and script files, and returns populated
        :class:`Skill` instances.

        Args:
            context: Contextual information about the agent and session
                requesting skills. Accepted for the source contract; this
                source discovers the same skills regardless of context.

        Returns:
            A list of discovered file-based skills.
        """
        mark_feature_used(FeatureIndex.CORE_FILE_SKILLS_SOURCE)
        skills: dict[str, FileSkill] = {}

        discovered = FileSkillsSource._discover_skill_directories(self._skill_paths)
        logger.info("Discovered %d potential skills", len(discovered))

        for skill_path in discovered:
            parsed = FileSkillsSource._read_and_parse_skill_file(skill_path)
            if parsed is None:
                continue

            frontmatter, content = parsed

            if frontmatter.name in skills:
                logger.warning(
                    "Duplicate skill name '%s': skill from '%s' skipped in favor of existing skill",
                    frontmatter.name,
                    skill_path,
                )
                continue

            # Discover file-based resources
            resources: list[SkillResource] = []
            for rn in self._discover_resource_files(skill_path, frontmatter.name):
                resource_full_path = FileSkillsSource._get_validated_resource_path(skill_path, rn)
                resources.append(FileSkillResource(name=rn, full_path=resource_full_path))

            # Discover file-based scripts (as relative paths)
            scripts: list[str] = []
            for sn in self._discover_script_files(skill_path, frontmatter.name):
                scripts.append(sn)

            file_skill = FileSkill(
                frontmatter=frontmatter,
                content=content,
                path=skill_path,
                resources=resources,
                scripts=scripts,
            )

            skills[file_skill.frontmatter.name] = file_skill
            logger.info("Loaded skill: %s", file_skill.frontmatter.name)

        logger.info("Successfully loaded %d skills", len(skills))
        return list(skills.values())

    @staticmethod
    def _normalize_resource_path(path: str) -> str:
        """Normalize a relative resource path to a canonical forward-slash form.

        Converts backslashes to forward slashes and strips leading ``./``
        prefixes so that ``./refs/doc.md`` and ``refs/doc.md`` resolve
        identically.

        Args:
            path: The relative path to normalize.

        Returns:
            A clean forward-slash-separated path string.
        """
        return PurePosixPath(path.replace("\\", "/")).as_posix()

    @staticmethod
    def _is_path_within_directory(path: str, directory: str) -> bool:
        """Return whether *path* resides under *directory*.

        Comparison uses :meth:`pathlib.Path.is_relative_to`, which respects
        per-platform case-sensitivity rules.

        Args:
            path: Absolute path to check.
            directory: Directory that must be an ancestor of *path*.

        Returns:
            ``True`` if *path* is a descendant of *directory*.
        """
        try:
            return Path(path).is_relative_to(directory)
        except (ValueError, OSError):
            return False

    @staticmethod
    def _has_link_or_reparse_point_in_path(path: str, directory: str) -> bool:
        """Detect links or reparse points in the portion of *path* below *directory*.

        Only segments below *directory* are inspected; the directory itself
        and anything above it are not checked.

        **Precondition:** *path* must be a descendant of *directory*.
        Call :meth:`_is_path_within_directory` first to verify containment.

        Args:
            path: Absolute path to inspect.
            directory: Root directory; segments above it are not checked.

        Returns:
            ``True`` if any segment below *directory* is a symbolic link,
            junction, other reparse point, or cannot be safely inspected.

        Raises:
            ValueError: If *path* is not relative to *directory*.
        """
        dir_path = Path(directory)
        try:
            relative = Path(path).relative_to(dir_path)
        except ValueError as exc:
            raise ValueError(f"path {path!r} does not start with directory {directory!r}") from exc

        current = dir_path
        for part in relative.parts:
            current = current / part
            try:
                is_link = is_link_or_reparse_point(current)
            except OSError:
                return True
            if is_link:
                return True
        return False

    def _discover_resource_files(
        self,
        skill_dir_path: str,
        skill_name: str,
    ) -> list[str]:
        """Recursively scan a skill directory for resource files matching configured extensions.

        Scans the skill directory up to the configured search depth for files
        whose extension matches the allowed resource extensions, excluding
        ``SKILL.md`` itself.  Each candidate is validated against path-traversal
        and symlink-escape checks; unsafe files are skipped with a warning.
        If a ``resource_filter`` predicate is configured, files that do not
        satisfy it are excluded.

        Args:
            skill_dir_path: Absolute path to the skill directory to scan.
            skill_name: The skill name (from frontmatter) for filter context.

        Returns:
            Sorted relative resource paths (forward-slash-separated) for every
            discovered file that passes security and filter checks.
        """
        skill_dir = Path(skill_dir_path).absolute()
        root_directory_path = str(skill_dir)
        resources: list[str] = []
        normalized_extensions = {e.lower() for e in self._resource_extensions}

        self._scan_directory_for_resources(
            target_dir=skill_dir,
            skill_dir=skill_dir,
            root_directory_path=root_directory_path,
            skill_name=skill_name,
            normalized_extensions=normalized_extensions,
            resources=resources,
            current_depth=1,
        )

        resources.sort()
        return resources

    def _scan_directory_for_resources(
        self,
        target_dir: Path,
        skill_dir: Path,
        root_directory_path: str,
        skill_name: str,
        normalized_extensions: set[str],
        resources: list[str],
        current_depth: int,
    ) -> None:
        """Recursively scan a directory for resource files.

        Args:
            target_dir: The directory to scan at this level.
            skill_dir: The skill root directory (for relative path computation).
            root_directory_path: String form of the skill root (for security checks).
            skill_name: Skill name for filter predicate context.
            normalized_extensions: Lowercased allowed extensions.
            resources: Accumulator list for discovered relative paths.
            current_depth: Current recursion depth (starts at 1).
        """
        if current_depth > self._search_depth:
            return

        is_root = target_dir == skill_dir

        # Directory-level symlink check for non-root directories
        if not is_root:
            resolved_target = str(Path(os.path.normpath(target_dir)).absolute())
            if not FileSkillsSource._is_path_within_directory(resolved_target, root_directory_path):
                logger.warning(
                    "Skipping resource directory '%s': resolves outside skill directory '%s'",
                    target_dir,
                    root_directory_path,
                )
                return

            if FileSkillsSource._has_link_or_reparse_point_in_path(resolved_target, root_directory_path):
                logger.warning(
                    "Skipping resource directory '%s': symbolic link or reparse point detected in path under "
                    "skill directory '%s'",
                    target_dir,
                    root_directory_path,
                )
                return

        try:
            entries = list(target_dir.iterdir())
        except OSError:
            logger.warning(
                "Failed to list resource directory '%s' in skill directory '%s'; skipping.",
                target_dir,
                root_directory_path,
            )
            return

        subdirectories: list[Path] = []

        for entry in entries:
            if entry.is_dir():
                subdirectories.append(entry)
                continue

            if not entry.is_file():
                continue

            if entry.name.upper() == SKILL_FILE_NAME.upper():
                continue

            if entry.suffix.lower() not in normalized_extensions:
                continue

            resource_full_path = str(Path(os.path.normpath(entry)).absolute())

            # Containment check: file must resolve within the skill directory
            if not FileSkillsSource._is_path_within_directory(resource_full_path, root_directory_path):
                logger.warning(
                    "Skipping resource '%s': resolves outside skill directory '%s'",
                    entry,
                    root_directory_path,
                )
                continue

            if FileSkillsSource._has_link_or_reparse_point_in_path(resource_full_path, root_directory_path):
                logger.warning(
                    "Skipping resource '%s': symbolic link or reparse point detected in path under "
                    "skill directory '%s'",
                    entry,
                    root_directory_path,
                )
                continue

            rel_path = FileSkillsSource._normalize_resource_path(str(entry.relative_to(skill_dir)))

            # Apply user-provided filter predicate
            if self._resource_filter is not None and not self._resource_filter(skill_name, rel_path):
                continue

            resources.append(rel_path)

        # Recurse into subdirectories if within depth limit.
        # Subdirectories that contain their own SKILL.md are NOT skipped: a nested
        # SKILL.md is not an independent skill (see _discover_skill_directories), so
        # its contents belong to this skill.
        if current_depth < self._search_depth:
            for subdir in subdirectories:
                self._scan_directory_for_resources(
                    target_dir=subdir,
                    skill_dir=skill_dir,
                    root_directory_path=root_directory_path,
                    skill_name=skill_name,
                    normalized_extensions=normalized_extensions,
                    resources=resources,
                    current_depth=current_depth + 1,
                )

    def _discover_script_files(
        self,
        skill_dir_path: str,
        skill_name: str,
    ) -> list[str]:
        """Recursively scan a skill directory for script files matching configured extensions.

        Scans the skill directory up to the configured search depth for files
        whose extension matches the allowed script extensions.  Each candidate
        is validated against path-traversal and symlink-escape checks; unsafe
        files are skipped with a warning.  If a ``script_filter`` predicate
        is configured, files that do not satisfy it are excluded.

        Args:
            skill_dir_path: Absolute path to the skill directory to scan.
            skill_name: The skill name (from frontmatter) for filter context.

        Returns:
            Sorted relative script paths (forward-slash-separated) for every
            discovered file that passes security and filter checks.
        """
        skill_dir = Path(skill_dir_path).absolute()
        root_directory_path = str(skill_dir)
        scripts: list[str] = []
        normalized_extensions = {e.lower() for e in self._script_extensions}

        self._scan_directory_for_scripts(
            target_dir=skill_dir,
            skill_dir=skill_dir,
            root_directory_path=root_directory_path,
            skill_name=skill_name,
            normalized_extensions=normalized_extensions,
            scripts=scripts,
            current_depth=1,
        )

        scripts.sort()
        return scripts

    def _scan_directory_for_scripts(
        self,
        target_dir: Path,
        skill_dir: Path,
        root_directory_path: str,
        skill_name: str,
        normalized_extensions: set[str],
        scripts: list[str],
        current_depth: int,
    ) -> None:
        """Recursively scan a directory for script files.

        Args:
            target_dir: The directory to scan at this level.
            skill_dir: The skill root directory (for relative path computation).
            root_directory_path: String form of the skill root (for security checks).
            skill_name: Skill name for filter predicate context.
            normalized_extensions: Lowercased allowed extensions.
            scripts: Accumulator list for discovered relative paths.
            current_depth: Current recursion depth (starts at 1).
        """
        if current_depth > self._search_depth:
            return

        is_root = target_dir == skill_dir

        # Directory-level symlink check for non-root directories
        if not is_root:
            resolved_target = str(Path(os.path.normpath(target_dir)).absolute())
            if not FileSkillsSource._is_path_within_directory(resolved_target, root_directory_path):
                logger.warning(
                    "Skipping script directory '%s': resolves outside skill directory '%s'",
                    target_dir,
                    root_directory_path,
                )
                return

            if FileSkillsSource._has_link_or_reparse_point_in_path(resolved_target, root_directory_path):
                logger.warning(
                    "Skipping script directory '%s': symbolic link or reparse point detected in path under "
                    "skill directory '%s'",
                    target_dir,
                    root_directory_path,
                )
                return

        try:
            entries = list(target_dir.iterdir())
        except OSError:
            logger.warning(
                "Failed to list script directory '%s' in skill directory '%s'; skipping.",
                target_dir,
                root_directory_path,
            )
            return

        subdirectories: list[Path] = []

        for entry in entries:
            if entry.is_dir():
                subdirectories.append(entry)
                continue

            if not entry.is_file():
                continue

            if entry.suffix.lower() not in normalized_extensions:
                continue

            script_full_path = str(Path(os.path.normpath(entry)).absolute())

            # Containment check: file must resolve within the skill directory
            if not FileSkillsSource._is_path_within_directory(script_full_path, root_directory_path):
                logger.warning(
                    "Skipping script '%s': resolves outside skill directory '%s'",
                    entry,
                    root_directory_path,
                )
                continue

            if FileSkillsSource._has_link_or_reparse_point_in_path(script_full_path, root_directory_path):
                logger.warning(
                    "Skipping script '%s': symbolic link or reparse point detected in path under skill directory '%s'",
                    entry,
                    root_directory_path,
                )
                continue

            rel_path = FileSkillsSource._normalize_resource_path(str(entry.relative_to(skill_dir)))

            # Apply user-provided filter predicate
            if self._script_filter is not None and not self._script_filter(skill_name, rel_path):
                continue

            scripts.append(rel_path)

        # Recurse into subdirectories if within depth limit.
        # Subdirectories that contain their own SKILL.md are NOT skipped: a nested
        # SKILL.md is not an independent skill (see _discover_skill_directories), so
        # its contents belong to this skill.
        if current_depth < self._search_depth:
            for subdir in subdirectories:
                self._scan_directory_for_scripts(
                    target_dir=subdir,
                    skill_dir=skill_dir,
                    root_directory_path=root_directory_path,
                    skill_name=skill_name,
                    normalized_extensions=normalized_extensions,
                    scripts=scripts,
                    current_depth=current_depth + 1,
                )

    @staticmethod
    def _get_validated_resource_path(skill_dir: str, resource_name: str) -> str:
        """Resolve and validate a resource file path within a skill directory.

        Normalizes *resource_name*, resolves it against *skill_dir*, and
        validates that the result stays within the skill directory and does
        not traverse any symlinks.

        Args:
            skill_dir: Absolute path to the owning skill directory.
            resource_name: Relative path of the resource within the skill directory.

        Returns:
            The validated absolute path to the resource file.

        Raises:
            ValueError: If *skill_dir* is not an absolute path, the resolved path
                escapes the skill directory, the file does not exist, or a symlink
                is detected in the path.
        """
        if not os.path.isabs(skill_dir):
            raise ValueError(f"skill_dir must be an absolute path, got: '{skill_dir}'")

        resource_name = FileSkillsSource._normalize_resource_path(resource_name)

        resource_full_path = os.path.normpath(Path(skill_dir) / resource_name)
        root_directory_path = os.path.normpath(skill_dir)

        if not FileSkillsSource._is_path_within_directory(resource_full_path, root_directory_path):
            raise ValueError(f"Resource file '{resource_name}' references a path outside the skill directory.")

        if not Path(resource_full_path).is_file():
            raise ValueError(f"Resource file '{resource_name}' not found in skill directory '{skill_dir}'.")

        if FileSkillsSource._has_link_or_reparse_point_in_path(resource_full_path, root_directory_path):
            raise ValueError(
                f"Resource file '{resource_name}' has a symbolic link or reparse point in its path; "
                "links and reparse points are not allowed."
            )

        return resource_full_path

    @staticmethod
    def _validate_skill_metadata(
        name: str | None,
        description: str | None,
        source: str,
        compatibility: str | None = None,
    ) -> str | None:
        """Validate a skill's name, description, and compatibility against naming rules.

        Enforces length limits, character-set restrictions, and non-emptiness
        for both file-based and code-defined skills.

        Args:
            name: Skill name to validate.
            description: Skill description to validate.
            source: Human-readable label for diagnostics (e.g. a file path
                or ``"code skill"``).
            compatibility: Optional compatibility value to validate.

        Returns:
            A diagnostic error string if validation fails, or ``None`` if valid.
        """
        if not name or not name.strip():
            return f"Skill from '{source}' is missing a name."

        if len(name) > MAX_NAME_LENGTH or not VALID_NAME_RE.match(name):
            return (
                f"Skill from '{source}' has an invalid name '{name}': Must be {MAX_NAME_LENGTH} characters or fewer, "
                "using only lowercase letters, numbers, and hyphens, and must not start or end with a hyphen "
                "or contain consecutive hyphens."
            )

        if not description or not description.strip():
            return f"Skill '{name}' from '{source}' is missing a description."

        if len(description) > MAX_DESCRIPTION_LENGTH:
            return (
                f"Skill '{name}' from '{source}' has an invalid description: "
                f"Must be {MAX_DESCRIPTION_LENGTH} characters or fewer."
            )

        if compatibility is not None and len(compatibility) > MAX_COMPATIBILITY_LENGTH:
            return (
                f"Skill '{name}' from '{source}' has an invalid compatibility: "
                f"Must be {MAX_COMPATIBILITY_LENGTH} characters or fewer."
            )

        return None

    @staticmethod
    def _extract_frontmatter(
        content: str,
        skill_file_path: str,
    ) -> SkillFrontmatter | None:
        """Extract and validate YAML frontmatter from a SKILL.md file.

        Parses the ``---``-delimited frontmatter block for all
        `agentskills.io specification <https://agentskills.io/specification>`_
        fields: ``name``, ``description``, ``license``, ``compatibility``,
        ``allowed-tools``, and ``metadata``.

        Args:
            content: Raw text content of the SKILL.md file.
            skill_file_path: Path to the file (used in diagnostic messages only).

        Returns:
            A :class:`SkillFrontmatter` on success, or ``None`` if the
            frontmatter is missing, malformed, or fails validation.
        """
        match = FRONTMATTER_RE.search(content)
        if not match:
            logger.error("SKILL.md at '%s' does not contain valid YAML frontmatter delimited by '---'", skill_file_path)
            return None

        yaml_content = match.group(1).strip()
        name: str | None = None
        description: str | None = None
        license_value: str | None = None
        compatibility: str | None = None
        allowed_tools: str | None = None

        for kv_match in YAML_KV_RE.finditer(yaml_content):
            key = kv_match.group(1)
            value = (
                kv_match.group(2) if kv_match.group(2) is not None else _parse_yaml_scalar_value(yaml_content, kv_match)
            )

            key_lower = key.lower()
            if key_lower == "name":
                name = value
            elif key_lower == "description":
                description = value
            elif key_lower == "license":
                license_value = value
            elif key_lower == "compatibility":
                compatibility = value
            elif key_lower == "allowed-tools":
                allowed_tools = value

        # Parse metadata block (indented key-value pairs under "metadata:").
        metadata: dict[str, str] | None = None
        metadata_match = YAML_METADATA_BLOCK_RE.search(yaml_content)
        if metadata_match:
            metadata = {}
            for kv_match in YAML_INDENTED_KV_RE.finditer(metadata_match.group(1)):
                mk = kv_match.group(1)
                mv = kv_match.group(2) if kv_match.group(2) is not None else kv_match.group(3)
                metadata[mk] = mv

        error = FileSkillsSource._validate_skill_metadata(name, description, skill_file_path, compatibility)
        if error:
            logger.error(error)
            return None

        # name and description are guaranteed non-None after validation;
        # SkillFrontmatter re-validates as a defense-in-depth invariant.
        return SkillFrontmatter(
            name=cast(str, name),
            description=cast(str, description),
            license=license_value,
            compatibility=compatibility,
            allowed_tools=allowed_tools,
            metadata=metadata,
        )

    @staticmethod
    def _read_and_parse_skill_file(
        skill_dir_path: str,
    ) -> tuple[SkillFrontmatter, str] | None:
        """Read and parse the SKILL.md file in *skill_dir_path*.

        Args:
            skill_dir_path: Absolute path to the directory containing ``SKILL.md``.

        Returns:
            A ``(frontmatter, content)`` tuple where *content* is the
            full raw file text, or ``None`` if the file cannot be read or
            its frontmatter is invalid.
        """
        skill_file = Path(skill_dir_path) / SKILL_FILE_NAME

        try:
            content = skill_file.read_text(encoding="utf-8")
        except OSError:
            logger.error("Failed to read SKILL.md at '%s'", skill_file)
            return None

        frontmatter = FileSkillsSource._extract_frontmatter(content, str(skill_file))
        if frontmatter is None:
            return None

        dir_name = Path(skill_dir_path).name
        if frontmatter.name != dir_name:
            logger.error(
                "SKILL.md at '%s' has frontmatter name '%s' that does not match the directory name '%s'; skipping.",
                skill_file,
                frontmatter.name,
                dir_name,
            )
            return None

        return frontmatter, content

    @staticmethod
    def _discover_skill_directories(skill_paths: Sequence[str]) -> list[str]:
        """Return absolute paths of all directories that contain a ``SKILL.md`` file.

        Recursively searches each root path up to :data:`MAX_SEARCH_DEPTH`. Once a
        ``SKILL.md`` is found in a directory, that directory is the skill root and the
        search does not descend into its subdirectories: everything beneath a skill
        boundary is part of that skill, not an independent skill root.

        Discovery fails closed on links: any entry below a configured root that is a
        symbolic link, junction, other reparse point, or cannot be inspected is skipped
        and never adopted as a skill root, and a directory whose ``SKILL.md`` is itself
        such a link is skipped too. Without this check a link below a root would become
        the skill root, and because every later guard treats the skill root as the trust
        boundary and only inspects segments below it, the link itself would never be
        inspected. The configured root paths are not checked: the host chose them
        explicitly, so they define the trust boundary rather than sit inside it.

        Args:
            skill_paths: Root directory paths to search.

        Returns:
            Absolute paths to directories containing ``SKILL.md``.
        """
        discovered: list[str] = []

        def _is_unsafe_link(path: Path) -> bool:
            try:
                return is_link_or_reparse_point(path)
            except OSError:
                return True

        def _search(directory: str, current_depth: int) -> None:
            dir_path = Path(directory)
            skill_file = dir_path / SKILL_FILE_NAME
            if skill_file.is_file():
                # This directory is a skill root. Subdirectories are part of this
                # skill and must not be treated as independent skill roots.
                if _is_unsafe_link(skill_file):
                    logger.warning(
                        "Skipping skill directory '%s': '%s' is a symbolic link or reparse point, "
                        "or could not be inspected",
                        directory,
                        SKILL_FILE_NAME,
                    )
                    return
                discovered.append(str(dir_path.absolute()))
                return

            if current_depth >= MAX_SEARCH_DEPTH:
                return

            try:
                entries = list(dir_path.iterdir())
            except OSError:
                return

            for entry in entries:
                if _is_unsafe_link(entry):
                    logger.warning(
                        "Skipping discovery entry '%s': symbolic link or reparse point detected, "
                        "or the entry could not be inspected",
                        entry,
                    )
                    continue
                if entry.is_dir():
                    _search(str(entry), current_depth + 1)

        for root_dir in skill_paths:
            if not root_dir or not root_dir.strip() or not Path(root_dir).is_dir():
                continue
            _search(root_dir, current_depth=0)

        return discovered
