"""Tests for the YOLO tool-approval rule and the harness factory rework."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from giskard import create_harness_agent
from giskard.core._feature_stage import ExperimentalWarning
from giskard.core.harness.file_access import FileAccessProvider, FileSystemAgentFileStore
from giskard.core.harness.file_memory import FileMemoryProvider
from giskard.core.harness.tool_approval import ToolApprovalRuleCallback, create_yolo_approval_rule
from giskard.core.types import Content
from giskard.tools.shell import LocalShellTool
from giskard.tools.web_search import ParallelSearchClient


def _function_call(name: str, arguments: dict | None = None) -> Content:
    return Content.from_function_call(call_id="c1", name=name, arguments=arguments or {})


@pytest.fixture()
def yolo() -> ToolApprovalRuleCallback:
    return create_yolo_approval_rule(Path.cwd().resolve())


class TestDestructiveShellDetection:
    @pytest.mark.parametrize(
        "command",
        [
            "rm -rf build",
            "rmdir tmp",
            "del file.txt",
            "erase file.txt",
            "rd /s /q build",
            "Remove-Item -Recurse -Force .",
            "remove-item foo",
            "cd foo && rm bar.txt",
            "Set-Location ..; Remove-Item x",
            "Get-ChildItem . | Remove-Item -Force",
        ],
    )
    def test_destructive_commands_escalate(self, yolo, command):
        assert yolo(_function_call("run_shell", {"command": command})) is False

    @pytest.mark.parametrize(
        "command",
        [
            "ls -la",
            "python script.py",
            "cd build && python -m pytest -q",
            "Get-ChildItem -Recurse -Filter *.py",
            "echo hello",
            "",
            "   ",
            "dir notes",
            "border 1",
            "echo rm -rf x",
        ],
    )
    def test_safe_commands_are_approved(self, yolo, command):
        assert yolo(_function_call("run_shell", {"command": command})) is True

    def test_shell_without_command_argument_escalates(self, yolo):
        assert yolo(_function_call("run_shell")) is False


class TestYoloApprovalMatrix:
    @pytest.mark.parametrize(
        "name",
        ["read_file", "ls", "glob", "grep", "web_search", "web_fetch",
         "write_file", "edit_file", "edit_file_lines",
         "file_memory_write", "file_memory_read", "file_memory_ls",
         "file_memory_grep", "file_memory_replace", "file_memory_replace_lines"],
    )
    def test_approved_tools(self, yolo, name):
        assert yolo(_function_call(name)) is True

    @pytest.mark.parametrize(
        "name",
        ["delete_file", "file_memory_delete", "some_unknown_mcp_tool"],
    )
    def test_escalated_tools(self, yolo, name):
        assert yolo(_function_call(name)) is False

    def test_none_name_escalates(self, yolo):
        # Runtime edge: a function_call content without a name escalates.
        # Constructed directly since from_function_call types name as str.
        assert yolo(Content("function_call", call_id="c1", name=None)) is False


class TestWorkdirThreading:
    def test_workdir_roots_file_access_store(self, tmp_path: Path) -> None:
        agent = create_harness_agent(MagicMock(), workdir=tmp_path)
        provider = next(p for p in agent.context_providers if isinstance(p, FileAccessProvider))
        assert isinstance(provider.store, FileSystemAgentFileStore)
        assert provider.store.root_path == tmp_path.resolve()

    def test_workdir_roots_file_memory_store(self, tmp_path: Path) -> None:
        agent = create_harness_agent(MagicMock(), workdir=tmp_path)
        provider = next(p for p in agent.context_providers if isinstance(p, FileMemoryProvider))
        assert isinstance(provider.store, FileSystemAgentFileStore)
        assert provider.store.root_path == (tmp_path.resolve() / "agent-file-memory")

    def test_workdir_reaches_default_shell_tool(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}

        class _SpyShellTool(LocalShellTool):
            def __init__(self, **kwargs: object) -> None:
                super().__init__(**kwargs)  # type: ignore[arg-type]
                captured.update(kwargs)

        monkeypatch.setattr("giskard.tools.shell.LocalShellTool", _SpyShellTool)
        create_harness_agent(MagicMock(), workdir=tmp_path)
        # The factory passes the resolved workdir through to the shell tool;
        # LocalShellTool normalizes it to a str via os.fspath internally.
        assert Path(captured["workdir"]) == tmp_path.resolve()


class TestToolApprovalRuleValidation:
    def test_yolo_with_disabled_auto_approval_raises(self) -> None:
        with pytest.raises(ValueError, match="disable_tool_auto_approval"):
            create_harness_agent(MagicMock(), tool_approval_rule="yolo", disable_tool_auto_approval=True)

    def test_unknown_rule_value_raises(self) -> None:
        with pytest.raises(ValueError, match="tool_approval_rule"):
            create_harness_agent(MagicMock(), tool_approval_rule="bogus")


class TestDefaultToolAssembly:
    def _tool_names(self, agent) -> list[str]:
        return [getattr(t, "name", "") for t in agent.default_options["tools"] or []]

    def test_default_web_search_tools_injected_without_client_protocol(self, tmp_path: Path) -> None:
        agent = create_harness_agent(MagicMock(), workdir=tmp_path)
        names = self._tool_names(agent)
        assert "web_search" in names
        assert "web_fetch" in names

    def test_disable_web_search_skips_web_tools(self, tmp_path: Path) -> None:
        agent = create_harness_agent(MagicMock(), workdir=tmp_path, disable_web_search=True)
        names = self._tool_names(agent)
        assert "web_search" not in names
        assert "web_fetch" not in names

    def test_supplied_web_search_client_is_used(self, tmp_path: Path) -> None:
        client = ParallelSearchClient()
        agent = create_harness_agent(MagicMock(), workdir=tmp_path, web_search_client=client)
        assert self._tool_names(agent).count("web_search") == 1
        web_search_tool = next(
            t for t in agent.default_options["tools"] if getattr(t, "name", "") == "web_search"
        )
        assert web_search_tool is client.get_tools()[0]

    def test_shell_tool_wired_without_supports_shell_tool(self, tmp_path: Path) -> None:
        # MagicMock does NOT implement SupportsShellTool; previously this path
        # logged a warning and skipped the shell tool entirely.
        agent = create_harness_agent(MagicMock(), workdir=tmp_path)
        assert "run_shell" in self._tool_names(agent)


class TestWorkdirPathValidation:
    """Shell writes outside workdir escalate; reads anywhere are approved."""

    @pytest.fixture()
    def scoped_yolo(self, tmp_path: Path) -> ToolApprovalRuleCallback:
        return create_yolo_approval_rule(tmp_path.resolve())

    @pytest.fixture()
    def outside(self, tmp_path: Path) -> str:
        # A directory guaranteed to be outside the tmp_path workdir.
        return str(tmp_path.parent / "yolo-outside-probe")

    @pytest.mark.parametrize(
        "command",
        [
            # Read-only commands may touch any path.
            "cat /etc/hosts",
            "grep root /etc/passwd",
            "ls C:\\Windows",
            "Get-Content C:\\Windows\\win.ini",
            # Writes/creates inside workdir are approved.
            "cp a.txt b.txt",
            "mkdir build",
            "touch new.txt",
            "echo hi > out.log",
            "echo hi >> out.log",
            "mv a b",
            "tar -cf backup.tar .",
            "Set-Content out.txt -Value x",
            "Copy-Item a.txt b.txt",
            "sed -i 's/a/b/' file.txt",
            # Executing scripts/interpreters stays approved (documented
            # limitation: static analysis cannot see what they write).
            "python script.py",
            "git status",
        ],
    )
    def test_non_outside_workdir_writes_approved(self, scoped_yolo, command):
        assert scoped_yolo(_function_call("run_shell", {"command": command})) is True

    @pytest.mark.parametrize(
        "command",
        [
            "touch /tmp/x",
            "echo hi > /etc/hosts",
            "echo x >> /var/log/app.log",
            "mv x ~",
            "cp a.txt ~/b",
            "cp a.txt ..\\b",
            "sed -i s/a/b/ /etc/fstab",
            "wget -O /etc/w https://example.com",
            "curl -o /etc/x https://example.com",
            # Deletions still escalate regardless of location.
            "rm -rf build",
            "Remove-Item C:\\Windows\\temp.txt",
        ],
    )
    def test_outside_workdir_or_destructive_writes_escalate(self, scoped_yolo, command):
        assert scoped_yolo(_function_call("run_shell", {"command": command})) is False

    def test_write_to_absolute_outside_path_escalates(self, scoped_yolo, outside):
        assert scoped_yolo(_function_call("run_shell", {"command": f"cp a.txt {outside}\\b.txt"})) is False

    def test_write_redirect_to_outside_path_escalates(self, scoped_yolo, outside):
        assert scoped_yolo(_function_call("run_shell", {"command": f"echo x > {outside}\\o.log"})) is False

    def test_set_content_outside_escalates(self, scoped_yolo, outside):
        assert scoped_yolo(_function_call("run_shell", {"command": f"Set-Content {outside}\\f.txt -Value y"})) is False

    def test_variable_path_reference_escalates_conservatively(self, scoped_yolo):
        # $HOME/x.log references a path via an unresolvable variable that
        # contains a separator: conservatively escalated.
        assert scoped_yolo(_function_call("run_shell", {"command": "echo hi > $HOME/x.log"})) is False

    def test_bare_variable_without_separator_is_ignored(self, scoped_yolo):
        # A bare variable (no path separator) is not treated as a path token;
        # the redirect target out.log is inside workdir, so this is approved.
        assert scoped_yolo(_function_call("run_shell", {"command": "echo $PATH > out.log"})) is True

    def test_workdir_relative_redirect_to_parent_escalates(self, scoped_yolo):
        # A relative path that climbs out of workdir via .. is escalated.
        # tmp_path workdirs are always strictly nested, so one level escapes.
        assert scoped_yolo(_function_call("run_shell", {"command": "echo x > ..\\escape.txt"})) is False


class TestYoloWiring:
    def test_yolo_rule_reaches_middleware(self, tmp_path: Path) -> None:
        agent = create_harness_agent(
            MagicMock(),
            workdir=tmp_path,
            tool_approval_rule="yolo",
        )
        from giskard.core.harness.tool_approval import ToolApprovalMiddleware

        middleware = next(m for m in agent.middleware if isinstance(m, ToolApprovalMiddleware))
        rule_names = {r.__name__ for r in middleware.auto_approval_rules}
        assert "_yolo_rule" in rule_names

    def test_experimental_warnings_removed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        with warnings.catch_warnings():
            # Only the factory parameter warning is forbidden; other
            # @experimental(HARNESS) suppliers wired by the factory (stores,
            # providers) warn legitimately once per process via the shared
            # dedup registry and must not fail this test.
            warnings.filterwarnings("ignore", category=ExperimentalWarning)
            warnings.filterwarnings(
                "error",
                category=ExperimentalWarning,
                message=r"\[HARNESS\] create_harness_agent\b.*",
            )
            # Clear the shared dedup registry so a re-introduced factory
            # warning is actually emitted (not suppressed by an earlier
            # test's HARNESS warning) in merged runs too.
            monkeypatch.setattr("giskard.core._feature_stage._WARNED_FEATURES", set())
            # loop_should_continue previously triggered the harness experimental warning.
            create_harness_agent(MagicMock(), workdir=tmp_path, loop_should_continue=lambda response: False)
