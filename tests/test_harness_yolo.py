"""Tests for the YOLO tool-approval rule and the harness factory rework."""

from __future__ import annotations

from pathlib import Path

import pytest

from giskard.core.harness.tool_approval import create_yolo_approval_rule
from giskard.core.types import Content


def _function_call(name: str, arguments: dict | None = None) -> Content:
    return Content.from_function_call(call_id="c1", name=name, arguments=arguments or {})


@pytest.fixture()
def yolo() -> object:
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
