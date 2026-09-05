import functools
import logging
import ast
import asyncio
import sys
from pathlib import Path
from typing import List, Optional
from giskard.core.tools import FunctionTool


logger = logging.getLogger("giskard")


@functools.lru_cache(maxsize=None)
def warn() -> None:
    logger.warning("PythonTools can run arbitrary code, please provide human supervision.")


class LocalPythonExecutor():
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        timeout: float = 60.0,
        **kwargs,
    ):
        self.base_dir: Path = (base_dir or Path.cwd()).resolve()
        self.timeout: float = timeout


    async def run_python_code(self, code: str) -> str:
        """Run Python code in an isolated subprocess and return its stdout.

        Each call runs in a fresh interpreter process, so state does not persist between calls.
        If successful, returns the captured stdout (or a success message if nothing was printed).
        If failed or timed out, returns an error message.

        :param code: The code to run.
        :return: captured stdout if successful, otherwise returns an error message.
        """
        # Fail fast on syntax errors without spawning a process
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise ValueError(
                f"Code parsing failed on line {e.lineno} due to: {type(e).__name__}: {str(e)}\n"
                f"{e.text}"
                f"{' ' * (e.offset or 0)}^"
            )

        try:
            warn()
            # -I: isolated mode (ignores env vars and user site-packages), -X utf8: force UTF-8 IO
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-I", "-X", "utf8", "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.base_dir,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return f"Error running python code: timed out after {self.timeout} seconds"

            out = stdout.decode("utf-8", errors="replace").replace("\r\n", "\n")
            err = stderr.decode("utf-8", errors="replace").replace("\r\n", "\n")

            if proc.returncode == 0:
                if out:
                    logger.debug(f"Code output: {out}")
                    return out
                return "successfully ran python code"
            return f"Error running python code (exit {proc.returncode}): {err}"
        except Exception as e:
            logger.exception("Error running python code")
            return f"Error running python code: {e}"

    def get_tools(self) -> List[FunctionTool]:
        return [
            FunctionTool(
                name="run_python_code",
                description=(
                    "Runs Python code in an isolated subprocess and returns its stdout. "
                    "Use print() to produce output. State does not persist between calls."
                ),
                func=self.run_python_code,
                input_model={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "The Python code to run."},
                    },
                    "required": ["code"],
                }
            )
        ]
