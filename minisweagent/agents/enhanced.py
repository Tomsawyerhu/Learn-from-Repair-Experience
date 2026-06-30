"""Enhanced agent adapted from upstream ``src/minisweagent/agents/default.py``.

Local modifications relative to the upstream mini-swe-agent implementation:
- adds ``ToolManager``-based tool dispatch with a bash fallback path;
- tracks recent bash executions and blocks repeated identical commands that loop;
- normalizes tool-call payloads and recovers from malformed JSON or empty arguments;
- augments empty command output with follow-up guidance and supports extra action formats;
- tightens ``submit_patch`` handling by validating that ``fix.patch`` exists before finishing.
"""

import json
import re
import subprocess
from json import JSONDecodeError
from typing import Any

from minisweagent import logger
from minisweagent.agents.default import (
    AgentConfig,
    DefaultAgent,
    ExecutionTimeoutError,
    FormatError,
    LimitsExceeded,
    NonTerminatingException,
    Submitted,
)
from minisweagent.tools.base import BashToolNotSupportedError, ToolManager
from minisweagent.utils.log import format_response_for_log, truncate_text


class InvalidParamError(NonTerminatingException):
    """Raised when the LM receive invalid parameters."""


class EnhancedAgent(DefaultAgent):
    def __init__(self, *args, tool_manager: ToolManager, config_class=AgentConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)
        self.tool_manager = tool_manager
        self._bash_history: list[dict[str, Any]] = []

    @staticmethod
    def _normalize_bash_command(command: str) -> str:
        normalized = command.strip()
        for prefix in ("```bash", "```"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].lstrip()
        for suffix in ("```bash", "```"):
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].rstrip()
        return normalized.strip()

    def _extract_bash_command(self, action: dict) -> str:
        try:
            arguments = json.loads(action.get("arguments", "{}"))
        except (TypeError, JSONDecodeError):
            return ""
        if not isinstance(arguments, dict):
            return ""
        cmd = arguments.get("cmd", "")
        if not isinstance(cmd, str):
            return ""
        return self._normalize_bash_command(cmd)

    def _record_bash_execution(self, command: str, output: dict) -> None:
        self._bash_history.append(
            {
                "command": command,
                "returncode": int(output.get("returncode", 0)),
                "output": str(output.get("output", "")),
            }
        )
        if len(self._bash_history) > 12:
            self._bash_history = self._bash_history[-12:]

    def _build_repeat_warning(self, command: str, previous_output: dict[str, Any]) -> dict[str, Any]:
        rendered_output = str(previous_output.get("output", "")).strip() or "<empty output>"
        if len(rendered_output) > 1200:
            rendered_output = rendered_output[:1200] + "\n<... truncated ...>"
        return {
            "returncode": 1,
            "output": (
                "Loop guard: you already ran the same bash command twice in a row and received the same result. "
                "Do not repeat it again without a new hypothesis.\n"
                f"Repeated command:\n{command}\n"
                f"Previous return code: {previous_output.get('returncode', 0)}\n"
                f"Previous output:\n{rendered_output}\n"
                "Choose a different next step: inspect `git diff`, inspect the edited file with `sed -n`, "
                "run build/repro if you changed code, or switch to a different search target."
            ),
        }

    def _check_repeated_bash_loop(self, command: str) -> dict[str, Any] | None:
        if not command or len(self._bash_history) < 2:
            return None
        previous = self._bash_history[-2:]
        if not all(entry["command"] == command for entry in previous):
            return None
        if previous[0]["returncode"] != previous[1]["returncode"]:
            return None
        if previous[0]["output"] != previous[1]["output"]:
            return None
        return self._build_repeat_warning(command, previous[-1])

    def _augment_empty_output(self, action: dict, output: dict[str, Any]) -> dict[str, Any]:
        rendered_output = output.get("output")
        if not isinstance(rendered_output, str) or rendered_output.strip():
            return output

        command = self._extract_bash_command(action) if action.get("name") == "bash" else ""
        guidance = "Use a different follow-up command instead of blindly repeating this one."
        if any(token in command for token in ("sed -i", "perl -0pi", "python - <<", "ed -s")):
            guidance = (
                "This looks like an edit command. Inspect the target file or run `git diff`, "
                "then build or reproduce before editing again."
            )
        elif command.startswith("find ") or command.startswith("grep ") or " grep " in f" {command} ":
            guidance = (
                "This search produced no visible result. Change the pattern or inspect a different path "
                "instead of repeating the same search."
            )
        elif "git diff" in command:
            guidance = "If you expected changes, inspect `git status -sb` or the edited file to confirm the edit landed."

        augmented = dict(output)
        augmented["output"] = (
            "Command completed with no stdout/stderr.\n"
            f"Command:\n{command or '<unknown>'}\n"
            f"Guidance: {guidance}"
        )
        return augmented

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded("time or money limit exceeded.")
        response = self.model.query(self.messages)
        logger.info("=" * 20 + " Assistant " + "=" * 20)
        logger.info(format_response_for_log(response))
        logger.info("=" * 20 + " Assistant " + "=" * 20)
        self.add_message("assistant", **response)
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        action = self.parse_action(response)
        output = self.execute_action(action)
        output = self._augment_empty_output(action, output)
        if "output" not in output:
            observation = "Please provide a tool call."
        else:
            observation = self.render_template(self.config.action_observation_template, output=output)
        logger.info("=" * 20 + " Observation " + "=" * 20)
        logger.info(truncate_text(observation))
        logger.info("=" * 20 + " Observation " + "=" * 20)
        self.add_message("user", observation)
        return output

    def execute_action(self, action: dict) -> dict:
        if "name" not in action:
            return {}

        try:
            # Be tolerant of empty payloads by treating them as an empty JSON object.
            arguments = action.get("arguments", "{}")
            if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
                arguments = "{}"

            parsed_args = json.loads(arguments)
            # Ensure the parsed result is a dictionary.
            if not isinstance(parsed_args, dict):
                parsed_args = {}

            output = self.tool_manager.call_tool(action["name"], **parsed_args)
        except BashToolNotSupportedError:
            actions = self._extract_bash_command(action)
            if loop_warning := self._check_repeated_bash_loop(actions):
                output = loop_warning
            else:
                output = self.env.execute(actions)
                self._record_bash_execution(actions, output)
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        except JSONDecodeError as jde:
            arguments = action.get("arguments", "{}")
            # Recovery path: if this is submit_patch, force empty arguments.
            if action.get("name") == "submit_patch":
                try:
                    output = self.tool_manager.call_tool(action["name"])
                    self.has_finished(output)
                    return output
                except Exception as e:
                    logger.error(f"Failed to force submit_patch: {e}")

            # Recovery path: use ast.literal_eval for single-quoted JSON-like Python dicts.
            try:
                import ast

                # Only attempt recovery when the payload looks like a dictionary.
                if isinstance(arguments, str) and arguments.strip().startswith("{"):
                    parsed_args = ast.literal_eval(arguments)
                    if isinstance(parsed_args, dict):
                        output = self.tool_manager.call_tool(action["name"], **parsed_args)
                        self.has_finished(output)
                        return output
            except Exception:
                pass

            output = {
                "returncode": 1,
                "output": f"❌ JSON Decode Error: {str(jde)}. Please ensure the arguments are valid JSON.",
            }
        except TimeoutError:
            raise ExecutionTimeoutError(self.render_template(self.config.timeout_template, action=action, output=""))
        self.has_finished(output)
        return output

    def _normalize_tool_arguments(self, action: dict) -> dict:
        """Normalize tool arguments across different formats.

        Handles:
        - Removes redundant "name" field
        - Unwraps "parameters" wrapper if it's the only field
        - Ensures arguments is properly formatted JSON string
        """
        if not isinstance(action, dict) or "arguments" not in action:
            return action

        arguments = action.get("arguments", "{}")

        # Parse arguments if it's a string
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                return action

        if not isinstance(arguments, dict):
            return action

        # Remove redundant "name" field (from Mistral or other sources)
        arguments.pop("name", None)

        # Unwrap "parameters" if it's the only remaining field
        if "parameters" in arguments and len(arguments) == 1:
            if isinstance(arguments["parameters"], dict):
                arguments = arguments["parameters"]

        # Reconstruct action with cleaned arguments
        return {"name": action["name"], "arguments": json.dumps(arguments)}

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action.

        Supports multiple formats:
        1. Standard OpenAI tool_calls (primary)
        2. Mistral-style [TOOL_CALLS]tool_name{json} format (fallback)
        3. Bash code block format ```bash...``` (legacy fallback)
        """
        # Priority 1: Standard OpenAI tool_calls
        if response["extra"]["has_tool_calls"]:
            if len(response["extra"]["tool_calls"]) == 1:
                action = response["extra"]["tool_calls"][0]
                return self._normalize_tool_arguments(action)
            raise FormatError(
                self.render_template(self.config.format_error_template, actions=response["extra"]["tool_calls"])
            )

        content = response["content"]

        # Priority 2: Mistral-style [TOOL_CALLS]tool_name{json} format
        mistral_match = re.search(r"\[TOOL_CALLS\](\w+)", content)
        if mistral_match:
            tool_name = mistral_match.group(1)
            json_start = mistral_match.end()
            remaining = content[json_start:]

            # Try to parse JSON arguments
            try:
                decoder = json.JSONDecoder()
                arguments_dict, _ = decoder.raw_decode(content, json_start)
                action = {"name": tool_name, "arguments": json.dumps(arguments_dict)}
                return self._normalize_tool_arguments(action)
            except json.JSONDecodeError as e:
                # Fallback 1: Fix invalid escape sequences (e.g., \' from Python-style strings)
                if "Invalid \\escape" in str(e) and remaining.startswith("{"):
                    try:
                        # Fix common invalid escapes: \' -> '
                        fixed_json = remaining.replace("\\'", "'")
                        arguments_dict = json.loads(fixed_json)
                        action = {"name": tool_name, "arguments": json.dumps(arguments_dict)}
                        return self._normalize_tool_arguments(action)
                    except json.JSONDecodeError:
                        pass  # Continue to next fallback

                # Fallback 2: Handle mixed format [TOOL_CALLS]bash\ncommand\n```
                if remaining.startswith("\n") or remaining.startswith(" "):
                    end_marker = remaining.find("```")
                    cmd = remaining[:end_marker].strip() if end_marker != -1 else remaining.strip()
                    if cmd:
                        return {"name": tool_name, "arguments": json.dumps({"cmd": cmd})}

        # Priority 3: Legacy bash code block format ```bash...```
        if "```bash" in content:
            return {
                "name": "bash",
                "arguments": json.dumps({"cmd": content.split("```bash")[1].split("```")[0].strip()}),
            }

        return {}

    def has_finished(self, output):
        """Raises Submitted exception with final output if the agent has finished its task."""
        if "output" in output and output["output"] == "Successfully submitted patch":
            patch_output = self.env.execute(
                "if [ -s fix.patch ]; then "
                "cat fix.patch; "
                "else "
                "echo \"submit_patch rejected: fix.patch is missing or empty. "
                "Generate it with git diff -- . ':(exclude)*test*' ':(exclude)*Test*' > fix.patch "
                "and inspect it before calling submit_patch again.\"; "
                "exit 1; "
                "fi"
            )
            if int(patch_output.get("returncode", 1)) != 0:
                output["returncode"] = patch_output.get("returncode", 1)
                output["output"] = patch_output.get("output", "")
                return
            raise Submitted(patch_output["output"])
