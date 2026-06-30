from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional

from minisweagent.tools.annotation import tool_schema


class ToolNotFoundException(Exception):
    pass


class ToolCallError(Exception):
    pass

class BashToolNotSupportedError(Exception):
    pass


@dataclass
class ToolManagerEnv:
    pass


class ToolManager:
    """
    Tool manager that auto-discovers, registers, and invokes methods decorated with @tool_schema.
    """

    def __init__(self, env: "ToolManagerEnv"):
        self.env = env
        self.tools: Dict[str, Dict[str, Any]] = {}  # name → {method, schema}
        self.register_all_tools()  # Automatically register all annotated methods at initialization.

    def _register_tool(self, method: Callable, name: str, schema: Dict[str, Any]):
        """
        Register a single tool internally.
        """
        if name in self.tools:
            raise ValueError(f"Tool with name '{name}' is already registered.")
        self.tools[name] = {
            "method": method,
            "schema": schema
        }

    def register_all_tools(self):
        """
        Scan all methods on the current instance and auto-register those decorated with @tool_schema.
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            # Ignore private methods, non-callables, and class self-methods such as __init__.
            if attr_name.startswith("_") or not callable(attr) or attr == self.__class__.__init__:
                continue
            # Check whether the method is decorated with @tool_schema.
            if hasattr(attr, "is_tool") and attr.is_tool:
                schema = attr.tool_schema
                tool_name = schema["function"]["name"]
                self._register_tool(method=attr, name=tool_name, schema=schema)

    def call_tool(self, tool_name: str, **kwargs):
        """
        Invoke a tool by name with the provided arguments.
        :return: Tool execution result
        :raises: ToolNotFoundException if tool not found, TypeError if missing args
        """
        if tool_name not in self.tools:
            raise ToolNotFoundException(f"Tool '{tool_name}' not found. Available: {list(self.tools.keys())}")

        if self.is_bash_tool(self.get_tool_schema(tool_name)):
            raise BashToolNotSupportedError

        method = self.tools[tool_name]["method"]
        try:
            return method(**kwargs)
        except Exception as e:
            raise ToolCallError(f"Error calling tool '{tool_name}' with arguments {kwargs}: {e}") from e

    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool.
        :param name: Tool name
        :return: Whether the tool was successfully unregistered
        """
        if name in self.tools:
            del self.tools[name]
            return True
        return False

    def get_tool_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a specific tool."""
        tool = self.tools.get(name)
        return tool["schema"] if tool else None

    def get_all_schemas(self) -> list[Dict[str, Any]]:
        """Get all tool schemas (can be used for the LLM tools parameter)."""
        return [self.get_tool_schema(name) for name in self.tools]

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self.tools.keys())

    @tool_schema(
        name="bash",
        description=(
                "Execute shell scripts on the local system. "
        ),
        parameters={
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "The shell script to execute, e.g., '```bash\nls -la\n```', must start with ```bash, end with ```"
                }
            },
            "required": ["cmd"],
            "additionalProperties": False
        },
    )
    def run_bash_command(self, cmd: str) -> str:
        """
        Execute a Bash command and return output wrapped in ```bash ... ```.
        This is suitable for file operations, code building, and log inspection.
        Not implemented here; this belongs in the environment layer.
        """
        pass

    def is_bash_tool(self,tool_schema:dict):
        if tool_schema['function']['name']=='bash':
            return True
        return False
