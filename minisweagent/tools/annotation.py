from typing import Dict, Any, Callable
from functools import wraps


def tool_schema(
        *,
        name: str,
        description: str,
        parameters: Dict[str, Any]
):
    """
    Decorator: attach tool schema metadata to a function.

    :param name: Tool name.
    :param description: Tool description.
    :param parameters: JSON-Schema-style parameter definition.
    """

    def decorator(func: Callable) -> Callable:
        # Bind schema information to function attributes.
        func.is_tool = True
        func.tool_schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
                "strict": True
            }
        }
        # Optional: keep raw fields for debugging.
        func.tool_name = name
        func.tool_description = description
        func.tool_parameters = parameters

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def extract_tool_schema(func: Callable) -> dict | None:
    """Safely extract a function's tool schema."""
    return getattr(func, "tool_schema", None)
