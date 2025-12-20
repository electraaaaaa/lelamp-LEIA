"""
Tool registry for extracting function tools from LeLamp agent.

Converts LiveKit @function_tool decorated methods to Ollama tool format.
"""

import inspect
import logging
from typing import Dict, Any, List, Callable, get_type_hints

logger = logging.getLogger(__name__)


def python_type_to_json_type(py_type) -> str:
    """Convert Python type to JSON schema type."""
    if py_type == int:
        return "integer"
    elif py_type == float:
        return "number"
    elif py_type == bool:
        return "boolean"
    elif py_type == str:
        return "string"
    elif py_type == list or getattr(py_type, "__origin__", None) == list:
        return "array"
    elif py_type == dict or getattr(py_type, "__origin__", None) == dict:
        return "object"
    else:
        return "string"


def extract_tools_from_agent(agent) -> List[Dict[str, Any]]:
    """
    Extract all @function_tool decorated methods from LeLamp agent.

    Returns:
        List of tool definitions in Ollama format with handlers
    """
    tools = []

    for name in dir(agent):
        if name.startswith("_"):
            continue

        try:
            method = getattr(agent, name)
        except Exception:
            continue

        if not callable(method):
            continue

        # Check if it's a function_tool decorated method
        # LiveKit's function_tool adds specific attributes
        is_tool = (
            hasattr(method, "_lk_function_info")
            or hasattr(method, "__wrapped__")
            and hasattr(method.__wrapped__, "_lk_function_info")
        )

        # Also check for methods in function mixin classes
        if not is_tool:
            # Check if this method comes from a Functions mixin
            for cls in type(agent).__mro__:
                if "Functions" in cls.__name__ and name in cls.__dict__:
                    # Check if the method has docstring (function tools always do)
                    if method.__doc__ and inspect.iscoroutinefunction(method):
                        is_tool = True
                        break

        if not is_tool:
            continue

        # Get docstring for description
        doc = method.__doc__ or f"Function {name}"
        description = doc.split("\n")[0].strip()

        # Get function signature
        try:
            sig = inspect.signature(method)
            hints = {}
            try:
                hints = get_type_hints(method)
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Could not get signature for {name}: {e}")
            continue

        # Build parameter schema
        parameters = {"type": "object", "properties": {}, "required": []}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # Get type from annotation or hints
            param_type = hints.get(param_name, param.annotation)
            json_type = "string"

            if param_type != inspect.Parameter.empty:
                json_type = python_type_to_json_type(param_type)

            # Extract parameter description from docstring if available
            param_desc = f"Parameter {param_name}"
            if doc:
                # Try to find param in docstring (Args section)
                for line in doc.split("\n"):
                    line = line.strip()
                    if line.startswith(f"{param_name}:") or line.startswith(
                        f"{param_name} ("
                    ):
                        param_desc = line.split(":", 1)[-1].strip()
                        break

            parameters["properties"][param_name] = {
                "type": json_type,
                "description": param_desc,
            }

            # Mark as required if no default
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)

        tools.append(
            {
                "name": name,
                "description": description,
                "parameters": parameters,
                "handler": method,
            }
        )
        logger.debug(f"Registered tool: {name}")

    logger.info(f"Extracted {len(tools)} function tools from agent")
    return tools


def tools_to_ollama_format(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert tool list to Ollama API format.

    Args:
        tools: List of tool dicts with name, description, parameters, handler

    Returns:
        List of tools in Ollama format (without handlers)
    """
    ollama_tools = []

    for tool in tools:
        ollama_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
        )

    return ollama_tools
