import json
import logging
from pathlib import Path

from rich.logging import RichHandler


def truncate_text(text: str, max_length: int = 500, head_length: int = 200) -> str:
    if text is None:
        return "None"
    text = str(text)
    if len(text) <= max_length:
        return text
    tail_length = max_length - head_length
    return f"{text[:head_length]}\n... [truncated {len(text) - max_length} chars] ...\n{text[-tail_length:]}"


def format_response_for_log(response: dict, max_content_length: int = 500) -> str:
    if not isinstance(response, dict):
        return truncate_text(str(response), max_content_length)

    content = response.get("content", "")
    extra = response.get("extra", {})

    lines = []
    if content:
        lines.append(f"Content: {truncate_text(content, max_content_length)}")

    if extra.get("has_tool_calls"):
        tool_calls = extra.get("tool_calls", [])
        for tc in tool_calls[:3]:
            name = tc.get("name", "unknown")
            args = tc.get("arguments", "{}")
            if isinstance(args, str) and len(args) > 100:
                try:
                    args_dict = json.loads(args)
                    args = json.dumps(args_dict, ensure_ascii=False)[:100]
                except:
                    args = args[:100]
            lines.append(f"Tool: {name}({args})")
        if len(tool_calls) > 3:
            lines.append(f"... and {len(tool_calls) - 3} more tool calls")

    lines.append(f"Calls: {extra.get('n_calls', 0)}, Cost: ${extra.get('cost', 0):.2f}")

    return "\n".join(lines)


def _setup_root_logger() -> None:
    logger = logging.getLogger("minisweagent")
    logger.setLevel(logging.DEBUG)
    _handler = RichHandler(
        show_path=False,
        show_time=False,
        show_level=False,
        markup=False,
    )
    _formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)


def add_file_handler(path: Path | str, level: int = logging.DEBUG, *, print_path: bool = True) -> None:
    logger = logging.getLogger("minisweagent")
    handler = logging.FileHandler(path)
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if print_path:
        print(f"Logging to '{path}'")


_setup_root_logger()
logger = logging.getLogger("minisweagent")


__all__ = ["logger", "truncate_text", "format_response_for_log"]
