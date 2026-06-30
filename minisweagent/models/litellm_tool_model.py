import json
import logging
import os
from collections.abc import Callable, Awaitable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Dict, List, Optional, Union

import litellm
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.litellm_model import LitellmModelConfig
from minisweagent.models.utils.cache_control import set_cache_control
from minisweagent.tools.base import ToolManager

logger = logging.getLogger("litellm_tool_model")


@dataclass
class LitellmToolModelConfig(LitellmModelConfig):
    tools: list[Dict[str, Any]] = field(default_factory=list)
    tool_choice: Literal["auto", "none", "all"] = "auto"


class LitellmToolModel:
    def __init__(
        self,
        *,
        config_class: Callable = LitellmToolModelConfig,
        **kwargs,
    ):
        """
        Initialize the model with optional tools.

        :param available_tools: A dict mapping tool name to callable functions.
                                e.g., {"get_weather": get_weather_fn}
        """
        self.config = config_class(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        self.tools = self.config.tools

        # Register custom models if registry exists
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    @retry(
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "4"))),
        wait=wait_exponential(multiplier=1, min=10, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                litellm.exceptions.UnsupportedParamsError,
                litellm.exceptions.NotFoundError,
                litellm.exceptions.PermissionDeniedError,
                litellm.exceptions.ContextWindowExceededError,
                litellm.exceptions.APIError,
                litellm.exceptions.AuthenticationError,
                KeyboardInterrupt,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs) -> litellm.ModelResponse:
        # Set default timeout to prevent hanging requests
        # Can be overridden via model_kwargs in config or kwargs at call time
        default_timeout = int(os.getenv("MSWEA_MODEL_TIMEOUT", "120"))
        merged_kwargs = {"timeout": default_timeout} | self.config.model_kwargs | kwargs

        try:
            response = litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=self.tools,
                tool_choice=self.config.tool_choice,
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                **merged_kwargs,
            )
            return response
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def query(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Query the model once and return the raw response.
        Does NOT handle or execute tool calls — returns them as-is for upper layers to process.
        """
        # Optionally apply cache control
        if self.config.set_cache_control:
            messages = set_cache_control(messages, mode=self.config.set_cache_control)

        # Make single API call
        response = self._query(messages, **kwargs)

        try:
            cost = litellm.cost_calculator.completion_cost(response)
            if cost <= 0.0:
                raise ValueError(f"Cost must be > 0.0, got {cost}")
        except Exception as e:
            cost = 0.0
            if self.config.cost_tracking != "ignore_errors":
                # msg = (
                #     f"Error calculating cost for model {self.config.model_name}: {e}, perhaps it's not registered? "
                #     "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                #     "globally with export MSWEA_COST_TRACKING='ignore_errors'. "
                #     "Alternatively check the 'Cost tracking' section in the documentation at "
                #     "https://klieret.short.gy/mini-local-models. "
                #     " Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
                # )
                # logger.critical(msg)
                # raise RuntimeError(msg) from e
                cost = 0.01  # minimal cost if error occurs

        # Update stats
        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)

        # Extract message
        message = response.choices[0].message
        model_name = self.config.model_name.split("/")[-1]

        # Simply return content + full response in extra (with tool_calls if any)
        return {
            "content": message.content or "",
            "extra": {
                "response": response.model_dump(),  # Contains everything: tool_calls, finish_reason, etc.
                "model_name": model_name,
                "n_calls": self.n_calls,
                "cost": self.cost,
                "has_tool_calls": hasattr(message, "tool_calls") and bool(message.tool_calls),
                "tool_calls": (
                    [
                        {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                            "id": tc.id,
                        }
                        for tc in message.tool_calls
                    ]
                    if hasattr(message, "tool_calls") and message.tool_calls
                    else []
                ),
            },
        }

    def get_template_vars(self) -> Dict[str, Any]:
        return asdict(self.config) | {
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
        }
