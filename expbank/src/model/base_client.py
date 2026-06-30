"""Abstract base class for model conversation clients"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseModelClient(ABC):
    """Base class for model conversation clients

    Defines unified interface, supports setting base_url, api_key, model selection,
    parameter adjustment, multi-turn conversation and other features.
    Concrete implementation classes should inherit from this class and implement abstract methods.
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None, model: Optional[str] = None, **kwargs):
        """Initialize client

        Args:
            api_key: API key
            base_url: Base URL, optional
            model: Model name, optional
            **kwargs: Other initialization parameters, e.g., default model parameters
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model or self.get_default_model()
        self._messages = []
        self._parameters = kwargs.get("default_parameters", {})
        self._last_cost: float = 0.0  # Last call cost

    def set_parameter(self, key: str, value: Any) -> None:
        """Set model parameter

        Args:
            key: Parameter name, e.g., "temperature"
            value: Parameter value
        """
        self._parameters[key] = value

    def set_system_prompt(self, prompt: str) -> None:
        """Set system prompt

        If system prompt already exists in message list, replace it; otherwise insert at the beginning.

        Args:
            prompt: System prompt content
        """
        for i, msg in enumerate(self._messages):
            if msg.get("role") == "system":
                self._messages[i]["content"] = prompt
                return
        self._messages.insert(0, {"role": "system", "content": prompt})

    def add_user_message(self, content: str) -> None:
        """Add user message

        Args:
            content: User message content
        """
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Add assistant message (can be used to manually add historical records)

        Args:
            content: Assistant message content
        """
        self._messages.append({"role": "assistant", "content": content})

    @abstractmethod
    async def generate(self, prompt: Optional[str] = None) -> str:
        """Generate response (single or multi-turn)

        If prompt is provided, will first call add_user_message to add the prompt.

        Args:
            prompt: Optional user prompt

        Returns:
            Assistant's response text

        Raises:
            Concrete implementation classes should define their own exception types
        """
        pass

    def clear_messages(self) -> None:
        """Clear message history, but keep system prompt (if exists)"""
        system_message = None
        for msg in self._messages:
            if msg.get("role") == "system":
                system_message = msg
                break
        if system_message:
            self._messages = [system_message]
        else:
            self._messages = []

    def get_messages(self) -> List[Dict[str, str]]:
        """Get current message history (read-only)"""
        return self._messages.copy()

    def get_parameters(self) -> Dict[str, Any]:
        """Get current model parameters (read-only)"""
        return self._parameters.copy()

    def get_model(self) -> Optional[str]:
        """Get currently used model name"""
        return self.model

    def get_default_model(self) -> str:
        """Return default model name (subclass can override)"""
        return "deepseek-reasoner"

    def get_last_cost(self) -> float:
        """Get last call cost

        Returns:
            Cost value (USD), default 0.0
        """
        return self._last_cost
