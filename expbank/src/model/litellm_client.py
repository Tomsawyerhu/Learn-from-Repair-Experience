"""Model conversation client based on LiteLLM library"""

import litellm
from typing import Optional, Dict, Any, List
from .base_client import BaseModelClient


class LiteLLMClient(BaseModelClient):
    """LiteLLM client implementation

    Uses litellm library, supports multiple model providers (OpenAI, Anthropic, Cohere, etc.).
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None, model: Optional[str] = None, **kwargs):
        """Initialize LiteLLM client

        Args:
            api_key: API key
            base_url: Custom endpoint, optional (litellm supports setting via api_base)
            model: Model name, optional, default "gpt-3.5-turbo"
            **kwargs: Other parameters, e.g., default_parameters dictionary
        """
        # 1. Call parent class __init__ method
        # 2. Save api_key, base_url, model and other attributes
        # 3. litellm library doesn't need explicit client instance, usually call litellm.completion directly
        # 4. Can configure litellm via environment variables or passing parameters
        # 5. Initialize message list and parameters dictionary

        super().__init__(api_key, base_url, model, **kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self.model = model or self.get_default_model()
        self._messages = []
        self._parameters = kwargs.get("default_parameters", {})

    def set_parameter(self, key: str, value: Any) -> None:
        """Set model parameter"""
        self._parameters[key] = value

    def set_system_prompt(self, prompt: str) -> None:
        """Set system prompt"""
        for i, msg in enumerate(self._messages):
            if msg.get("role") == "system":
                self._messages[i]["content"] = prompt
                return
        self._messages.insert(0, {"role": "system", "content": prompt})

    def add_user_message(self, content: str) -> None:
        """Add user message"""
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Add assistant message"""
        self._messages.append({"role": "assistant", "content": content})

    async def generate(self, prompt: Optional[str] = None) -> str:
        """Generate response

        Args:
            prompt: Optional user prompt

        Returns:
            Assistant's response text

        Raises:
            Exception: litellm call failed
        """
        # If prompt is provided, add user message
        if prompt is not None:
            self.add_user_message(prompt)

        # Build call parameters, merge with self._parameters
        completion_kwargs = {
            "model": self.model,
            "messages": self._messages.copy(),
        }
        if self.base_url:
            completion_kwargs["api_base"] = self.base_url
        if self.api_key:
            completion_kwargs["api_key"] = self.api_key
        # Merge other parameters (e.g., temperature, max_tokens, etc.)
        completion_kwargs.update(self._parameters)

        # Call litellm.acompletion (true async call, won't block event loop)
        response = await litellm.acompletion(**completion_kwargs)

        # Calculate and save cost
        try:
            self._last_cost = litellm.completion_cost(completion_response=response)
        except Exception:
            self._last_cost = 0.0

        # Extract response content (litellm returns dictionary)
        content = response["choices"][0]["message"]["content"]

        # Add assistant response to message history
        self.add_assistant_message(content)
        return content

    def clear_messages(self) -> None:
        """Clear message history, keep system prompt"""
        # Find system prompt message
        system_message = None
        for msg in self._messages:
            if msg.get("role") == "system":
                system_message = msg
                break
        # Reset message list
        if system_message:
            self._messages = [system_message]
        else:
            self._messages = []

    def get_messages(self) -> List[Dict[str, str]]:
        """Get message history copy"""
        return self._messages.copy()

    def get_parameters(self) -> Dict[str, Any]:
        """Get parameters copy"""
        return self._parameters.copy()

    def get_model(self) -> Optional[str]:
        """Get current model name"""
        return self.model

    def get_default_model(self) -> str:
        """Return default model name"""
        return "deepseek-reasoner"

    def get_last_cost(self) -> float:
        """Get last call cost

        Returns:
            Cost value (USD), default 0.0
        """
        return self._last_cost
