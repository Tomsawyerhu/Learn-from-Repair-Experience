"""Model conversation client based on OpenAI library"""

import openai
from typing import Optional, Dict, Any, List
from .base_client import BaseModelClient


class OpenAIClient(BaseModelClient):
    """OpenAI client implementation

    Uses openai>=1.0 library, supports official OpenAI API or custom endpoints.
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None, model: Optional[str] = None, **kwargs):
        """Initialize OpenAI client

        Args:
            api_key: OpenAI API key
            base_url: Custom endpoint, optional
            model: Model name, optional, default "gpt-3.5-turbo"
            **kwargs: Other parameters, e.g., default_parameters dictionary
        """
        # TODO: Complete initialization logic
        # Hints:
        # 1. Call parent class __init__ method (use super().__init__(...))
        # 2. Create openai.OpenAI client instance, save as self._client
        #    - Need to pass api_key and base_url (if provided)
        # 3. Set self.model to passed model or default value
        # 4. Extract default_parameters from kwargs and update self._parameters
        # 5. Initialize message list self._messages
        # Code example (partial):
        super().__init__(api_key, base_url, model, **kwargs)
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**client_kwargs)
        self.model = model or self.get_default_model()
        self._messages = []
        self._parameters = kwargs.get("default_parameters", {})

    def set_parameter(self, key: str, value: Any) -> None:
        """Set model parameter"""
        self._parameters[key] = value

    def set_system_prompt(self, prompt: str) -> None:
        """Set system prompt"""
        # Find existing system prompt
        for i, msg in enumerate(self._messages):
            if msg.get("role") == "system":
                self._messages[i]["content"] = prompt
                return
        # Not found, insert at beginning
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
            openai.APIError: OpenAI API call failed
        """
        # If prompt is provided, add user message
        if prompt is not None:
            self.add_user_message(prompt)

        # Build call parameters, merge with self._parameters
        completion_kwargs = {
            "model": self.model,
            "messages": self._messages.copy(),
        }
        completion_kwargs.update(self._parameters)

        # Call OpenAI API
        response = self._client.chat.completions.create(**completion_kwargs)

        # Extract response content
        content = response.choices[0].message.content

        # Add assistant response to message history
        self.add_assistant_message(content)
        return content

    def clear_messages(self) -> None:
        """Clear message history, keep system prompt"""
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
        # TODO: Return a reasonable default model, e.g., "gpt-3.5-turbo"
        # Hint: Consider different versions of OpenAI models
        return "deepseek-reasoner"
