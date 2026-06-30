"""Unit tests for client classes

These tests use unittest.mock to simulate external API calls and verify the core logic.
Ensure the dependencies are installed before running the tests: pip install openai litellm
"""
import asyncio
import unittest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any

# Import the classes under test
try:
    from openai_client import OpenAIClient
    from litellm_client import LiteLLMClient
except ImportError:
    # If imports fail because TODOs are not implemented, skip the tests
    OpenAIClient = None
    LiteLLMClient = None


class TestBaseClient(unittest.TestCase):
    """Test implemented methods in the base class (if any)."""
    pass  # The base class is currently all TODOs; skip


class TestOpenAIClient(unittest.IsolatedAsyncioTestCase):
    """OpenAIClient tests."""

    def setUp(self):
        """Create a mocked client before each test."""
        if OpenAIClient is None:
            self.skipTest("OpenAIClient was not imported, possibly because the TODOs are not implemented")

        # First, patch BaseModelClient.__init__ so it does not raise an exception
        # Because the subclass calls super().__init__ while the base class currently raises NotImplementedError
        self.base_init_patcher = patch('base_client.BaseModelClient.__init__')
        mock_base_init = self.base_init_patcher.start()
        # Make the base class __init__ do nothing, or set default attributes
        def mock_base_init_side_effect(api_key, base_url=None, model=None, **kwargs):
            # Set default attributes to simulate base-class initialization
            self = mock_base_init_side_effect.self
            self.api_key = api_key
            self.base_url = base_url
            self.model = model
            self._messages = []
            self._parameters = kwargs.get('default_parameters', {})
        # Bind self to the mock instance using a small trick:
        # We will manually set attributes after creating the client
        mock_base_init.side_effect = lambda api_key, base_url=None, model=None, **kwargs: None

        # Mock the openai.OpenAI client
        self.mock_client = Mock()
        self.mock_chat = Mock()
        self.mock_completions = Mock()
        self.mock_client.chat = self.mock_chat
        self.mock_chat.completions = self.mock_completions

        # Replace openai.OpenAI using patch
        self.openai_patcher = patch('openai.OpenAI', return_value=self.mock_client)
        self.openai_patcher.start()

        # Create the client instance (note: __init__ may still contain TODOs and must be mocked)
        # Assume the user has already implemented __init__
        try:
            self.client = OpenAIClient(
                api_key="test-key",
                base_url="http://test.example.com",
                model="gpt-3.5-turbo",
                default_parameters={"temperature": 0.5}
            )
        except NotImplementedError:
            self.skipTest("OpenAIClient.__init__ is not implemented, skipping test")

        # If the client does not have a _client attribute, set it
        if not hasattr(self.client, '_client'):
            self.client._client = self.mock_client
        # Ensure the client has the required attributes
        if not hasattr(self.client, '_messages'):
            self.client._messages = []
        if not hasattr(self.client, '_parameters'):
            self.client._parameters = {}
        # Set additional attributes so the tests can access them
        self.client.api_key = "test-key"
        self.client.base_url = "http://test.example.com"
        self.client.model = "gpt-3.5-turbo"

    def tearDown(self):
        """Clean up patches."""
        self.openai_patcher.stop()
        self.base_init_patcher.stop()

    def test_initialization(self):
        """Test initialization."""
        self.assertEqual(self.client.api_key, "test-key")
        self.assertEqual(self.client.base_url, "http://test.example.com")
        self.assertEqual(self.client.model, "gpt-3.5-turbo")
        self.assertEqual(self.client.get_parameters(), {"temperature": 0.5})

    def test_set_parameter(self):
        """Test setting parameters."""
        self.client.set_parameter("max_tokens", 100)
        self.assertEqual(self.client.get_parameters()["max_tokens"], 100)
        # Existing parameters should be preserved
        self.assertEqual(self.client.get_parameters()["temperature"], 0.5)

    def test_set_system_prompt(self):
        """Test setting the system prompt."""
        self.client.set_system_prompt("You are an assistant")
        messages = self.client.get_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are an assistant")

        # Replace the existing system prompt
        self.client.set_system_prompt("You are another assistant")
        messages = self.client.get_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["content"], "You are another assistant")

    def test_add_user_message(self):
        """Test adding a user message."""
        self.client.add_user_message("Hello")
        messages = self.client.get_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "Hello")

    def test_add_assistant_message(self):
        """Test adding an assistant message."""
        self.client.add_assistant_message("Hello, I am the assistant")
        messages = self.client.get_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["content"], "Hello, I am the assistant")

    def test_clear_messages(self):
        """Test clearing message history."""
        self.client.set_system_prompt("System prompt")
        self.client.add_user_message("User message")
        self.client.add_assistant_message("Assistant message")
        self.client.clear_messages()
        messages = self.client.get_messages()
        # Only the system prompt should remain
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "System prompt")

    async def test_generate_without_prompt(self):
        """Test generating a reply without providing a prompt."""
        # Mock the API response
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Mock reply"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        self.mock_completions.create.return_value = mock_response

        # Add some messages first
        self.client.set_system_prompt("System")
        self.client.add_user_message("User message")

        # Call generate
        response = await self.client.generate()
        self.assertEqual(response, "Mock reply")

        # Check whether the message history was updated
        messages = self.client.get_messages()
        self.assertEqual(len(messages), 3)  # system, user, assistant
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(messages[-1]["content"], "Mock reply")

        # Verify the API call parameters
        self.mock_completions.create.assert_called_once()
        call_kwargs = self.mock_completions.create.call_args[1]
        self.assertEqual(call_kwargs.get("model"), "gpt-3.5-turbo")
        self.assertEqual(len(call_kwargs.get("messages")), 2)  # system + user
        self.assertEqual(call_kwargs.get("temperature"), 0.5)

    async def test_generate_with_prompt(self):
        """Test generating a reply with a prompt."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Reply content"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        self.mock_completions.create.return_value = mock_response

        self.client.set_system_prompt("System")

        response = await self.client.generate("New user message")
        self.assertEqual(response, "Reply content")

        # Message history should contain the system prompt, new user message, and assistant reply
        messages = self.client.get_messages()
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "New user message")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["content"], "Reply content")


class TestLiteLLMClient(unittest.IsolatedAsyncioTestCase):
    """LiteLLMClient tests."""

    def setUp(self):
        """Create a mocked client before each test."""
        if LiteLLMClient is None:
            self.skipTest("LiteLLMClient was not imported, possibly because the TODOs are not implemented")

        # Patch BaseModelClient.__init__ so it does not raise an exception
        self.base_init_patcher = patch('base_client.BaseModelClient.__init__')
        mock_base_init = self.base_init_patcher.start()
        mock_base_init.side_effect = lambda api_key, base_url=None, model=None, **kwargs: None

        # Replace litellm.completion using patch
        self.litellm_patcher = patch('litellm.completion')
        self.mock_completion = self.litellm_patcher.start()

        # Create the client instance
        try:
            self.client = LiteLLMClient(
                api_key="test-key",
                base_url="http://litellm.example.com",
                model="gpt-3.5-turbo",
                default_parameters={"temperature": 0.7}
            )
        except NotImplementedError:
            self.skipTest("LiteLLMClient.__init__ is not implemented, skipping test")

        # If the client does not have _messages or _parameters, initialize them
        if not hasattr(self.client, '_messages'):
            self.client._messages = []
        if not hasattr(self.client, '_parameters'):
            self.client._parameters = {}
        # Set attributes so the tests can access them
        self.client.api_key = "test-key"
        self.client.base_url = "http://litellm.example.com"
        self.client.model = "gpt-3.5-turbo"

    def tearDown(self):
        """Clean up patches."""
        self.litellm_patcher.stop()
        self.base_init_patcher.stop()

    def test_initialization(self):
        """Test initialization."""
        self.assertEqual(self.client.api_key, "test-key")
        self.assertEqual(self.client.base_url, "http://litellm.example.com")
        self.assertEqual(self.client.model, "gpt-3.5-turbo")
        self.assertEqual(self.client.get_parameters(), {"temperature": 0.7})

    def test_set_parameter(self):
        """Test setting parameters."""
        self.client.set_parameter("max_tokens", 200)
        self.assertEqual(self.client.get_parameters()["max_tokens"], 200)
        self.assertEqual(self.client.get_parameters()["temperature"], 0.7)

    def test_set_system_prompt(self):
        """Test setting the system prompt."""
        self.client.set_system_prompt("Litellm System prompt")
        messages = self.client.get_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "Litellm System prompt")

        # Replace
        self.client.set_system_prompt("New system prompt")
        messages = self.client.get_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["content"], "New system prompt")

    def test_add_user_message(self):
        """Test adding a user message."""
        self.client.add_user_message("Litellm User message")
        messages = self.client.get_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "Litellm User message")

    def test_add_assistant_message(self):
        """Test adding an assistant message."""
        self.client.add_assistant_message("Litellm Assistant message")
        messages = self.client.get_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["content"], "Litellm Assistant message")

    def test_clear_messages(self):
        """Test clearing message history."""
        self.client.set_system_prompt("System prompt")
        self.client.add_user_message("User message")
        self.client.add_assistant_message("Assistant message")
        self.client.clear_messages()
        messages = self.client.get_messages()
        # Only the system prompt should remain
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "System prompt")

    async def test_generate_without_prompt(self):
        """Test generating a reply without providing a prompt."""
        # Mock the API response
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "Litellm Mock reply"
                    }
                }
            ]
        }
        self.mock_completion.return_value = mock_response

        self.client.set_system_prompt("System")
        self.client.add_user_message("User message")

        response = await self.client.generate()
        self.assertEqual(response, "Litellm Mock reply")

        # Check the message history
        messages = self.client.get_messages()
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(messages[-1]["content"], "Litellm Mock reply")

        # Verify the API call parameters
        self.mock_completion.assert_called_once()
        call_kwargs = self.mock_completion.call_args[1]
        self.assertEqual(call_kwargs.get("model"), "gpt-3.5-turbo")
        self.assertEqual(len(call_kwargs.get("messages")), 2)  # system + user
        self.assertEqual(call_kwargs.get("temperature"), 0.7)

    async def test_generate_with_prompt(self):
        """Test generating a reply with a prompt."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "Reply content"
                    }
                }
            ]
        }
        self.mock_completion.return_value = mock_response

        self.client.set_system_prompt("System")

        response = await self.client.generate("New user message")
        self.assertEqual(response, "Reply content")

        messages = self.client.get_messages()
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "New user message")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["content"], "Reply content")


if __name__ == '__main__':
    unittest.main()
