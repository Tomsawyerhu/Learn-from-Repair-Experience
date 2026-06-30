"""
Dialog Agent class implementation, responsible for configuration management, template processing and client switching.
"""

import os
import yaml
import logging
from typing import Optional, Dict, Any, Type, Union
from string import Template
from pathlib import Path

# Import client implementations
from src.model.base_client import BaseModelClient
from src.model.litellm_client import LiteLLMClient
from src.model.openai_client import OpenAIClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import re


class CustomTemplate:
    """
    Custom template class, supports {{variable}} syntax.
    """

    def __init__(self, template_str: str):
        self.template_str = template_str

    def substitute(self, mapping: Dict[str, Any]) -> str:
        """
        Replace variables, raise KeyError if missing.
        """
        pattern = r"\{\{([a-zA-Z0-9_]+)\}\}"

        def replace(match):
            key = match.group(1)
            if key in mapping:
                return str(mapping[key])
            else:
                raise KeyError(key)

        return re.sub(pattern, replace, self.template_str)

    def safe_substitute(self, mapping: Dict[str, Any]) -> str:
        """
        Safe replacement, keep original if missing.
        """
        pattern = r"\{\{([a-zA-Z0-9_]+)\}\}"

        def replace(match):
            key = match.group(1)
            return str(mapping.get(key, match.group(0)))

        return re.sub(pattern, replace, self.template_str)


class Agent:
    """
    Dialog Agent class, encapsulates model interaction logic.

    Features:
    1. Read configuration from YAML
    2. Support template rendering
    3. Support runtime switching of underlying client (LiteLLM / OpenAI)
    """

    def __init__(self, config_path: str, client_type: str = "openai"):
        """
        Initialize Agent.

        Args:
            config_path: Config file path (YAML)
            client_type: Initial client type to use ("openai" or "litellm")
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)

        self.api_key = self.config.get("api_key", "")
        self.base_url = self.config.get("base_url")
        self.model_name = self.config.get("model_name", "gpt-3.5-turbo")
        self.temperature = self.config.get("temperature", 0.7)
        self.templates = self.config.get("templates", {})

        self.client: Optional[BaseModelClient] = None
        self.client_type = client_type

        # Initialize client
        self.switch_client(client_type)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load YAML config file.

        Args:
            config_path: File path

        Returns:
            Config dictionary

        Raises:
            FileNotFoundError: Config file does not exist
            yaml.YAMLError: YAML parsing error
        """
        path = Path(config_path)
        if not path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config or {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise

    def switch_client(self, client_type: str) -> None:
        """
        Switch underlying model client.

        Args:
            client_type: "openai" or "litellm"

        Raises:
            ValueError: Unsupported client type
        """
        logger.info(f"Switching client to: {client_type}")

        default_params = {"temperature": self.temperature}

        if client_type.lower() == "openai":
            self.client = OpenAIClient(
                api_key=self.api_key, base_url=self.base_url, model=self.model_name, default_parameters=default_params
            )
        elif client_type.lower() == "litellm":
            self.client = LiteLLMClient(
                api_key=self.api_key, base_url=self.base_url, model=self.model_name, default_parameters=default_params
            )
        else:
            logger.error(f"Unsupported client type: {client_type}")
            raise ValueError(f"Unsupported client type: {client_type}. Use 'openai' or 'litellm'.")

        self.client_type = client_type

    def build_prompt(self, context_dict: Dict[str, Any], template_name: str = "default") -> str:
        """
        Build Prompt.
        Based on template name and context variables, generate final user prompt string.
        Also sets System Prompt (if configured).

        Args:
            context_dict: Dictionary containing template variables
            template_name: Template name, corresponds to key under templates in config file

        Returns:
            Formatted user Prompt string

        Raises:
            ValueError: Template does not exist or variable is missing
        """
        if template_name not in self.templates:
            error_msg = f"Template '{template_name}' not found in configuration."
            logger.error(error_msg)
            raise ValueError(error_msg)

        template_config = self.templates[template_name]

        # Process System Template
        system_tmpl_str = template_config.get("system_template")
        if system_tmpl_str:
            try:
                # Use safe_substitute to allow missing variables in system prompt (or can also enforce)
                # Here for flexibility, assume System Prompt may not need variables or variables are optional
                # But the problem requires handling missing variable exceptions, so we are stricter on User Template
                system_prompt = CustomTemplate(system_tmpl_str).safe_substitute(context_dict)
                self.client.set_system_prompt(system_prompt)
            except Exception as e:
                logger.warning(f"Error formatting system template: {e}")

        # Process User Template
        user_tmpl_str = template_config.get("user_template", "")
        if not user_tmpl_str:
            logger.warning(f"User template for '{template_name}' is empty.")
            return ""

        try:
            # Use substitute, will raise KeyError if variable is missing
            user_prompt = CustomTemplate(user_tmpl_str).substitute(context_dict)
            return user_prompt
        except KeyError as e:
            error_msg = f"Missing variable in context for template '{template_name}': {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Error formatting user template: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    async def chat(self, context_dict: Dict[str, Any], template_name: str = "default") -> str:
        """
        Execute dialog.
        Build Prompt and call client to generate response.

        Args:
            context_dict: Context variables
            template_name: Template name

        Returns:
            Model response
        """
        prompt = self.build_prompt(context_dict, template_name)
        if not prompt:
            logger.warning("Generated empty prompt, skipping generation.")
            return ""

        try:
            response = await self.client.generate(prompt)
            return response
        except Exception as e:
            logger.error(f"Error during chat generation: {e}")
            raise

    def get_history(self) -> list:
        """Get current dialog history"""
        if self.client:
            return self.client.get_messages()
        return []

    def clear_history(self):
        """Clear dialog history"""
        if self.client:
            self.client.clear_messages()

    def get_last_cost(self) -> float:
        """Get last call cost

        Returns:
            Cost value (USD), litellm client returns actual cost, openai client returns 0
        """
        if self.client:
            return self.client.get_last_cost()
        return 0.0
