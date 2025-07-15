"""
config_manager.py
Module for loading and managing project configuration files.
"""

import json
import os
from typing import Any, Dict, Optional

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")

class ConfigManager:
    """
    Centralized manager for loading and accessing project configuration files.
    """
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = config_dir or CONFIG_DIR
        self._model_config = None
        self._neo4j_config = None
        self._paths_config = None

    def load_json(self, filename: str) -> Dict[str, Any]:
        """
        Load a JSON configuration file from the config directory.
        """
        path = os.path.join(self.config_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def model_config(self) -> Dict[str, Any]:
        """
        Get the model configuration as a dictionary.
        """
        if self._model_config is None:
            self._model_config = self.load_json("model_config.json")
        return self._model_config

    @property
    def neo4j_config(self) -> Dict[str, Any]:
        """
        Get the Neo4j configuration as a dictionary.
        """
        if self._neo4j_config is None:
            self._neo4j_config = self.load_json("neo4j_config.json")
        return self._neo4j_config

    @property
    def paths_config(self) -> Dict[str, Any]:
        """
        Get the paths configuration as a dictionary.
        """
        if self._paths_config is None:
            self._paths_config = self.load_json("paths.json")
        return self._paths_config

    def get_llm_config(self, name: str = "default") -> Dict[str, Any]:
        """
        Get the LLM model configuration by name, falling back to 'default' if not found.
        """
        llm_config = self.model_config.get("llm", {})
        config = llm_config.get(name, llm_config.get("default"))
        if config is None:
            raise ValueError(f"No LLM config found for name '{name}' and no default provided.")
        return config

# Singleton instance for convenience
config_manager = ConfigManager()

def load_config() -> Dict[str, Any]:
    """
    Load the entire model configuration (for backward compatibility).
    """
    return config_manager.model_config

def llm_config_selector(func):
    """
    Decorator to inject the appropriate LLM model configuration into the decorated function.
    Uses the 'name' keyword argument (or 'default') to select the model config,
    and passes it as 'model_config' to the function.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        name = kwargs.get("name") or "default"
        model_config = config_manager.get_llm_config(name)
        kwargs["model_config"] = model_config
        return func(*args, **kwargs)
    return wrapper 