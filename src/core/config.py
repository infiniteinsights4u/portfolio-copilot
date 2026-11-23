"""Configuration management for LLM providers and compatibility layer for application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Final, Literal

from dotenv import load_dotenv

from src.config.settings import Settings, get_settings, load_settings

load_dotenv()

# Environment variable names
ENV_LLM_PROVIDER = "LLM_PROVIDER"
ENV_OLLAMA_BASE_URL = "OLLAMA_BASE_URL"
ENV_OLLAMA_MODEL = "OLLAMA_MODEL"
ENV_CLAUDE_MODEL = "CLAUDE_MODEL"

# Provider identifiers
PROVIDER_OLLAMA: Final = "ollama"
PROVIDER_CLAUDE: Final = "claude"

# Default configuration values
DEFAULT_PROVIDER: str = PROVIDER_OLLAMA
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"

ProviderType = Literal[PROVIDER_OLLAMA, PROVIDER_CLAUDE]


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""

    provider: ProviderType = DEFAULT_PROVIDER
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    claude_model: str = DEFAULT_CLAUDE_MODEL

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Load configuration delegating to core application settings."""
        settings = get_settings()
        provider = settings.llm_provider.lower()
        if provider not in (PROVIDER_OLLAMA, PROVIDER_CLAUDE):
            provider = DEFAULT_PROVIDER

        return cls(
            provider=provider,  # type: ignore[arg-type]
            ollama_base_url=settings.ollama_base_url or DEFAULT_OLLAMA_BASE_URL,
            ollama_model=settings.ollama_model or DEFAULT_OLLAMA_MODEL,
            claude_model=settings.claude_model or DEFAULT_CLAUDE_MODEL,
        )


# Global config instance (singleton pattern)
_config: LLMConfig | None = None


def get_config() -> LLMConfig:
    """Get the current LLM configuration.

    Returns a singleton instance, delegating to application settings.
    """
    global _config
    if _config is None:
        _config = LLMConfig.from_env()
    return _config


def get_app_settings(**overrides: Any) -> Settings:
    """Convenience accessor to core application settings."""
    return get_settings(**overrides)


def set_provider(
    provider: ProviderType,
    model: str | None = None,
) -> None:
    """Override the current provider configuration.

    Used by CLI to set provider from command-line arguments.

    Args:
        provider: The provider to use ("ollama" or "claude")
        model: Optional model name override
    """
    global _config
    config = get_config()

    ollama_model = model if model and provider == "ollama" else config.ollama_model
    claude_model = model if model and provider == "claude" else config.claude_model

    # Sync into global settings singleton
    get_settings(
        force_reload=True,
        llm_provider=provider,
        ollama_model=ollama_model,
        claude_model=claude_model,
    )

    # Create a new config with updated values
    _config = LLMConfig(
        provider=provider,
        ollama_base_url=config.ollama_base_url,
        ollama_model=ollama_model,
        claude_model=claude_model,
    )


def reset_config() -> None:
    """Reset configuration to reload from environment.

    Useful for testing.
    """
    global _config
    _config = None
