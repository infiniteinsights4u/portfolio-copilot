"""Application settings management enforcing strict configuration precedence.

Precedence order:
    1. CLI flags (explicit overrides)
    2. Environment variables
    3. .env file
    4. Code defaults (safe defaults: DRY_RUN=True, DDPI warning)
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dotenv import dotenv_values, find_dotenv
import structlog

logger = structlog.get_logger(__name__)

# Constants & Defaults
DEFAULT_DRY_RUN: bool = True
DEFAULT_KITE_MCP_URL: str = "https://mcp.kite.trade/mcp"
DEFAULT_ZERODHA_DDPI_ENABLED: bool = False
DEFAULT_DB_PATH: Path = Path("data/copilot.db")
DEFAULT_AUDIT_LOG_PATH: Path = Path("data/trade_audit.jsonl")
DEFAULT_MAX_RISK_PER_TRADE_PCT: float = 0.01  # 1% worst-case loss cap
DEFAULT_MAX_POSITION_SIZE_PCT: float = 0.10   # 10% portfolio cap per single stock
DEFAULT_MAX_SECTOR_SIZE_PCT: float = 0.25     # 25% portfolio cap per sector
DEFAULT_DUPLICATE_ORDER_WINDOW_SEC: int = 30  # 30-second duplicate suppression


def _str_to_bool(val: Any, default: bool = False) -> bool:
    """Parse string representation of boolean."""
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    str_val = str(val).strip().lower()
    if str_val in ("1", "true", "yes", "on", "t"):
        return True
    if str_val in ("0", "false", "no", "off", "f"):
        return False
    return default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings container."""

    dry_run: bool = DEFAULT_DRY_RUN
    kite_mcp_url: str = DEFAULT_KITE_MCP_URL
    zerodha_ddpi_enabled: bool = DEFAULT_ZERODHA_DDPI_ENABLED
    db_path: Path = DEFAULT_DB_PATH
    audit_log_path: Path = DEFAULT_AUDIT_LOG_PATH
    max_risk_per_trade_pct: float = DEFAULT_MAX_RISK_PER_TRADE_PCT
    max_position_size_pct: float = DEFAULT_MAX_POSITION_SIZE_PCT
    max_sector_size_pct: float = DEFAULT_MAX_SECTOR_SIZE_PCT
    duplicate_order_window_sec: int = DEFAULT_DUPLICATE_ORDER_WINDOW_SEC
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    claude_model: str = "claude-sonnet-4-20250514"
    slack_webhook_url: str | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    schedule_timezone: str = "Asia/Kolkata"

    def check_ddpi_status(self) -> bool:
        """Check DDPI status and emit diagnostic warning if disabled.

        Returns:
            True if DDPI is enabled, False otherwise.
        """
        if not self.zerodha_ddpi_enabled:
            msg = (
                "[SECURITY/OPERATIONAL WARNING] ZERODHA_DDPI_ENABLED is False! "
                "Triggered CNC sell GTTs (Stage 2 Stop-Loss/Target) will fail execution "
                "without daily CDSL TPIN/eDIS pre-authorization. "
                "Enable DDPI online in Zerodha Console (₹100 + GST) or authorize daily after 7:00 AM IST."
            )
            logger.warning("ddpi_disabled_warning", message=msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
            return False
        return True


def load_settings(
    cli_overrides: Mapping[str, Any] | None = None,
    dotenv_path: str | Path | None = None,
) -> Settings:
    """Load settings according to CLI > Env > .env > Defaults precedence.

    Args:
        cli_overrides: Mapping of explicit CLI flags or runtime overrides.
        dotenv_path: Optional custom path to .env file.

    Returns:
        Fully resolved Settings instance.
    """
    cli = dict(cli_overrides or {})

    # Load from .env file if available
    resolved_dotenv = dotenv_path or find_dotenv(usecwd=True)
    file_vars = dotenv_values(resolved_dotenv) if resolved_dotenv else {}

    def get_val(key: str, env_key: str, default: Any, caster: Any = str) -> Any:
        # 1. CLI flags (explicit overrides)
        if key in cli and cli[key] is not None:
            return caster(cli[key])
        if env_key in cli and cli[env_key] is not None:
            return caster(cli[env_key])

        # 2. Environment variables
        if env_key in os.environ and os.environ[env_key] != "":
            return caster(os.environ[env_key])

        # 3. .env file
        if env_key in file_vars and file_vars[env_key] is not None and file_vars[env_key] != "":
            return caster(file_vars[env_key])

        # 4. Defaults
        return default

    settings = Settings(
        dry_run=get_val("dry_run", "DRY_RUN", DEFAULT_DRY_RUN, _str_to_bool),
        kite_mcp_url=get_val("kite_mcp_url", "KITE_MCP_URL", DEFAULT_KITE_MCP_URL, str),
        zerodha_ddpi_enabled=get_val(
            "zerodha_ddpi_enabled",
            "ZERODHA_DDPI_ENABLED",
            DEFAULT_ZERODHA_DDPI_ENABLED,
            _str_to_bool,
        ),
        db_path=get_val("db_path", "DB_PATH", DEFAULT_DB_PATH, Path),
        audit_log_path=get_val("audit_log_path", "AUDIT_LOG_PATH", DEFAULT_AUDIT_LOG_PATH, Path),
        max_risk_per_trade_pct=get_val(
            "max_risk_per_trade_pct",
            "MAX_RISK_PER_TRADE_PCT",
            DEFAULT_MAX_RISK_PER_TRADE_PCT,
            float,
        ),
        max_position_size_pct=get_val(
            "max_position_size_pct",
            "MAX_POSITION_SIZE_PCT",
            DEFAULT_MAX_POSITION_SIZE_PCT,
            float,
        ),
        max_sector_size_pct=get_val(
            "max_sector_size_pct",
            "MAX_SECTOR_SIZE_PCT",
            DEFAULT_MAX_SECTOR_SIZE_PCT,
            float,
        ),
        duplicate_order_window_sec=get_val(
            "duplicate_order_window_sec",
            "DUPLICATE_ORDER_WINDOW_SEC",
            DEFAULT_DUPLICATE_ORDER_WINDOW_SEC,
            int,
        ),
        llm_provider=get_val("llm_provider", "LLM_PROVIDER", "ollama", str),
        ollama_base_url=get_val("ollama_base_url", "OLLAMA_BASE_URL", "http://localhost:11434", str),
        ollama_model=get_val("ollama_model", "OLLAMA_MODEL", "llama3", str),
        claude_model=get_val("claude_model", "CLAUDE_MODEL", "claude-sonnet-4-20250514", str),
        slack_webhook_url=get_val("slack_webhook_url", "SLACK_WEBHOOK_URL", None, str),
        telegram_bot_token=get_val("telegram_bot_token", "TELEGRAM_BOT_TOKEN", None, str),
        telegram_chat_id=get_val("telegram_chat_id", "TELEGRAM_CHAT_ID", None, str),
        schedule_timezone=get_val("schedule_timezone", "SCHEDULE_TIMEZONE", "Asia/Kolkata", str),
    )

    return settings


# Global cached settings instance
_SETTINGS_INSTANCE: Settings | None = None


def get_settings(force_reload: bool = False, **cli_overrides: Any) -> Settings:
    """Retrieve global cached settings or reload with fresh overrides."""
    global _SETTINGS_INSTANCE
    if _SETTINGS_INSTANCE is None or force_reload or cli_overrides:
        _SETTINGS_INSTANCE = load_settings(cli_overrides=cli_overrides)
    return _SETTINGS_INSTANCE
