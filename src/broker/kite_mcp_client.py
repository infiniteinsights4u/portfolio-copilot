"""Hardened Zerodha Kite MCP client using Streamable HTTP with persistent sessions.

Enforces:
1. GTT-Only Brokerage Scope: Strictly allows GTT mutations (place_gtt_order,
   modify_gtt_order, delete_gtt_order) while prohibiting regular order tools
   (place_order, modify_order, cancel_order) for SEBI compliance exemption.
2. Persistent Session: Holds a long-lived HTTP session binding on mcp.kite.trade.
3. ToolCallInterceptor: Exponential backoff on HTTP 429 (rate limit) and 522 (Cloudflare timeout).
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Set

import structlog

logger = structlog.get_logger(__name__)

# Strict Tool Allowlist for Zerodha Kite MCP
ALLOWED_TOOLS: Set[str] = {
    # Authentication & Profile
    "login",
    "get_profile",
    # Account & Portfolio Reads
    "get_margins",
    "get_holdings",
    "get_positions",
    "get_orders",
    # Ephemeral Market Data Reads
    "get_quotes",
    "get_ltp",
    "get_ohlc",
    "get_historical_data",
    "search_instruments",
    # GTT Order Lifecycle (Exempt from SEBI Static IP & Algo ID rules)
    "get_gtts",
    "place_gtt_order",
    "modify_gtt_order",
    "delete_gtt_order",
}

# Explicitly Prohibited Tools (Hard Safety Boundary)
PROHIBITED_TOOLS: Set[str] = {
    "place_order",
    "modify_order",
    "cancel_order",
}


class ProhibitedToolError(PermissionError):
    """Raised when an attempt is made to call a forbidden order tool."""


class AuthenticationError(Exception):
    """Raised when Kite OAuth session has expired or user is not logged in."""


@dataclass
class LoginResult:
    """Result from Kite login tool."""

    url: str
    message: str


def extract_tool_result_text(result: Any) -> str:
    """Extract plain text string from MCP tool result objects."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if hasattr(result, "content") and result.content:
        content = result.content
        if isinstance(content, (list, tuple)) and len(content) > 0:
            first = content[0]
            if hasattr(first, "text"):
                return str(first.text)
            return str(first)
        return str(content)
    return str(result)


def parse_json_result(text: str, default: Any = None) -> Any:
    """Safely parse JSON from tool output text."""
    if not text:
        return default
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


class ToolCallInterceptor:
    """Interceptor executing tool calls with retry and backoff on transient errors."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_backoff_sec: float = 0.5,
        backoff_multiplier: float = 2.0,
    ) -> None:
        self.max_retries = max_retries
        self.initial_backoff_sec = initial_backoff_sec
        self.backoff_multiplier = backoff_multiplier

    async def execute_with_retry(
        self,
        call_fn: Callable[..., Any],
        tool_name: str,
        args: dict[str, Any],
    ) -> Any:
        """Execute a tool call with exponential backoff on HTTP 429/522."""
        backoff = self.initial_backoff_sec

        for attempt in range(1, self.max_retries + 1):
            try:
                return await call_fn(tool_name, args)
            except Exception as e:
                err_msg = str(e).lower()
                is_rate_limited = "429" in err_msg or "rate limit" in err_msg or "too many requests" in err_msg
                is_timeout = "522" in err_msg or "cloudflare" in err_msg or "gateway timeout" in err_msg

                if (is_rate_limited or is_timeout) and attempt < self.max_retries:
                    logger.warning(
                        "mcp_transient_error_retry",
                        tool=tool_name,
                        attempt=attempt,
                        backoff=backoff,
                        error=str(e),
                    )
                    await asyncio.sleep(backoff)
                    backoff *= self.backoff_multiplier
                    continue

                if "log in first" in err_msg or "session" in err_msg:
                    raise AuthenticationError(f"Kite session expired or not authenticated: {e}") from e

                raise


class KiteMCPClient:
    """Hardened Kite MCP client with allowlist enforcement and persistent session."""

    def __init__(
        self,
        server_url: str = "https://mcp.kite.trade/mcp",
        interceptor: ToolCallInterceptor | None = None,
    ) -> None:
        self.server_url = server_url
        self.interceptor = interceptor or ToolCallInterceptor()
        self._connected: bool = False
        self._logged_in: bool = False
        self._raw_client: Any = None
        self._lock = asyncio.Lock()

    def validate_tool_allowed(self, tool_name: str) -> None:
        """Validate tool name against strict allowlist.

        Raises:
            ProhibitedToolError: If tool is in PROHIBITED_TOOLS or not in ALLOWED_TOOLS.
        """
        if tool_name in PROHIBITED_TOOLS:
            msg = (
                f"FATAL: Tool '{tool_name}' is strictly prohibited! "
                "SEBI exemption mandates GTT-only orders; regular order execution is disabled."
            )
            logger.critical("prohibited_tool_blocked", tool=tool_name)
            raise ProhibitedToolError(msg)

        if tool_name not in ALLOWED_TOOLS:
            msg = f"Tool '{tool_name}' is not in the authorized Zerodha Kite MCP allowlist."
            logger.error("unauthorized_tool_blocked", tool=tool_name)
            raise ProhibitedToolError(msg)

    async def connect(self) -> None:
        """Establish or ensure connection to Kite MCP server."""
        async with self._lock:
            if not self._connected:
                try:
                    # Attempt connection via fastmcp client with HTTP streamable transport
                    from fastmcp import Client
                    self._raw_client = Client(self.server_url)
                    await self._raw_client.__aenter__()
                    self._connected = True
                    logger.info("kite_mcp_connected", server_url=self.server_url)
                except Exception as e:
                    logger.error("kite_mcp_connect_failed", error=str(e))
                    raise

    async def disconnect(self) -> None:
        """Cleanly close MCP session."""
        async with self._lock:
            if self._connected and self._raw_client:
                try:
                    await self._raw_client.__aexit__(None, None, None)
                except Exception:
                    pass
                self._connected = False
                self._raw_client = None
                logger.info("kite_mcp_disconnected")

    async def __aenter__(self) -> "KiteMCPClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.disconnect()

    async def call_tool(self, tool_name: str, args: dict[str, Any] | None = None) -> str:
        """Enforce allowlist and execute tool call via interceptor."""
        self.validate_tool_allowed(tool_name)
        call_args = args or {}

        async def _raw_call(name: str, a: dict[str, Any]) -> str:
            await self.connect()
            result = await self._raw_client.call_tool(name, a)
            return extract_tool_result_text(result)

        return await self.interceptor.execute_with_retry(_raw_call, tool_name, call_args)

    # --- High-level typed helper methods ---

    async def login(self) -> LoginResult:
        """Request OAuth login URL."""
        text = await self.call_tool("login")
        url_match = re.search(r"https://kite\.zerodha\.com/connect/login[^\s\)]+", text)
        url = url_match.group(0) if url_match else ""
        return LoginResult(url=url, message=text)

    async def is_logged_in(self) -> bool:
        """Verify if current session is active by querying user profile."""
        if self._logged_in:
            return True
        try:
            profile = await self.get_profile()
            if profile and profile.get("user_id"):
                self._logged_in = True
                return True
        except (AuthenticationError, Exception):
            pass
        self._logged_in = False
        return False

    def mark_logged_in(self) -> None:
        """Set local logged in status."""
        self._logged_in = True

    async def get_profile(self) -> dict[str, Any]:
        """Fetch user profile."""
        text = await self.call_tool("get_profile")
        data = parse_json_result(text, {})
        return data if isinstance(data, dict) else {}

    async def get_holdings(self) -> list[dict[str, Any]]:
        """Fetch long-term portfolio holdings."""
        text = await self.call_tool("get_holdings")
        data = parse_json_result(text, [])
        return data if isinstance(data, list) else []

    async def get_positions(self) -> dict[str, Any]:
        """Fetch net and day trading positions."""
        text = await self.call_tool("get_positions")
        data = parse_json_result(text, {"net": [], "day": []})
        return data if isinstance(data, dict) else {"net": [], "day": []}

    async def get_margins(self, segment: str | None = None) -> dict[str, Any]:
        """Fetch cash and buying power margins."""
        args = {"segment": segment} if segment else {}
        text = await self.call_tool("get_margins", args)
        data = parse_json_result(text, {})
        return data if isinstance(data, dict) else {}

    async def get_quotes(self, instruments: list[str]) -> dict[str, Any]:
        """Fetch real-time quotes."""
        text = await self.call_tool("get_quotes", {"instruments": instruments})
        data = parse_json_result(text, {})
        return data if isinstance(data, dict) else {}

    async def get_ltp(self, instruments: list[str]) -> dict[str, Any]:
        """Fetch last traded price."""
        text = await self.call_tool("get_ltp", {"instruments": instruments})
        data = parse_json_result(text, {})
        return data if isinstance(data, dict) else {}

    async def search_instruments(self, query: str) -> list[dict[str, Any]]:
        """Search instruments."""
        text = await self.call_tool("search_instruments", {"search_q": query})
        data = parse_json_result(text, [])
        return data if isinstance(data, list) else []

    # --- GTT Methods ---

    async def get_gtts(self) -> list[dict[str, Any]]:
        """Fetch active Good Till Triggered orders."""
        text = await self.call_tool("get_gtts")
        data = parse_json_result(text, [])
        return data if isinstance(data, list) else []

    async def place_gtt_order(
        self,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,
        trigger_type: str,
        trigger_values: list[float],
        limit_prices: list[float],
        quantities: list[int],
        product: str = "CNC",
    ) -> dict[str, Any]:
        """Place a GTT order (Stage 1 Single-Leg Buy or Stage 2 Two-Leg Sell OCO)."""
        args = {
            "tradingsymbol": tradingsymbol.upper(),
            "exchange": exchange.upper(),
            "transaction_type": transaction_type.upper(),
            "trigger_type": trigger_type.lower(),
            "trigger_values": [float(v) for v in trigger_values],
            "limit_prices": [float(p) for p in limit_prices],
            "quantities": [int(q) for q in quantities],
            "product": product.upper(),
        }
        text = await self.call_tool("place_gtt_order", args)
        data = parse_json_result(text, {"raw": text})
        return data if isinstance(data, dict) else {"raw": text}

    async def modify_gtt_order(
        self,
        trigger_id: int,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,
        trigger_type: str,
        trigger_values: list[float],
        limit_prices: list[float],
        quantities: list[int],
        product: str = "CNC",
    ) -> dict[str, Any]:
        """Modify an active GTT order."""
        if not isinstance(trigger_id, int):
            trigger_id = int(trigger_id)

        args = {
            "trigger_id": trigger_id,
            "tradingsymbol": tradingsymbol.upper(),
            "exchange": exchange.upper(),
            "transaction_type": transaction_type.upper(),
            "trigger_type": trigger_type.lower(),
            "trigger_values": [float(v) for v in trigger_values],
            "limit_prices": [float(p) for p in limit_prices],
            "quantities": [int(q) for q in quantities],
            "product": product.upper(),
        }
        text = await self.call_tool("modify_gtt_order", args)
        data = parse_json_result(text, {"raw": text})
        return data if isinstance(data, dict) else {"raw": text}

    async def delete_gtt_order(self, trigger_id: int) -> dict[str, Any]:
        """Delete an active GTT order ensuring trigger_id is an integer."""
        if not isinstance(trigger_id, int):
            trigger_id = int(trigger_id)

        args = {"trigger_id": trigger_id}
        text = await self.call_tool("delete_gtt_order", args)
        data = parse_json_result(text, {"raw": text})
        return data if isinstance(data, dict) else {"raw": text}
