"""Backwards compatibility adapter for Kite MCP client.

Re-exports KiteMCPClient from src.broker.kite_mcp_client as KiteClient.
"""

from __future__ import annotations

from src.broker.kite_mcp_client import (
    ALLOWED_TOOLS,
    PROHIBITED_TOOLS,
    AuthenticationError,
    KiteMCPClient,
    LoginResult,
    ProhibitedToolError,
    ToolCallInterceptor,
    extract_tool_result_text as extract_text,
    parse_json_result as parse_json_response,
)

# Alias KiteMCPClient as KiteClient for drop-in compatibility
KiteClient = KiteMCPClient

__all__ = [
    "KiteClient",
    "KiteMCPClient",
    "AuthenticationError",
    "ProhibitedToolError",
    "LoginResult",
    "ALLOWED_TOOLS",
    "PROHIBITED_TOOLS",
    "extract_text",
    "parse_json_response",
]
