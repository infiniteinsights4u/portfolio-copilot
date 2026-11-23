#!/usr/bin/env python3
"""Local zerodha/kite-mcp-server failover and management script.

Enables seamless switching between hosted https://mcp.kite.trade/mcp and a
locally self-hosted Go binary (zerodha/kite-mcp-server) on http://localhost:8080/mcp.

MANDATORY SAFETY RULE:
Whenever running the local kite-mcp-server, EXCLUDED_TOOLS must be set to:
  EXCLUDED_TOOLS=place_order,modify_order,cancel_order
to permanently block non-GTT trading endpoints at the server layer.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request

HOSTED_KITE_MCP_URL = "https://mcp.kite.trade/mcp"
LOCAL_KITE_MCP_URL = "http://localhost:8080/mcp"
MANDATORY_EXCLUDED_TOOLS = "place_order,modify_order,cancel_order"


def get_project_root() -> Path:
    """Find the project root containing .env or pyproject.toml."""
    current = Path(__file__).resolve().parent
    for parent in [current, current.parent, current.parent.parent]:
        if (parent / ".env").exists() or (parent / "pyproject.toml").exists():
            return parent
    return current.parent


def get_current_mcp_url(env_path: Path) -> str:
    """Read the active KITE_MCP_URL from environment or .env file."""
    if "KITE_MCP_URL" in os.environ:
        return os.environ["KITE_MCP_URL"]

    if env_path.exists():
        content = env_path.read_text(encoding="utf-8")
        match = re.search(r"^KITE_MCP_URL\s*=\s*([^\r\n#]+)", content, re.MULTILINE)
        if match:
            return match.group(1).strip().strip("'\"")

    return HOSTED_KITE_MCP_URL


def update_env_file(env_path: Path, target_url: str) -> None:
    """Update or append KITE_MCP_URL in the .env file."""
    if not env_path.exists():
        env_path.write_text(f"KITE_MCP_URL={target_url}\n", encoding="utf-8")
        print(f"[OK] Created {env_path.name} with KITE_MCP_URL={target_url}")
        return

    content = env_path.read_text(encoding="utf-8")
    pattern = r"^KITE_MCP_URL\s*=.*$"

    if re.search(pattern, content, re.MULTILINE):
        new_content = re.sub(pattern, f"KITE_MCP_URL={target_url}", content, flags=re.MULTILINE)
    else:
        new_content = content.rstrip() + f"\nKITE_MCP_URL={target_url}\n"

    env_path.write_text(new_content, encoding="utf-8")
    print(f"[OK] Updated {env_path.name} with KITE_MCP_URL={target_url}")


def check_health(url: str, timeout: float = 3.0) -> bool:
    """Probe an MCP endpoint to check if the server is reachable."""
    print(f"[*] Probing health on {url} ...")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "DipCopilot-HealthProbe/1.0"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            print(f"[+] Health probe succeeded: HTTP {resp.status}")
            return True
    except urllib.error.HTTPError as e:
        # MCP streamable HTTP often returns 405 Method Not Allowed on plain GET, or 400, which confirms reachability
        if e.code in (200, 204, 400, 404, 405):
            print(f"[+] Server reachable (HTTP {e.code})")
            return True
        print(f"[-] Server returned HTTP {e.code}: {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"[-] Connection failed: {e.reason}")
        return False
    except Exception as e:
        print(f"[-] Unexpected error probing health: {e}")
        return False


def print_status(env_path: Path) -> None:
    """Print current configuration status."""
    url = get_current_mcp_url(env_path)
    is_local = "localhost" in url or "127.0.0.1" in url

    print("=" * 60)
    print("Zerodha Kite MCP Configuration Status")
    print("=" * 60)
    print(f"Active KITE_MCP_URL   : {url}")
    print(f"Mode                  : {'LOCAL SELF-HOSTED' if is_local else 'HOSTED (mcp.kite.trade)'}")
    print(f"Required Exclusions   : EXCLUDED_TOOLS={MANDATORY_EXCLUDED_TOOLS}")
    print("=" * 60)


def generate_local_launch_command(binary_path: str = "kite-mcp-server") -> str:
    """Generate shell command to launch local kite-mcp-server with mandatory exclusions."""
    return (
        f"export EXCLUDED_TOOLS={MANDATORY_EXCLUDED_TOOLS}\n"
        f"export PORT=8080\n"
        f"{binary_path}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Switch between hosted mcp.kite.trade and local kite-mcp-server"
    )
    parser.add_argument(
        "action",
        choices=["status", "switch-local", "switch-hosted", "check-health", "print-cmd"],
        nargs="?",
        default="status",
        help="Action to perform",
    )
    parser.add_argument(
        "--url",
        help="Custom URL override for local switch",
        default=LOCAL_KITE_MCP_URL,
    )

    args = parser.parse_args()
    project_root = get_project_root()
    env_file = project_root / ".env"

    if args.action == "status":
        print_status(env_file)
    elif args.action == "switch-local":
        print("[*] Switching KITE_MCP_URL to local server...")
        update_env_file(env_file, args.url)
        print("\nIMPORTANT: Ensure your local Go server is started with:")
        print(f"  EXCLUDED_TOOLS={MANDATORY_EXCLUDED_TOOLS}")
        print_status(env_file)
    elif args.action == "switch-hosted":
        print("[*] Switching KITE_MCP_URL to hosted mcp.kite.trade...")
        update_env_file(env_file, HOSTED_KITE_MCP_URL)
        print_status(env_file)
    elif args.action == "check-health":
        active_url = get_current_mcp_url(env_file)
        alive = check_health(active_url)
        if not alive:
            return 1
    elif args.action == "print-cmd":
        print("# Command to run local zerodha/kite-mcp-server binary safely:")
        print(generate_local_launch_command())

    return 0


if __name__ == "__main__":
    sys.exit(main())
