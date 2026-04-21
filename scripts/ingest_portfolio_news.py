import asyncio
import sys

from src.data.ingestion import ingest_news
from src.mcp.kite_client import AuthenticationError, KiteClient


async def main(limit: int) -> None:
    kite = KiteClient()
    await kite.connect()

    try:
        holdings = await kite.get_holdings()
        symbols = sorted(
            {
                (h.get("tradingsymbol") or "").upper().strip()
                for h in (holdings or [])
                if (h.get("tradingsymbol") or "").strip()
            }
        )

        if not symbols:
            print("No portfolio symbols found from Kite holdings.")
            return

        stats = await ingest_news(symbols=symbols, limit=limit)

        print(f"Ingestion complete for {len(symbols)} symbols (limit={limit}).")
        print(f"  Articles fetched: {stats.articles_fetched}")
        print(f"  Chunks stored: {stats.chunks_stored}")
        if stats.errors:
            print(f"  Errors: {len(stats.errors)} (showing up to 5)")
            for err in stats.errors[:5]:
                print(f"    - {err}")
    finally:
        await kite.disconnect()


if __name__ == "__main__":
    # Usage: python scripts/ingest_portfolio_news.py [limit]
    # LIMIT controls number of articles per source/symbol (latest only).
    try:
        limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    except ValueError:
        limit = 10

    try:
        asyncio.run(main(limit))
    except AuthenticationError as e:
        print(str(e))
        print("Run `make web` (and log in) or the CLI `login` command first.")
        raise

