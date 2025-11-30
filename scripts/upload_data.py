#!/usr/bin/env python3
"""
Script to upload OHLCV data to ClickHouse for a specific symbol and date.

Usage:
    python scripts/upload_data.py BTCUSDT 2024-01-15
    python scripts/upload_data.py ETHUSDT 2024-01-15
"""

import sys
from datetime import datetime

from clickhouse_driver import Client
from pybit.unified_trading import HTTP

from flying_trading.config import config
from flying_trading.data_collection import upload_to_clickhouse


def main():
    """Main function to upload data to ClickHouse."""
    if len(sys.argv) != 3:
        print("Usage: python scripts/upload_data.py <SYMBOL> <DATE>")
        print("Example: python scripts/upload_data.py BTCUSDT 2024-01-15")
        sys.exit(1)

    symbol = sys.argv[1].upper()
    date_str = sys.argv[2]

    # Validate date format
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        print(f"Error: Invalid date format. Expected YYYY-MM-DD, got: {date_str}")
        sys.exit(1)

    print(f"Uploading data for {symbol} on {date_str}...")

    try:
        # Connect to ClickHouse
        ch_client = Client.from_url(config.clickhouse_url)
        print(f"✓ Connected to ClickHouse: {config.clickhouse_url}")

        # Create HTTP client for Bybit
        http_client = HTTP(
            api_key=config.bybit_api_key,
            api_secret=config.bybit_api_secret,
        )
        print("✓ Connected to Bybit API")

        # Upload data
        upload_to_clickhouse(
            ch_client=ch_client,
            http_client=http_client,
            date=date_str,
            symbol=symbol,
        )

        print(f"✓ Successfully uploaded data for {symbol} on {date_str}")

    except ValueError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
