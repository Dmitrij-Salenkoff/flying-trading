#!/usr/bin/env python3
"""
Script to upload OHLCV data from Bybit to StarRocks.

Usage:
    # Upload single day:
    python scripts/upload_starrocks.py BTCUSDT 2024-01-15

    # Upload date range:
    python scripts/upload_starrocks.py BTCUSDT 2024-01-01 2024-01-31

Examples:
    python scripts/upload_starrocks.py BTCUSDT 2024-12-01
    python scripts/upload_starrocks.py ETHUSDT 2024-11-01 2024-11-30
"""

import sys
from datetime import datetime

from flying_trading.starrocks_loader import (
    get_starrocks_connection,
    upload_date_range,
    upload_to_starrocks,
)


def main():
    """Main function to upload data to StarRocks."""
    if len(sys.argv) < 3:
        print("Usage: python scripts/upload_starrocks.py <SYMBOL> <DATE> [END_DATE]")
        print("Examples:")
        print("  python scripts/upload_starrocks.py BTCUSDT 2024-01-15")
        print("  python scripts/upload_starrocks.py BTCUSDT 2024-01-01 2024-01-31")
        sys.exit(1)

    symbol = sys.argv[1].upper()
    start_date = sys.argv[2]
    end_date = sys.argv[3] if len(sys.argv) > 3 else None

    # Validate date format
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        print(f"Error: Invalid date format. Expected YYYY-MM-DD: {e}")
        sys.exit(1)

    try:
        # Test connection
        connection = get_starrocks_connection()
        print(f"✓ Connected to StarRocks")

        if end_date:
            print(f"Uploading {symbol} data from {start_date} to {end_date}...")
            rows = upload_date_range(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                connection=connection,
            )
        else:
            print(f"Uploading {symbol} data for {start_date}...")
            rows = upload_to_starrocks(
                symbol=symbol,
                date=start_date,
                connection=connection,
            )

        print(f"✓ Successfully uploaded {rows} rows")

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

