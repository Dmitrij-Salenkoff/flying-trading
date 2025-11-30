from datetime import datetime, timedelta

from clickhouse_driver import Client
from pybit.unified_trading import HTTP

from .config import config


def upload_to_clickhouse(
    ch_client: Client,
    http_client: HTTP,
    date: str,  # YYYY-MM-DD
    symbol: str,
):
    """
    Uploads OHLCV data to Clickhouse.
    The data is fetched from the HTTP client and uploaded to the Clickhouse client.
    All data is stored in a unified 'klines' table with 1-minute resolution.
    The data is uploaded in the following format:
    - symbol
    - timestamp
    - open
    - high
    - low
    - close
    - volume
    """
    # Parse date string to datetime
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    start_dt = date_obj.replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt = start_dt + timedelta(days=1)

    # Convert to milliseconds timestamp
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    # Always use 1-minute interval for storage
    # Data is always stored with 1-minute resolution
    interval_clean = "1"
    interval_ms = 60 * 1000  # 1 minute in milliseconds

    # Fetch data from Bybit
    all_rows = []
    current_start = start_ms
    limit = 200  # Bybit API limit

    while current_start < end_ms:
        resp = http_client.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval_clean,
            start=current_start,
            limit=limit,
        )

        # Check API response
        if resp.get("retCode") != 0:
            raise ValueError(f"Bybit API error: {resp.get('retMsg', 'Unknown error')}")

        result = resp.get("result", {})
        rows = result.get("list", [])

        if not rows:
            break

        # Bybit returns data from newest to oldest, sort by timestamp
        rows_sorted = sorted(rows, key=lambda r: int(r[0]))

        for row in rows_sorted:
            ts = int(row[0])
            if ts < start_ms or ts >= end_ms:
                continue
            all_rows.append(row)

        if not rows_sorted:
            break

        last_ts = int(rows_sorted[-1][0])

        # If we've reached the end, break
        if last_ts + interval_ms >= end_ms:
            break

        current_start = last_ts + interval_ms

    if not all_rows:
        return

    # Prepare data for ClickHouse
    # Format: [timestamp, open, high, low, close, volume, turnover?]
    data_to_insert = []
    symbol_lower = symbol.lower()
    for row in all_rows:
        timestamp = int(row[0])
        open_price = float(row[1])
        high = float(row[2])
        low = float(row[3])
        close = float(row[4])
        volume = float(row[5])

        data_to_insert.append(
            (
                symbol_lower,
                timestamp,
                open_price,
                high,
                low,
                close,
                volume,
            )
        )

    # Insert into unified klines table
    ch_client.execute(
        "INSERT INTO trading.klines (symbol, timestamp, open, high, low, close, volume) VALUES",
        data_to_insert,
    )


if __name__ == "__main__":
    ch_client = Client.from_url(config.clickhouse_url)
    http_client = HTTP(api_key=config.bybit_api_key, api_secret=config.bybit_api_secret)
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    upload_to_clickhouse(
        ch_client,
        http_client,
        yesterday,
        "BTCUSDT",
    )
