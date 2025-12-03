"""
Module for loading OHLCV data from Bybit into StarRocks.

This module provides functionality to fetch kline (candlestick) data from the
Bybit exchange and insert it into a StarRocks database table.

StarRocks uses MySQL-compatible protocol, so we use pymysql for database connections.
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pymysql
from pymysql.connections import Connection
from pybit.unified_trading import HTTP

from .config import config
from .logger import get_logger

logger = get_logger(__name__)


def get_starrocks_connection() -> Connection:
    """Create and return a connection to StarRocks database."""
    return pymysql.connect(
        host=config.starrocks_host,
        port=config.starrocks_port,
        user=config.starrocks_user,
        password=config.starrocks_password,
        database=config.starrocks_database,
    )


def fetch_bybit_ohlcv(
    http_client: HTTP,
    symbol: str,
    start_ms: int,
    end_ms: int,
    interval: str = "1",
    limit: int = 200,
) -> list[dict]:
    """
    Fetch OHLCV data from Bybit API.

    Args:
        http_client: Bybit HTTP client
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        start_ms: Start timestamp in milliseconds
        end_ms: End timestamp in milliseconds
        interval: Kline interval (default "1" for 1 minute)
        limit: Number of candles per request (max 200)

    Returns:
        List of OHLCV dictionaries
    """
    all_rows = []
    current_start = start_ms
    interval_ms = 60 * 1000  # 1 minute in milliseconds

    while current_start < end_ms:
        resp = http_client.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            start=current_start,
            limit=limit,
        )

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
            all_rows.append({
                "timestamp_ms": ts,
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5],
                "turnover": row[6],
            })

        if not rows_sorted:
            break

        last_ts = int(rows_sorted[-1][0])

        if last_ts + interval_ms >= end_ms:
            break

        current_start = last_ts + interval_ms

    return all_rows


def upload_to_starrocks(
    symbol: str,
    date: str,
    http_client: HTTP | None = None,
    connection: Connection | None = None,
) -> int:
    """
    Upload OHLCV data from Bybit to StarRocks.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        date: Date in YYYY-MM-DD format
        http_client: Optional Bybit HTTP client (creates one if not provided)
        connection: Optional StarRocks connection (creates one if not provided)

    Returns:
        Number of rows inserted
    """
    # Create clients if not provided
    if http_client is None:
        http_client = HTTP(
            api_key=config.bybit_api_key,
            api_secret=config.bybit_api_secret,
        )

    if connection is None:
        connection = get_starrocks_connection()

    # Parse date and calculate time range
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    start_dt = date_obj.replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt = start_dt + timedelta(days=1)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    logger.info(f"Fetching {symbol} data for {date} ({start_dt} to {end_dt})")

    # Fetch data from Bybit
    ohlcv_data = fetch_bybit_ohlcv(http_client, symbol, start_ms, end_ms)

    if not ohlcv_data:
        logger.warning(f"No data received for {symbol} on {date}")
        return 0

    logger.info(f"Fetched {len(ohlcv_data)} candles for {symbol}")

    # Prepare data for insertion
    rows_to_insert = []
    for row in ohlcv_data:
        event_time = datetime.fromtimestamp(row["timestamp_ms"] / 1000)
        rows_to_insert.append((
            symbol,
            event_time,
            Decimal(row["open"]),
            Decimal(row["high"]),
            Decimal(row["low"]),
            Decimal(row["close"]),
            Decimal(row["volume"]),
            Decimal(row["turnover"]),
        ))

    # Insert data into StarRocks
    # Note: Table uses PRIMARY KEY(symbol, event_time), so duplicates are auto-replaced
    cursor = connection.cursor()
    try:
        insert_sql = """
            INSERT INTO bybit_ohlcv_1m 
            (symbol, event_time, open_price, high_price, low_price, close_price, volume, turnover)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        cursor.executemany(insert_sql, rows_to_insert)
        connection.commit()

        logger.info(f"Successfully inserted {len(rows_to_insert)} rows for {symbol} on {date}")
        return len(rows_to_insert)

    except Exception as e:
        logger.error(f"Failed to insert data: {e}")
        connection.rollback()
        raise
    finally:
        cursor.close()


def upload_date_range(
    symbol: str,
    start_date: str,
    end_date: str,
    http_client: HTTP | None = None,
    connection: Connection | None = None,
) -> int:
    """
    Upload OHLCV data for a date range.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (inclusive)
        http_client: Optional Bybit HTTP client
        connection: Optional StarRocks connection

    Returns:
        Total number of rows inserted
    """
    # Create clients if not provided
    if http_client is None:
        http_client = HTTP(
            api_key=config.bybit_api_key,
            api_secret=config.bybit_api_secret,
        )

    if connection is None:
        connection = get_starrocks_connection()

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    total_rows = 0
    current_dt = start_dt

    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y-%m-%d")
        try:
            rows = upload_to_starrocks(
                symbol=symbol,
                date=date_str,
                http_client=http_client,
                connection=connection,
            )
            total_rows += rows
        except Exception as e:
            logger.error(f"Failed to upload data for {date_str}: {e}")

        current_dt += timedelta(days=1)

    logger.info(f"Total rows inserted for {symbol}: {total_rows}")
    return total_rows


if __name__ == "__main__":
    # Example usage
    from datetime import timedelta

    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Uploading BTCUSDT data for {yesterday}...")
    rows = upload_to_starrocks("BTCUSDT", yesterday)
    print(f"Uploaded {rows} rows")

