CREATE DATABASE IF NOT EXISTS trading;

CREATE TABLE IF NOT EXISTS trading.klines
(
    symbol String,     -- Trading symbol (e.g., 'BTCUSDT', 'ETHUSDT')
    timestamp UInt64,  -- Timestamp in milliseconds
    open Float64,      -- Open price
    high Float64,      -- High price
    low Float64,       -- Low price
    close Float64,     -- Close price
    volume Float64     -- Volume
)
ENGINE = MergeTree()
PARTITION BY (symbol, toYYYYMM(toDateTime(timestamp / 1000)))  -- Partition by symbol and month
ORDER BY (symbol, timestamp)
SETTINGS index_granularity = 8192;