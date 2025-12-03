-- Migration: Recreate bybit_ohlcv_1m with PRIMARY KEY for automatic deduplication
-- 
-- IMPORTANT: This will DROP the existing table with all data!
-- Run this only if you want to start fresh or backup data first.

-- Step 1: Drop old table
DROP TABLE IF EXISTS bybit_ohlcv_1m;

-- Step 2: Create new table with PRIMARY KEY (auto-dedup on insert)
CREATE TABLE IF NOT EXISTS bybit_ohlcv_1m (
    -- Primary Key columns
    symbol VARCHAR(32) NOT NULL COMMENT "Trading pair, e.g. BTCUSDT",
    event_time DATETIME NOT NULL COMMENT "Candle start time",
    
    -- Value columns (will be updated on duplicate key)
    open_price DECIMAL(38, 10) COMMENT "Open price",
    high_price DECIMAL(38, 10) COMMENT "High price",
    low_price DECIMAL(38, 10) COMMENT "Low price",
    close_price DECIMAL(38, 10) COMMENT "Close price",
    volume DECIMAL(38, 10) COMMENT "Volume in base currency (BTC)",
    turnover DECIMAL(38, 10) COMMENT "Turnover in quote currency (USDT)"
)
ENGINE=OLAP
PRIMARY KEY(symbol, event_time)
PARTITION BY RANGE(event_time) (
    PARTITION p202001 VALUES LESS THAN ("2020-02-01"),
)
DISTRIBUTED BY HASH(symbol) BUCKETS 10 
PROPERTIES (
    "replication_num" = "1",
    
    "dynamic_partition.enable" = "true",
    "dynamic_partition.time_unit" = "MONTH",
    "dynamic_partition.start" = "-48",
    "dynamic_partition.end" = "6",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "10"
);

