CREATE TABLE queue_trades (
    event_time String,
    symbol String,
    price Float64,
    quantity Float64,
    trade_id UInt64,
    is_buyer_maker Bool
) ENGINE = Kafka
SETTINGS kafka_broker_list = 'redpanda:9092',
       kafka_topic_list = 'crypto_trades',
       kafka_group_name = 'ch_consumer_group',
       kafka_format = 'JSONEachRow';


CREATE TABLE trades (
    ts DateTime64(3),
    symbol LowCardinality(String),
    price Float64,
    quantity Float64,
    trade_id UInt64,
    side Enum8('buy'=1, 'sell'=-1),
    ingested_at DateTime64(3)
) ENGINE = MergeTree()
ORDER BY (symbol, ts);


CREATE MATERIALIZED VIEW mv_trades TO trades AS
SELECT
    parseDateTime64BestEffort(event_time, 3) AS ts,
    symbol,
    price,
    quantity,
    trade_id,
    if(is_buyer_maker, 'sell', 'buy') AS side,
    now64(3) as ingested_at
FROM queue_trades;