{{
    config(
        materialized='table'
    )
}}

with source as (
    select
        symbol,
        event_time,
        open_price,
        high_price,
        low_price,
        close_price,
        volume,
        turnover
    from {{ source('raw', 'bybit_ohlcv_1m') }}
),

with_window as (
    select
        symbol,
        date_trunc('minute', event_time) - interval (minute(event_time) % 5) minute as event_time_5m,
        event_time as original_event_time,
        open_price,
        high_price,
        low_price,
        close_price,
        volume,
        turnover,
        row_number() over (
            partition by symbol, date_trunc('minute', event_time) - interval (minute(event_time) % 5) minute 
            order by event_time asc
        ) as rn_asc,
        row_number() over (
            partition by symbol, date_trunc('minute', event_time) - interval (minute(event_time) % 5) minute 
            order by event_time desc
        ) as rn_desc
    from source
),

aggregated as (
    select
        symbol,
        event_time_5m as event_time,
        max(case when rn_asc = 1 then open_price end) as open_price,
        max(high_price) as high_price,
        min(low_price) as low_price,
        max(case when rn_desc = 1 then close_price end) as close_price,
        sum(volume) as volume,
        sum(turnover) as turnover
    from with_window
    group by symbol, event_time_5m
)

select
    symbol,
    event_time,
    open_price,
    high_price,
    low_price,
    close_price,
    volume,
    turnover
from aggregated
order by symbol, event_time

