from dataclasses import dataclass
import enum


class Side(enum.Enum):
    BUY = "BUY"
    SELL = "SELL"
    NONE = "NONE"


@dataclass
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float


@dataclass
class Position:
    size: float
    entry_price: float
    side: Side
    positionIdx: int
    leverage: int
    unrealized_pnl: float
    cur_realized_pnl: float


@dataclass
class StrategyParams:
    interval: str = "5"

    bb_width_min: float = 0.007
    rsi_period: int = 14
    rsi_lower: float = 40.0
    rsi_upper: float = 60.0
    bb_period: int = 20
    bb_std: float = 2.0
    ema_filter_period: int = 200
    vol_multiplier: float = 1.1
    atr_period: int = 14
    atr_multiplier_sl: float = 2.5
    atr_multiplier_trail: float = 3.5
    min_bars: int = 250
    cooldown_minutes: int = 5

    min_turnover: float = 1_000.0

    ema_fast_len: int = 50
    ema_slow_len: int = 200

    # Настройки StochRSI
    stoch_rsi_len: int = 14
    stoch_rsi_k: int = 3
    stoch_rsi_d: int = 3

    # ATR для стопов
    atr_length: int = 14

    adx_length: int = 14


@dataclass
class Signal:
    action: Side
    sl_price: None | float = None
    trailing_dist: None | float = None
    reason: None | str = None  # TODO: think whether we need this field
