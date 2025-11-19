import time
from flying_trading.domain.models import StrategyParams, Signal, Candle, Side
from flying_trading.adapters.indicator_service import IndicatorService
from flying_trading.logger import get_logger

logger = get_logger(__name__)


class TrendStrategyLogic:
    def __init__(self, params: StrategyParams):
        self.params = params
        self.indicators = IndicatorService()
        self.last_trade_ts = 0

    def analyze(self, candles: list[Candle], current_pos_size: float) -> Signal:
        # 1. Check cooldown
        current_candle = candles[-1]

        time_since_trade = (current_candle.ts / 1000) - self.last_trade_ts
        if current_pos_size == 0 and time_since_trade < (
            self.params.cooldown_minutes * 60
        ):
            logger.debug(f"Cooldown active: {time_since_trade:.0f}s remaining")
            return Signal(action=Side.NONE, reason="Cooldown")

        data = self.indicators.calculate(candles, self.params)

        close = data["close"]
        open_price = data["open"]
        ema_200 = data["ema_200"]
        rsi = data["rsi"]
        atr = data["atr"]
        adx = data["adx"]
        # 3. Логика определения тренда
        # Используем буфер 0.1%, чтобы не торговать прямо на линии EMA
        is_uptrend = close > ema_200
        is_downtrend = close < ema_200

        is_strong_trend = adx > 25

        if not is_strong_trend:
            return Signal(Side.NONE, reason="Weak Trend (ADX < 25)")

        # Фильтр цвета свечи (Price Action)
        is_green_candle = close > open_price
        is_red_candle = close < open_price

        # 4. Генерация сигналов
        if current_pos_size == 0:
            # --- LONG SETUP ---
            # Цена выше EMA 200 (тренд вверх)
            # RSI опустился в зону "дешево для тренда" (35-55)
            # Свеча зеленая (начался откуп)
            if is_uptrend and (35 < rsi < 55) and is_green_candle:
                # Stop Loss: 2 ATR от цены входа
                sl = close - (atr * self.params.atr_multiplier_sl)

                # Защита: SL не должен быть выше EMA 200 (иначе выбьет об поддержку)
                if sl > ema_200:
                    sl = ema_200 * 0.998  # Ставим чуть ниже EMA

                logger.info(
                    f"LONG Signal: Price {close}, RSI {rsi:.1f}, EMA {ema_200:.1f}"
                )
                return Signal(
                    Side.BUY,
                    sl_price=sl,
                    trailing_dist=atr * self.params.atr_multiplier_trail,
                )

            # --- SHORT SETUP ---
            # Цена ниже EMA 200 (тренд вниз)
            # RSI поднялся в зону "дорого для тренда" (45-65)
            # Свеча красная (начались продажи)
            elif is_downtrend and (45 < rsi < 65) and is_red_candle:
                sl = close + (atr * self.params.atr_multiplier_sl)

                # Защита: SL не должен быть ниже EMA 200
                if sl < ema_200:
                    sl = ema_200 * 1.002

                logger.info(
                    f"SHORT Signal: Price {close}, RSI {rsi:.1f}, EMA {ema_200:.1f}"
                )
                return Signal(
                    Side.SELL,
                    sl_price=sl,
                    trailing_dist=atr * self.params.atr_multiplier_trail,
                )

        return Signal(Side.NONE)

    def notify_trade_closed(self):
        self.last_trade_ts = time.time()


"""
        # 2. Calculate indicators
        data = self.indicators.calculate(candles, self.params)

        close = data["close"]
        bb_width = (data["bb_upper"] - data["bb_lower"]) / close

        # 3. Filters
        if bb_width < self.params.bb_width_min:
            return Signal(action=Side.NONE, reason=f"Flat (Width {bb_width:.4f})")

        if current_candle.turnover < self.params.min_turnover:
            if data["volume"] < (data["vol_sma"] * self.params.vol_multiplier):
                return Signal(
                    action=Side.NONE,
                    reason=f"Low Liquidity ({current_candle.turnover:.0f}$): Volume {data['volume']:.0f} < {data['vol_sma'] * self.params.vol_multiplier:.0f}",
                )

        # 4. Entry logic
        atr = data["atr"]

        if atr > (close * 0.02):
            return Signal(
                action=Side.NONE,
                reason=f"Extreme Volatility (ATR {atr:.2f}): ATR > {close * 0.02:.2f}",
            )

        l_trend = current_candle.close > h1_ema_trend
        l_pattern = (prev_candle.close < data["bb_lower"]) and (
            current_candle.close > data["bb_lower"]
        )
        l_candle_green = current_candle.close > current_candle.open
        l_rsi = data["rsi"] < self.params.rsi_lower
        long_cond = l_trend and l_pattern and l_candle_green and l_rsi

                
        s_trend = current_candle.close < h1_ema_trend
        s_pattern = (prev_candle.close > data["bb_upper"]) and (
            current_candle.close < data["bb_upper"]
        )
        s_candle_red = current_candle.close < current_candle.open
        s_rsi = data["rsi"] > self.params.rsi_upper
        short_cond = s_trend and s_pattern and s_candle_red and s_rsi

        if current_pos_size == 0:
            if long_cond:
                sl = close - (atr * self.params.atr_multiplier_sl)
                trail = atr * self.params.atr_multiplier_trail
                logger.info(
                    f"LONG signal: price={close:.2f}, RSI={data['rsi']:.2f}, "
                    f"BB_lower={data['bb_lower']:.2f}, EMA200={h1_ema_trend:.2f}, "
                    f"SL={sl:.2f}, trail={trail:.2f}"
                )
                return Signal(Side.BUY, sl_price=sl, trailing_dist=trail)
            elif short_cond:
                sl = close + (atr * self.params.atr_multiplier_sl)
                trail = atr * self.params.atr_multiplier_trail
                logger.info(
                    f"SHORT signal: price={close:.2f}, RSI={data['rsi']:.2f}, "
                    f"BB_upper={data['bb_upper']:.2f}, EMA200={h1_ema_trend:.2f}, "
                    f"SL={sl:.2f}, trail={trail:.2f}"
                )
                return Signal(Side.SELL, sl_price=sl, trailing_dist=trail)
            else:
                return Signal(Side.NONE, reason="No entry signal")

        return Signal(Side.NONE, reason="In trade")
"""
