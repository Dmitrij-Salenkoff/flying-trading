import pandas as pd
import pandas_ta as ta  # noqa: F401

from flying_trading.domain.models import Candle, StrategyParams


class IndicatorService:
    def calculate(self, candles: list[Candle], params: StrategyParams):
        """
        Calculates indicators for a list of candles.
        Returns a dictionary with the indicator values.
        The dictionary contains the following keys:
        - ema_trend: EMA trend
        - rsi: RSI
        - bb_upper: Bollinger Bands upper
        - bb_lower: Bollinger Bands lower
        - atr: ATR
        - vol_sma: Volume SMA
        """
        df = pd.DataFrame([vars(c) for c in candles])

        # EMA Trend
        df["ema_trend"] = (
            df["close"].ewm(span=params.ema_filter_period, adjust=False).mean()
        )
        df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / params.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / params.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger
        sma = df["close"].rolling(window=params.bb_period).mean()
        std = df["close"].rolling(window=params.bb_period).std()
        df["bb_upper"] = sma + (std * params.bb_std)
        df["bb_lower"] = sma - (std * params.bb_std)

        # ATR
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr"] = tr.rolling(window=params.atr_period).mean()

        # ADX
        df["adx"] = df.ta.adx(length=params.adx_length)

        # Vol SMA
        df["vol_sma"] = df["volume"].rolling(window=20).mean()

        return df.iloc[-1]

    def calculate_trend_strategy(
        self, candles: list[Candle], params: StrategyParams
    ) -> dict:
        df = pd.DataFrame([vars(c) for c in candles])

        if len(df) < params.ema_slow_len + 50:
            raise ValueError(
                f"Not enough candles. Need > {params.ema_slow_len + 50}, got {len(df)}"
            )

        ema_fast = df.ta.ema(length=params.ema_fast_len)
        ema_slow = df.ta.ema(length=params.ema_slow_len)

        # --- ATR (Average True Range) ---
        atr = df.ta.atr(length=params.atr_length)

        # --- Stochastic RSI ---
        # stochrsi возвращает DataFrame с двумя колонками: K и D
        # Имена колонок обычно выглядят как: STOCHRSIk_14_14_3_3 и STOCHRSId_14_14_3_3
        stoch_rsi_df = df.ta.stochrsi(
            length=params.stoch_rsi_len,
            rsi_length=params.stoch_rsi_len,
            k=params.stoch_rsi_k,
            d=params.stoch_rsi_d,
        )

        # Чтобы не зависеть от сгенерированных имен, берем по индексу колонок
        # 0 - это K (быстрая), 1 - это D (медленная)
        stoch_k = stoch_rsi_df.iloc[:, 0]
        stoch_d = stoch_rsi_df.iloc[:, 1]

        # 3. Формирование результата
        # Нам нужны значения текущей свечи (-1) и предыдущей (-2)
        idx_curr = -1
        idx_prev = -2

        return {
            # Текущие значения (для проверки условий)
            "close": df.iloc[idx_curr]["close"],
            "volume": df.iloc[idx_curr]["volume"],
            "ema_50": ema_fast.iloc[idx_curr],
            "ema_200": ema_slow.iloc[idx_curr],
            "atr": atr.iloc[idx_curr],
            "stoch_k": stoch_k.iloc[idx_curr],
            "stoch_d": stoch_d.iloc[idx_curr],
            # Значения прошлой свечи (для проверки пересечений/кроссоверов)
            "prev_stoch_k": stoch_k.iloc[idx_prev],
            "prev_stoch_d": stoch_d.iloc[idx_prev],
            "prev_close": df.iloc[idx_prev]["close"],
        }
