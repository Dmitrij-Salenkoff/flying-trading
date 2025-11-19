import threading
import time
import math
from collections import deque
from dataclasses import dataclass
import datetime

import pandas as pd
from pybit.unified_trading import HTTP, WebSocket
import requests
import click

# Твой конфиг
from flying_trading.config import config


@dataclass
class StrategyParams:
    # Индикаторы
    rsi_period: int = 14
    rsi_lower: float = 30.0
    rsi_upper: float = 70.0

    bb_period: int = 20
    bb_std: float = 2.0

    ema_filter_period: int = 200  # Тренд

    # НОВОЕ: Фильтр объема
    vol_multiplier: float = 1.2  # Объем должен быть в 1.5 раза выше среднего

    # Риск-менеджмент
    atr_period: int = 14
    atr_multiplier_sl: float = 2.0
    atr_multiplier_trail: float = 2.5  # Дистанция трейлинга (вместо TP)

    risk_per_trade: float = 0.01  # Риск 1% от депозита на сделку

    min_bars: int = 201
    cooldown_minutes: int = 5

    leverage: str = "10"


class ProfessionalTrendBot:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str = "SOLUSDT",
        qty_usdt: float = 20.0,
        testnet: bool = True,
    ):
        self.symbol = symbol
        self.qty_usdt = qty_usdt
        self.testnet = testnet

        self.session = HTTP(testnet=testnet, api_key=api_key, api_secret=api_secret)
        self.ws = WebSocket(testnet=testnet, channel_type="linear")
        self.params = StrategyParams()

        # Увеличили буфер истории для EMA 200
        self.max_bars = 500
        self.history = deque(maxlen=self.max_bars)

        # Состояние
        self.position = 0  # 0, 1 (long), -1 (short)
        self.entry_price = 0.0
        self.sl_price = 0.0  # Храним текущий стоп, чтобы проверять БУ
        self.tp_price = 0.0
        self.is_breakeven_set = False  # Флаг, переставлен ли стоп в БУ

        self.last_trade_time = 0  # timestamp закрытия последней сделки

        # Метаданные
        self.price_precision = 4
        self.qty_precision = 2
        self.tick_size = 0.0001
        self.qty_step = 0.01
        self.taker_fee_rate = 0.001

        self.global_trend_ema = 0.0  # Переменная для хранения EMA с часовика
        self.trend_ready = False  # Флаг, что EMA посчиталась

        self.set_leverage(self.params.leverage)

    # --------- Инициализация ---------
    def _trend_loop(self):
        """Этот метод будет крутиться в отдельном потоке"""
        print("[THREAD] Запущен фоновый анализ H1 тренда...")

        while True:
            try:
                # 1. Запрашиваем 300 свечей по 60 минут (1 час)
                # end=int(time.time()*1000) берет данные на текущий момент
                resp = self.session.get_kline(
                    category="linear",
                    symbol=self.symbol,
                    interval="60",  # 60 минут
                    limit=300,
                )

                rows = resp["result"]["list"]
                # Сортируем: старые -> новые
                rows = sorted(rows, key=lambda x: int(x[0]))

                # Делаем DataFrame
                df = pd.DataFrame(
                    rows,
                    columns=["ts", "open", "high", "low", "close", "vol", "turnover"],
                )
                df["close"] = df["close"].astype(float)

                # 2. Считаем EMA 200
                # Используем adjust=False, чтобы совпадало с TradingView
                ema_series = (
                    df["close"]
                    .ewm(span=self.params.ema_filter_period, adjust=False)
                    .mean()
                )

                # 3. Обновляем "глобальную" переменную
                new_ema = ema_series.iloc[-1]

                # Логируем только если изменение значительное (чтобы не спамить)
                if abs(new_ema - self.global_trend_ema) > (new_ema * 0.001):
                    print(f"[TREND H1] EMA200 Updated: {new_ema:.4f}")

                self.global_trend_ema = new_ema
                self.trend_ready = True

            except Exception as e:
                print(f"[THREAD ERR] Ошибка обновления тренда: {e}")

            # Спим 60 секунд перед следующим обновлением.
            # Часовая свеча меняется медленно, чаще не нужно.
            time.sleep(60)

    def fetch_instrument_info(self):
        try:
            resp = self.session.get_instruments_info(
                category="linear", symbol=self.symbol
            )
            info = resp["result"]["list"][0]
            self.tick_size = float(info["priceFilter"]["tickSize"])
            self.qty_step = float(info["lotSizeFilter"]["qtyStep"])
            self.price_precision = int(abs(math.log10(self.tick_size)))
            self.qty_precision = int(abs(math.log10(self.qty_step)))
            print(f"[INFO] {self.symbol}: Precisions loaded.")
        except Exception as e:
            print(f"[ERROR] Info Error: {e}")

    def fetch_fee_rate(self):
        try:
            resp = self.session.get_fee_rate(category="linear", symbol=self.symbol)
            self.taker_fee_rate = float(resp["result"]["list"][0]["takerFeeRate"])
            print(f"[INFO] Fee Rate: {self.taker_fee_rate * 100:.4f}%")
        except Exception as e:
            print(f"[ERROR] Fee Rate Error: {e}")
            print(f"[INFO] Default Fee Rate: {self.taker_fee_rate * 100:.4f}%")
            pass

    def round_price(self, price: float) -> float:
        return round(price, self.price_precision)

    def round_qty(self, qty: float) -> float:
        steps = math.floor(qty / self.qty_step)
        return round(steps * self.qty_step, self.qty_precision)

    def calculate_qty(self, price: float) -> float:
        return self.round_qty(self.qty_usdt / price)

    # --------- Расчет индикаторов (Pandas) ---------

    def compute_indicators(self):
        if len(self.history) < self.params.min_bars:
            return None

        df = pd.DataFrame(self.history)

        # 1. EMA 200 (Фильтр тренда)
        df["ema_trend"] = (
            df["close"].ewm(span=self.params.ema_filter_period, adjust=False).mean()
        )

        # 2. RSI (Wilder)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        alpha = 1.0 / self.params.rsi_period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # 3. Bollinger Bands
        sma = df["close"].rolling(window=self.params.bb_period).mean()
        std = df["close"].rolling(window=self.params.bb_period).std()
        df["bb_upper"] = sma + (std * self.params.bb_std)
        df["bb_lower"] = sma - (std * self.params.bb_std)

        # 4. ATR
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr"] = tr.rolling(window=self.params.atr_period).mean()

        # 5. Volume SMA
        df["vol_sma"] = df["volume"].rolling(window=20).mean()

        last = df.iloc[-1]
        if (
            pd.isna(last["ema_trend"])
            or pd.isna(last["atr"])
            or pd.isna(last["vol_sma"])
        ):
            return None

        return last

    def set_leverage(self, leverage: str = "10"):
        try:
            self.session.set_leverage(
                category="linear",
                symbol=self.symbol,
                buyLeverage=leverage,
                sellLeverage=leverage,
            )
            print(f"[INIT] Плечо установлено: {leverage}x")
        except Exception as e:
            # Код ошибки 110043 означает, что плечо уже установлено - это ок
            if "110043" not in str(e):
                print(f"[WARN] Ошибка установки плеча: {e}")

    def start_trend_updater(self):
        # Создаем поток
        t = threading.Thread(target=self._trend_loop)
        t.daemon = (
            True  # Daemon означает, что поток умрет сам, если закроется основной скрипт
        )
        t.start()

    # --------- Основной цикл (On Bar) ---------

    def on_bar(self, candle: dict):
        # Добавляем свечу
        self.history.append(candle)

        # Актуализируем позицию
        self.sync_position()

        # Расчет индикаторов
        data = self.compute_indicators()
        if data is None:
            return

        close = data["close"]
        ema_trend = self.global_trend_ema
        rsi = data["rsi"]
        atr = data["atr"]
        bb_upper = data["bb_upper"]
        bb_lower = data["bb_lower"]
        vol_sma = data["vol_sma"]

        # Логирование состояния
        trend_status = "UP" if close > ema_trend else "DOWN"
        print(
            f"[{datetime.datetime.now().strftime('%H:%M')}] Price:{close:.{self.price_precision}f} | Trend:{trend_status} (EMA:{ema_trend:.{self.price_precision}f}) | RSI:{rsi:.1f} | ATR:{atr:.{self.price_precision}f} | BB:{bb_lower:.{self.price_precision}f}-{bb_upper:.{self.price_precision}f}"
        )

        # --- Проверка Кулдауна ---
        # Если мы недавно закрыли сделку, ждем N минут
        time_since_trade = (candle["ts"] / 1000) - self.last_trade_time
        if time_since_trade < (self.params.cooldown_minutes * 60):
            return

        bb_width = (data["bb_upper"] - data["bb_lower"]) / data["close"]

        # Если ширина меньше 0.5% (для крипты это очень узко)
        if bb_width < 0.005:
            print(
                f"[WARN] Market is sleeping (Flat). Waiting for breakout... BB Width: {bb_width:.{self.price_precision}f}"
            )
            return None  # Не торгуем внутри

        # --- Стратегия: Pullback Trading (Торговля откатов) ---

        # 1. Фильтр объема (VSA)
        # Если текущий объем < (средний * 1.5), значит движения слабые, пропускаем
        vol_ok = data["volume"] > (vol_sma * self.params.vol_multiplier)

        if not vol_ok:
            print(
                f"[WARN] Skip: Low Volume. Volume: {data['volume']:.{self.price_precision}f}. Vol SMA: {vol_sma:.{self.price_precision}f}"
            )  # Можно раскомментить для отладки
            return

        # 2. Фильтр комиссий (ATR должен быть больше 2.5x комиссии)
        expected_move = atr * self.params.atr_multiplier_trail
        if (expected_move / close) < (self.taker_fee_rate * 2.5):
            print(
                f"[WARN] Skip: Low Expected Move. Expected Move: {expected_move:.{self.price_precision}f}. Close: {close:.{self.price_precision}f}. Taker Fee Rate: {self.taker_fee_rate:.{self.price_precision}f}"
            )  # Можно раскомментить для отладки
            return

        # 3. Логика входа
        # ЛОНГ: Глобальный тренд ВВЕРХ + Локальная перепроданность (цена упала)
        long_condition = (
            (close > ema_trend)  # Только по тренду
            and (close < bb_lower)  # Отскок от нижней границы
            and (rsi < self.params.rsi_lower)  # Перепроданность
        )

        # ШОРТ: Глобальный тренд ВНИЗ + Локальная перекупленность (цена выросла)
        short_condition = (
            (close < ema_trend)  # Только по тренду
            and (close > bb_upper)  # Отскок от верхней границы
            and (rsi > self.params.rsi_upper)  # Перекупленность
        )

        if self.position == 0:
            if long_condition:
                # Считаем уровни
                sl_price = close - (atr * self.params.atr_multiplier_sl)
                trailing_dist = atr * self.params.atr_multiplier_trail

                # Умный расчет лота (1% риска)
                qty = self.calculate_dynamic_qty(close, sl_price)

                if qty > 0:
                    self.place_order_with_trailing("Buy", qty, sl_price, trailing_dist)

            elif short_condition:
                sl_price = close + (atr * self.params.atr_multiplier_sl)
                trailing_dist = atr * self.params.atr_multiplier_trail

                qty = self.calculate_dynamic_qty(close, sl_price)

                if qty > 0:
                    self.place_order_with_trailing("Sell", qty, sl_price, trailing_dist)

    # --------- Управление ордерами ---------

    def place_order_with_trailing(
        self, side: str, qty: float, sl: float, trailing_dist: float
    ):
        sl = self.round_price(sl)
        trailing_dist = self.round_price(trailing_dist)
        entry_price = self.history[-1]["close"]  # Для логов

        msg = f"[TRADE] {self.symbol} {side}\nPrice: {entry_price}\nQty: {qty}\nSL: {sl}\nTrailing Dist: {trailing_dist}"
        print(msg)
        self.send_telegram(msg)

        try:
            # 1. Открываем позицию + Сразу ставим Стоп-лосс (без Тейка!)
            self.session.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=str(qty),
                stopLoss=str(sl),
                slTriggerBy="LastPrice",
                positionIdx=0,
            )

            # 2. Включаем нативный Трейлинг-стоп
            # Он активируется сразу (activePrice=0) и тянется на расстоянии trailing_dist
            time.sleep(0.5)  # Небольшая пауза, чтобы ордер успел пройти
            self.session.set_trading_stop(
                category="linear",
                symbol=self.symbol,
                trailingStop=str(trailing_dist),
                activePrice="0",  # Активация сразу
                positionIdx=0,
            )
            print(f"--> Trailing Stop активирован (Dist: {trailing_dist})")

        except Exception as e:
            err_msg = f"Ошибка ордера: {e}"
            print(err_msg)
            self.send_telegram(err_msg)

    def modify_sl(self, new_sl: float):
        new_sl = self.round_price(new_sl)
        print(f"--> Moving SL to Breakeven: {new_sl}")
        try:
            self.session.set_trading_stop(
                category="linear",
                symbol=self.symbol,
                stopLoss=str(new_sl),
                slTriggerBy="LastPrice",
                positionIdx=0,
            )
            self.sl_price = new_sl
            self.is_breakeven_set = True
        except Exception as e:
            print(f"[WARN] Could not move SL: {e}")

    def sync_position(self):
        """Проверяем позицию и обновляем статус"""
        try:
            res = self.session.get_positions(category="linear", symbol=self.symbol)
            data = res["result"]["list"][0]
            size = float(data["size"])

            old_pos = self.position

            if size > 0:
                self.position = 1 if data["side"] == "Buy" else -1
                self.entry_price = float(data["avgPrice"])
            else:
                self.position = 0
                self.entry_price = 0.0

            # Если позиция только что закрылась (была, а теперь нет)
            if old_pos != 0 and self.position == 0:
                print(
                    f"--- Trade Closed. Cooldown {self.params.cooldown_minutes}m started. ---"
                )
                self.last_trade_time = time.time()

        except Exception:
            pass

    # --------- Запуск ---------

    def preload_history(self):
        print("[INIT] Preloading history...")
        end_time = int(time.time() * 1000)
        # Грузим больше, так как EMA200 требует разгона
        resp = self.session.get_kline(
            category="linear", symbol=self.symbol, interval="1", limit=500, end=end_time
        )
        rows = sorted(resp["result"]["list"], key=lambda x: int(x[0]))
        self.history.clear()
        for r in rows:
            self.history.append(
                {
                    "ts": int(r[0]),
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                    "volume": float(r[5]),
                }
            )

    def handle_message(self, message):
        if "data" in message and "kline" in message["topic"]:
            data = message["data"][0]
            if data["confirm"]:
                candle = {
                    "ts": int(data["end"]),
                    "open": float(data["open"]),
                    "high": float(data["high"]),
                    "low": float(data["low"]),
                    "close": float(data["close"]),
                    "volume": float(data["volume"]),
                }
                self.on_bar(candle)

    def calculate_dynamic_qty(
        self, entry_price: float, sl_price: float, risk_per_trade_pct: float = 0.01
    ):
        """
        Считает объем позиции так, чтобы при срабатывании SL потерять ровно 1% от депозита.
        """
        try:
            # 1. Узнаем баланс (USDT)
            wallet = self.session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            equity = float(wallet["result"]["list"][0]["coin"][0]["equity"])

            # 2. Сколько денег готовы потерять (Risk Amount)
            risk_amount = equity * risk_per_trade_pct  # Например, 1000$ * 0.01 = 10$

            # 3. Дистанция до стопа в процентах
            # abs(100 - 98) / 100 = 0.02 (2%)
            dist_pct = abs(entry_price - sl_price) / entry_price

            # 4. Размер позиции (USDT) = Риск / %Стопа
            # 10$ / 0.02 = 500$ (размер позиции с плечом)
            pos_size_usdt = risk_amount / dist_pct

            # 5. Переводим в монеты
            qty = pos_size_usdt / entry_price

            return self.round_qty(qty)

        except Exception as e:
            print(f"[Risk Error] {e}, using default qty")
            return self.calculate_qty(entry_price)  # Фоллбек на старый метод

    def send_telegram(self, message: str):
        bot_token = config.telegram_bot_token

        # 2. Вставь Chat ID (число, которое узнал выше)
        # Можно как число (12345), можно как строку ("12345")
        chat_id = "880801285"

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        try:
            requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=2)
        except Exception as e:
            print(f"[TG ERR] {e}")

    def run(self):
        self.fetch_instrument_info()
        self.fetch_fee_rate()
        self.preload_history()
        self.start_trend_updater()

        print(f"Strategy: Trend Following + Dip Buying. Symbol: {self.symbol}")
        self.ws.kline_stream(
            interval="1", symbol=self.symbol, callback=self.handle_message
        )
        while True:
            time.sleep(1)


@click.command()
@click.option(
    "--symbol",
    help="Trading symbol (e.g., BTCUSDT, ETHUSDT). Required.",
)
@click.option(
    "--qty-usdt",
    default=100.0,
    type=float,
    help="Quantity in USDT per trade (default: 100.0)",
)
@click.option(
    "--testnet/--no-testnet",
    default=False,
    help="Use testnet (default: False)",
)
def main(symbol: str, qty_usdt: float, testnet: bool):
    """Run the Professional Trend Bot trading strategy."""

    bot = ProfessionalTrendBot(
        api_key=config.bybit_api_key,
        api_secret=config.bybit_api_secret,
        symbol=symbol,
        qty_usdt=qty_usdt,
        testnet=testnet,
    )
    bot.run()


if __name__ == "__main__":
    main()
