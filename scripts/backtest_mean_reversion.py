import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import pandas as pd
from pybit.unified_trading import HTTP

from flying_trading.config import config

# ======================= ПАРАМЕТРЫ СТРАТЕГИИ =======================


@dataclass
class StrategyParams:
    rsi_period: int = 14
    rsi_lower: float = 30.0
    rsi_upper: float = 70.0

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    tp_pct: float = 0.006  # 0.6% take-profit
    sl_pct: float = 0.004  # 0.4% stop-loss

    min_bars: int = 100  # минимальное кол-во баров до начала торговли

    start_equity: float = 1000.0
    fee_rate: float = 0.001  # комиссия за одну сторону (0.06% как пример)


# ======================= ЗАГРУЗКА ДАННЫХ С BYBIT =======================


def fetch_klines_bybit(
    session: HTTP,
    symbol: str,
    interval: str,
    start_ts_ms: int,
    end_ts_ms: int,
    category: str = "linear",
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Загружает исторические свечи с Bybit V5 (unified_trading).

    Работает для любых диапазонов > 1000 свечей:
    - берет пакеты по `limit` штук, двигая end назад.
    """

    all_rows = []
    cur_end = end_ts_ms

    while True:
        resp = session.get_kline(
            category=category,
            symbol=symbol,
            interval=interval,
            end=cur_end,
            limit=limit,
        )
        if resp["retCode"] != 0:
            raise RuntimeError(f"Bybit error: {resp['retCode']} {resp['retMsg']}")

        rows = resp["result"]["list"]
        if not rows:
            break

        # list отсортирован в обратном порядке по startTime:
        # rows[0] - самый НОВЫЙ бар, rows[-1] - самый СТАРЫЙ в этом пакете.
        for r in rows:
            ts = int(r[0])

            # Если бар старше нужного диапазона — просто пропускаем его
            if ts < start_ts_ms:
                continue
            # Если бар новее end_ts_ms (на всякий случай) — тоже пропускаем
            if ts > end_ts_ms:
                continue

            all_rows.append(
                {
                    "start": ts,
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                    "volume": float(r[5]),
                    "turnover": float(r[6]),
                }
            )

        # Самый старый бар в этом ответе — последний элемент.
        oldest_ts = int(rows[-1][0])

        # Если он уже раньше или ровно на границе start_ts_ms — дальше смысла нет.
        if oldest_ts <= start_ts_ms:
            break

        # Если получили меньше, чем limit — значит, данных дальше назад уже нет.
        if len(rows) < limit:
            break

        # Сдвигаем end чуть раньше самого старого бара, чтобы следующий запрос
        # вернул ещё более старые свечи и не было перекрытия.
        cur_end = oldest_ts - 1

    if not all_rows:
        raise RuntimeError("Не получили ни одной свечи от Bybit")

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["start"])
    df.sort_values("start", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["timestamp"] = pd.to_datetime(df["start"], unit="ms", utc=True)

    return df[["timestamp", "open", "high", "low", "close", "volume"]]


# ======================= ИНДИКАТОРЫ: RSI + MACD =======================


def add_indicators(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    s = df["close"].astype(float)

    # --- RSI ---
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=params.rsi_period).mean()
    avg_loss = loss.rolling(window=params.rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df["rsi"] = rsi

    # --- MACD ---
    ema_fast = s.ewm(span=params.macd_fast, adjust=False).mean()
    ema_slow = s.ewm(span=params.macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=params.macd_signal, adjust=False).mean()
    hist = macd - signal

    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = hist

    df["macd_hist_prev"] = df["macd_hist"].shift(1)

    return df


# ======================= БЭКТЕСТ =======================


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str  # "long" / "short"
    entry_price: float
    exit_price: float
    bars_held: int
    gross_return: float  # до комиссий
    net_return: float  # после комиссий


def backtest_mean_reversion(
    df: pd.DataFrame,
    params: StrategyParams,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    df: с колонками timestamp, open, high, low, close, volume, rsi, macd_hist, macd_hist_prev
    """
    trades: List[Trade] = []

    position = 0  # 0 = flat, +1 = long, -1 = short
    entry_price = None
    entry_time = None
    entry_index = None

    equity = params.start_equity
    equity_curve = [equity]

    fees_per_side = params.fee_rate  # комиссия за одну сторону
    total_bars = len(df)

    for i in range(total_bars):
        row = df.iloc[i]

        # Пропускаем, пока нет индикаторов
        if i < params.min_bars:
            equity_curve.append(equity)
            continue

        close_price = float(row["close"])
        rsi = float(row["rsi"])
        hist = float(row["macd_hist"])
        prev_hist = (
            float(row["macd_hist_prev"])
            if not math.isnan(row["macd_hist_prev"])
            else None
        )
        ts = row["timestamp"]

        if prev_hist is None or math.isnan(rsi) or math.isnan(hist):
            equity_curve.append(equity)
            continue

        # --- Сигналы входа ---
        long_entry = (rsi < params.rsi_lower) and (hist > prev_hist)
        short_entry = (rsi > params.rsi_upper) and (hist < prev_hist)

        # --- Управление позицией ---
        if position == 0:
            # Вне позиции: ищем вход
            if long_entry:
                position = +1
                entry_price = close_price
                entry_time = ts
                entry_index = i
                # комиссия за вход
                equity *= 1 - fees_per_side
            elif short_entry:
                position = -1
                entry_price = close_price
                entry_time = ts
                entry_index = i
                equity *= 1 - fees_per_side

        else:
            # Уже в позиции: проверяем выход
            should_exit = False

            # Тейк/стоп по цене
            if position == +1:
                tp_price = entry_price * (1 + params.tp_pct)
                sl_price = entry_price * (1 - params.sl_pct)
                if close_price >= tp_price or close_price <= sl_price:
                    should_exit = True
                # mean reversion фильтр
                if rsi > 50:
                    should_exit = True
            elif position == -1:
                tp_price = entry_price * (1 - params.tp_pct)
                sl_price = entry_price * (1 + params.sl_pct)
                if close_price <= tp_price or close_price >= sl_price:
                    should_exit = True
                if rsi < 50:
                    should_exit = True

            if should_exit:
                # Выход по close
                exit_price = close_price
                exit_time = ts
                bars_held = i - entry_index

                if position == +1:
                    gross_ret = exit_price / entry_price - 1.0
                else:  # short
                    gross_ret = entry_price / exit_price - 1.0

                # комиссия за выход
                net_ret = (1 + gross_ret) * (1 - fees_per_side) - 1

                # Обновляем капитал
                equity *= 1 + net_ret

                trades.append(
                    Trade(
                        entry_time=entry_time,
                        exit_time=exit_time,
                        side="long" if position == +1 else "short",
                        entry_price=entry_price,
                        exit_price=exit_price,
                        bars_held=bars_held,
                        gross_return=gross_ret,
                        net_return=net_ret,
                    )
                )

                # Сбрасываем позицию
                position = 0
                entry_price = None
                entry_time = None
                entry_index = None

        equity_curve.append(equity)

    # Если позиция осталась открыта в конце — можно принудительно закрыть по последнему close
    # (по желанию; сейчас оставляем незакрытую как есть)

    # DataFrame по сделкам
    if trades:
        trades_df = pd.DataFrame([t.__dict__ for t in trades])
    else:
        trades_df = pd.DataFrame(
            columns=[
                "entry_time",
                "exit_time",
                "side",
                "entry_price",
                "exit_price",
                "bars_held",
                "gross_return",
                "net_return",
            ]
        )

    # Equity curve
    equity_df = pd.DataFrame(
        {
            "timestamp": df["timestamp"].reindex(range(len(equity_curve))).ffill(),
            "equity": equity_curve,
        }
    )

    # --- Статистика ---
    stats = compute_stats(trades_df, equity_df, params)

    return trades_df, equity_df, stats


# ======================= СТАТИСТИКА =======================


def compute_stats(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    params: StrategyParams,
) -> Dict[str, Any]:
    if trades_df.empty:
        return {
            "total_trades": 0,
            "winrate": None,
            "avg_gain": None,
            "avg_loss": None,
            "profit_factor": None,
            "max_drawdown_pct": None,
            "final_equity": equity_df["equity"].iloc[-1],
            "total_return_pct": equity_df["equity"].iloc[-1] / params.start_equity - 1,
        }

    total_trades = len(trades_df)
    wins = trades_df[trades_df["net_return"] > 0]
    losses = trades_df[trades_df["net_return"] <= 0]

    winrate = len(wins) / total_trades if total_trades > 0 else None
    avg_gain = wins["net_return"].mean() if not wins.empty else None
    avg_loss = losses["net_return"].mean() if not losses.empty else None

    profit_factor = None
    if not wins.empty and not losses.empty:
        gross_profit = wins["net_return"].sum()
        gross_loss = losses["net_return"].sum()
        if gross_loss != 0:
            profit_factor = gross_profit / abs(gross_loss)

    # Max drawdown
    equity_series = equity_df["equity"].astype(float)
    running_max = equity_series.cummax()
    drawdown = equity_series / running_max - 1.0
    max_drawdown_pct = drawdown.min()

    final_equity = float(equity_series.iloc[-1])
    total_return_pct = final_equity / params.start_equity - 1.0

    # Sharpe по сделкам (очень грубо)
    r = trades_df["net_return"]
    if r.std() != 0:
        sharpe = r.mean() / r.std() * (len(r) ** 0.5)
    else:
        sharpe = None

    return {
        "total_trades": total_trades,
        "winrate": winrate,
        "avg_gain": avg_gain,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown_pct,
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "sharpe_trades": sharpe,
    }


# ======================= MAIN: ПРИМЕР ЗАПУСКА =======================

if __name__ == "__main__":
    # --- Настройки ---
    SYMBOL = "SOLUSDT"
    INTERVAL = "1"  # 1m
    CATEGORY = "linear"

    # Период бэктеста: последние X дней
    days_back = 30
    
    end = pd.Timestamp.utcnow().floor("min")
    start = end - pd.Timedelta(days=days_back)

    start_ts_ms = int(start.timestamp() * 1000)
    end_ts_ms = int(end.timestamp() * 1000)

    api_key = config.bybit_api_key
    api_secret = config.bybit_api_secret

    if not api_key or not api_secret:
        raise RuntimeError("Нужно указать BYBIT_API_KEY и BYBIT_API_SECRET в окружении")

    session = HTTP(
        testnet=True,  # сначала бэктестить тоже можно с testnet свечами
        api_key=api_key,
        api_secret=api_secret,
    )

    print(f"Загружаю {SYMBOL} {INTERVAL}m c {start} по {end} ...")
    df = fetch_klines_bybit(
        session=session,
        symbol=SYMBOL,
        interval=INTERVAL,
        start_ts_ms=start_ts_ms,
        end_ts_ms=end_ts_ms,
        category=CATEGORY,
        limit=1000,
    )
    print(f"Получено {len(df)} свечей")

    params = StrategyParams()
    df = add_indicators(df, params)

    trades_df, equity_df, stats = backtest_mean_reversion(df, params)

    print("\n=== Статистика бэктеста ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    print("\nПервые сделки:")
    print(trades_df.head())

    # Можешь сохранить csv, чтобы покрутить в pandas / excel
    trades_df.to_csv("trades_mean_reversion.csv", index=False)
    equity_df.to_csv("equity_curve_mean_reversion.csv", index=False)
    print("\nРезультаты сохранены в CSV.")
