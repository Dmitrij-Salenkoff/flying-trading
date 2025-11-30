"""
Бэктест для TrendStrategyLogic из flying_trading.application.strategy
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from pybit.unified_trading import HTTP

from flying_trading.application.strategy import TrendStrategyLogic
from flying_trading.config import config
from flying_trading.domain.models import Candle, Side, StrategyParams

# ======================= ЗАГРУЗКА ДАННЫХ =======================


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

        for r in rows:
            ts = int(r[0])
            if ts < start_ts_ms:
                continue
            if ts > end_ts_ms:
                continue

            all_rows.append(
                {
                    "ts": ts,
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                    "volume": float(r[5]),
                    "turnover": float(r[6]),
                }
            )

        oldest_ts = int(rows[-1][0])
        if oldest_ts <= start_ts_ms:
            break
        if len(rows) < limit:
            break

        cur_end = oldest_ts - 1

    if not all_rows:
        raise RuntimeError("Не получили ни одной свечи от Bybit")

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["ts"])
    df.sort_values("ts", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ======================= БЭКТЕСТ =======================


@dataclass
class Trade:
    entry_time: int  # timestamp в ms
    exit_time: int
    side: str  # "long" / "short"
    entry_price: float
    exit_price: float
    sl_price: float
    trailing_dist: float
    bars_held: int
    gross_return: float
    net_return: float
    exit_reason: str  # "stop_loss", "trailing_stop", "end_of_data"


@dataclass
class Position:
    side: Side
    entry_price: float
    entry_time: int
    entry_index: int
    sl_price: float
    trailing_dist: float
    highest_price: float  # для long
    lowest_price: float  # для short


def backtest_trend_strategy(
    df: pd.DataFrame,
    params: StrategyParams,
    start_equity: float = 1000.0,
    fee_rate: float = 0.001,  # 0.1% за одну сторону
    risk_per_trade: float = 0.01,  # 1% риска на сделку
    verbose: bool = False,  # Выводить диагностику
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Бэктест для TrendStrategyLogic.

    Args:
        df: DataFrame с колонками ts, open, high, low, close, volume, turnover
        params: Параметры стратегии
        start_equity: Начальный капитал
        fee_rate: Комиссия за одну сторону
        risk_per_trade: Риск на сделку (доля от капитала)

    Returns:
        trades_df: DataFrame со сделками
        equity_df: DataFrame с кривой капитала
        stats: Словарь со статистикой
    """
    strategy = TrendStrategyLogic(params)
    trades: list[Trade] = []

    position: Position | None = None
    equity = start_equity
    equity_curve = [equity]

    total_bars = len(df)

    # Диагностика: считаем причины отсутствия сигналов
    diagnostics = {
        "total_checks": 0,
        "cooldown": 0,
        "no_trend": 0,
        "rsi_out_of_range": 0,
        "wrong_candle_color": 0,
        "signals": 0,
    }

    # Конвертируем DataFrame в список Candle для стратегии
    candles_list: list[Candle] = []

    for i in range(total_bars):
        row = df.iloc[i]
        ts = int(row["ts"])

        candle = Candle(
            ts=ts,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            turnover=float(row.get("turnover", 0)),
        )
        candles_list.append(candle)

        # Пропускаем, пока нет достаточно данных для расчета индикаторов
        # Для calculate() нужно минимум 200 свечей для EMA 200 и 14 для RSI
        min_required = max(200, params.rsi_period, params.atr_period, params.bb_period)
        if len(candles_list) < min_required:
            equity_curve.append(equity)
            continue

        current_pos_size = (
            0.0 if position is None else (1.0 if position.side == Side.BUY else -1.0)
        )

        # Проверяем выход по trailing stop или stop loss
        if position is not None:
            should_exit = False
            exit_price = None
            exit_reason = None

            if position.side == Side.BUY:
                # Обновляем highest price для trailing stop
                if candle.high > position.highest_price:
                    position.highest_price = candle.high

                # Trailing stop: если цена упала на trailing_dist от highest
                trailing_stop_price = position.highest_price - position.trailing_dist
                if candle.low <= trailing_stop_price:
                    should_exit = True
                    exit_price = trailing_stop_price
                    exit_reason = "trailing_stop"
                # Stop loss
                elif candle.low <= position.sl_price:
                    should_exit = True
                    exit_price = position.sl_price
                    exit_reason = "stop_loss"

            else:  # SHORT
                # Обновляем lowest price для trailing stop
                if candle.low < position.lowest_price:
                    position.lowest_price = candle.low

                # Trailing stop: если цена выросла на trailing_dist от lowest
                trailing_stop_price = position.lowest_price + position.trailing_dist
                if candle.high >= trailing_stop_price:
                    should_exit = True
                    exit_price = trailing_stop_price
                    exit_reason = "trailing_stop"
                # Stop loss
                elif candle.high >= position.sl_price:
                    should_exit = True
                    exit_price = position.sl_price
                    exit_reason = "stop_loss"

            if should_exit:
                # Выход из позиции
                bars_held = i - position.entry_index

                if position.side == Side.BUY:
                    gross_ret = exit_price / position.entry_price - 1.0
                else:  # SHORT
                    gross_ret = position.entry_price / exit_price - 1.0

                # Комиссия за выход
                net_ret = (1 + gross_ret) * (1 - fee_rate) - 1.0
                equity *= 1 + net_ret

                trades.append(
                    Trade(
                        entry_time=position.entry_time,
                        exit_time=ts,
                        side="long" if position.side == Side.BUY else "short",
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        sl_price=position.sl_price,
                        trailing_dist=position.trailing_dist,
                        bars_held=bars_held,
                        gross_return=gross_ret,
                        net_return=net_ret,
                        exit_reason=exit_reason,
                    )
                )

                # Обновляем cooldown с timestamp свечи (в секундах)
                strategy.last_trade_ts = ts / 1000
                position = None

        # Если нет позиции, проверяем сигналы входа
        if position is None:
            try:
                # Диагностика условий входа
                if verbose and len(candles_list) >= min_required:
                    from flying_trading.adapters.indicator_service import (
                        IndicatorService,
                    )

                    ind_service = IndicatorService()
                    data = ind_service.calculate(candles_list, params)

                    close = data["close"]
                    open_price = data["open"]
                    ema_200 = data["ema_200"]
                    rsi = data["rsi"]

                    is_uptrend = close > (ema_200 * 1.001)
                    is_downtrend = close < (ema_200 * 0.999)
                    is_green_candle = close > open_price
                    is_red_candle = close < open_price

                    diagnostics["total_checks"] += 1

                    if not is_uptrend and not is_downtrend:
                        diagnostics["no_trend"] += 1
                    elif is_uptrend:
                        if not (35 < rsi < 55):
                            diagnostics["rsi_out_of_range"] += 1
                        elif not is_green_candle:
                            diagnostics["wrong_candle_color"] += 1
                    elif is_downtrend:
                        if not (45 < rsi < 65):
                            diagnostics["rsi_out_of_range"] += 1
                        elif not is_red_candle:
                            diagnostics["wrong_candle_color"] += 1

                signal = strategy.analyze(candles_list, current_pos_size)

                if signal.action != Side.NONE:
                    diagnostics["signals"] += 1
                elif verbose and signal.reason == "Cooldown":
                    diagnostics["cooldown"] += 1

            except (ValueError, IndexError, KeyError) as e:
                # Недостаточно данных для расчета индикаторов
                if verbose:
                    print(f"Ошибка на свече {i}: {e}")
                equity_curve.append(equity)
                continue

            if signal.action == Side.BUY and signal.sl_price is not None:
                # Вход в LONG
                entry_price = candle.close
                sl_price = signal.sl_price
                trailing_dist = signal.trailing_dist

                position = Position(
                    side=Side.BUY,
                    entry_price=entry_price,
                    entry_time=ts,
                    entry_index=i,
                    sl_price=sl_price,
                    trailing_dist=trailing_dist,
                    highest_price=entry_price,
                    lowest_price=0.0,
                )

                # Комиссия за вход
                equity *= 1 - fee_rate

            elif signal.action == Side.SELL and signal.sl_price is not None:
                # Вход в SHORT
                entry_price = candle.close
                sl_price = signal.sl_price
                trailing_dist = signal.trailing_dist

                position = Position(
                    side=Side.SELL,
                    entry_price=entry_price,
                    entry_time=ts,
                    entry_index=i,
                    sl_price=sl_price,
                    trailing_dist=trailing_dist,
                    highest_price=0.0,
                    lowest_price=entry_price,
                )

                # Комиссия за вход
                equity *= 1 - fee_rate

        equity_curve.append(equity)

    # Закрываем последнюю позицию, если она осталась открытой
    if position is not None:
        last_candle = candles_list[-1]
        exit_price = last_candle.close
        bars_held = total_bars - 1 - position.entry_index

        if position.side == Side.BUY:
            gross_ret = exit_price / position.entry_price - 1.0
        else:
            gross_ret = position.entry_price / exit_price - 1.0

        net_ret = (1 + gross_ret) * (1 - fee_rate) - 1.0
        equity *= 1 + net_ret

        trades.append(
            Trade(
                entry_time=position.entry_time,
                exit_time=last_candle.ts,
                side="long" if position.side == Side.BUY else "short",
                entry_price=position.entry_price,
                exit_price=exit_price,
                sl_price=position.sl_price,
                trailing_dist=position.trailing_dist,
                bars_held=bars_held,
                gross_return=gross_ret,
                net_return=net_ret,
                exit_reason="end_of_data",
            )
        )

    # DataFrame по сделкам
    if trades:
        trades_df = pd.DataFrame([t.__dict__ for t in trades])
        trades_df["entry_time"] = pd.to_datetime(
            trades_df["entry_time"], unit="ms", utc=True
        )
        trades_df["exit_time"] = pd.to_datetime(
            trades_df["exit_time"], unit="ms", utc=True
        )
    else:
        trades_df = pd.DataFrame(
            columns=[
                "entry_time",
                "exit_time",
                "side",
                "entry_price",
                "exit_price",
                "sl_price",
                "trailing_dist",
                "bars_held",
                "gross_return",
                "net_return",
                "exit_reason",
            ]
        )

    # Equity curve
    equity_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df["ts"], unit="ms", utc=True),
            "equity": equity_curve[: len(df)],
        }
    )

    # Статистика
    stats = compute_stats(trades_df, equity_df, start_equity)

    # Добавляем диагностику в статистику
    if verbose and diagnostics["total_checks"] > 0:
        stats["diagnostics"] = {
            "total_checks": diagnostics["total_checks"],
            "cooldown_pct": diagnostics["cooldown"] / diagnostics["total_checks"] * 100,
            "no_trend_pct": diagnostics["no_trend"] / diagnostics["total_checks"] * 100,
            "rsi_out_of_range_pct": diagnostics["rsi_out_of_range"]
            / diagnostics["total_checks"]
            * 100,
            "wrong_candle_color_pct": diagnostics["wrong_candle_color"]
            / diagnostics["total_checks"]
            * 100,
            "signals": diagnostics["signals"],
        }

    return trades_df, equity_df, stats


def compute_stats(
    trades_df: pd.DataFrame, equity_df: pd.DataFrame, start_equity: float
) -> dict[str, Any]:
    """Вычисляет статистику бэктеста."""
    if trades_df.empty:
        return {
            "total_trades": 0,
            "winrate": None,
            "avg_gain": None,
            "avg_loss": None,
            "profit_factor": None,
            "max_drawdown_pct": None,
            "final_equity": equity_df["equity"].iloc[-1],
            "total_return_pct": equity_df["equity"].iloc[-1] / start_equity - 1,
            "sharpe_ratio": None,
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
        gross_loss = abs(losses["net_return"].sum())
        if gross_loss != 0:
            profit_factor = gross_profit / gross_loss

    # Max drawdown
    equity_series = equity_df["equity"].astype(float)
    running_max = equity_series.cummax()
    drawdown = equity_series / running_max - 1.0
    max_drawdown_pct = drawdown.min()

    final_equity = float(equity_series.iloc[-1])
    total_return_pct = final_equity / start_equity - 1.0

    # Sharpe ratio
    r = trades_df["net_return"]
    if len(r) > 1 and r.std() != 0:
        sharpe = r.mean() / r.std() * (len(r) ** 0.5)
    else:
        sharpe = None

    # Дополнительная статистика
    long_trades = trades_df[trades_df["side"] == "long"]
    short_trades = trades_df[trades_df["side"] == "short"]

    return {
        "total_trades": total_trades,
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "winrate": winrate,
        "avg_gain": avg_gain,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown_pct,
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "sharpe_ratio": sharpe,
        "avg_bars_held": trades_df["bars_held"].mean() if not trades_df.empty else None,
    }


# ======================= ВИЗУАЛИЗАЦИЯ =======================


def plot_backtest_results(
    df: pd.DataFrame,
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    stats: dict[str, Any],
    symbol: str,
    interval: str,
    output_dir: str = ".",
):
    """
    Строит графики по результатам бэктеста.

    Args:
        df: DataFrame с ценовыми данными
        trades_df: DataFrame со сделками
        equity_df: DataFrame с кривой капитала
        stats: Словарь со статистикой
        symbol: Торговый символ
        interval: Таймфрейм
        output_dir: Директория для сохранения графиков
    """
    if trades_df.empty:
        print("Нет сделок для визуализации")
        return

    # Подготовка данных
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    trades_df["entry_time"] = pd.to_datetime(
        trades_df["entry_time"], unit="ms", utc=True
    )
    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], unit="ms", utc=True)

    # Создаем фигуру с несколькими subplot'ами
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

    # 1. График цены с отметками входов/выходов
    ax1 = fig.add_subplot(gs[0:2, :])
    ax1.plot(
        df["timestamp"],
        df["close"],
        label="Цена",
        linewidth=0.8,
        alpha=0.7,
        color="black",
    )

    # Отмечаем входы и выходы
    long_trades = trades_df[trades_df["side"] == "long"]
    short_trades = trades_df[trades_df["side"] == "short"]

    # Long входы (зеленые стрелки вверх)
    for _, trade in long_trades.iterrows():
        ax1.scatter(
            trade["entry_time"],
            trade["entry_price"],
            color="green",
            marker="^",
            s=100,
            zorder=5,
            alpha=0.7,
        )
        ax1.scatter(
            trade["exit_time"],
            trade["exit_price"],
            color="red" if trade["net_return"] < 0 else "green",
            marker="v",
            s=100,
            zorder=5,
            alpha=0.7,
        )
        # Линия от входа до выхода
        ax1.plot(
            [trade["entry_time"], trade["exit_time"]],
            [trade["entry_price"], trade["exit_price"]],
            color="green" if trade["net_return"] > 0 else "red",
            alpha=0.3,
            linewidth=1,
        )

    # Short входы (красные стрелки вниз)
    for _, trade in short_trades.iterrows():
        ax1.scatter(
            trade["entry_time"],
            trade["entry_price"],
            color="red",
            marker="v",
            s=100,
            zorder=5,
            alpha=0.7,
        )
        ax1.scatter(
            trade["exit_time"],
            trade["exit_price"],
            color="green" if trade["net_return"] < 0 else "red",
            marker="^",
            s=100,
            zorder=5,
            alpha=0.7,
        )
        # Линия от входа до выхода
        ax1.plot(
            [trade["entry_time"], trade["exit_time"]],
            [trade["entry_price"], trade["exit_price"]],
            color="green" if trade["net_return"] > 0 else "red",
            alpha=0.3,
            linewidth=1,
        )

    ax1.set_title(
        f"{symbol} {interval}m - Цена и сделки", fontsize=14, fontweight="bold"
    )
    ax1.set_xlabel("Время")
    ax1.set_ylabel("Цена")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))

    # 2. Кривая капитала
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.plot(equity_df["timestamp"], equity_df["equity"], linewidth=1.5, color="blue")
    ax2.axhline(
        y=stats.get("final_equity", 0),
        color="green",
        linestyle="--",
        alpha=0.5,
        label="Финальный капитал",
    )
    ax2.set_title("Кривая капитала", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Время")
    ax2.set_ylabel("Капитал")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # 3. Drawdown
    ax3 = fig.add_subplot(gs[2, 1])
    equity_series = equity_df["equity"].astype(float)
    running_max = equity_series.cummax()
    drawdown = (equity_series / running_max - 1.0) * 100
    ax3.fill_between(equity_df["timestamp"], drawdown, 0, alpha=0.3, color="red")
    ax3.plot(equity_df["timestamp"], drawdown, linewidth=1, color="red")
    ax3.set_title("Drawdown", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Время")
    ax3.set_ylabel("Drawdown (%)")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # 4. Распределение прибылей/убытков
    ax4 = fig.add_subplot(gs[3, 0])
    returns = trades_df["net_return"] * 100
    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    if not wins.empty:
        ax4.hist(
            wins, bins=30, alpha=0.6, color="green", label=f"Прибыли ({len(wins)})"
        )
    if not losses.empty:
        ax4.hist(
            losses, bins=30, alpha=0.6, color="red", label=f"Убытки ({len(losses)})"
        )

    ax4.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax4.set_title("Распределение прибылей/убытков", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Доходность (%)")
    ax4.set_ylabel("Количество")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Кумулятивная доходность по сделкам
    ax5 = fig.add_subplot(gs[3, 1])
    cumulative_returns = (1 + trades_df["net_return"]).cumprod() - 1
    cumulative_returns_pct = cumulative_returns * 100
    ax5.plot(
        range(len(cumulative_returns_pct)),
        cumulative_returns_pct,
        linewidth=1.5,
        color="blue",
    )
    ax5.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax5.set_title("Кумулятивная доходность", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Номер сделки")
    ax5.set_ylabel("Кумулятивная доходность (%)")
    ax5.grid(True, alpha=0.3)

    # Добавляем статистику в заголовок
    stats_text = (
        f"Сделок: {stats.get('total_trades', 0)} | "
        f"Winrate: {stats.get('winrate', 0) * 100:.1f}% | "
        f"Прибыль: {stats.get('total_return_pct', 0) * 100:.2f}% | "
        f"Max DD: {stats.get('max_drawdown_pct', 0) * 100:.2f}%"
    )
    fig.suptitle(stats_text, fontsize=10, y=0.995)

    # Сохраняем график
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_file = output_path / f"backtest_plot_{symbol}_{interval}m.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"  График: {plot_file}")

    plt.close()


# ======================= MAIN =======================


def main():
    """Основная функция для запуска бэктеста."""
    import argparse

    parser = argparse.ArgumentParser(description="Backtest TrendStrategyLogic")
    parser.add_argument("--symbol", default="SOLUSDT", help="Trading symbol")
    parser.add_argument("--interval", default="5", help="Trading interval (minutes)")
    parser.add_argument("--days", type=int, default=30, help="Days to backtest")
    parser.add_argument(
        "--start-equity", type=float, default=1000.0, help="Starting equity"
    )
    parser.add_argument(
        "--fee-rate", type=float, default=0.001, help="Fee rate per side"
    )
    parser.add_argument(
        "--output-dir", default=".", help="Output directory for CSV files"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show diagnostic information"
    )
    args = parser.parse_args()

    SYMBOL = args.symbol
    INTERVAL = args.interval
    CATEGORY = "linear"

    # Период бэктеста
    end = pd.Timestamp.utcnow().floor("min")
    start = end - pd.Timedelta(days=args.days)

    start_ts_ms = int(start.timestamp() * 1000)
    end_ts_ms = int(end.timestamp() * 1000)

    api_key = config.bybit_api_key
    api_secret = config.bybit_api_secret

    if not api_key or not api_secret:
        raise RuntimeError("Нужно указать BYBIT_API_KEY и BYBIT_API_SECRET в .env")

    session = HTTP(
        testnet=config.bybit_testnet,
        api_key=api_key,
        api_secret=api_secret,
    )

    print(f"Загружаю {SYMBOL} {INTERVAL}m свечи с {start} по {end}...")
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

    # Параметры стратегии
    params = StrategyParams(interval=INTERVAL)

    # Запускаем бэктест
    print("Запускаю бэктест...")
    trades_df, equity_df, stats = backtest_trend_strategy(
        df=df,
        params=params,
        start_equity=args.start_equity,
        fee_rate=args.fee_rate,
        verbose=args.verbose,
    )

    # Выводим статистику
    print("\n" + "=" * 60)
    print("СТАТИСТИКА БЭКТЕСТА")
    print("=" * 60)
    for k, v in stats.items():
        if v is not None:
            if isinstance(v, float):
                if "pct" in k or "return" in k or "drawdown" in k:
                    print(f"{k:25s}: {v * 100:.2f}%")
                elif "ratio" in k or "factor" in k:
                    print(f"{k:25s}: {v:.4f}")
                else:
                    print(f"{k:25s}: {v:.2f}")
            else:
                print(f"{k:25s}: {v}")
        else:
            print(f"{k:25s}: N/A")

    print("\nПервые 10 сделок:")
    print(trades_df.head(10).to_string())

    # Выводим диагностику, если запрошена
    if args.verbose and "diagnostics" in stats:
        print("\n" + "=" * 60)
        print("ДИАГНОСТИКА: Почему мало сделок?")
        print("=" * 60)
        diag = stats["diagnostics"]
        print(f"Всего проверок: {diag['total_checks']}")
        print(f"Cooldown: {diag['cooldown_pct']:.1f}%")
        print(f"Нет тренда: {diag['no_trend_pct']:.1f}%")
        print(f"RSI вне диапазона: {diag['rsi_out_of_range_pct']:.1f}%")
        print(f"Неправильный цвет свечи: {diag['wrong_candle_color_pct']:.1f}%")
        print(f"Всего сигналов: {diag['signals']}")

    # Сохраняем результаты
    output_dir = args.output_dir
    trades_file = f"{output_dir}/trades_trend_strategy_{SYMBOL}_{INTERVAL}m.csv"
    equity_file = f"{output_dir}/equity_trend_strategy_{SYMBOL}_{INTERVAL}m.csv"

    trades_df.to_csv(trades_file, index=False)
    equity_df.to_csv(equity_file, index=False)

    print("\nРезультаты сохранены:")
    print(f"  Сделки: {trades_file}")
    print(f"  Кривая капитала: {equity_file}")

    # Строим графики
    if not trades_df.empty:
        print("\nСтрою графики...")
        plot_backtest_results(
            df=df,
            trades_df=trades_df,
            equity_df=equity_df,
            stats=stats,
            symbol=SYMBOL,
            interval=INTERVAL,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
