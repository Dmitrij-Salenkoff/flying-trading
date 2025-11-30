import os
import time

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

from flying_trading.train import FEATURE_COLS, add_features

load_dotenv()

TESTNET = False  # для безопасности лучше начинать с True
SYMBOL = "SOLUSDT"
CATEGORY = "linear"
INTERVAL = "1"  # тот же, что в тренинге
KLINE_LIMIT = 1000  # сколько последних свечей брать
POSITION_MODE = "hedge"  # мы явно работаем в hedge-режиме

MODEL_PATH = f"models/{SYMBOL}_{INTERVAL}m_logreg.pkl"

EDGE_THRESHOLD = 0.05  # порог сигнала
TRADE_QTY = 0.1  # базовый размер по модулю
MAX_POSITION = 5  # ограничение net-позиции
SLEEP_SECONDS = 10  # частота проверки (для 1m достаточно)


# ==============================
#   Загрузка последних свечей
# ==============================


def fetch_recent_klines(
    session: HTTP, symbol: str, category: str, interval: str, limit: int
) -> pd.DataFrame:
    resp = session.get_kline(
        category=category,
        symbol=symbol,
        interval=interval,
        limit=limit,
    )
    result = resp.get("result", {})
    rows = result.get("list", [])
    if not rows:
        raise ValueError("No kline data from Bybit")

    # сортируем по времени
    rows_sorted = sorted(rows, key=lambda r: int(r[0]))
    n_cols = len(rows_sorted[0])
    if n_cols == 7:
        cols = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
    elif n_cols == 6:
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
    else:
        raise ValueError(f"Unexpected kline row length: {n_cols}")

    df = pd.DataFrame(rows_sorted, columns=cols)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"].astype("int64"), unit="ms", utc=True
    )
    for c in cols[1:]:
        df[c] = df[c].astype(float)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ==============================
#    Состояние позиций + PnL
# ==============================


def get_position_state(session: HTTP, symbol: str):
    """
    Возвращает словарь:
      {
        "long_size": float,
        "short_size": float,
        "net_pos": float,
        "unrealised_pnl": float,
        "cum_realised_pnl": float,
      }
    В hedge-режиме long = positionIdx=1, short = positionIdx=2.
    """
    resp = session.get_positions(
        category=CATEGORY,
        symbol=symbol,
    )
    positions = resp.get("result", {}).get("list", []) or []

    long_size = 0.0
    short_size = 0.0
    long_upnl = 0.0
    short_upnl = 0.0
    long_real = 0.0
    short_real = 0.0

    for pos in positions:
        if pos.get("symbol") != symbol:
            continue

        size = float(pos.get("size") or 0.0)
        upnl = float(pos.get("unrealisedPnl") or 0.0)
        crpnl = float(pos.get("cumRealisedPnl") or 0.0)
        idx = int(pos.get("positionIdx", 0))

        if POSITION_MODE == "hedge":
            if idx == 1:
                long_size = size
                long_upnl = upnl
                long_real = crpnl
            elif idx == 2:
                short_size = size
                short_upnl = upnl
                short_real = crpnl
        else:
            # one-way (idx=0) — для совместимости, вдруг пригодится
            side = pos.get("side", "")
            if side == "Buy":
                long_size = size
                long_upnl = upnl
                long_real = crpnl
            elif side == "Sell":
                short_size = size
                short_upnl = upnl
                short_real = crpnl

    net_pos = long_size - short_size
    unrealised = long_upnl + short_upnl
    cum_real = long_real + short_real

    return {
        "long_size": long_size,
        "short_size": short_size,
        "net_pos": net_pos,
        "unrealised_pnl": unrealised,
        "cum_realised_pnl": cum_real,
    }


# ==============================
#   Отправка маркет-ордеров
# ==============================


def place_market_order(
    session: HTTP,
    symbol: str,
    side: str,
    qty: float,
    reduce_only: bool = False,
    position_idx: int | None = None,
):
    """
    side: "Buy" или "Sell"
    qty: размер в контрактах/монетах (строкой)
    position_idx:
      - hedge mode: обязателен (1 = long-сторона, 2 = short-сторона)
      - one_way: игнорируется, всегда 0
    """
    qty = float(qty)
    if qty <= 0:
        return

    kwargs = {
        "category": CATEGORY,
        "symbol": symbol,
        "side": side,
        "orderType": "Market",
        "qty": str(qty),
        "reduceOnly": reduce_only,
    }

    if POSITION_MODE == "hedge":
        if position_idx is None:
            raise ValueError("position_idx is required in hedge mode")
        kwargs["positionIdx"] = position_idx
    elif POSITION_MODE == "one_way":
        kwargs["positionIdx"] = 0
    else:
        raise ValueError(f"Unknown POSITION_MODE={POSITION_MODE}")

    resp = session.place_order(**kwargs)
    return resp


# ==============================
#      Логика принятия решения
# ==============================


def decide_and_trade(session: HTTP, state: dict, edge: float):
    """
    state: результат get_position_state
    edge: сигнал модели (p_up - p_down)
    """

    long_size = state["long_size"]
    short_size = state["short_size"]
    # net_pos = state["net_pos"]

    # желаемая net-позиция по сигналу
    if edge > EDGE_THRESHOLD:
        desired_net = TRADE_QTY
    elif edge < -EDGE_THRESHOLD:
        desired_net = -TRADE_QTY
    else:
        desired_net = 0.0

    # ограничиваем по модулю
    desired_net = max(-MAX_POSITION, min(MAX_POSITION, desired_net))

    # превращаем desired_net в target_long/target_short:
    if desired_net > 0:
        target_long = desired_net
        target_short = 0.0
    elif desired_net < 0:
        target_long = 0.0
        target_short = -desired_net
    else:
        target_long = 0.0
        target_short = 0.0

    # если и так близко к целям — ничего не делаем
    if abs(long_size - target_long) < 1e-6 and abs(short_size - target_short) < 1e-6:
        print(
            f"[TRADE] keep positions: long={long_size}, short={short_size}, edge={edge:.3f}"
        )
        return

    print(
        f"[TRADE] cur_long={long_size}, cur_short={short_size}, "
        f"target_long={target_long}, target_short={target_short}, edge={edge:.3f}"
    )

    # 1) сначала уменьшаем лишний long (Sell по idx=1, reduceOnly=True)
    if long_size > target_long + 1e-8:
        reduce_qty = long_size - target_long
        print(
            f"[ORDER] close/reduce LONG: Sell {reduce_qty} {SYMBOL} (idx=1, reduceOnly)"
        )
        try:
            place_market_order(
                session,
                SYMBOL,
                side="Sell",
                qty=reduce_qty,
                reduce_only=True,
                position_idx=1,
            )
        except Exception as e:
            print("Error reducing long:", e)

    # 2) потом уменьшаем лишний short (Buy по idx=2, reduceOnly=True)
    if short_size > target_short + 1e-8:
        reduce_qty = short_size - target_short
        print(
            f"[ORDER] close/reduce SHORT: Buy {reduce_qty} {SYMBOL} (idx=2, reduceOnly)"
        )
        try:
            place_market_order(
                session,
                SYMBOL,
                side="Buy",
                qty=reduce_qty,
                reduce_only=True,
                position_idx=2,
            )
        except Exception as e:
            print("Error reducing short:", e)

    # пересчитывать long/short после reduce можно, но для маленьких размеров
    # можно просто добавлять поверх (ошибка не будет критичной),
    # но давай аккуратно — запросим состояние заново:
    new_state = get_position_state(session, SYMBOL)
    long_size = new_state["long_size"]
    short_size = new_state["short_size"]

    # 3) если нужно увеличить long (Buy idx=1)
    if long_size < target_long - 1e-8:
        add_qty = target_long - long_size
        print(f"[ORDER] open/increase LONG: Buy {add_qty} {SYMBOL} (idx=1)")
        try:
            place_market_order(
                session,
                SYMBOL,
                side="Buy",
                qty=add_qty,
                reduce_only=False,
                position_idx=1,
            )
        except Exception as e:
            print("Error increasing long:", e)

    # 4) если нужно увеличить short (Sell idx=2)
    if short_size < target_short - 1e-8:
        add_qty = target_short - short_size
        print(f"[ORDER] open/increase SHORT: Sell {add_qty} {SYMBOL} (idx=2)")
        try:
            place_market_order(
                session,
                SYMBOL,
                side="Sell",
                qty=add_qty,
                reduce_only=False,
                position_idx=2,
            )
        except Exception as e:
            print("Error increasing short:", e)


# ==============================
#              MAIN
# ==============================


def main():
    # 1. Грузим модель
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")

    # 2. Создаём HTTP с ключами (для приватных методов нужен api_key/api_secret)
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Set BYBIT_API_KEY and BYBIT_API_SECRET env vars")

    session = HTTP(
        testnet=TESTNET,
        api_key=api_key,
        api_secret=api_secret,
    )

    last_bar_time = None
    start_cum_realised = None  # базовая точка для PnL

    while True:
        try:
            df_raw = fetch_recent_klines(
                session, SYMBOL, CATEGORY, INTERVAL, KLINE_LIMIT
            )
            df_feat = add_features(df_raw)

            # Убираем inf, -inf → в NaN
            df_feat = df_feat.replace([np.inf, -np.inf], np.nan)

            # Берём только строки, где все нужные фичи не NaN
            df_live = df_feat.dropna(subset=FEATURE_COLS)

            if df_live.empty:
                print(
                    "[LIVE] Not enough clean feature rows yet (NaN in features). Waiting..."
                )
                time.sleep(SLEEP_SECONDS)
                continue

            # Берём последнюю нормальную строку
            row = df_live.iloc[-1]

            # Чтобы не делать лишних сделок — проверяем, изменилась ли свеча
            ts = row["timestamp"]
            if last_bar_time is not None and ts <= last_bar_time:
                time.sleep(SLEEP_SECONDS)
                continue

            last_bar_time = ts

            # Формируем вектор признаков
            x = row[FEATURE_COLS].to_frame().T
            probas = model.predict_proba(x)[0]
            classes = model.classes_.tolist()

            if 1 in classes and -1 in classes:
                idx_up = classes.index(1)
                idx_down = classes.index(-1)
                p_up = probas[idx_up]
                p_down = probas[idx_down]
                edge = p_up - p_down
            else:
                if 1 in classes:
                    idx_up = classes.index(1)
                    p_up = probas[idx_up]
                    edge = p_up
                else:
                    edge = 0.0

            print(f"[{ts}] close={row['close']:.4f} edge={edge:.3f}")

            # Состояние позиций и PnL
            state = get_position_state(session, SYMBOL)
            net_pos = state["net_pos"]
            upnl = state["unrealised_pnl"]
            cum_real = state["cum_realised_pnl"]

            if start_cum_realised is None:
                start_cum_realised = cum_real

            realised_delta = cum_real - start_cum_realised
            total_pnl = realised_delta + upnl

            print(
                f"[PnL] total={total_pnl:.6f}  realised={realised_delta:.6f}  unrealised={upnl:.6f}"
            )
            print(
                f"Current positions: long={state['long_size']}, "
                f"short={state['short_size']}, net={net_pos}"
            )

            # Решаем и торгуем (теперь с правильным закрытием ног)
            decide_and_trade(session, state, edge)

        except Exception as e:
            print("Error in main loop:", e)

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
