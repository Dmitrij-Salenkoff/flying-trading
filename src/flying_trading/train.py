from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import os
from datetime import timedelta

import joblib

FEATURE_COLS = [
    "return_1",
    "return_5",
    "vol_5",
    "vol_20",
    "vol_ratio",
    "volume_z",
    "ma_ratio",
    "rsi_14",
]

# ==============================
#           CONFIG
# ==============================

TESTNET = False  # переключатель окружения Bybit
SYMBOL = "SOLUSDT"
CATEGORY = "linear"  # для фьючей/перпетулей
INTERVAL = "1"  # "1","3","5","15","60","240","D","W" ...
START_DATE = "2025-09-16 00:00"
END_DATE = "2025-11-16 00:00"

HORIZON_BARS = 5  # прогноз через N баров вперёд
TARGET_THRESHOLD = 0.0005  # 0.05% — порог для up/down
EDGE_THRESHOLD = 0.5  # порог уверенности модели для входа
TAKER_FEE = 0.00036  # 0.036% на вход + 0.036% на выход -> 2*TAKER_FEE


# ==============================
#    UTILS: time & interval
# ==============================


def parse_utc(date_str: str) -> datetime:
    """
    Преобразует строку 'YYYY-MM-DD HH:MM' в datetime UTC.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
    return dt.replace(tzinfo=timezone.utc)


def interval_to_ms(interval: str) -> int:
    """
    Bybit v5 kline intervals:
    1,3,5,15,30,60,120,240,360,720,'D','W','M'
    Нас интересуют минуты и день/неделя.
    """
    s = str(interval)
    if s.isdigit():
        minutes = int(s)
        return minutes * 60_000
    mapping = {
        "D": 24 * 60 * 60_000,
        "W": 7 * 24 * 60 * 60_000,
        "M": 30 * 24 * 60 * 60_000,  # условный месяц
    }
    if s in mapping:
        return mapping[s]
    raise ValueError(f"Unsupported interval: {interval}")


# ==============================
#     FETCH: kline history
# ==============================


def fetch_klines_range(
    session: HTTP,
    symbol: str,
    category: str,
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
    limit: int = 1000,
    cache_dir: str = "kline_cache",
) -> pd.DataFrame:
    """
    Качаем свечи Bybit /v5/market/kline за период [start_dt, end_dt)
    с пагинацией только по start и кэшированием целых дней в parquet.

    Для каждого полного дня внутри диапазона:
      - если parquet уже есть -> читаем его;
      - если нет -> качаем день, сохраняем в parquet.
    Огрызки (до первого полного дня и после последнего) качаем как раньше.
    """

    # убеждаемся, что всё в UTC
    start_dt = start_dt.astimezone(timezone.utc)
    end_dt = end_dt.astimezone(timezone.utc)

    interval_ms = interval_to_ms(interval)

    os.makedirs(cache_dir, exist_ok=True)

    # --------- вспомогательная функция без кэша ---------
    def _fetch_raw(start_dt_inner: datetime, end_dt_inner: datetime) -> pd.DataFrame:
        start_ms = int(start_dt_inner.timestamp() * 1000)
        end_ms = int(end_dt_inner.timestamp() * 1000)

        all_rows: List[List[str]] = []
        current_start = start_ms

        while current_start < end_ms:
            resp = session.get_kline(
                category=category,
                symbol=symbol,
                interval=interval,
                start=current_start,
                limit=limit,
            )
            result = resp.get("result", {})
            rows = result.get("list", [])
            if not rows:
                # дальше данных нет
                break

            # Bybit отдаёт обычно от новых к старым -> сортируем по времени
            rows_sorted = sorted(rows, key=lambda r: int(r[0]))

            for row in rows_sorted:
                ts = int(row[0])
                if ts < start_ms or ts >= end_ms:
                    continue
                all_rows.append(row)

            last_ts = int(rows_sorted[-1][0])

            # если мы уже почти упёрлись в конец – выходим
            if last_ts + interval_ms >= end_ms:
                break

            # сдвигаем окно вперёд
            current_start = last_ts + interval_ms

        if not all_rows:
            return pd.DataFrame()

        # убираем дубли и приводим к DataFrame
        all_rows_sorted = sorted(all_rows, key=lambda r: int(r[0]))
        seen = set()
        unique_rows: List[List[str]] = []
        for row in all_rows_sorted:
            ts = int(row[0])
            if ts in seen:
                continue
            seen.add(ts)
            unique_rows.append(row)

        n_cols = len(unique_rows[0])
        if n_cols == 7:
            cols = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
        elif n_cols == 6:
            cols = ["timestamp", "open", "high", "low", "close", "volume"]
        else:
            raise ValueError(f"Unexpected kline row length: {n_cols}")

        df_local = pd.DataFrame(unique_rows, columns=cols)
        df_local["timestamp"] = pd.to_datetime(
            df_local["timestamp"].astype("int64"),
            unit="ms",
            utc=True,
        )
        for c in cols[1:]:
            df_local[c] = df_local[c].astype(float)

        df_local = df_local.sort_values("timestamp").reset_index(drop=True)
        return df_local

    # --------- определяем полные дни внутри диапазона ---------
    # Работать будем по UTC-датам
    from datetime import date

    def day_start(d: date) -> datetime:
        return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)

    def day_end(d: date) -> datetime:
        return day_start(d) + timedelta(days=1)

    # последний включённый момент -> (end_dt - микро) даёт последнюю дату
    last_date = (end_dt - timedelta(microseconds=1)).date()
    cur_date = start_dt.date()

    full_days = []  # список (day_start_dt, day_end_dt, date_obj)

    while cur_date <= last_date:
        ds = day_start(cur_date)
        de = day_end(cur_date)
        # полный день, если он целиком лежит внутри [start_dt, end_dt)
        if ds >= start_dt and de <= end_dt:
            full_days.append((ds, de, cur_date))
        cur_date = cur_date + timedelta(days=1)

    frames = []

    # --------- кусок до первого полного дня ---------
    if full_days:
        first_full_start = full_days[0][0]
        if start_dt < first_full_start:
            pre_df = _fetch_raw(start_dt, first_full_start)
            if not pre_df.empty:
                frames.append(pre_df)
    else:
        # Нет ни одного полного дня — просто качаем весь диапазон и выходим
        df_whole = _fetch_raw(start_dt, end_dt)
        if df_whole.empty:
            raise ValueError(
                "No kline data fetched; check symbol/category/interval/time range"
            )
        return df_whole

    # --------- полные дни с кэшем ---------
    for ds, de, d in full_days:
        day_str = d.strftime("%Y%m%d")
        fname = f"{symbol}_{category}_{interval}_{day_str}.parquet"
        path = os.path.join(cache_dir, fname)

        if os.path.exists(path):
            # читаем из кэша
            df_day = pd.read_parquet(path)
            # на всякий случай фильтруем по интервалу, если вдруг больше
            df_day = df_day[(df_day["timestamp"] >= ds) & (df_day["timestamp"] < de)]
        else:
            # качаем этот день и сохраняем
            df_day = _fetch_raw(ds, de)
            if not df_day.empty:
                df_day.to_parquet(path)

        if not df_day.empty:
            frames.append(df_day)

    # --------- кусок после последнего полного дня ---------
    last_full_end = full_days[-1][1]
    if last_full_end < end_dt:
        post_df = _fetch_raw(last_full_end, end_dt)
        if not post_df.empty:
            frames.append(post_df)

    if not frames:
        raise ValueError(
            "No kline data fetched; check symbol/category/interval/time range"
        )

    # Склеиваем всё, убираем дубли, сортируем
    df = pd.concat(frames, ignore_index=True)
    df = (
        df.drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return df


# ==============================
#    FEATURE ENGINEERING
# ==============================


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["mid"] = (df["high"] + df["low"]) / 2.0

    # Returns
    df["return_1"] = df["close"].pct_change()
    df["return_5"] = df["close"].pct_change(5)

    # Volatility
    df["vol_5"] = df["return_1"].rolling(5).std()
    df["vol_20"] = df["return_1"].rolling(20).std()
    df["vol_ratio"] = df["vol_5"] / df["vol_20"]

    # Volume anomaly
    vol_mean = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std()
    df["volume_z"] = (df["volume"] - vol_mean) / vol_std

    # Moving averages
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_50"] = df["close"].rolling(50).mean()
    df["ma_ratio"] = df["ma_10"] / df["ma_50"] - 1.0

    # RSI 14
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll = 14
    roll_up = up.rolling(roll).mean()
    roll_down = down.rolling(roll).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    df["rsi_14"] = 100 - (100 / (1 + rs))

    return df


# ==============================
#        TARGET (LABEL)
# ==============================


def add_target(
    df: pd.DataFrame,
    horizon_bars: int = HORIZON_BARS,
    threshold: float = TARGET_THRESHOLD,
) -> pd.DataFrame:
    df = df.copy()
    df["future_mid"] = df["mid"].shift(-horizon_bars)
    df["ret_horizon"] = (df["future_mid"] - df["mid"]) / df["mid"]

    def label(r: float) -> int:
        if r > threshold:
            return 1
        if r < -threshold:
            return -1
        return 0

    df["y"] = df["ret_horizon"].apply(label)
    df = df.dropna()
    return df


# ==============================
#           BACKTEST
# ==============================


def backtest_directional(
    df_test: pd.DataFrame,
    edge: np.ndarray,
    taker_fee: float = TAKER_FEE,
    edge_threshold: float = EDGE_THRESHOLD,
) -> Tuple[pd.DataFrame, float]:
    """
    Очень грубый бэктест:
    - если edge > edge_threshold -> long на горизонте H, pnl = ret_horizon - 2*fee
    - если edge < -edge_threshold -> short, pnl = -ret_horizon - 2*fee
    - иначе -> 0
    """
    df_bt = df_test.copy()
    df_bt["edge"] = edge

    long_mask = df_bt["edge"] > edge_threshold
    short_mask = df_bt["edge"] < -edge_threshold

    df_bt["strategy_ret"] = 0.0

    # long: profit = future_ret - fee_entry - fee_exit
    df_bt.loc[long_mask, "strategy_ret"] = (
        df_bt.loc[long_mask, "ret_horizon"] - 2 * taker_fee
    )

    # short: profit = -future_ret - fee_entry - fee_exit
    df_bt.loc[short_mask, "strategy_ret"] = (
        -df_bt.loc[short_mask, "ret_horizon"] - 2 * taker_fee
    )

    # equity-curve (без плеча)
    df_bt["equity"] = (1 + df_bt["strategy_ret"]).cumprod()

    total_pnl = df_bt["strategy_ret"].sum()
    return df_bt, total_pnl


# ==============================
#             MAIN
# ==============================


def main():
    # 1. Создаём HTTP-сессию (public методы get_kline не требуют ключей)
    session = HTTP(testnet=TESTNET)

    start_dt = parse_utc(START_DATE)
    end_dt = parse_utc(END_DATE)

    print(f"Downloading klines {SYMBOL} {INTERVAL} from {start_dt} to {end_dt} ...")
    df_raw = fetch_klines_range(
        session=session,
        symbol=SYMBOL,
        category=CATEGORY,
        interval=INTERVAL,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    print("Raw klines:", df_raw.shape)

    # 2. Фичи + таргет
    df_feat = add_features(df_raw)
    df_all = add_target(df_feat, horizon_bars=HORIZON_BARS, threshold=TARGET_THRESHOLD)

    df_all = df_all.dropna(subset=FEATURE_COLS + ["y", "ret_horizon"])
    print("Dataset after features & target:", df_all.shape)

    X = df_all[FEATURE_COLS]
    y = df_all["y"]

    # 3. Train/test split (важно: shuffle=False, чтобы не ломать время)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

    split_idx = X_train.shape[0]
    df_train = df_all.iloc[:split_idx]
    df_test = df_all.iloc[split_idx:]

    print("Train size:", X_train.shape, "Test size:", X_test.shape)

    # 4. Модель: логистическая регрессия в pipeline со StandardScaler
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, multi_class="auto"),
    )

    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test  accuracy: {test_acc:.3f}")

    # 5. Сохраняем модель
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{SYMBOL}_{INTERVAL}m_logreg.pkl"
    joblib.dump(clf, model_path)
    print(f"Saved model to {model_path}")

    # 5. Прогнозы и edge
    probas = clf.predict_proba(X_test)
    classes = clf.classes_.tolist()

    # Находим индексы классов 1 и -1 (если их нет — надо упрощать задачу)
    if 1 in classes and -1 in classes:
        idx_up = classes.index(1)
        idx_down = classes.index(-1)
        edge = probas[:, idx_up] - probas[:, idx_down]

        import numpy as np
        print("edge stats on test:")
        print("min :", edge.min())
        print("max :", edge.max())
        print("mean:", edge.mean())
        print("q90 :", np.quantile(edge, 0.9))
        print("q95 :", np.quantile(edge, 0.95))
        print("q99 :", np.quantile(edge, 0.99))
    else:
        # fallback: бинарная задача, например только up vs not-up
        # здесь просто берём p(up) как edge
        if 1 in classes:
            idx_up = classes.index(1)
            edge = probas[:, idx_up]
        else:
            edge = np.zeros(len(X_test))
            print("Warning: class 1 not present in test set; edge=0")

    # 6. Бэктест по ret_horizon
    df_bt, total_pnl = backtest_directional(
        df_test=df_test,
        edge=edge,
        taker_fee=TAKER_FEE,
        edge_threshold=EDGE_THRESHOLD,
    )

    n_trades = (df_bt["strategy_ret"] != 0).sum()
    mean_pnl = df_bt["strategy_ret"].mean()

    print("========== BACKTEST RESULT ==========")
    print(f"Trades count:       {n_trades}")
    print(f"Mean PnL per trade: {mean_pnl:.6f}")
    print(f"Total PnL (no lev): {total_pnl:.6f}")
    print(f"Final equity:       {df_bt['equity'].iloc[-1]:.4f}")
    print("=====================================")

    # При желании можно сохранить всё в CSV
    # df_bt.to_csv("backtest_results.csv", index=False)


if __name__ == "__main__":
    main()
