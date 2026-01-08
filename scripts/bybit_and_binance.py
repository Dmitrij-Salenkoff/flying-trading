import asyncio
import websockets
import json
import logging
from datetime import datetime, timezone, UTC
from aiokafka import AIOKafkaProducer

# Настройка логгера
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

KAFKA_TOPIC = "crypto_trades"
KAFKA_TOPIC_FUNDING = "crypto_funding"
KAFKA_BOOTSTRAP_SERVERS = "localhost:19092"  # Или os.getenv...


# === 1. ЛОГИКА НОРМАЛИЗАЦИИ ===
# Мы приводим все к этому формату перед отправкой в Кафку
def create_trade_payload(exchange, symbol, price, quantity, timestamp_ms, side):
    return {
        "exchange": exchange,  # Добавили поле источника!
        "symbol": symbol,
        "price": float(price),
        "quantity": float(quantity),
        # event_time - in utc
        "event_time": datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat(),
        "side": side,  # 'buy' или 'sell'
    }


async def listen_binance_funding(producer):
    # !markPrice@arr@1s возвращает массив данных по ВСЕМ парам раз в секунду
    url = "wss://fstream.binance.com/ws/!markPrice@arr@1s"  # Обрати внимание: fstream (Futures), а не stream
    logging.info("Starting Binance Funding Rate listener...")

    async with websockets.connect(url) as ws:
        while True:
            try:
                msg = await ws.recv()
                data_list = json.loads(msg)

                # data_list - это список из сотен монет.
                # Нам нужно пройтись по нему и вытащить интересные.

                for item in data_list:
                    symbol = item["s"]

                    # Оптимизация: Берем только USDT пары, чтобы не засорять базу мусором типа BTCQuarterly
                    if not symbol.endswith("USDT"):
                        continue

                    payload = {
                        "exchange": "Binance",
                        "symbol": symbol,
                        "mark_price": float(item["p"]),
                        "funding_rate": float(item["r"]),  # Самое важное поле!
                        "next_funding_time": item["T"],
                        "event_time": datetime.now(
                            tz=UTC
                        ).isoformat(),  # Время получения
                    }

                    # Отправляем в ОТДЕЛЬНЫЙ топик
                    await producer.send_and_wait(
                        KAFKA_TOPIC_FUNDING, json.dumps(payload).encode("utf-8")
                    )

            except Exception as e:
                logging.error(f"Error in Funding Listener: {e}")
                await asyncio.sleep(5)


# === 2. СЛУШАТЕЛЬ BINANCE ===
async def listen_binance(producer):
    url = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    exchange_name = "Binance"

    logging.info(f"Starting {exchange_name} listener...")
    async with websockets.connect(url) as ws:
        while True:
            try:
                msg = await ws.recv()
                data = json.loads(msg)

                # Маппинг полей Binance
                normalized = create_trade_payload(
                    exchange=exchange_name,
                    symbol="BTCUSDT",
                    price=data["p"],
                    quantity=data["q"],
                    timestamp_ms=data["E"],
                    side="buy"
                    if not data["m"]
                    else "sell",  # m=True значит Maker (продавец инициатор?) -> Sell
                )

                await producer.send_and_wait(
                    KAFKA_TOPIC, json.dumps(normalized).encode("utf-8")
                )

            except Exception as e:
                logging.error(f"Error in {exchange_name}: {e}")
                await asyncio.sleep(
                    5
                )  # Пауза перед реконнектом (в реальном коде нужен цикл while True снаружи)


# === 3. СЛУШАТЕЛЬ BYBIT (Сложнее: требует подписки) ===
async def listen_bybit(producer):
    # Публичный V5 стрим
    url = "wss://stream.bybit.com/v5/public/linear"
    exchange_name = "Bybit"

    logging.info(f"Starting {exchange_name} listener...")
    async with websockets.connect(url) as ws:
        # 3.1 ОТПРАВКА ЗАПРОСА НА ПОДПИСКУ
        subscribe_msg = {"op": "subscribe", "args": ["publicTrade.BTCUSDT"]}
        await ws.send(json.dumps(subscribe_msg))
        logging.info(f"Subscribed to {exchange_name}")

        while True:
            try:
                msg = await ws.recv()
                data = json.loads(msg)

                # Игнорируем сообщения подтверждения подписки
                if "topic" not in data or data["topic"] != "publicTrade.BTCUSDT":
                    continue

                # Bybit присылает массив сделок (batch)
                for trade in data["data"]:
                    normalized = create_trade_payload(
                        exchange=exchange_name,
                        symbol="BTCUSDT",
                        price=trade[
                            "p"
                        ],  # У Bybit поля тоже p/v или price/size в зависимости от версии
                        quantity=trade["v"],
                        timestamp_ms=trade["T"],  # T - timestamp
                        side=trade["S"].lower(),  # 'Buy' -> 'buy'
                    )
                    await producer.send_and_wait(
                        KAFKA_TOPIC, json.dumps(normalized).encode("utf-8")
                    )

            except Exception as e:
                logging.error(f"Error in {exchange_name}: {e}")
                await asyncio.sleep(5)


# === 4. ORCHESTRATOR ===
async def main():
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
    await producer.start()
    try:
        # Запускаем обе функции ПАРАЛЛЕЛЬНО
        await asyncio.gather(
            listen_binance(producer),
            listen_bybit(producer),
            listen_binance_funding(producer),
        )
    finally:
        await producer.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
