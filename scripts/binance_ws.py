import asyncio
import websockets
import json
from datetime import datetime
from aiokafka import AIOKafkaProducer  # Импортируем продюсера

BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"
KAFKA_TOPIC = "crypto_trades"
# В docker-compose мы выставили внешний порт 19092
KAFKA_BOOTSTRAP_SERVERS = "localhost:19092"


async def process_message(message: str, producer: AIOKafkaProducer):
    try:
        data = json.loads(message)

        normalized_data = {
            "event_time": datetime.fromtimestamp(data["E"] / 1000).isoformat(),
            "symbol": data["s"],
            "price": float(data["p"]),
            "quantity": float(data["q"]),
            "trade_id": data["t"],
            "is_buyer_maker": data["m"],
        }

        # 1. Сериализация: превращаем dict в bytes
        value_json = json.dumps(normalized_data).encode("utf-8")

        # 2. Отправка в Kafka
        # Мы не ждем подтверждения от брокера (fire and forget) для скорости,
        # но await гарантирует, что сообщение ушло в буфер библиотеки.
        await producer.send_and_wait(KAFKA_TOPIC, value_json)

        print(
            f"Sent to Kafka: {normalized_data['symbol']} | Price: {normalized_data['price']}"
        )

    except Exception as e:
        print(f"Error processing message: {e}")


async def run_listener():
    # Инициализация продюсера
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
    # Стартуем соединение с брокером
    await producer.start()
    print("Kafka Producer started.")

    try:
        print(f"Connecting to {BINANCE_WS_URL}...")
        async with websockets.connect(BINANCE_WS_URL) as websocket:
            print("Connected! Listening for trades...")
            while True:
                message = await websocket.recv()
                # Передаем producer внутрь функции обработки
                await process_message(message, producer)
    finally:
        # Очень важно: закрываем соединение при выходе, чтобы сбросить буферы
        await producer.stop()
        print("Kafka Producer stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(run_listener())
    except KeyboardInterrupt:
        print("\nStopped by user.")
