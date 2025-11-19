from pybit.unified_trading import HTTP
from flying_trading.config import config

price = "137"
qty = "0.1"


session = HTTP(
    testnet=False,  # True для тестнета, False для боевой биржи
    api_key=config.bybit_api_key,
    api_secret=config.bybit_api_secret,
)

resp = session.place_order(
    category="linear",
    symbol="SOLUSDT",
    side="Sell",
    orderType="Limit",
    qty=qty,
    price=price,
    positionIdx=0,
    timeInForce="GTC",  # GTC по умолчанию
    reduceOnly=False,  # True, если хочешь только закрывать позицию
    closeOnTrigger=False,  # для некоторых стоп-ордеров
    takeProfit="134",  # TP цена
    stopLoss="140",  # SL цена
    tpTriggerBy="LastPrice",  # по какой цене триггерить
    slTriggerBy="LastPrice",
)
print(resp)
