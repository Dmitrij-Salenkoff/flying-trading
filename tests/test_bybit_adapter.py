from flying_trading.adapters.bybit_adapter import BybitAdapter
from flying_trading.config import config


def test_get_current_price():
    adapter = BybitAdapter(
        api_key=config.bybit_api_key,
        api_secret=config.bybit_api_secret,
        testnet=config.bybit_testnet,
    )
    assert adapter.get_current_price("BTCUSDT") > 0


def test_get_position():
    adapter = BybitAdapter(
        api_key=config.bybit_api_key,
        api_secret=config.bybit_api_secret,
        testnet=config.bybit_testnet,
    )
    position = adapter.get_position("BTCUSDT")
    assert position is not None
