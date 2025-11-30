import math
from pprint import pprint

from pybit.unified_trading import HTTP

from flying_trading.config import config
from flying_trading.domain.interfaces import IExchange
from flying_trading.domain.models import Candle, Position, Side
from flying_trading.logger import get_logger

logger = get_logger(__name__)

CATEGORY = "linear"


class BybitAdapter(IExchange):
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.session = HTTP(testnet=testnet, api_key=api_key, api_secret=api_secret)
        self.qty_step = {}
        self.qty_precision = {}
        self.price_precision = {}

    def load_precisions(self, symbol: str) -> None:
        resp = self.session.get_instruments_info(category=CATEGORY, symbol=symbol)
        info = resp["result"]["list"][0]
        self.qty_step[symbol] = float(info["lotSizeFilter"]["qtyStep"])
        self.qty_precision[symbol] = int(abs(math.log10(self.qty_step[symbol])))
        self.price_precision[symbol] = int(
            abs(math.log10(float(info["priceFilter"]["tickSize"])))
        )
        logger.info(
            f"Loaded precisions for {symbol}: qty={self.qty_precision[symbol]}, "
            f"price={self.price_precision[symbol]}"
        )

    def get_price_precision(self, symbol: str) -> int:
        return self.price_precision[symbol]

    def get_qty_precision(self, symbol: str) -> int:
        return self.qty_precision[symbol]

    def get_qty_step(self, symbol: str) -> float:
        return self.qty_step[symbol]

    def get_history(
        self,
        symbol: str,
        interval: str,
        limit: int,
        start_ms: None | int = None,
        end_ms: None | int = None,
    ) -> list[Candle]:
        if start_ms and end_ms:
            raise NotImplementedError("Not implemented")  # TODO: implement this
        else:
            resp = self.session.get_kline(
                category=CATEGORY, symbol=symbol, interval=interval, limit=limit
            )
        rows = sorted(resp["result"]["list"], key=lambda x: int(x[0]))
        return [
            Candle(
                ts=int(r[0]),
                open=float(r[1]),
                high=float(r[2]),
                low=float(r[3]),
                close=float(r[4]),
                volume=float(r[5]),
                turnover=float(r[6]),
            )
            for r in rows
        ]

    def get_position(self, symbol: str) -> Position:
        res = self.session.get_positions(category=CATEGORY, symbol=symbol)
        data = res["result"]["list"][0]
        size = float(data["size"])
        side = Side.BUY if size > 0 else (Side.SELL if size < 0 else Side.NONE)
        unrealized_pnl = (
            0 if data["unrealisedPnl"] == "" else float(data["unrealisedPnl"])
        )
        return Position(
            size=abs(size),
            entry_price=float(data["avgPrice"]),
            side=side,
            positionIdx=int(data["positionIdx"]),
            leverage=int(data["leverage"]),
            unrealized_pnl=unrealized_pnl,
            cur_realized_pnl=float(data["curRealisedPnl"]),
        )

    def place_market_order(self, symbol, side, qty, sl):
        qty_str = f"{qty:.{self.qty_precision[symbol]}f}"
        sl_str = f"{sl:.{self.price_precision[symbol]}f}"
        logger.info(
            f"Placing {side} market order: {symbol}, qty={qty_str}, sl={sl_str}"
        )
        try:
            result = self.session.place_order(
                category=CATEGORY,
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=qty_str,
                stopLoss=sl_str,
                slTriggerBy="LastPrice",
                positionIdx=0,
            )
            logger.info(f"Order placed successfully: {result.get('retMsg', 'OK')}")
        except Exception as e:
            logger.error(f"Failed to place order: {e}", exc_info=True)
            raise

    def set_trailing_stop(self, symbol, dist):
        dist_str = f"{dist:.{self.price_precision[symbol]}f}"
        logger.info(f"Setting trailing stop for {symbol}: {dist_str}")
        try:
            result = self.session.set_trading_stop(
                category=CATEGORY,
                symbol=symbol,
                trailingStop=dist_str,
                activePrice="0",
                positionIdx=0,
            )
            logger.info(f"Trailing stop set successfully: {result.get('retMsg', 'OK')}")
        except Exception as e:
            logger.error(f"Failed to set trailing stop: {e}", exc_info=True)
            raise

    def get_current_price(self, symbol: str) -> float:
        resp = self.session.get_orderbook(category=CATEGORY, symbol=symbol, limit=1)
        return float(resp["result"]["b"][0][0])

    def get_wallet_balance(self) -> float:
        resp = self.session.get_wallet_balance(accountType="UNIFIED")
        return float(resp["result"]["list"][0]["totalAvailableBalance"])


if __name__ == "__main__":
    adapter = BybitAdapter(
        api_key=config.bybit_api_key,
        api_secret=config.bybit_api_secret,
        testnet=config.bybit_testnet,
    )
    pprint(adapter.get_current_price("SOLUSDT"))
