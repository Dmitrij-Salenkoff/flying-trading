from flying_trading.domain.interfaces import IExchange
from flying_trading.logger import get_logger

logger = get_logger(__name__)


class RiskManager:
    def __init__(self, exchange: IExchange, risk_per_trade: float = 0.01):
        self.exchange = exchange
        self.risk_per_trade = risk_per_trade
        self.logger = get_logger(f"{__name__}.RiskManager")

    def calculate_qty(self, entry_price: float, sl_price: float) -> float:
        balance = self.exchange.get_wallet_balance()
        risk_amount = balance * self.risk_per_trade

        dist_pct = abs(entry_price - sl_price) / entry_price
        if dist_pct == 0:
            self.logger.warning("Stop loss equals entry price, returning 0 qty")
            return 0.0

        pos_usdt = risk_amount / dist_pct
        qty = pos_usdt / entry_price
        
        self.logger.debug(
            f"Qty calculation: balance={balance:.2f}, risk={risk_amount:.2f}, "
            f"dist_pct={dist_pct:.4f}, qty={qty:.4f}"
        )
        return qty
