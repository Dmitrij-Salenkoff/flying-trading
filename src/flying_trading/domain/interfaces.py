# domain/interfaces.py
from abc import ABC, abstractmethod

from .models import Candle, Position, Side


class IExchange(ABC):
    @abstractmethod
    def get_history(self, symbol: str, interval: str, limit: int) -> list[Candle]: ...

    @abstractmethod
    def get_current_price(self, symbol: str) -> float: ...

    @abstractmethod
    def load_precisions(self, symbol: str): ...

    @abstractmethod
    def get_price_precision(self, symbol: str) -> int: ...

    @abstractmethod
    def get_qty_precision(self, symbol: str) -> int: ...

    @abstractmethod
    def get_qty_step(self, symbol: str) -> float: ...

    @abstractmethod
    def get_position(self, symbol: str) -> Position: ...

    @abstractmethod
    def place_market_order(
        self,
        symbol: str,
        side: Side,
        qty: float,
        sl: None | float = None,
        trailing_dist: None | float = None,
    ): ...

    @abstractmethod
    def set_trailing_stop(self, symbol: str, dist: float): ...

    @abstractmethod
    def get_wallet_balance(self) -> float: ...


class INotifier(ABC):
    @abstractmethod
    def send(self, message: str): ...
