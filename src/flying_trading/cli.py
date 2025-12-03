import time
from datetime import datetime, timedelta

import click
from pybit.unified_trading import WebSocket

from flying_trading.adapters.bybit_adapter import BybitAdapter
from flying_trading.adapters.telegram_adapter import TelegramAdapter
from flying_trading.application.risk_manager import RiskManager
from flying_trading.application.strategy import TrendStrategyLogic
from flying_trading.config import config
from flying_trading.domain.interfaces import IExchange, INotifier
from flying_trading.domain.models import Candle, Side, StrategyParams
from flying_trading.logger import get_logger, setup_logging

logger = get_logger(__name__)


class TradingBot:
    def __init__(self, exchange: IExchange, notifier: INotifier, symbol: str):
        self.exchange = exchange
        self.notifier = notifier
        self.symbol = symbol

        self.params = StrategyParams()
        self.strategy = TrendStrategyLogic(self.params)
        self.risk_manager = RiskManager(exchange)

        self.history = []
        self.logger = get_logger(f"{__name__}.TradingBot")

    def on_candle_update(self, candle: Candle):
        self.history.append(candle)
        if len(self.history) > 500:
            self.history = self.history[-500:]
        pos = self.exchange.get_position(self.symbol)

        signal = self.strategy.analyze(self.history, pos.size)
        self.logger.info(f"Signal: {signal}")

        if signal.action == Side.BUY:
            qty = self.risk_manager.calculate_qty(candle.close, signal.sl_price)

            try:
                self.exchange.place_market_order(
                    self.symbol, "Buy", qty, signal.sl_price
                )
                time.sleep(0.5)
                self.exchange.set_trailing_stop(self.symbol, signal.trailing_dist)
                self.logger.info(
                    f"Order placed and trailing stop set for {self.symbol}"
                )
                self.notifier.send(
                    f"GO LONG! {self.symbol} {qty:.{self.exchange.get_qty_precision(self.symbol)}f} @ {candle.close:.{self.exchange.get_price_precision(self.symbol)}f}"
                )
            except Exception as e:
                self.logger.error(f"Failed to place BUY order: {e}", exc_info=True)

        elif signal.action == Side.SELL:
            qty = self.risk_manager.calculate_qty(candle.close, signal.sl_price)

            try:
                self.exchange.place_market_order(
                    self.symbol, "Sell", qty, signal.sl_price
                )
                time.sleep(0.5)
                self.exchange.set_trailing_stop(self.symbol, signal.trailing_dist)
                self.logger.info(
                    f"Order placed and trailing stop set for {self.symbol}"
                )
                self.notifier.send(
                    f"GO SHORT! {self.symbol} {qty:.{self.exchange.get_qty_precision(self.symbol)}f} @ {candle.close:.{self.exchange.get_price_precision(self.symbol)}f}"
                )
            except Exception as e:
                self.logger.error(f"Failed to place SELL order: {e}", exc_info=True)

    def run(self):
        self.logger.info("Starting trading bot...")

        try:
            self.exchange.load_precisions(self.symbol)
            self.logger.info(f"Loaded precisions for {self.symbol}")
        except Exception as e:
            self.logger.error(f"Failed to load precisions: {e}", exc_info=True)
            raise

        try:
            self.history = self.exchange.get_history(
                self.symbol, self.params.interval, 500
            )
            self.logger.info(f"Loaded {len(self.history)} candles from history")
        except Exception as e:
            self.logger.error(f"Failed to load history: {e}", exc_info=True)
            raise

        ws = WebSocket(testnet=config.bybit_testnet, channel_type="linear")
        self.logger.info("WebSocket connection created")

        def ws_handler(msg):
            try:
                if "data" in msg:
                    d = msg["data"][0]
                    if d.get("confirm"):
                        from flying_trading.domain.models import Candle

                        c = Candle(
                            int(d["end"]),
                            float(d["open"]),
                            float(d["high"]),
                            float(d["low"]),
                            float(d["close"]),
                            float(d["volume"]),
                            float(d.get("turnover", 0)),
                        )
                        self.logger.debug(f"New candle: {c.close:.2f} @ {c.ts}")
                        self.on_candle_update(c)
            except Exception as e:
                self.logger.error(f"WebSocket handler error: {e}", exc_info=True)

        try:
            self.logger.info(f"Subscribing to kline stream for {self.symbol}")
            ws.kline_stream(
                interval=self.params.interval, symbol=self.symbol, callback=ws_handler
            )

            self.logger.info("Trading bot is running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("\nShutting down...")
            try:
                ws.exit()
                self.logger.info("WebSocket closed")
            except Exception as e:
                self.logger.warning(f"Error closing WebSocket: {e}")
        except Exception as e:
            self.logger.error(f"Error in run loop: {e}", exc_info=True)
            try:
                ws.exit()
            except Exception:
                pass
            raise


@click.group()
def cli():
    """Flying Trading CLI - Trading bot and data management tools."""
    pass


@cli.command()
@click.option("--symbol", default="SOLUSDT", help="Trading symbol (e.g., SOLUSDT)")
@click.option(
    "--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)"
)
@click.option("--log-file", default=None, help="Log file name (optional)")
def run(symbol: str, log_level: str, log_file: str | None):
    """Run the trading bot."""
    # Настройка логирования
    setup_logging(log_level=log_level, log_file=log_file)
    logger = get_logger(__name__)

    logger.info("=" * 50)
    logger.info("Flying Trading Bot Starting")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Testnet: {config.bybit_testnet}")
    logger.info("=" * 50)

    try:
        exchange = BybitAdapter(
            api_key=config.bybit_api_key,
            api_secret=config.bybit_api_secret,
            testnet=config.bybit_testnet,
        )
        logger.info("Bybit adapter initialized")

        notifier = TelegramAdapter(
            token=config.telegram_bot_token, chat_id=config.telegram_chat_id
        )
        logger.info("Telegram adapter initialized")

        bot = TradingBot(exchange, notifier, symbol)
        bot.run()
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        raise


@cli.command()
@click.argument("symbol")
@click.argument("start_date")
@click.argument("end_date", required=False)
@click.option(
    "--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)"
)
def upload(symbol: str, start_date: str, end_date: str | None, log_level: str):
    """
    Upload OHLCV data from Bybit to StarRocks.

    SYMBOL: Trading pair symbol (e.g., BTCUSDT)

    START_DATE: Start date in YYYY-MM-DD format

    END_DATE: Optional end date in YYYY-MM-DD format (inclusive)

    Examples:

        flt upload BTCUSDT 2024-12-01

        flt upload BTCUSDT 2024-11-01 2024-11-30
    """
    from flying_trading.starrocks_loader import (
        get_starrocks_connection,
        upload_date_range,
        upload_to_starrocks,
    )

    setup_logging(log_level=log_level)

    symbol = symbol.upper()

    # Validate date format
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        click.echo(f"Error: Invalid date format. Expected YYYY-MM-DD: {e}", err=True)
        raise SystemExit(1) from e

    try:
        connection = get_starrocks_connection()
        click.echo("✓ Connected to StarRocks")

        if end_date:
            click.echo(f"Uploading {symbol} data from {start_date} to {end_date}...")
            rows = upload_date_range(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                connection=connection,
            )
        else:
            click.echo(f"Uploading {symbol} data for {start_date}...")
            rows = upload_to_starrocks(
                symbol=symbol,
                date=start_date,
                connection=connection,
            )

        click.echo(f"✓ Successfully uploaded {rows} rows")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise SystemExit(1) from e


@cli.command()
@click.argument("symbol")
@click.option("--days", default=7, help="Number of days to backfill")
@click.option(
    "--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)"
)
def backfill(symbol: str, days: int, log_level: str):
    """
    Backfill OHLCV data for the last N days.

    SYMBOL: Trading pair symbol (e.g., BTCUSDT)

    Example:

        flt backfill BTCUSDT --days 30
    """
    from flying_trading.starrocks_loader import (
        get_starrocks_connection,
        upload_date_range,
    )

    setup_logging(log_level=log_level)

    symbol = symbol.upper()
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        connection = get_starrocks_connection()
        click.echo("✓ Connected to StarRocks")
        click.echo(f"Backfilling {symbol} data from {start_date} to {end_date}...")

        rows = upload_date_range(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            connection=connection,
        )

        click.echo(f"✓ Successfully uploaded {rows} rows")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise SystemExit(1) from e


if __name__ == "__main__":
    cli()
