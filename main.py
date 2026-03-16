"""Bot de trading para Binance Testnet con estrategia SMA Crossover."""

import os
import signal
import sys
import time
from datetime import datetime, timezone

from config.settings import APP_CONFIG, BINANCE_CONFIG, TRADING_CONFIG
from models.trade import TradeRepository
from services.binance_client import BinanceClient
from services.data_fetcher import DataFetcher
from services.order_manager import OrderManager
from strategies.sma_crossover import SmaCrossoverStrategy
from utils.logger import setup_logger

logger = setup_logger("main")


class TradingBot:
    """Orquestador principal del bot de trading."""

    def __init__(self) -> None:
        self._running = False
        self._trade_repo = TradeRepository(APP_CONFIG.db_path)

        self._binance_client = BinanceClient(BINANCE_CONFIG)
        self._data_fetcher = DataFetcher(self._binance_client, TRADING_CONFIG)
        self._strategy = SmaCrossoverStrategy(
            short_period=TRADING_CONFIG.sma_short_period,
            long_period=TRADING_CONFIG.sma_long_period,
        )
        self._order_manager = OrderManager(
            self._binance_client, self._trade_repo, TRADING_CONFIG
        )

    def start(self) -> None:
        """Inicia el bot de trading."""
        self._running = True
        logger.info("=" * 60)
        logger.info("Bot de Trading iniciado")
        logger.info("Par: %s | Intervalo: %s", TRADING_CONFIG.symbol, TRADING_CONFIG.interval)
        logger.info("SMA: %d/%d", TRADING_CONFIG.sma_short_period, TRADING_CONFIG.sma_long_period)
        logger.info("SL: %.1f%% | TP: %.1f%%", TRADING_CONFIG.stop_loss_pct * 100, TRADING_CONFIG.take_profit_pct * 100)
        logger.info("Modo: %s", "DRY-RUN" if TRADING_CONFIG.dry_run else "LIVE")
        logger.info("=" * 60)

        self._data_fetcher.start_price_stream()

        # Esperar a recibir el primer precio
        logger.info("Esperando datos de precio...")
        for _ in range(30):
            if self._data_fetcher.current_price > 0:
                break
            time.sleep(1)

        if self._data_fetcher.current_price == 0:
            logger.warning("No se recibieron precios del WebSocket, usando API REST")

        try:
            while self._running:
                self._tick()
                self._print_dashboard()
                time.sleep(APP_CONFIG.dashboard_refresh_seconds)
        except KeyboardInterrupt:
            logger.info("Interrupción por teclado")
        finally:
            self.stop()

    def stop(self) -> None:
        """Detiene el bot de forma ordenada."""
        self._running = False
        self._data_fetcher.stop()
        logger.info("Bot detenido")

    def _tick(self) -> None:
        """Ejecuta un ciclo de análisis y trading."""
        try:
            prices = self._data_fetcher.get_closing_prices(
                limit=TRADING_CONFIG.sma_long_period + 10
            )
            if not prices:
                logger.warning("No se pudieron obtener precios")
                return

            current_price = self._data_fetcher.current_price
            if current_price == 0:
                current_price = prices[-1]

            # Verificar stop-loss / take-profit
            self._order_manager.check_stop_loss_take_profit(current_price)

            # Analizar estrategia
            result = self._strategy.analyze(prices)
            self._order_manager.process_signal(result.signal, current_price)

        except Exception as e:
            logger.error("Error en ciclo de trading: %s", e)

    def _print_dashboard(self) -> None:
        """Muestra el dashboard en consola."""
        try:
            os.system("cls" if os.name == "nt" else "clear")
        except Exception:
            print("\n" * 3)

        price = self._data_fetcher.current_price
        position = self._order_manager.position
        total_pnl = self._trade_repo.total_pnl(TRADING_CONFIG.symbol)
        recent_trades = self._trade_repo.find_all(TRADING_CONFIG.symbol, limit=5)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        mode = "DRY-RUN" if TRADING_CONFIG.dry_run else "LIVE"

        print("=" * 60)
        print(f"  TRADING BOT - {TRADING_CONFIG.symbol} [{mode}]")
        print(f"  {now}")
        print("=" * 60)
        print(f"  Precio actual:  ${price:,.2f}")

        try:
            usdt_balance = self._binance_client.get_account_balance("USDT")
            print(f"  Balance USDT:   ${usdt_balance:,.2f}")
        except Exception:
            print("  Balance USDT:   N/A")

        print("-" * 60)

        if position:
            unrealized_pnl = (price - position.entry_price) * position.quantity
            pnl_pct = ((price / position.entry_price) - 1) * 100 if position.entry_price else 0
            pnl_indicator = "+" if unrealized_pnl >= 0 else ""
            print("  POSICION ABIERTA:")
            print(f"    Entrada:    ${position.entry_price:,.2f}")
            print(f"    Cantidad:   {position.quantity:.6f}")
            print(f"    Stop-Loss:  ${position.stop_loss:,.2f}")
            print(f"    Take-Profit:${position.take_profit:,.2f}")
            print(f"    PnL no realizado: {pnl_indicator}${unrealized_pnl:,.2f} ({pnl_indicator}{pnl_pct:.2f}%)")
        else:
            print("  Sin posiciones abiertas")

        print("-" * 60)
        pnl_indicator = "+" if total_pnl >= 0 else ""
        print(f"  PnL Total: {pnl_indicator}${total_pnl:,.2f}")
        print("-" * 60)

        if recent_trades:
            print("  ULTIMAS OPERACIONES:")
            for t in recent_trades:
                pnl_str = f" | PnL: ${t.pnl:,.2f}" if t.pnl is not None else ""
                print(f"    {t.timestamp[:19]} | {t.side:4s} | ${t.price:,.2f} | {t.quantity:.6f}{pnl_str}")
        else:
            print("  Sin operaciones registradas")

        print("=" * 60)
        print("  Ctrl+C para detener")
        print("=" * 60)


def main() -> None:
    """Punto de entrada del bot."""
    logger.info("Iniciando bot de trading...")

    if not BINANCE_CONFIG.api_key or not BINANCE_CONFIG.api_secret:
        logger.error(
            "Configura BINANCE_API_KEY y BINANCE_API_SECRET en el archivo .env"
        )
        sys.exit(1)

    bot = TradingBot()

    def handle_shutdown(signum, frame):
        logger.info("Señal de terminación recibida")
        bot.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    bot.start()


if __name__ == "__main__":
    main()
