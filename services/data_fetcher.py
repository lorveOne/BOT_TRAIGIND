"""Obtención de datos de mercado en tiempo real y velas históricas."""

import json
import threading
from typing import Callable, Optional

from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException

from config.settings import BinanceConfig, TradingConfig
from services.binance_client import BinanceClient
from utils.logger import setup_logger

logger = setup_logger("services.data_fetcher")


class DataFetcher:
    """Gestiona la obtención de datos de mercado."""

    def __init__(
        self, binance_client: BinanceClient, trading_config: TradingConfig
    ) -> None:
        self._client = binance_client
        self._config = trading_config
        self._current_price: float = 0.0
        self._price_lock = threading.Lock()
        self._twm: Optional[ThreadedWebsocketManager] = None
        self._price_callbacks: list[Callable[[float], None]] = []

    @property
    def current_price(self) -> float:
        with self._price_lock:
            return self._current_price

    def on_price_update(self, callback: Callable[[float], None]) -> None:
        """Registra un callback para actualizaciones de precio."""
        self._price_callbacks.append(callback)

    def get_closing_prices(self, limit: int = 100) -> list[float]:
        """Obtiene los precios de cierre de las últimas velas."""
        klines = self._client.get_klines(
            symbol=self._config.symbol,
            interval=self._config.interval,
            limit=limit,
        )
        return [float(k[4]) for k in klines]  # index 4 = close price

    def start_price_stream(self) -> None:
        """Inicia el WebSocket para recibir precios en tiempo real."""
        try:
            self._twm = ThreadedWebsocketManager(
                api_key=self._client.client.API_KEY,
                api_secret=self._client.client.API_SECRET,
                testnet=True,
            )
            self._twm.start()

            symbol_lower = self._config.symbol.lower()
            self._twm.start_symbol_ticker_socket(
                callback=self._handle_price_message,
                symbol=symbol_lower,
            )
            logger.info(
                "WebSocket iniciado para %s", self._config.symbol
            )
        except Exception as e:
            logger.error("Error al iniciar WebSocket: %s", e)
            raise

    def _handle_price_message(self, msg: dict) -> None:
        """Procesa mensajes del WebSocket de precio."""
        if msg.get("e") == "error":
            logger.error("Error en WebSocket: %s", msg)
            return

        if "c" in msg:  # 'c' = current price en el ticker
            price = float(msg["c"])
            with self._price_lock:
                self._current_price = price
            for callback in self._price_callbacks:
                try:
                    callback(price)
                except Exception as e:
                    logger.error("Error en callback de precio: %s", e)

    def stop(self) -> None:
        """Detiene el WebSocket."""
        if self._twm:
            try:
                self._twm.stop()
                logger.info("WebSocket detenido")
            except Exception as e:
                logger.warning("Error al detener WebSocket: %s", e)
