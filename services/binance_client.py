"""Wrapper para la API de Binance Testnet."""

from binance.client import Client
from binance.exceptions import BinanceAPIException

from config.settings import BinanceConfig
from utils.logger import setup_logger

logger = setup_logger("services.binance_client")


class BinanceClient:
    """Encapsula la conexión y operaciones con Binance Testnet."""

    def __init__(self, config: BinanceConfig) -> None:
        self._config = config
        self._client = self._create_client()

    def _create_client(self) -> Client:
        if not self._config.api_key or not self._config.api_secret:
            raise ValueError(
                "BINANCE_API_KEY y BINANCE_API_SECRET deben estar configurados en .env"
            )
        client = Client(
            api_key=self._config.api_key,
            api_secret=self._config.api_secret,
            testnet=self._config.testnet,
        )
        logger.info("Conectado a Binance Testnet")
        return client

    @property
    def client(self) -> Client:
        return self._client

    def get_account_balance(self, asset: str = "USDT") -> float:
        """Obtiene el balance disponible de un activo."""
        try:
            account = self._client.get_account()
            for balance in account["balances"]:
                if balance["asset"] == asset:
                    return float(balance["free"])
            return 0.0
        except BinanceAPIException as e:
            logger.error("Error al obtener balance: %s", e)
            raise

    def get_all_balances(self) -> dict[str, float]:
        """Retorna todos los balances con monto > 0."""
        try:
            account = self._client.get_account()
            return {
                b["asset"]: float(b["free"])
                for b in account["balances"]
                if float(b["free"]) > 0 or float(b["locked"]) > 0
            }
        except BinanceAPIException as e:
            logger.error("Error al obtener balances: %s", e)
            raise

    def get_symbol_price(self, symbol: str) -> float:
        """Obtiene el precio actual de un par."""
        try:
            ticker = self._client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except BinanceAPIException as e:
            logger.error("Error al obtener precio de %s: %s", symbol, e)
            raise

    def get_klines(
        self, symbol: str, interval: str, limit: int = 100
    ) -> list[list]:
        """Obtiene velas históricas."""
        try:
            return self._client.get_klines(
                symbol=symbol, interval=interval, limit=limit
            )
        except BinanceAPIException as e:
            logger.error("Error al obtener klines: %s", e)
            raise

    def place_market_order(
        self, symbol: str, side: str, quantity: float
    ) -> dict:
        """Ejecuta una orden de mercado."""
        try:
            order = self._client.create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity,
            )
            logger.info(
                "Orden ejecutada: %s %s %s @ mercado", side, quantity, symbol
            )
            return order
        except BinanceAPIException as e:
            logger.error("Error al ejecutar orden: %s", e)
            raise

    def get_symbol_info(self, symbol: str) -> dict:
        """Obtiene información del par (filtros, precisión, etc.)."""
        try:
            info = self._client.get_symbol_info(symbol)
            if info is None:
                raise ValueError(f"Símbolo no encontrado: {symbol}")
            return info
        except BinanceAPIException as e:
            logger.error("Error al obtener info de %s: %s", symbol, e)
            raise
