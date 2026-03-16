"""Descarga de datos históricos paginados desde Binance."""

import time

from config.settings import TradingConfig
from services.binance_client import BinanceClient
from utils.logger import setup_logger

logger = setup_logger("services.historical_data")


class HistoricalDataFetcher:
    """Obtiene grandes volúmenes de datos históricos de Binance."""

    def __init__(
        self, binance_client: BinanceClient, trading_config: TradingConfig
    ) -> None:
        self._client = binance_client
        self._config = trading_config

    def fetch_training_data(
        self, limit: int = 4320
    ) -> tuple[list[float], list[float]]:
        """Descarga velas históricas con paginación.

        Binance limita a 1000 velas por request, así que paginamos.

        Args:
            limit: Número total de velas a descargar (~4320 = 6 meses de 1h).

        Returns:
            Tupla (closes, volumes) con listas paralelas.
        """
        all_klines: list[list] = []
        remaining = limit
        end_time = None

        logger.info(
            "Descargando %d velas de %s (%s)...",
            limit,
            self._config.symbol,
            self._config.interval,
        )

        while remaining > 0:
            batch_size = min(remaining, 1000)
            kwargs = {
                "symbol": self._config.symbol,
                "interval": self._config.interval,
                "limit": batch_size,
            }
            if end_time is not None:
                kwargs["endTime"] = end_time

            klines = self._client.client.get_klines(**kwargs)
            if not klines:
                break

            all_klines = klines + all_klines
            end_time = klines[0][0] - 1  # timestamp anterior a la primera vela
            remaining -= len(klines)

            logger.info(
                "Descargadas %d/%d velas", len(all_klines), limit
            )
            time.sleep(0.5)  # respetar rate limits

        closes = [float(k[4]) for k in all_klines]
        volumes = [float(k[5]) for k in all_klines]

        logger.info(
            "Descarga completada: %d velas de %s",
            len(closes),
            self._config.symbol,
        )
        return closes, volumes
