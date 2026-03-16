"""Estrategia de cruce de medias móviles simples (SMA)."""

from dataclasses import dataclass
from enum import Enum

from utils.logger import setup_logger

logger = setup_logger("strategies.sma_crossover")


class Signal(Enum):
    """Señales de trading generadas por la estrategia."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class StrategyResult:
    """Resultado inmutable del análisis de la estrategia."""

    signal: Signal
    sma_short: float
    sma_long: float
    current_price: float
    reason: str


def calculate_sma(prices: list[float], period: int) -> float:
    """Calcula la media móvil simple para un período dado."""
    if len(prices) < period:
        raise ValueError(
            f"Se necesitan al menos {period} precios, se recibieron {len(prices)}"
        )
    return sum(prices[-period:]) / period


class SmaCrossoverStrategy:
    """Estrategia de cruce de SMA corta sobre SMA larga.

    - BUY: SMA corta cruza por encima de SMA larga (golden cross)
    - SELL: SMA corta cruza por debajo de SMA larga (death cross)
    - HOLD: No hay cruce
    """

    def __init__(self, short_period: int = 20, long_period: int = 50) -> None:
        if short_period >= long_period:
            raise ValueError("short_period debe ser menor que long_period")
        self._short_period = short_period
        self._long_period = long_period
        self._previous_short_above: bool | None = None

    @property
    def short_period(self) -> int:
        return self._short_period

    @property
    def long_period(self) -> int:
        return self._long_period

    def analyze(self, prices: list[float]) -> StrategyResult:
        """Analiza los precios y genera una señal de trading.

        Args:
            prices: Lista de precios de cierre (más antiguo primero).

        Returns:
            StrategyResult con la señal y datos del análisis.
        """
        if len(prices) < self._long_period:
            return StrategyResult(
                signal=Signal.HOLD,
                sma_short=0.0,
                sma_long=0.0,
                current_price=prices[-1] if prices else 0.0,
                reason=f"Datos insuficientes ({len(prices)}/{self._long_period})",
            )

        sma_short = calculate_sma(prices, self._short_period)
        sma_long = calculate_sma(prices, self._long_period)
        current_price = prices[-1]
        short_above = sma_short > sma_long

        signal = Signal.HOLD
        reason = f"SMA{self._short_period}={sma_short:.2f}, SMA{self._long_period}={sma_long:.2f}"

        if self._previous_short_above is not None:
            if short_above and not self._previous_short_above:
                signal = Signal.BUY
                reason = f"Golden cross: SMA{self._short_period} ({sma_short:.2f}) cruzó por encima de SMA{self._long_period} ({sma_long:.2f})"
                logger.info("SENAL BUY: %s", reason)
            elif not short_above and self._previous_short_above:
                signal = Signal.SELL
                reason = f"Death cross: SMA{self._short_period} ({sma_short:.2f}) cruzó por debajo de SMA{self._long_period} ({sma_long:.2f})"
                logger.info("SENAL SELL: %s", reason)

        self._previous_short_above = short_above

        return StrategyResult(
            signal=signal,
            sma_short=sma_short,
            sma_long=sma_long,
            current_price=current_price,
            reason=reason,
        )

    def reset(self) -> None:
        """Reinicia el estado de la estrategia."""
        self._previous_short_above = None
