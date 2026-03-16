"""Estrategia ensemble que combina SMA Crossover + LSTM."""

from dataclasses import dataclass
from typing import Optional

from strategies.lstm_predictor import LstmPrediction, LstmPredictor
from strategies.sma_crossover import Signal, SmaCrossoverStrategy, StrategyResult
from utils.logger import setup_logger

logger = setup_logger("strategies.ensemble")


@dataclass(frozen=True)
class EnsembleResult:
    """Resultado inmutable de la decisión ensemble."""

    final_signal: Signal
    sma_result: StrategyResult
    lstm_prediction: Optional[LstmPrediction]
    reason: str


class EnsembleStrategy:
    """Combina las señales de SMA y LSTM para una decisión más robusta.

    Reglas:
    - Si SMA y LSTM coinciden (con confianza >= threshold) -> ejecutar señal
    - Si no coinciden -> HOLD (conservador)
    - Si LSTM no está listo -> usar solo SMA como fallback
    """

    def __init__(
        self,
        sma_strategy: SmaCrossoverStrategy,
        lstm_predictor: LstmPredictor,
        confidence_threshold: float = 0.70,
    ) -> None:
        self._sma = sma_strategy
        self._lstm = lstm_predictor
        self._threshold = confidence_threshold

    def analyze(
        self, prices: list[float], volumes: list[float]
    ) -> EnsembleResult:
        """Analiza precios y volúmenes con ambas estrategias.

        Args:
            prices: Precios de cierre históricos.
            volumes: Volúmenes históricos.

        Returns:
            EnsembleResult con la decisión final.
        """
        sma_result = self._sma.analyze(prices)

        if not self._lstm.is_ready():
            logger.debug("LSTM no disponible, usando solo SMA")
            return EnsembleResult(
                final_signal=sma_result.signal,
                sma_result=sma_result,
                lstm_prediction=None,
                reason=f"Solo SMA (LSTM no listo): {sma_result.reason}",
            )

        lstm_pred = self._lstm.predict(prices, volumes)

        if sma_result.signal == Signal.HOLD:
            return EnsembleResult(
                final_signal=Signal.HOLD,
                sma_result=sma_result,
                lstm_prediction=lstm_pred,
                reason="SMA=HOLD -> HOLD",
            )

        if lstm_pred.confidence < self._threshold:
            reason = (
                f"LSTM baja confianza ({lstm_pred.confidence:.1%} < {self._threshold:.0%}): "
                f"SMA={sma_result.signal.value}, LSTM={lstm_pred.direction.value} -> HOLD"
            )
            logger.info(reason)
            return EnsembleResult(
                final_signal=Signal.HOLD,
                sma_result=sma_result,
                lstm_prediction=lstm_pred,
                reason=reason,
            )

        if sma_result.signal == lstm_pred.direction:
            reason = (
                f"ACUERDO: SMA={sma_result.signal.value} + "
                f"LSTM={lstm_pred.direction.value} ({lstm_pred.confidence:.1%}) "
                f"-> {sma_result.signal.value}"
            )
            logger.info(reason)
            return EnsembleResult(
                final_signal=sma_result.signal,
                sma_result=sma_result,
                lstm_prediction=lstm_pred,
                reason=reason,
            )

        reason = (
            f"DESACUERDO: SMA={sma_result.signal.value} vs "
            f"LSTM={lstm_pred.direction.value} ({lstm_pred.confidence:.1%}) -> HOLD"
        )
        logger.info(reason)
        return EnsembleResult(
            final_signal=Signal.HOLD,
            sma_result=sma_result,
            lstm_prediction=lstm_pred,
            reason=reason,
        )
