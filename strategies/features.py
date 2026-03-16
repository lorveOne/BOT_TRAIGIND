"""Feature engineering para el modelo LSTM."""

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils.logger import setup_logger

logger = setup_logger("strategies.features")


@dataclass
class FeatureSet:
    """Conjunto de datos preparado para entrenamiento LSTM."""

    X: np.ndarray  # (samples, sequence_length, n_features)
    y: np.ndarray  # (samples,) - labels binarios
    scaler: MinMaxScaler


def calculate_rsi(prices: list[float], period: int = 14) -> list[float]:
    """Calcula el RSI (Relative Strength Index) usando suavizado de Wilder.

    Retorna una lista del mismo largo que prices, con NaN en las primeras posiciones.
    """
    if len(prices) < period + 1:
        return [float("nan")] * len(prices)

    rsi_values = [float("nan")] * period
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

    gains = [max(d, 0) for d in deltas[:period]]
    losses = [abs(min(d, 0)) for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        rsi_values.append(100.0)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(100.0 - (100.0 / (1.0 + rs)))

    for i in range(period, len(deltas)):
        delta = deltas[i]
        gain = max(delta, 0)
        loss = abs(min(delta, 0))
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100.0 - (100.0 / (1.0 + rs)))

    return rsi_values


def calculate_price_change_pct(prices: list[float]) -> list[float]:
    """Calcula el porcentaje de cambio entre velas consecutivas."""
    result = [0.0]
    for i in range(1, len(prices)):
        if prices[i - 1] != 0:
            result.append((prices[i] - prices[i - 1]) / prices[i - 1])
        else:
            result.append(0.0)
    return result


def build_features(
    closes: list[float], volumes: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Construye la matriz de features y labels binarios.

    Features: close, volume, RSI(14), price_change_pct
    Labels: 1 si el siguiente close > close actual, 0 si no
    """
    rsi = calculate_rsi(closes, period=14)
    pct_change = calculate_price_change_pct(closes)

    features = []
    labels = []
    start_idx = 15  # RSI necesita 15 valores iniciales

    for i in range(start_idx, len(closes) - 1):
        if np.isnan(rsi[i]):
            continue
        features.append([closes[i], volumes[i], rsi[i], pct_change[i]])
        labels.append(1.0 if closes[i + 1] > closes[i] else 0.0)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32)


def create_sequences(
    features: np.ndarray, labels: np.ndarray, sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Crea ventanas deslizantes para entrada LSTM.

    Retorna:
        X: (n_samples, sequence_length, n_features)
        y: (n_samples,)
    """
    if len(features) <= sequence_length:
        raise ValueError(
            f"Se necesitan al menos {sequence_length + 1} muestras, "
            f"se recibieron {len(features)}"
        )

    X, y = [], []
    for i in range(sequence_length, len(features)):
        X.append(features[i - sequence_length : i])
        y.append(labels[i - 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_training_data(
    closes: list[float], volumes: list[float], sequence_length: int
) -> FeatureSet:
    """Pipeline completo: features -> normalización -> secuencias.

    Args:
        closes: Precios de cierre históricos.
        volumes: Volúmenes históricos.
        sequence_length: Largo de la ventana para LSTM.

    Returns:
        FeatureSet con datos listos para entrenar.
    """
    features, labels = build_features(closes, volumes)

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    X, y = create_sequences(features_scaled, labels, sequence_length)

    logger.info(
        "Datos preparados: %d muestras, ventana=%d, features=%d",
        X.shape[0],
        sequence_length,
        X.shape[2],
    )
    return FeatureSet(X=X, y=y, scaler=scaler)
