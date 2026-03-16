"""Feature engineering mejorado para el modelo LSTM."""

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
    """Calcula el RSI usando suavizado de Wilder."""
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


def calculate_ema(prices: list[float], period: int) -> list[float]:
    """Calcula la media móvil exponencial."""
    if len(prices) < period:
        return [float("nan")] * len(prices)

    ema = [float("nan")] * (period - 1)
    sma = sum(prices[:period]) / period
    ema.append(sma)
    multiplier = 2.0 / (period + 1)

    for i in range(period, len(prices)):
        val = (prices[i] - ema[-1]) * multiplier + ema[-1]
        ema.append(val)

    return ema


def calculate_macd(
    prices: list[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[list[float], list[float], list[float]]:
    """Calcula MACD, Signal line y Histograma."""
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)

    macd_line = []
    for i in range(len(prices)):
        if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
            macd_line.append(float("nan"))
        else:
            macd_line.append(ema_fast[i] - ema_slow[i])

    # Signal line = EMA del MACD
    valid_macd = [v for v in macd_line if not np.isnan(v)]
    if len(valid_macd) < signal:
        return macd_line, [float("nan")] * len(prices), [float("nan")] * len(prices)

    signal_line = [float("nan")] * len(prices)
    histogram = [float("nan")] * len(prices)

    first_valid = next(i for i, v in enumerate(macd_line) if not np.isnan(v))
    valid_portion = macd_line[first_valid:]
    ema_signal = calculate_ema(valid_portion, signal)

    for i, val in enumerate(ema_signal):
        idx = first_valid + i
        signal_line[idx] = val
        if not np.isnan(val) and not np.isnan(macd_line[idx]):
            histogram[idx] = macd_line[idx] - val

    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    prices: list[float], period: int = 20, num_std: float = 2.0
) -> tuple[list[float], list[float], list[float]]:
    """Calcula las Bandas de Bollinger (upper, middle, lower)."""
    upper, middle, lower = [], [], []

    for i in range(len(prices)):
        if i < period - 1:
            upper.append(float("nan"))
            middle.append(float("nan"))
            lower.append(float("nan"))
        else:
            window = prices[i - period + 1 : i + 1]
            mean = sum(window) / period
            std = (sum((x - mean) ** 2 for x in window) / period) ** 0.5
            middle.append(mean)
            upper.append(mean + num_std * std)
            lower.append(mean - num_std * std)

    return upper, middle, lower


def calculate_bollinger_pct(
    prices: list[float], period: int = 20, num_std: float = 2.0
) -> list[float]:
    """Calcula %B de Bollinger (posición del precio dentro de las bandas)."""
    upper, middle, lower = calculate_bollinger_bands(prices, period, num_std)
    pct_b = []
    for i in range(len(prices)):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            pct_b.append(float("nan"))
        elif upper[i] == lower[i]:
            pct_b.append(0.5)
        else:
            pct_b.append((prices[i] - lower[i]) / (upper[i] - lower[i]))
    return pct_b


def calculate_relative_volume(volumes: list[float], period: int = 20) -> list[float]:
    """Calcula el volumen relativo (actual vs promedio)."""
    result = []
    for i in range(len(volumes)):
        if i < period:
            result.append(1.0)
        else:
            avg = sum(volumes[i - period : i]) / period
            result.append(volumes[i] / avg if avg > 0 else 1.0)
    return result


def build_features(
    closes: list[float], volumes: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Construye la matriz de features mejorada y labels binarios.

    Features (10):
        close, volume, RSI(14), price_change_pct,
        MACD, MACD_signal, MACD_histogram,
        Bollinger_%B, relative_volume, EMA_ratio(12/26)
    """
    rsi = calculate_rsi(closes, period=14)
    pct_change = calculate_price_change_pct(closes)
    macd_line, signal_line, histogram = calculate_macd(closes)
    boll_pct = calculate_bollinger_pct(closes, period=20)
    rel_volume = calculate_relative_volume(volumes, period=20)
    ema12 = calculate_ema(closes, 12)
    ema26 = calculate_ema(closes, 26)

    features = []
    labels = []
    start_idx = 30  # Esperar a que todos los indicadores tengan datos

    for i in range(start_idx, len(closes) - 1):
        if any(
            np.isnan(v)
            for v in [
                rsi[i], macd_line[i], signal_line[i],
                histogram[i], boll_pct[i], ema12[i], ema26[i],
            ]
        ):
            continue

        ema_ratio = ema12[i] / ema26[i] if ema26[i] != 0 else 1.0

        features.append([
            closes[i],
            volumes[i],
            rsi[i],
            pct_change[i],
            macd_line[i],
            signal_line[i],
            histogram[i],
            boll_pct[i],
            rel_volume[i],
            ema_ratio,
        ])
        labels.append(1.0 if closes[i + 1] > closes[i] else 0.0)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32)


def create_sequences(
    features: np.ndarray, labels: np.ndarray, sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Crea ventanas deslizantes para entrada LSTM."""
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
    """Pipeline completo: features -> normalización -> secuencias."""
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
