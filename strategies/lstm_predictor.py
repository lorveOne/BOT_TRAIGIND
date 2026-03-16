"""Predictor basado en red neuronal LSTM."""

import os
import pickle
from dataclasses import dataclass

import numpy as np

from config.settings import LstmConfig
from strategies.features import (
    build_features,
    prepare_training_data,
)
from strategies.sma_crossover import Signal
from utils.logger import setup_logger

logger = setup_logger("strategies.lstm_predictor")


@dataclass(frozen=True)
class LstmPrediction:
    """Resultado inmutable de la predicción LSTM."""

    direction: Signal
    confidence: float  # 0.0 a 1.0
    predicted_probability: float  # salida raw del sigmoid
    reason: str


class LstmPredictor:
    """Modelo LSTM para predicción de dirección de precio."""

    def __init__(self, config: LstmConfig) -> None:
        self._config = config
        self._model = None
        self._scaler = None

    def is_ready(self) -> bool:
        return self._model is not None and self._scaler is not None

    def build_model(self, input_shape: tuple) -> None:
        """Construye el modelo LSTM con Keras."""
        from tensorflow import keras

        model = keras.Sequential()

        model.add(
            keras.layers.LSTM(
                self._config.lstm_units,
                return_sequences=True,
                input_shape=input_shape,
            )
        )
        model.add(keras.layers.Dropout(0.2))

        model.add(
            keras.layers.LSTM(
                self._config.lstm_units,
                return_sequences=False,
            )
        )
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(1, activation="sigmoid"))

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        self._model = model
        logger.info("Modelo LSTM construido: input_shape=%s", input_shape)

    def train(
        self, closes: list[float], volumes: list[float]
    ) -> dict[str, float]:
        """Entrena el modelo con datos históricos.

        Returns:
            Métricas de entrenamiento (loss, accuracy, val_loss, val_accuracy).
        """
        from tensorflow import keras

        feature_set = prepare_training_data(
            closes, volumes, self._config.sequence_length
        )
        self._scaler = feature_set.scaler

        split = int(len(feature_set.X) * 0.8)
        X_train, X_val = feature_set.X[:split], feature_set.X[split:]
        y_train, y_val = feature_set.y[:split], feature_set.y[split:]

        input_shape = (X_train.shape[1], X_train.shape[2])
        self.build_model(input_shape)

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=0,
        )

        logger.info(
            "Entrenando LSTM: %d train, %d val, %d epochs",
            len(X_train),
            len(X_val),
            self._config.epochs,
        )

        history = self._model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self._config.epochs,
            batch_size=self._config.batch_size,
            callbacks=[early_stop],
            verbose=0,
        )

        final_epoch = len(history.history["loss"]) - 1
        metrics = {
            "loss": history.history["loss"][final_epoch],
            "accuracy": history.history["accuracy"][final_epoch],
            "val_loss": history.history["val_loss"][final_epoch],
            "val_accuracy": history.history["val_accuracy"][final_epoch],
            "epochs_trained": final_epoch + 1,
        }

        logger.info(
            "Entrenamiento completado: loss=%.4f, acc=%.4f, val_loss=%.4f, val_acc=%.4f (%d epochs)",
            metrics["loss"],
            metrics["accuracy"],
            metrics["val_loss"],
            metrics["val_accuracy"],
            metrics["epochs_trained"],
        )
        return metrics

    def predict(
        self, closes: list[float], volumes: list[float]
    ) -> LstmPrediction:
        """Predice la dirección del siguiente precio.

        Args:
            closes: Últimos precios de cierre (al menos sequence_length + 15).
            volumes: Últimos volúmenes correspondientes.

        Returns:
            LstmPrediction con dirección, confianza y razón.
        """
        if not self.is_ready():
            return LstmPrediction(
                direction=Signal.HOLD,
                confidence=0.0,
                predicted_probability=0.5,
                reason="Modelo LSTM no entrenado",
            )

        try:
            features, _ = build_features(closes, volumes)

            if len(features) < self._config.sequence_length:
                return LstmPrediction(
                    direction=Signal.HOLD,
                    confidence=0.0,
                    predicted_probability=0.5,
                    reason=f"Datos insuficientes ({len(features)}/{self._config.sequence_length})",
                )

            last_sequence = features[-self._config.sequence_length :]
            scaled = self._scaler.transform(last_sequence)
            X = np.expand_dims(scaled, axis=0)

            prob = float(self._model.predict(X, verbose=0)[0][0])
            direction = Signal.BUY if prob > 0.5 else Signal.SELL
            confidence = abs(prob - 0.5) * 2  # mapea 0.5->0.0, 1.0->1.0

            return LstmPrediction(
                direction=direction,
                confidence=confidence,
                predicted_probability=prob,
                reason=f"LSTM: prob={prob:.3f}, conf={confidence:.3f}",
            )
        except Exception as e:
            logger.error("Error en predicción LSTM: %s", e)
            return LstmPrediction(
                direction=Signal.HOLD,
                confidence=0.0,
                predicted_probability=0.5,
                reason=f"Error: {e}",
            )

    def save_model(self, path: str) -> None:
        """Guarda el modelo y el scaler en disco."""
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, "lstm_model.keras")
        scaler_path = os.path.join(path, "scaler.pkl")

        self._model.save(model_path)
        with open(scaler_path, "wb") as f:
            pickle.dump(self._scaler, f)

        logger.info("Modelo guardado en %s", path)

    def load_model(self, path: str) -> bool:
        """Carga el modelo y el scaler desde disco."""
        from tensorflow import keras

        model_path = os.path.join(path, "lstm_model.keras")
        scaler_path = os.path.join(path, "scaler.pkl")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logger.warning("No se encontró modelo guardado en %s", path)
            return False

        try:
            self._model = keras.models.load_model(model_path)
            with open(scaler_path, "rb") as f:
                self._scaler = pickle.load(f)
            logger.info("Modelo cargado desde %s", path)
            return True
        except Exception as e:
            logger.error("Error al cargar modelo: %s", e)
            return False
