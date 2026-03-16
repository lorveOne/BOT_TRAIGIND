"""Pipeline de entrenamiento del modelo LSTM."""

from config.settings import LstmConfig
from services.historical_data import HistoricalDataFetcher
from strategies.lstm_predictor import LstmPredictor
from utils.logger import setup_logger

logger = setup_logger("services.model_trainer")


class ModelTrainer:
    """Orquesta el entrenamiento o carga del modelo LSTM."""

    def __init__(
        self,
        lstm_predictor: LstmPredictor,
        historical_fetcher: HistoricalDataFetcher,
        config: LstmConfig,
    ) -> None:
        self._predictor = lstm_predictor
        self._fetcher = historical_fetcher
        self._config = config

    def train_or_load(self) -> bool:
        """Carga un modelo existente o entrena uno nuevo.

        Returns:
            True si el modelo está listo, False en caso de error.
        """
        if not self._config.retrain_on_startup:
            if self._predictor.load_model(self._config.model_dir):
                logger.info("Modelo LSTM cargado desde disco")
                return True
            logger.info("No hay modelo guardado, entrenando nuevo...")

        return self._train_fresh()

    def _train_fresh(self) -> bool:
        """Descarga datos y entrena el modelo desde cero."""
        try:
            closes, volumes = self._fetcher.fetch_training_data(
                self._config.training_candles
            )

            min_required = self._config.sequence_length + 100
            if len(closes) < min_required:
                logger.error(
                    "Datos insuficientes: %d velas (mínimo %d)",
                    len(closes),
                    min_required,
                )
                return False

            logger.info("Entrenando modelo LSTM con %d velas...", len(closes))
            metrics = self._predictor.train(closes, volumes)

            self._predictor.save_model(self._config.model_dir)

            logger.info(
                "Modelo entrenado y guardado. Accuracy: %.2f%%",
                metrics["val_accuracy"] * 100,
            )
            return True

        except Exception as e:
            logger.error("Error en pipeline de entrenamiento: %s", e)
            return False
