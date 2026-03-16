"""Tests para el ensemble, features y LSTM predictor."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from strategies.features import (
    build_features,
    calculate_price_change_pct,
    calculate_rsi,
    create_sequences,
)
from strategies.ensemble import EnsembleStrategy, Signal
from strategies.sma_crossover import StrategyResult


# --- Tests de Features ---


class TestCalculateRsi(unittest.TestCase):
    def test_rsi_all_gains(self):
        prices = list(range(1, 20))  # subida constante
        rsi = calculate_rsi(prices, period=14)
        valid = [v for v in rsi if not np.isnan(v)]
        self.assertTrue(all(v > 90 for v in valid))

    def test_rsi_all_losses(self):
        prices = list(range(20, 1, -1))  # bajada constante
        rsi = calculate_rsi(prices, period=14)
        valid = [v for v in rsi if not np.isnan(v)]
        self.assertTrue(all(v < 10 for v in valid))

    def test_rsi_length_matches_input(self):
        prices = [float(x) for x in range(100)]
        rsi = calculate_rsi(prices, period=14)
        self.assertEqual(len(rsi), len(prices))

    def test_rsi_insufficient_data(self):
        prices = [10.0, 11.0, 12.0]
        rsi = calculate_rsi(prices, period=14)
        self.assertTrue(all(np.isnan(v) for v in rsi))

    def test_rsi_range_0_100(self):
        np.random.seed(42)
        prices = list(np.cumsum(np.random.randn(200)) + 100)
        rsi = calculate_rsi(prices)
        valid = [v for v in rsi if not np.isnan(v)]
        self.assertTrue(all(0 <= v <= 100 for v in valid))


class TestPriceChangePct(unittest.TestCase):
    def test_basic_change(self):
        prices = [100.0, 110.0, 99.0]
        result = calculate_price_change_pct(prices)
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 0.10)
        self.assertAlmostEqual(result[2], -0.1, places=2)

    def test_length_matches(self):
        prices = [1.0, 2.0, 3.0, 4.0]
        result = calculate_price_change_pct(prices)
        self.assertEqual(len(result), len(prices))


class TestBuildFeatures(unittest.TestCase):
    def test_output_shapes(self):
        closes = [float(x) for x in range(100)]
        volumes = [float(x * 10) for x in range(100)]
        features, labels = build_features(closes, volumes)
        self.assertEqual(features.shape[1], 10)  # 10 features
        self.assertEqual(len(features), len(labels))

    def test_labels_binary(self):
        np.random.seed(42)
        closes = list(np.cumsum(np.random.randn(200)) + 100)
        volumes = [100.0] * 200
        _, labels = build_features(closes, volumes)
        unique = set(labels.tolist())
        self.assertTrue(unique.issubset({0.0, 1.0}))


class TestCreateSequences(unittest.TestCase):
    def test_output_shape(self):
        features = np.random.rand(100, 4).astype(np.float32)
        labels = np.random.randint(0, 2, 100).astype(np.float32)
        X, y = create_sequences(features, labels, sequence_length=10)
        self.assertEqual(X.shape[1], 10)
        self.assertEqual(X.shape[2], 4)
        self.assertEqual(len(X), len(y))

    def test_insufficient_data_raises(self):
        features = np.random.rand(5, 4).astype(np.float32)
        labels = np.random.randint(0, 2, 5).astype(np.float32)
        with self.assertRaises(ValueError):
            create_sequences(features, labels, sequence_length=10)


# --- Tests del Ensemble ---


class TestEnsembleStrategy(unittest.TestCase):
    def setUp(self):
        self.mock_sma = MagicMock()
        self.mock_lstm = MagicMock()
        self.ensemble = EnsembleStrategy(
            self.mock_sma, self.mock_lstm, confidence_threshold=0.70
        )
        self.prices = [float(x) for x in range(100)]
        self.volumes = [100.0] * 100

    def _sma_result(self, signal):
        return StrategyResult(
            signal=signal,
            sma_short=50.0,
            sma_long=48.0,
            current_price=51.0,
            reason="test",
        )

    def _lstm_prediction(self, direction, confidence):
        pred = MagicMock()
        pred.direction = direction
        pred.confidence = confidence
        pred.predicted_probability = 0.8 if direction == Signal.BUY else 0.2
        pred.reason = "test"
        return pred

    def test_both_agree_buy(self):
        self.mock_sma.analyze.return_value = self._sma_result(Signal.BUY)
        self.mock_lstm.is_ready.return_value = True
        self.mock_lstm.predict.return_value = self._lstm_prediction(Signal.BUY, 0.85)

        result = self.ensemble.analyze(self.prices, self.volumes)
        self.assertEqual(result.final_signal, Signal.BUY)

    def test_both_agree_sell(self):
        self.mock_sma.analyze.return_value = self._sma_result(Signal.SELL)
        self.mock_lstm.is_ready.return_value = True
        self.mock_lstm.predict.return_value = self._lstm_prediction(Signal.SELL, 0.90)

        result = self.ensemble.analyze(self.prices, self.volumes)
        self.assertEqual(result.final_signal, Signal.SELL)

    def test_disagree_high_confidence_lstm_wins(self):
        self.mock_sma.analyze.return_value = self._sma_result(Signal.BUY)
        self.mock_lstm.is_ready.return_value = True
        self.mock_lstm.predict.return_value = self._lstm_prediction(Signal.SELL, 0.85)

        result = self.ensemble.analyze(self.prices, self.volumes)
        self.assertEqual(result.final_signal, Signal.SELL)

    def test_disagree_low_confidence_returns_hold(self):
        self.mock_sma.analyze.return_value = self._sma_result(Signal.BUY)
        self.mock_lstm.is_ready.return_value = True
        self.mock_lstm.predict.return_value = self._lstm_prediction(Signal.SELL, 0.30)

        result = self.ensemble.analyze(self.prices, self.volumes)
        self.assertEqual(result.final_signal, Signal.HOLD)

    def test_low_confidence_returns_hold(self):
        self.mock_sma.analyze.return_value = self._sma_result(Signal.BUY)
        self.mock_lstm.is_ready.return_value = True
        self.mock_lstm.predict.return_value = self._lstm_prediction(Signal.BUY, 0.50)

        result = self.ensemble.analyze(self.prices, self.volumes)
        self.assertEqual(result.final_signal, Signal.HOLD)

    def test_sma_hold_lstm_high_confidence_executes(self):
        self.mock_sma.analyze.return_value = self._sma_result(Signal.HOLD)
        self.mock_lstm.is_ready.return_value = True
        self.mock_lstm.predict.return_value = self._lstm_prediction(Signal.BUY, 0.95)

        result = self.ensemble.analyze(self.prices, self.volumes)
        self.assertEqual(result.final_signal, Signal.BUY)

    def test_sma_hold_lstm_low_confidence_holds(self):
        self.mock_sma.analyze.return_value = self._sma_result(Signal.HOLD)
        self.mock_lstm.is_ready.return_value = True
        self.mock_lstm.predict.return_value = self._lstm_prediction(Signal.BUY, 0.30)

        result = self.ensemble.analyze(self.prices, self.volumes)
        self.assertEqual(result.final_signal, Signal.HOLD)

    def test_lstm_not_ready_falls_back_to_sma(self):
        self.mock_sma.analyze.return_value = self._sma_result(Signal.BUY)
        self.mock_lstm.is_ready.return_value = False

        result = self.ensemble.analyze(self.prices, self.volumes)
        self.assertEqual(result.final_signal, Signal.BUY)
        self.assertIsNone(result.lstm_prediction)

    def test_result_contains_both(self):
        self.mock_sma.analyze.return_value = self._sma_result(Signal.BUY)
        self.mock_lstm.is_ready.return_value = True
        self.mock_lstm.predict.return_value = self._lstm_prediction(Signal.BUY, 0.85)

        result = self.ensemble.analyze(self.prices, self.volumes)
        self.assertIsNotNone(result.sma_result)
        self.assertIsNotNone(result.lstm_prediction)
        self.assertIsNotNone(result.reason)


if __name__ == "__main__":
    unittest.main()
