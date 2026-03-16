"""Tests para la estrategia SMA Crossover."""

import unittest

from strategies.sma_crossover import (
    Signal,
    SmaCrossoverStrategy,
    calculate_sma,
)


class TestCalculateSma(unittest.TestCase):
    """Tests para la función calculate_sma."""

    def test_sma_basic(self):
        prices = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = calculate_sma(prices, 5)
        self.assertAlmostEqual(result, 30.0)

    def test_sma_uses_last_n_prices(self):
        prices = [100.0, 1.0, 2.0, 3.0]
        result = calculate_sma(prices, 3)
        self.assertAlmostEqual(result, 2.0)

    def test_sma_insufficient_data_raises(self):
        prices = [10.0, 20.0]
        with self.assertRaises(ValueError):
            calculate_sma(prices, 5)

    def test_sma_single_value(self):
        prices = [42.0]
        result = calculate_sma(prices, 1)
        self.assertAlmostEqual(result, 42.0)


class TestSmaCrossoverStrategy(unittest.TestCase):
    """Tests para la estrategia de cruce de medias."""

    def setUp(self):
        self.strategy = SmaCrossoverStrategy(short_period=3, long_period=5)

    def test_invalid_periods_raises(self):
        with self.assertRaises(ValueError):
            SmaCrossoverStrategy(short_period=10, long_period=5)

    def test_equal_periods_raises(self):
        with self.assertRaises(ValueError):
            SmaCrossoverStrategy(short_period=5, long_period=5)

    def test_insufficient_data_returns_hold(self):
        prices = [1.0, 2.0, 3.0]
        result = self.strategy.analyze(prices)
        self.assertEqual(result.signal, Signal.HOLD)
        self.assertIn("Datos insuficientes", result.reason)

    def test_first_analysis_returns_hold(self):
        """La primera llamada solo establece el estado, no genera señal."""
        prices = [10.0, 11.0, 12.0, 13.0, 14.0]
        result = self.strategy.analyze(prices)
        self.assertEqual(result.signal, Signal.HOLD)

    def test_golden_cross_generates_buy(self):
        """SMA corta cruza por encima de SMA larga -> BUY."""
        # Estado inicial: SMA corta < SMA larga
        prices_below = [50.0, 40.0, 30.0, 20.0, 10.0]
        self.strategy.analyze(prices_below)

        # Ahora SMA corta > SMA larga (golden cross)
        prices_above = [10.0, 20.0, 30.0, 50.0, 60.0]
        result = self.strategy.analyze(prices_above)
        self.assertEqual(result.signal, Signal.BUY)

    def test_death_cross_generates_sell(self):
        """SMA corta cruza por debajo de SMA larga -> SELL."""
        # Estado inicial: SMA corta > SMA larga
        prices_above = [10.0, 20.0, 30.0, 50.0, 60.0]
        self.strategy.analyze(prices_above)

        # Ahora SMA corta < SMA larga (death cross)
        prices_below = [60.0, 50.0, 30.0, 20.0, 10.0]
        result = self.strategy.analyze(prices_below)
        self.assertEqual(result.signal, Signal.SELL)

    def test_no_cross_returns_hold(self):
        """Sin cruce mantiene HOLD."""
        # Ambas llamadas con SMA corta > SMA larga
        prices1 = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.strategy.analyze(prices1)

        prices2 = [15.0, 25.0, 35.0, 45.0, 55.0]
        result = self.strategy.analyze(prices2)
        self.assertEqual(result.signal, Signal.HOLD)

    def test_reset_clears_state(self):
        """Reset limpia el estado previo."""
        prices = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.strategy.analyze(prices)
        self.strategy.reset()
        result = self.strategy.analyze(prices)
        self.assertEqual(result.signal, Signal.HOLD)

    def test_result_contains_sma_values(self):
        prices = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = self.strategy.analyze(prices)
        self.assertGreater(result.sma_short, 0)
        self.assertGreater(result.sma_long, 0)
        self.assertEqual(result.current_price, 50.0)


if __name__ == "__main__":
    unittest.main()
