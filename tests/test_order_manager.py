"""Tests para el OrderManager."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock

from config.settings import TradingConfig
from models.trade import TradeRepository
from services.order_manager import OrderManager
from strategies.sma_crossover import Signal


class TestOrderManager(unittest.TestCase):
    """Tests para la gestión de órdenes."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.trade_repo = TradeRepository(self.db_path)

        self.mock_client = MagicMock()
        self.mock_client.get_account_balance.return_value = 10000.0
        self.mock_client.get_symbol_info.return_value = {
            "filters": [
                {"filterType": "LOT_SIZE", "stepSize": "0.00001"}
            ]
        }

        self.config = TradingConfig(
            symbol="BTCUSDT",
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            position_size_pct=0.10,
            dry_run=True,
        )

        self.manager = OrderManager(
            binance_client=self.mock_client,
            trade_repo=self.trade_repo,
            config=self.config,
        )

    def tearDown(self):
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_initial_state_no_position(self):
        self.assertFalse(self.manager.has_position)
        self.assertIsNone(self.manager.position)

    def test_buy_signal_opens_position(self):
        trade = self.manager.process_signal(Signal.BUY, 50000.0)
        self.assertIsNotNone(trade)
        self.assertTrue(self.manager.has_position)
        self.assertEqual(trade.side, "BUY")
        self.assertEqual(trade.price, 50000.0)

    def test_buy_signal_with_existing_position_ignored(self):
        self.manager.process_signal(Signal.BUY, 50000.0)
        trade = self.manager.process_signal(Signal.BUY, 51000.0)
        self.assertIsNone(trade)

    def test_sell_signal_closes_position(self):
        self.manager.process_signal(Signal.BUY, 50000.0)
        trade = self.manager.process_signal(Signal.SELL, 52000.0)
        self.assertIsNotNone(trade)
        self.assertFalse(self.manager.has_position)
        self.assertEqual(trade.side, "SELL")
        self.assertIsNotNone(trade.pnl)

    def test_sell_signal_without_position_sells_existing_balance(self):
        trade = self.manager.process_signal(Signal.SELL, 50000.0)
        self.assertIsNotNone(trade)
        self.assertEqual(trade.side, "SELL")

    def test_sell_signal_without_position_no_balance_ignored(self):
        self.mock_client.get_account_balance.return_value = 0.0
        trade = self.manager.process_signal(Signal.SELL, 50000.0)
        self.assertIsNone(trade)

    def test_hold_signal_no_action(self):
        trade = self.manager.process_signal(Signal.HOLD, 50000.0)
        self.assertIsNone(trade)

    def test_stop_loss_triggered(self):
        self.manager.process_signal(Signal.BUY, 50000.0)
        # SL = 50000 * 0.98 = 49000
        trade = self.manager.check_stop_loss_take_profit(48500.0)
        self.assertIsNotNone(trade)
        self.assertFalse(self.manager.has_position)
        self.assertLess(trade.pnl, 0)

    def test_take_profit_triggered(self):
        self.manager.process_signal(Signal.BUY, 50000.0)
        # TP = 50000 * 1.04 = 52000
        trade = self.manager.check_stop_loss_take_profit(52500.0)
        self.assertIsNotNone(trade)
        self.assertFalse(self.manager.has_position)
        self.assertGreater(trade.pnl, 0)

    def test_price_within_range_no_action(self):
        self.manager.process_signal(Signal.BUY, 50000.0)
        trade = self.manager.check_stop_loss_take_profit(50500.0)
        self.assertIsNone(trade)
        self.assertTrue(self.manager.has_position)

    def test_sl_tp_without_position_no_action(self):
        trade = self.manager.check_stop_loss_take_profit(50000.0)
        self.assertIsNone(trade)

    def test_position_quantity_calculated(self):
        self.manager.process_signal(Signal.BUY, 50000.0)
        position = self.manager.position
        # 10000 * 0.10 / 50000 = 0.02
        self.assertAlmostEqual(position.quantity, 0.02, places=5)

    def test_position_stop_loss_level(self):
        self.manager.process_signal(Signal.BUY, 50000.0)
        position = self.manager.position
        self.assertAlmostEqual(position.stop_loss, 49000.0)

    def test_position_take_profit_level(self):
        self.manager.process_signal(Signal.BUY, 50000.0)
        position = self.manager.position
        self.assertAlmostEqual(position.take_profit, 52000.0)

    def test_trades_persisted_in_db(self):
        self.manager.process_signal(Signal.BUY, 50000.0)
        self.manager.process_signal(Signal.SELL, 51000.0)
        trades = self.trade_repo.find_all()
        self.assertEqual(len(trades), 2)

    def test_pnl_calculated_correctly(self):
        self.manager.process_signal(Signal.BUY, 50000.0)
        trade = self.manager.process_signal(Signal.SELL, 52000.0)
        # PnL = (52000 - 50000) * 0.02 = 40.0
        self.assertAlmostEqual(trade.pnl, 40.0, places=2)

    def test_negative_pnl(self):
        self.manager.process_signal(Signal.BUY, 50000.0)
        trade = self.manager.process_signal(Signal.SELL, 48000.0)
        # PnL = (48000 - 50000) * 0.02 = -40.0
        self.assertAlmostEqual(trade.pnl, -40.0, places=2)

    def test_dry_run_does_not_call_api(self):
        self.manager.process_signal(Signal.BUY, 50000.0)
        self.mock_client.place_market_order.assert_not_called()

    def test_insufficient_balance(self):
        self.mock_client.get_account_balance.return_value = 0.0
        trade = self.manager.process_signal(Signal.BUY, 50000.0)
        self.assertIsNone(trade)
        self.assertFalse(self.manager.has_position)


if __name__ == "__main__":
    unittest.main()
