"""Gestión de órdenes con stop-loss, take-profit y modo dry-run."""

import math
from dataclasses import dataclass
from typing import Optional

from config.settings import TradingConfig
from models.trade import Trade, TradeRepository, now_utc_iso
from services.binance_client import BinanceClient
from strategies.sma_crossover import Signal
from utils.logger import setup_logger

logger = setup_logger("services.order_manager")


@dataclass(frozen=True)
class Position:
    """Posición abierta inmutable."""

    symbol: str
    side: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float


class OrderManager:
    """Gestiona la ejecución de órdenes y el seguimiento de posiciones."""

    def __init__(
        self,
        binance_client: BinanceClient,
        trade_repo: TradeRepository,
        config: TradingConfig,
    ) -> None:
        self._client = binance_client
        self._repo = trade_repo
        self._config = config
        self._position: Optional[Position] = None

    @property
    def position(self) -> Optional[Position]:
        return self._position

    @property
    def has_position(self) -> bool:
        return self._position is not None

    def process_signal(self, signal: Signal, current_price: float) -> Optional[Trade]:
        """Procesa una señal de la estrategia y ejecuta la orden correspondiente."""
        if signal == Signal.BUY and not self.has_position:
            return self._open_position(current_price)
        elif signal == Signal.SELL and self.has_position:
            return self._close_position(current_price, reason="Señal SELL")
        return None

    def check_stop_loss_take_profit(self, current_price: float) -> Optional[Trade]:
        """Verifica si se debe cerrar la posición por SL/TP."""
        if not self.has_position:
            return None

        position = self._position
        if current_price <= position.stop_loss:
            logger.warning(
                "STOP-LOSS activado: precio=%.2f, SL=%.2f",
                current_price,
                position.stop_loss,
            )
            return self._close_position(current_price, reason="Stop-Loss")

        if current_price >= position.take_profit:
            logger.info(
                "TAKE-PROFIT activado: precio=%.2f, TP=%.2f",
                current_price,
                position.take_profit,
            )
            return self._close_position(current_price, reason="Take-Profit")

        return None

    def _open_position(self, price: float) -> Optional[Trade]:
        """Abre una nueva posición de compra."""
        balance = self._client.get_account_balance("USDT")
        trade_amount = balance * self._config.position_size_pct
        quantity = self._calculate_quantity(trade_amount, price)

        if quantity <= 0:
            logger.warning("Balance insuficiente para abrir posición")
            return None

        stop_loss = price * (1 - self._config.stop_loss_pct)
        take_profit = price * (1 + self._config.take_profit_pct)

        if self._config.dry_run:
            logger.info(
                "[DRY-RUN] Compra: %.6f %s @ %.2f | SL=%.2f | TP=%.2f",
                quantity,
                self._config.symbol,
                price,
                stop_loss,
                take_profit,
            )
        else:
            try:
                self._client.place_market_order(
                    symbol=self._config.symbol,
                    side="BUY",
                    quantity=quantity,
                )
            except Exception as e:
                logger.error("Error al ejecutar orden de compra: %s", e)
                return None

        self._position = Position(
            symbol=self._config.symbol,
            side="BUY",
            entry_price=price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        trade = Trade(
            id=None,
            timestamp=now_utc_iso(),
            symbol=self._config.symbol,
            side="BUY",
            price=price,
            quantity=quantity,
            order_type="MARKET",
            status="FILLED",
        )
        saved_trade = self._repo.create(trade)
        logger.info("Posición abierta: %s", saved_trade)
        return saved_trade

    def _close_position(self, price: float, reason: str) -> Optional[Trade]:
        """Cierra la posición abierta."""
        if not self.has_position:
            return None

        position = self._position
        pnl = (price - position.entry_price) * position.quantity

        if self._config.dry_run:
            logger.info(
                "[DRY-RUN] Venta (%s): %.6f %s @ %.2f | PnL=%.2f",
                reason,
                position.quantity,
                position.symbol,
                price,
                pnl,
            )
        else:
            try:
                self._client.place_market_order(
                    symbol=position.symbol,
                    side="SELL",
                    quantity=position.quantity,
                )
            except Exception as e:
                logger.error("Error al ejecutar orden de venta: %s", e)
                return None

        trade = Trade(
            id=None,
            timestamp=now_utc_iso(),
            symbol=position.symbol,
            side="SELL",
            price=price,
            quantity=position.quantity,
            order_type="MARKET",
            status="FILLED",
            pnl=pnl,
        )
        saved_trade = self._repo.create(trade)
        self._position = None
        logger.info("Posición cerrada (%s): PnL=%.2f USDT", reason, pnl)
        return saved_trade

    def _calculate_quantity(self, usdt_amount: float, price: float) -> float:
        """Calcula la cantidad ajustada a la precisión del par."""
        raw_qty = usdt_amount / price
        try:
            info = self._client.get_symbol_info(self._config.symbol)
            for f in info.get("filters", []):
                if f["filterType"] == "LOT_SIZE":
                    step_size = float(f["stepSize"])
                    precision = int(round(-math.log10(step_size)))
                    return math.floor(raw_qty * 10**precision) / 10**precision
        except Exception as e:
            logger.warning("No se pudo obtener precisión, usando 6 decimales: %s", e)
        return math.floor(raw_qty * 1e6) / 1e6
