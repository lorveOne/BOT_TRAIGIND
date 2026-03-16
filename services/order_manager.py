"""Gestión de órdenes con trailing stop-loss, break-even y tiempo máximo."""

import math
import time
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
    opened_at: float = 0.0  # timestamp de apertura


class OrderManager:
    """Gestiona la ejecución de órdenes y el seguimiento de posiciones."""

    SELL_COOLDOWN_SECONDS = 600  # Esperar 10 min entre ventas de balance existente
    MAX_BALANCE_SELLS_PER_SESSION = 3  # Máximo de ventas de balance por sesión

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
        self._last_sell_time: float = 0
        self._highest_price: float = 0.0  # Máximo alcanzado (para trailing)
        self._trailing_active: bool = False  # Break-even activado
        self._balance_sell_count: int = 0  # Contador de ventas de balance en sesión
        self._trade_timestamps: list = []  # Timestamps de operaciones (rate limiting)

    @property
    def position(self) -> Optional[Position]:
        return self._position

    @property
    def has_position(self) -> bool:
        return self._position is not None

    def restore_open_position(self) -> bool:
        """Restaura posición abierta desde la DB al iniciar el bot."""
        last_buy = self._repo.find_last_open_buy(self._config.symbol)
        if not last_buy:
            return False

        price = last_buy.price
        stop_loss = price * (1 - self._config.stop_loss_pct)
        take_profit = price * (1 + self._config.take_profit_pct)

        # Parsear timestamp para calcular tiempo transcurrido
        from datetime import datetime, timezone
        try:
            opened_at = datetime.fromisoformat(last_buy.timestamp).timestamp()
        except (ValueError, TypeError):
            opened_at = time.time()

        self._position = Position(
            symbol=self._config.symbol,
            side="BUY",
            entry_price=price,
            quantity=last_buy.quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            opened_at=opened_at,
        )
        self._highest_price = price
        self._trailing_active = False

        elapsed = (time.time() - opened_at) / 60
        logger.info(
            "Posición restaurada: %.6f %s @ %.2f (hace %.0f min) | SL=%.2f | TP=%.2f",
            last_buy.quantity, self._config.symbol, price,
            elapsed, stop_loss, take_profit,
        )
        return True

    def process_signal(self, signal: Signal, current_price: float) -> Optional[Trade]:
        """Procesa una señal de la estrategia y ejecuta la orden correspondiente."""
        # Protección: límite de operaciones por hora
        if not self._check_trade_rate_limit():
            return None

        if signal == Signal.BUY and not self.has_position:
            return self._open_position(current_price)
        elif signal == Signal.SELL and self.has_position:
            return self._close_position(current_price, reason="Señal SELL")
        # Deshabilitada la venta automática de balances existentes.
        # Esta función causó ventas en bucle destructivas.
        # Solo se vende balance existente cuando hay una posición tracked.
        return None

    def _check_trade_rate_limit(self) -> bool:
        """Verifica que no se exceda el límite de operaciones por hora."""
        now = time.time()
        one_hour_ago = now - 3600

        # Limpiar timestamps antiguos
        self._trade_timestamps = [
            ts for ts in self._trade_timestamps if ts > one_hour_ago
        ]

        if len(self._trade_timestamps) >= self._config.max_trades_per_hour:
            logger.warning(
                "Límite de operaciones por hora alcanzado (%d/%d). Esperando...",
                len(self._trade_timestamps), self._config.max_trades_per_hour,
            )
            return False
        return True

    def _record_trade_timestamp(self) -> None:
        """Registra el timestamp de una operación ejecutada."""
        self._trade_timestamps.append(time.time())

    def _sell_existing_balance(self, price: float) -> Optional[Trade]:
        """Vende balance existente del activo base (sin posición tracked).

        Protecciones:
        - Cooldown de 10 minutos entre ventas
        - Máximo de ventas por sesión (evita bucle destructivo)
        - Validación de monto mínimo notional
        - Vende TODO el balance de una vez (no en porciones)
        """
        # Protección 1: límite de ventas por sesión
        if self._balance_sell_count >= self.MAX_BALANCE_SELLS_PER_SESSION:
            return None

        # Protección 2: cooldown entre ventas
        now = time.time()
        if now - self._last_sell_time < self.SELL_COOLDOWN_SECONDS:
            return None

        base_asset = self._config.symbol.replace("USDT", "")
        try:
            balance = self._client.get_account_balance(base_asset)
        except Exception:
            return None

        if balance <= 0:
            return None

        # Vender TODO el balance disponible (no en porciones)
        quantity = self._adjust_quantity(balance)

        if quantity <= 0:
            return None

        # Protección 3: validar monto mínimo notional
        notional_value = quantity * price
        try:
            min_notional = self._client.get_min_notional(self._config.symbol)
            if notional_value < min_notional:
                logger.info(
                    "Balance existente de %s muy pequeño (%.2f USDT < %.2f min). Ignorando.",
                    base_asset, notional_value, min_notional,
                )
                return None
        except Exception:
            pass

        logger.info(
            "Vendiendo balance existente completo: %.6f %s @ %.2f (venta %d/%d)",
            quantity, base_asset, price,
            self._balance_sell_count + 1, self.MAX_BALANCE_SELLS_PER_SESSION,
        )

        if not self._config.dry_run:
            try:
                self._client.place_market_order(
                    symbol=self._config.symbol,
                    side="SELL",
                    quantity=quantity,
                )
            except Exception as e:
                logger.error("Error al vender balance existente: %s", e)
                return None

        trade = Trade(
            id=None,
            timestamp=now_utc_iso(),
            symbol=self._config.symbol,
            side="SELL",
            price=price,
            quantity=quantity,
            order_type="MARKET",
            status="FILLED",
        )
        saved_trade = self._repo.create(trade)
        self._last_sell_time = now
        self._balance_sell_count += 1
        logger.info("Balance vendido: %s (ventas restantes: %d)",
                     saved_trade, self.MAX_BALANCE_SELLS_PER_SESSION - self._balance_sell_count)
        return saved_trade

    def _adjust_quantity(self, raw_qty: float) -> float:
        """Ajusta la cantidad a la precisión del par."""
        try:
            info = self._client.get_symbol_info(self._config.symbol)
            for f in info.get("filters", []):
                if f["filterType"] == "LOT_SIZE":
                    step_size = float(f["stepSize"])
                    precision = int(round(-math.log10(step_size)))
                    return math.floor(raw_qty * 10**precision) / 10**precision
        except Exception as e:
            logger.warning("No se pudo obtener precisión: %s", e)
        return math.floor(raw_qty * 1e6) / 1e6

    def check_stop_loss_take_profit(self, current_price: float) -> Optional[Trade]:
        """Verifica SL/TP con trailing stop-loss, break-even y tiempo máximo."""
        if not self.has_position:
            return None

        position = self._position

        # 1. Verificar tiempo máximo de operación
        elapsed_minutes = (time.time() - position.opened_at) / 60
        if self._config.max_trade_minutes > 0 and elapsed_minutes >= self._config.max_trade_minutes:
            pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            logger.info(
                "TIEMPO MÁXIMO alcanzado (%.0f min): precio=%.2f, PnL=%.2f%%",
                elapsed_minutes, current_price, pnl_pct,
            )
            return self._close_position(current_price, reason=f"Tiempo máximo ({int(elapsed_minutes)}min)")

        # 2. Take-profit fijo
        if current_price >= position.take_profit:
            logger.info(
                "TAKE-PROFIT activado: precio=%.2f, TP=%.2f",
                current_price, position.take_profit,
            )
            return self._close_position(current_price, reason="Take-Profit")

        # 3. Actualizar precio máximo alcanzado
        if current_price > self._highest_price:
            self._highest_price = current_price

        # 4. Activar break-even cuando sube suficiente
        gain_pct = (current_price - position.entry_price) / position.entry_price
        if not self._trailing_active and gain_pct >= self._config.break_even_pct:
            self._trailing_active = True
            logger.info(
                "BREAK-EVEN activado: ganancia=%.2f%%, SL movido a entrada (%.2f)",
                gain_pct * 100, position.entry_price,
            )

        # 5. Calcular stop-loss dinámico
        if self._trailing_active:
            # Trailing: SL sigue al máximo alcanzado
            trailing_sl = self._highest_price * (1 - self._config.trailing_stop_pct)
            # Nunca bajar el SL por debajo del precio de entrada (break-even garantizado)
            effective_sl = max(trailing_sl, position.entry_price)
        else:
            effective_sl = position.stop_loss

        # 6. Verificar stop-loss
        if current_price <= effective_sl:
            if self._trailing_active:
                logger.info(
                    "TRAILING STOP activado: precio=%.2f, SL=%.2f (máximo fue %.2f)",
                    current_price, effective_sl, self._highest_price,
                )
                return self._close_position(current_price, reason="Trailing-Stop")
            else:
                logger.warning(
                    "STOP-LOSS activado: precio=%.2f, SL=%.2f",
                    current_price, effective_sl,
                )
                return self._close_position(current_price, reason="Stop-Loss")

        return None

    def _open_position(self, price: float) -> Optional[Trade]:
        """Abre una nueva posición de compra usando quoteOrderQty."""
        balance = self._client.get_account_balance("USDT")

        # Protección: no operar con balance insuficiente
        if balance < self._config.min_balance_usdt:
            logger.warning(
                "Balance USDT insuficiente: %.2f < %.2f mínimo. No se abre posición.",
                balance, self._config.min_balance_usdt,
            )
            return None

        trade_amount = balance * self._config.position_size_pct
        min_notional = self._client.get_min_notional(self._config.symbol)

        if trade_amount < min_notional:
            logger.warning(
                "Balance insuficiente para abrir posición: "
                "balance=%.4f USDT, monto=%.4f USDT (%.0f%%), "
                "mínimo notional=%.2f USDT. "
                "Necesitas al menos %.2f USDT",
                balance, trade_amount,
                self._config.position_size_pct * 100, min_notional,
                min_notional / self._config.position_size_pct,
            )
            return None

        # Redondear a 2 decimales para quoteOrderQty
        trade_amount = math.floor(trade_amount * 100) / 100
        quantity = trade_amount / price

        stop_loss = price * (1 - self._config.stop_loss_pct)
        take_profit = price * (1 + self._config.take_profit_pct)

        if self._config.dry_run:
            logger.info(
                "[DRY-RUN] Compra: %.2f USDT (~%.6f %s) @ %.2f | SL=%.2f | TP=%.2f",
                trade_amount,
                quantity,
                self._config.symbol,
                price,
                stop_loss,
                take_profit,
            )
        else:
            try:
                order = self._client.place_market_order_quote(
                    symbol=self._config.symbol,
                    side="BUY",
                    quote_quantity=trade_amount,
                )
                # Obtener cantidad real ejecutada desde la respuesta
                fills = order.get("fills", [])
                if fills:
                    quantity = sum(float(f["qty"]) for f in fills)
                    price = sum(
                        float(f["price"]) * float(f["qty"]) for f in fills
                    ) / quantity
                else:
                    executed = order.get("executedQty")
                    if executed:
                        quantity = float(executed)
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
            opened_at=time.time(),
        )
        self._highest_price = price
        self._trailing_active = False

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
        self._record_trade_timestamp()
        logger.info(
            "Posición abierta: %.2f USDT -> %.6f %s @ %.2f",
            trade_amount, quantity, self._config.symbol, price,
        )
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
        self._highest_price = 0.0
        self._trailing_active = False
        self._record_trade_timestamp()
        logger.info("Posición cerrada (%s): PnL=%.4f USDT", reason, pnl)
        return saved_trade

    def _calculate_quantity(self, usdt_amount: float, price: float) -> float:
        """Calcula la cantidad ajustada a la precisión del par."""
        raw_qty = usdt_amount / price
        return self._adjust_quantity(raw_qty)
