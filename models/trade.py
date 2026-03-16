"""Modelo de operaciones y gestión de la base de datos SQLite."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from utils.logger import setup_logger

logger = setup_logger("models.trade")


@dataclass(frozen=True)
class Trade:
    """Registro inmutable de una operación de trading."""

    id: Optional[int]
    timestamp: str
    symbol: str
    side: str  # BUY o SELL
    price: float
    quantity: float
    order_type: str  # MARKET, LIMIT
    status: str  # FILLED, PENDING, CANCELLED
    pnl: Optional[float] = None


class TradeRepository:
    """Repositorio para persistir operaciones en SQLite."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    order_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    pnl REAL
                )
            """)
            conn.commit()
            logger.info("Base de datos inicializada: %s", self._db_path)
        finally:
            conn.close()

    def create(self, trade: Trade) -> Trade:
        """Inserta una operación y retorna una copia con el ID asignado."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                INSERT INTO trades (timestamp, symbol, side, price, quantity, order_type, status, pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.timestamp,
                    trade.symbol,
                    trade.side,
                    trade.price,
                    trade.quantity,
                    trade.order_type,
                    trade.status,
                    trade.pnl,
                ),
            )
            conn.commit()
            return Trade(
                id=cursor.lastrowid,
                timestamp=trade.timestamp,
                symbol=trade.symbol,
                side=trade.side,
                price=trade.price,
                quantity=trade.quantity,
                order_type=trade.order_type,
                status=trade.status,
                pnl=trade.pnl,
            )
        finally:
            conn.close()

    def find_all(self, symbol: Optional[str] = None, limit: int = 100) -> list[Trade]:
        """Retorna las últimas operaciones, opcionalmente filtradas por símbolo."""
        conn = self._get_connection()
        try:
            if symbol:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE symbol = ? ORDER BY id DESC LIMIT ?",
                    (symbol, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            return [self._row_to_trade(row) for row in rows]
        finally:
            conn.close()

    def total_pnl(self, symbol: Optional[str] = None) -> float:
        """Calcula el P&L total de las operaciones cerradas."""
        conn = self._get_connection()
        try:
            if symbol:
                row = conn.execute(
                    "SELECT COALESCE(SUM(pnl), 0) as total FROM trades WHERE symbol = ? AND pnl IS NOT NULL",
                    (symbol,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COALESCE(SUM(pnl), 0) as total FROM trades WHERE pnl IS NOT NULL"
                ).fetchone()
            return float(row["total"])
        finally:
            conn.close()

    @staticmethod
    def _row_to_trade(row: sqlite3.Row) -> Trade:
        return Trade(
            id=row["id"],
            timestamp=row["timestamp"],
            symbol=row["symbol"],
            side=row["side"],
            price=row["price"],
            quantity=row["quantity"],
            order_type=row["order_type"],
            status=row["status"],
            pnl=row["pnl"],
        )


def now_utc_iso() -> str:
    """Retorna el timestamp actual en formato ISO 8601 UTC."""
    return datetime.now(timezone.utc).isoformat()
