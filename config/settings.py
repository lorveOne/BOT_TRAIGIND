"""Configuración y constantes del bot de trading."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class BinanceConfig:
    """Configuración de conexión a Binance."""

    api_key: str = field(default_factory=lambda: os.environ.get("BINANCE_API_KEY", ""))
    api_secret: str = field(
        default_factory=lambda: os.environ.get("BINANCE_API_SECRET", "")
    )
    testnet: bool = False


@dataclass(frozen=True)
class TradingConfig:
    """Parámetros de la estrategia de trading."""

    symbol: str = "EURUSDT"
    interval: str = "15m"
    sma_short_period: int = 7
    sma_long_period: int = 21
    stop_loss_pct: float = 0.008
    take_profit_pct: float = 0.015
    position_size_pct: float = 0.95
    dry_run: bool = False
    # Trailing stop-loss: cuando sube X%, mover SL al precio de entrada (break-even)
    break_even_pct: float = 0.005  # +0.5% activa break-even (SL = entrada)
    # Trailing: SL sigue al precio a esta distancia
    trailing_stop_pct: float = 0.004  # 0.4% debajo del máximo
    # Tiempo máximo de operación en minutos
    max_trade_minutes: int = 120  # 2 horas máximo
    # Protecciones de seguridad
    min_balance_usdt: float = 5.0  # No operar si balance < 5 USDT
    max_daily_loss_usdt: float = 50.0  # Detener operaciones si pérdida diaria > 50 USDT
    max_trades_per_hour: int = 10  # Máximo de operaciones por hora


# Configuración multi-par
MULTI_PAIR_SYMBOLS = ["PAXGUSDT"]


@dataclass(frozen=True)
class AppConfig:
    """Configuración general de la aplicación."""

    db_path: str = str(BASE_DIR / "trading_bot.db")
    log_dir: str = str(BASE_DIR / "logs")
    dashboard_refresh_seconds: int = 5


@dataclass(frozen=True)
class LstmConfig:
    """Configuración de la red neuronal LSTM."""

    enabled: bool = True
    sequence_length: int = 60
    lstm_units: int = 64
    num_layers: int = 2
    epochs: int = 20
    batch_size: int = 32
    confidence_threshold: float = 0.01
    training_candles: int = 10000  # ~3.5 días de velas de 5m
    model_dir: str = str(BASE_DIR / "models" / "trained")
    retrain_on_startup: bool = False
    retrain_interval_minutes: int = 30


# Instancias globales inmutables
BINANCE_CONFIG = BinanceConfig()
TRADING_CONFIG = TradingConfig()
APP_CONFIG = AppConfig()
LSTM_CONFIG = LstmConfig()
