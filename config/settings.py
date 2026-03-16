"""Configuración y constantes del bot de trading."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class BinanceConfig:
    """Configuración de conexión a Binance Testnet."""

    api_key: str = field(default_factory=lambda: os.environ.get("BINANCE_API_KEY", ""))
    api_secret: str = field(
        default_factory=lambda: os.environ.get("BINANCE_API_SECRET", "")
    )
    testnet: bool = True
    testnet_url: str = "https://testnet.binance.vision"
    testnet_ws_url: str = "wss://testnet.binance.vision/ws"


@dataclass(frozen=True)
class TradingConfig:
    """Parámetros de la estrategia de trading."""

    symbol: str = "BTCUSDT"
    interval: str = "1h"
    sma_short_period: int = 20
    sma_long_period: int = 50
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    position_size_pct: float = 0.10
    dry_run: bool = True


@dataclass(frozen=True)
class AppConfig:
    """Configuración general de la aplicación."""

    db_path: str = str(BASE_DIR / "trading_bot.db")
    log_dir: str = str(BASE_DIR / "logs")
    dashboard_refresh_seconds: int = 5


# Instancias globales inmutables
BINANCE_CONFIG = BinanceConfig()
TRADING_CONFIG = TradingConfig()
APP_CONFIG = AppConfig()
