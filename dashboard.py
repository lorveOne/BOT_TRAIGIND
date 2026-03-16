"""Dashboard web multi-par para monitorear el bot de trading."""

import copy
import json
import os
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from config.settings import (
    APP_CONFIG, BINANCE_CONFIG, LSTM_CONFIG, MULTI_PAIR_SYMBOLS, TRADING_CONFIG,
    LstmConfig, TradingConfig,
)
from models.trade import TradeRepository
from services.binance_client import BinanceClient
from services.data_fetcher import DataFetcher
from services.historical_data import HistoricalDataFetcher
from services.model_trainer import ModelTrainer
from services.order_manager import OrderManager
from strategies.ensemble import EnsembleStrategy
from strategies.lstm_predictor import LstmPredictor
from strategies.sma_crossover import SmaCrossoverStrategy
from utils.logger import setup_logger

logger = setup_logger("dashboard")

# Estado global multi-par
bot_states = {}
global_state = {"running": False, "balance_usdt": 0, "last_update": ""}
state_lock = threading.Lock()


def make_pair_state():
    return {
        "symbol": "",
        "price": 0,
        "position": None,
        "sma_short": 0,
        "sma_long": 0,
        "sma_signal": "HOLD",
        "lstm_direction": "N/A",
        "lstm_confidence": 0,
        "final_signal": "HOLD",
        "reason": "",
        "trades": [],
        "total_pnl": 0,
        "lstm_ready": False,
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with state_lock:
                data = {"global": global_state, "pairs": bot_states}
            self.wfile.write(json.dumps(data).encode())
        elif self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def run_pair_bot(symbol: str, client: BinanceClient, trade_repo: TradeRepository):
    """Ejecuta un bot para un par específico."""
    pair_config = TradingConfig(
        symbol=symbol,
        interval=TRADING_CONFIG.interval,
        sma_short_period=TRADING_CONFIG.sma_short_period,
        sma_long_period=TRADING_CONFIG.sma_long_period,
        stop_loss_pct=TRADING_CONFIG.stop_loss_pct,
        take_profit_pct=TRADING_CONFIG.take_profit_pct,
        position_size_pct=TRADING_CONFIG.position_size_pct / len(MULTI_PAIR_SYMBOLS),
        dry_run=TRADING_CONFIG.dry_run,
    )

    fetcher = DataFetcher(client, pair_config, binance_config=BINANCE_CONFIG)
    sma = SmaCrossoverStrategy(pair_config.sma_short_period, pair_config.sma_long_period)

    lstm_model_dir = os.path.join(LSTM_CONFIG.model_dir, symbol)
    lstm_config = LstmConfig(
        enabled=LSTM_CONFIG.enabled,
        sequence_length=LSTM_CONFIG.sequence_length,
        lstm_units=LSTM_CONFIG.lstm_units,
        num_layers=LSTM_CONFIG.num_layers,
        epochs=LSTM_CONFIG.epochs,
        batch_size=LSTM_CONFIG.batch_size,
        confidence_threshold=LSTM_CONFIG.confidence_threshold,
        training_candles=LSTM_CONFIG.training_candles,
        model_dir=lstm_model_dir,
        retrain_on_startup=LSTM_CONFIG.retrain_on_startup,
        retrain_interval_minutes=LSTM_CONFIG.retrain_interval_minutes,
    )

    lstm = LstmPredictor(lstm_config)
    ensemble = EnsembleStrategy(sma, lstm, lstm_config.confidence_threshold)
    order_mgr = OrderManager(client, trade_repo, pair_config)

    # Restaurar posición abierta si existe en la DB
    if order_mgr.restore_open_position():
        logger.info("[%s] Posición abierta restaurada desde DB", symbol)

    # Entrenar LSTM
    if lstm_config.enabled:
        logger.info("[%s] Entrenando LSTM...", symbol)
        hist_fetcher = HistoricalDataFetcher(client, pair_config)
        trainer = ModelTrainer(lstm, hist_fetcher, lstm_config)
        success = trainer.train_or_load()
        with state_lock:
            bot_states[symbol]["lstm_ready"] = success
        logger.info("[%s] LSTM: %s", symbol, "listo" if success else "no disponible")

    fetcher.start_price_stream()
    time.sleep(3)

    last_retrain_time = time.time()
    logger.info("[%s] Bot iniciado", symbol)

    while global_state["running"]:
        try:
            # Re-entrenar LSTM periódicamente
            elapsed = time.time() - last_retrain_time
            if elapsed >= lstm_config.retrain_interval_minutes * 60:
                logger.info("[%s] Re-entrenando LSTM...", symbol)
                hist_fetcher = HistoricalDataFetcher(client, pair_config)
                trainer = ModelTrainer(lstm, hist_fetcher, lstm_config)
                trainer.train()
                last_retrain_time = time.time()
                logger.info("[%s] Re-entrenamiento completado", symbol)

            data_limit = max(
                pair_config.sma_long_period + 10,
                lstm_config.sequence_length + 100,
            )
            ohlcv = fetcher.get_ohlcv(data_limit)
            prices = ohlcv["closes"]
            volumes = ohlcv["volumes"]
            if not prices:
                time.sleep(5)
                continue

            current_price = fetcher.current_price
            if current_price == 0:
                current_price = prices[-1]

            order_mgr.check_stop_loss_take_profit(current_price)
            result = ensemble.analyze(
                prices, volumes,
                highs=ohlcv["highs"],
                lows=ohlcv["lows"],
            )
            order_mgr.process_signal(result.final_signal, current_price)

            trades_list = []
            for t in trade_repo.find_all(symbol, limit=5):
                trades_list.append({
                    "timestamp": t.timestamp[:19],
                    "side": t.side,
                    "price": round(t.price, 4),
                    "quantity": t.quantity,
                    "pnl": round(t.pnl, 4) if t.pnl else None,
                })

            pos = order_mgr.position
            pos_data = None
            if pos:
                unrealized = (current_price - pos.entry_price) * pos.quantity
                pos_data = {
                    "entry_price": round(pos.entry_price, 4),
                    "quantity": pos.quantity,
                    "stop_loss": round(pos.stop_loss, 4),
                    "take_profit": round(pos.take_profit, 4),
                    "unrealized_pnl": round(unrealized, 4),
                    "pnl_pct": round(((current_price / pos.entry_price) - 1) * 100, 2),
                }

            with state_lock:
                bot_states[symbol].update({
                    "symbol": symbol,
                    "price": round(current_price, 4),
                    "position": pos_data,
                    "sma_short": round(result.sma_result.sma_short, 4),
                    "sma_long": round(result.sma_result.sma_long, 4),
                    "sma_signal": result.sma_result.signal.value,
                    "lstm_direction": result.lstm_prediction.direction.value if result.lstm_prediction else "N/A",
                    "lstm_confidence": round(result.lstm_prediction.confidence * 100, 1) if result.lstm_prediction else 0,
                    "final_signal": result.final_signal.value,
                    "reason": result.reason,
                    "trades": trades_list,
                    "total_pnl": round(trade_repo.total_pnl(symbol), 4),
                })

        except Exception as e:
            logger.error("[%s] Error: %s", symbol, e)

        time.sleep(5)

    fetcher.stop()
    logger.info("[%s] Bot detenido", symbol)


def run_all_bots():
    """Inicia todos los bots multi-par."""
    client = BinanceClient(BINANCE_CONFIG)
    trade_repo = TradeRepository(APP_CONFIG.db_path)

    with state_lock:
        global_state["running"] = True
        for sym in MULTI_PAIR_SYMBOLS:
            bot_states[sym] = make_pair_state()
            bot_states[sym]["symbol"] = sym

    threads = []
    for sym in MULTI_PAIR_SYMBOLS:
        t = threading.Thread(target=run_pair_bot, args=(sym, client, trade_repo), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(2)  # escalonar para no saturar la API

    # Actualizar balances globales
    while global_state["running"]:
        try:
            usdt = client.get_account_balance("USDT")
            with state_lock:
                global_state["balance_usdt"] = round(usdt, 2)
                global_state["last_update"] = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        except Exception:
            pass
        time.sleep(10)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Trading Bot Multi-Par</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',monospace;padding:15px}
.header{text-align:center;padding:12px;border-bottom:2px solid #f0b90b;margin-bottom:15px}
.header h1{color:#f0b90b;font-size:22px}
.header .sub{color:#8b949e;font-size:13px;margin-top:4px}
.global{display:flex;gap:15px;margin-bottom:15px;justify-content:center}
.global .card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 25px;text-align:center}
.global .card h3{color:#8b949e;font-size:11px;text-transform:uppercase;margin-bottom:4px}
.global .card .value{font-size:22px;font-weight:bold}
.pairs{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:15px}
.pair-card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;position:relative}
.pair-card h2{color:#f0b90b;font-size:16px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center}
.pair-card .price{font-size:20px;font-weight:bold;color:#e6edf3;margin-bottom:8px}
.row{display:flex;justify-content:space-between;padding:4px 0;font-size:13px;border-bottom:1px solid #21262d}
.row .label{color:#8b949e}
.decision-box{background:#1c2333;border:1px solid #f0b90b;border-radius:6px;padding:8px;text-align:center;margin:8px 0}
.decision-box .final{font-size:22px;font-weight:bold}
.decision-box .reason{font-size:10px;color:#8b949e;margin-top:3px}
.conf-bar{display:inline-block;width:80px;height:8px;background:#21262d;border-radius:4px;overflow:hidden;vertical-align:middle}
.conf-fill{height:100%;border-radius:4px;transition:width 0.5s}
.green{color:#3fb950}.red{color:#f85149}.yellow{color:#f0b90b}
.pos-box{background:#0d1117;border:1px solid #30363d;border-radius:4px;padding:6px;margin-top:6px;font-size:12px}
.trades-section{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px}
.trades-section h3{color:#f0b90b;margin-bottom:8px;font-size:14px}
.trades-section table{width:100%;border-collapse:collapse}
.trades-section th{color:#8b949e;text-align:left;padding:6px;border-bottom:1px solid #30363d;font-size:11px}
.trades-section td{padding:6px;border-bottom:1px solid #21262d;font-size:12px}
.buy-tag{background:#238636;color:white;padding:1px 6px;border-radius:3px;font-size:11px}
.sell-tag{background:#da3633;color:white;padding:1px 6px;border-radius:3px;font-size:11px}
.pulse{animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.5}}
.status{text-align:right;color:#8b949e;font-size:11px;margin-top:8px}
@media(max-width:900px){.pairs{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="header">
    <h1>TRADING BOT MULTI-PAR</h1>
    <div class="sub">BTC + ORO + EUR | BiLSTM+Attention (16 features) | Binance LIVE | 5min</div>
</div>

<div class="global">
    <div class="card"><h3>Balance USDT</h3><div class="value yellow" id="g-usdt">--</div></div>
    <div class="card"><h3>P&L Total</h3><div class="value" id="g-pnl">--</div></div>
    <div class="card"><h3>Pares Activos</h3><div class="value green" id="g-pairs">--</div></div>
</div>

<div class="pairs" id="pairs-container"></div>

<div class="trades-section">
    <h3>Todas las Operaciones</h3>
    <table>
        <thead><tr><th>Par</th><th>Hora</th><th>Tipo</th><th>Precio</th><th>Cantidad</th><th>PnL</th></tr></thead>
        <tbody id="all-trades"><tr><td colspan="6" style="color:#8b949e">Esperando operaciones...</td></tr></tbody>
    </table>
</div>

<div class="status">
    <span class="pulse" style="color:#3fb950;">&#9679;</span> Actualizando cada 3s | <span id="last-update">--</span>
</div>

<script>
const PAIR_NAMES = {'BTCUSDT':'Bitcoin','PAXGUSDT':'Oro (PAXG)','EURUSDT':'Euro'};
const PAIR_ICONS = {'BTCUSDT':'&#8383;','PAXGUSDT':'&#9733;','EURUSDT':'&#8364;'};

function sigClass(s){return s==='BUY'?'green':s==='SELL'?'red':'yellow'}

function renderPair(sym, d) {
    let posHtml = '<div style="color:#8b949e;font-size:12px">Sin posicion</div>';
    if (d.position) {
        const p = d.position;
        const s = p.unrealized_pnl >= 0 ? '+' : '';
        const c = p.unrealized_pnl >= 0 ? 'green' : 'red';
        posHtml = '<div class="pos-box">' +
            '<div class="row"><span class="label">Entrada</span><span>$'+p.entry_price+'</span></div>' +
            '<div class="row"><span class="label">SL / TP</span><span class="red">$'+p.stop_loss+'</span> / <span class="green">$'+p.take_profit+'</span></div>' +
            '<div class="row"><span class="label">PnL</span><span class="'+c+'">'+s+'$'+p.unrealized_pnl+' ('+s+p.pnl_pct+'%)</span></div></div>';
    }
    const confW = Math.min(d.lstm_confidence, 100);
    const confColor = confW >= 50 ? '#3fb950' : confW >= 20 ? '#f0b90b' : '#f85149';

    return '<div class="pair-card">' +
        '<h2><span>'+(PAIR_ICONS[sym]||'')+' '+(PAIR_NAMES[sym]||sym)+'</span><span style="font-size:12px;color:#8b949e">'+sym+'</span></h2>' +
        '<div class="price">$'+d.price.toLocaleString()+'</div>' +
        '<div class="row"><span class="label">SMA</span><span>'+d.sma_short+' / '+d.sma_long+'</span></div>' +
        '<div class="row"><span class="label">SMA Senal</span><span class="'+sigClass(d.sma_signal)+'">'+d.sma_signal+'</span></div>' +
        '<div class="row"><span class="label">LSTM</span><span class="'+sigClass(d.lstm_direction)+'">'+d.lstm_direction+'</span> <span class="conf-bar"><span class="conf-fill" style="width:'+confW+'%;background:'+confColor+'"></span></span> '+d.lstm_confidence+'%</div>' +
        '<div class="decision-box"><div style="color:#8b949e;font-size:10px">DECISION</div><div class="final '+sigClass(d.final_signal)+'">'+d.final_signal+'</div><div class="reason">'+d.reason+'</div></div>' +
        '<div class="row"><span class="label">PnL cerrado</span><span class="'+(d.total_pnl>=0?'green':'red')+'">'+(d.total_pnl>=0?'+':'')+d.total_pnl+'</span></div>' +
        posHtml + '</div>';
}

function update() {
    fetch('/api/state').then(r=>r.json()).then(data => {
        const g = data.global;
        const pairs = data.pairs;

        document.getElementById('g-usdt').textContent = '$' + g.balance_usdt.toLocaleString();
        document.getElementById('last-update').textContent = g.last_update;

        const syms = Object.keys(pairs);
        document.getElementById('g-pairs').textContent = syms.length + '/3';

        let totalPnl = 0;
        let pairsHtml = '';
        let allTrades = [];

        syms.forEach(sym => {
            const d = pairs[sym];
            totalPnl += d.total_pnl || 0;
            pairsHtml += renderPair(sym, d);
            (d.trades||[]).forEach(t => { t.symbol = sym; allTrades.push(t); });
        });

        document.getElementById('pairs-container').innerHTML = pairsHtml;

        const pnlEl = document.getElementById('g-pnl');
        const s = totalPnl >= 0 ? '+' : '';
        pnlEl.textContent = s + '$' + totalPnl.toFixed(4);
        pnlEl.className = 'value ' + (totalPnl >= 0 ? 'green' : 'red');

        const tbody = document.getElementById('all-trades');
        if (allTrades.length > 0) {
            allTrades.sort((a,b) => b.timestamp.localeCompare(a.timestamp));
            tbody.innerHTML = allTrades.slice(0,15).map(t => {
                const tag = t.side==='BUY'?'<span class="buy-tag">BUY</span>':'<span class="sell-tag">SELL</span>';
                const pnl = t.pnl!==null?((t.pnl>=0?'+':'')+t.pnl):'-';
                const pc = t.pnl!==null?(t.pnl>=0?'green':'red'):'';
                return '<tr><td>'+t.symbol+'</td><td>'+t.timestamp+'</td><td>'+tag+'</td><td>$'+t.price+'</td><td>'+t.quantity+'</td><td class="'+pc+'">'+pnl+'</td></tr>';
            }).join('');
        }
    }).catch(()=>{});
}
setInterval(update, 3000);
update();
</script>
</body>
</html>"""


if __name__ == "__main__":
    bot_thread = threading.Thread(target=run_all_bots, daemon=True)
    bot_thread.start()

    port = 8080
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    logger.info("Dashboard Multi-Par en http://localhost:%d", port)
    print("=" * 50)
    print(f"  Dashboard: http://localhost:{port}")
    print(f"  Pares: {', '.join(MULTI_PAIR_SYMBOLS)}")
    print("  Ctrl+C para detener")
    print("=" * 50)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        with state_lock:
            global_state["running"] = False
        server.shutdown()
        logger.info("Dashboard detenido")
