# app.py
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, send_file, jsonify, url_for
from flask_socketio import SocketIO, emit, join_room, leave_room
import requests
import pandas as pd
import os, csv, time, math, random, traceback
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo 
import threading
import re

# Import Blueprint
from chat_api import bp as chat_bp 

# -------- App setup --------
app = Flask(__name__)
app.register_blueprint(chat_bp)
app.config['SECRET_KEY'] = os.environ.get("FLASK_SECRET", "secret!")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# -------- Files and folders --------
CSV_FILE = "market_data.csv"
LIVE_LOG_DIR = "live_logs"
os.makedirs(LIVE_LOG_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

# -------- Global session & headers (NSE) --------
_session = requests.Session()
_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com",
    "Accept-Language": "en-US,en;q=0.9"
}

def prime_nse_session():
    try:
        _session.get("https://www.nseindia.com", headers=_headers, timeout=8)
    except Exception as e:
        print("[nse] prime cookies error:", e)

prime_nse_session()

# -------- Utilities --------
def now_iso_utc():
    return datetime.utcnow().replace(microsecond=0, tzinfo=timezone.utc).isoformat()

def now_ist_str():
    try:
        ist = ZoneInfo("Asia/Kolkata")
        return datetime.now(tz=ist).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
        return ist.strftime("%Y-%m-%d %H:%M:%S") + " IST"

# -------- NSE JSON fetch --------
def fetch_json_from_nse(url, max_attempts=4, backoff_base=0.5):
    for attempt in range(1, max_attempts + 1):
        try:
            r = _session.get(url, headers=_headers, timeout=10)
            r.raise_for_status()
            text = r.text.strip()
            if not text: raise ValueError("empty response")
            try:
                return r.json()
            except Exception:
                raise ValueError("non-json response")
        except Exception as e:
            print(f"[nse] fetch attempt {attempt} failed for {url!r}: {e}")
            try:
                _session.get("https://www.nseindia.com", headers=_headers, timeout=6)
            except Exception: pass
            sleep_t = backoff_base * (2 ** (attempt - 1)) + random.random() * 0.5
            time.sleep(sleep_t)
    return {}

# -------- Quote fetching --------
def fetch_quote(symbol):
    symbol = symbol.upper()
    url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
    j = fetch_json_from_nse(url)
    if isinstance(j, dict):
        price_info = j.get("priceInfo") or {}
        last = price_info.get("lastPrice")
        if last is None:
            last = price_info.get("close") or j.get("lastPrice") or j.get("price")
        try:
            last_f = float(last) if last is not None else None
        except Exception:
            last_f = None
        return {
            "raw": j,
            "lastPrice": last_f,
            "priceInfo": price_info
        }
    return {"raw": j, "lastPrice": None, "priceInfo": {}}

# -------- CSV tick logging helpers --------
def _csv_file(symbol):
    return os.path.join(LIVE_LOG_DIR, f"{symbol.upper()}.csv")

def append_tick(symbol, ts_iso, price, qty=0):
    fname = _csv_file(symbol)
    file_exists = os.path.exists(fname)
    with open(fname, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if not file_exists:
            w.writerow(["timestamp", "price", "qty"])
        w.writerow([ts_iso, f"{price:.4f}", qty])

def read_tick_df(symbol, since=None):
    fname = _csv_file(symbol)
    if not os.path.exists(fname):
        return pd.DataFrame(columns=["timestamp", "price", "qty"])
    df = pd.read_csv(fname, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    if since is not None:
        df = df[df["timestamp"] >= since]
    return df

def ticks_to_ohlcv_df(df, freq='1min'):
    if df.empty: return pd.DataFrame()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp").resample(freq).agg({
        "price": ["first", "max", "min", "last"],
        "qty": "sum"
    })
    df.columns = ["open", "high", "low", "close", "volume"]
    df = df.dropna(subset=["open"])
    df = df.reset_index()
    return df

# -------- Background logger threads (MANAGED) --------
_logger_threads = {}
_logger_flags = {}
_subscriber_counts = {} # Tracks how many users are watching a symbol
_thread_lock = threading.Lock() # Ensures thread safety

def _logger_worker(symbol, poll_interval=3.0):
    symbol = symbol.upper()
    flag = _logger_flags.get(symbol)
    while flag is not None and not flag.is_set():
        try:
            q = fetch_quote(symbol)
            price = q.get("lastPrice")
            if price is None:
                # Attempt to get last known price from CSV if live fetch fails
                df = read_tick_df(symbol)
                if not df.empty:
                    try:
                        price = float(df.iloc[-1]["price"])
                    except: price = None

            if price is not None:
                ts_iso = now_iso_utc()
                append_tick(symbol, ts_iso, float(price), qty=0)
                payload = {
                    "symbol": symbol,
                    "timestamp": ts_iso,
                    "price": float(price),
                    "time_ist": now_ist_str()
                }
                socketio.emit("price_update", payload, room=symbol)
                print(f"[logger:{symbol}] {ts_iso} price={price}")
            else:
                print(f"[logger:{symbol}] no price (will retry)")
        except Exception as exc:
            print(f"[logger:{symbol}] exception:", exc)
            traceback.print_exc()
        time.sleep(poll_interval)

def start_logger(symbol):
    symbol = symbol.upper()
    with _thread_lock:
        # Increment subscriber count
        _subscriber_counts[symbol] = _subscriber_counts.get(symbol, 0) + 1
        
        # Only start thread if it's the first subscriber
        if symbol not in _logger_threads:
            stop_event = threading.Event()
            _logger_flags[symbol] = stop_event
            t = threading.Thread(target=_logger_worker, args=(symbol,), daemon=True)
            _logger_threads[symbol] = t
            t.start()
            print(f"[logger] started for {symbol} (subscribers: {_subscriber_counts[symbol]})")

def stop_logger(symbol):
    symbol = symbol.upper()
    with _thread_lock:
        if symbol in _subscriber_counts:
            _subscriber_counts[symbol] -= 1
            
            # Only stop thread if NO ONE is watching anymore
            if _subscriber_counts[symbol] <= 0:
                flag = _logger_flags.get(symbol)
                if flag: flag.set()
                _logger_threads.pop(symbol, None)
                _logger_flags.pop(symbol, None)
                _subscriber_counts.pop(symbol, None)
                print(f"[logger] stopped for {symbol} (0 subscribers left)")
            else:
                print(f"[logger] keeping {symbol} alive (subscribers: {_subscriber_counts[symbol]})")

# -------- SocketIO handlers --------
@socketio.on("subscribe")
def handle_subscribe(data):
    symbol = (data.get("symbol") or "").upper()
    if not symbol:
        emit("error", {"message": "symbol required"})
        return
    join_room(symbol)
    start_logger(symbol) # This now intelligently handles multiple users
    
    # Send history immediately
    df = read_tick_df(symbol)
    if not df.empty:
        tail = df.tail(200)
        for _, row in tail.iterrows():
            try:
                ts = pd.to_datetime(row["timestamp"]).isoformat()
            except:
                ts = str(row["timestamp"])
            emit("price_update", {"symbol": symbol, "timestamp": ts, "price": float(row["price"])})
    emit("subscribed", {"symbol": symbol})

@socketio.on("unsubscribe")
def handle_unsubscribe(data):
    symbol = (data.get("symbol") or "").upper()
    if not symbol: return
    leave_room(symbol)
    stop_logger(symbol) # This now only stops if you are the last user
    emit("unsubscribed", {"symbol": symbol})

# -------- Chart data endpoint --------
@app.route("/chart-data/<symbol>")
def chart_data(symbol):
    rng = request.args.get("range", "1D").upper()
    symbol = symbol.upper()

    range_map = {
        "1D": {"days": 1, "freq": "1min"},
        "1W": {"days": 7, "freq": "1D"},
        "1M": {"days": 30, "freq": "1D"},
        "1Y": {"days": 365, "freq": "1D"},
    }
    cfg = range_map.get(rng, range_map["1D"])
    since_dt = datetime.utcnow() - timedelta(days=cfg["days"])

    try:
        q = fetch_quote(symbol).get("raw", {}) or {}
        ph = q.get("priceHistory") or q.get("data", {})
        hist = None
        if isinstance(ph, dict) and ph.get("data"): hist = ph.get("data")
        elif isinstance(ph, list) and ph: hist = ph
        
        if hist and isinstance(hist, list):
            out = []
            for r in hist:
                t = r.get("t") or r.get("timestamp") or r.get("date") or r.get("time")
                c = r.get("c") or r.get("close") or r.get("last") or r.get("closePrice")
                o = r.get("o") or r.get("open") or r.get("openPrice")
                h = r.get("h") or r.get("high") or r.get("highPrice")
                l = r.get("l") or r.get("low") or r.get("lowPrice")
                v = r.get("v") or r.get("volume") or 0
                try: t_iso = pd.to_datetime(t, errors="coerce").isoformat()
                except: t_iso = str(t)
                out.append({"t": t_iso, "o": float(o or c or 0), "h": float(h or c or 0),
                            "l": float(l or c or 0), "c": float(c or 0), "v": float(v or 0)})
            
            # Filter
            out_filtered = []
            for item in out:
                dt = pd.to_datetime(item["t"], errors="coerce")
                if pd.isna(dt) or dt.to_pydatetime() >= since_dt:
                    out_filtered.append(item)
            return jsonify(out_filtered if out_filtered else out)
    except Exception as e:
        print("[chart] parse priceHistory failed:", e)

    # Fallback to local
    try:
        df = read_tick_df(symbol)
        if not df.empty:
            df = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df = df[df["timestamp"] >= since_dt]
            
            if df.empty: df = read_tick_df(symbol).tail(5000)

            freq = cfg["freq"]
            ohlc = ticks_to_ohlcv_df(df, freq='1min' if freq == '1min' else '1D')
            if not ohlc.empty:
                out = []
                for _, r in ohlc.iterrows():
                    out.append({"t": r["timestamp"].isoformat(), "o": float(r["open"]),
                                "h": float(r["high"]), "l": float(r["low"]), "c": float(r["close"]), "v": float(r["volume"])})
                return jsonify(out)
    except Exception as e:
        print("[chart] tick aggregation failed:", e)

    # Simulation Fallback
    try:
        now = datetime.utcnow().replace(second=0, microsecond=0)
        npoints = 120 if cfg["days"] == 1 else min(120, cfg["days"] * 4)
        base = 100 + (hash(symbol) % 200)
        points = []
        for i in range(npoints):
            ts = (now - timedelta(minutes=(npoints - i - 1))) if cfg["days"] == 1 else (now - timedelta(days=(npoints - i - 1)))
            p = base + math.sin(i / 6.0) * 3 + (i % 5 - 2) * 0.6
            points.append({"t": ts.isoformat(), "o": round(p - 0.4, 2), "h": round(p + 1.2, 2), "l": round(p - 1.2, 2), "c": round(p + 0.2, 2), "v": float(1000 + i)})
        return jsonify(points)
    except:
        return jsonify([])

# -------- Index route --------
@app.route("/", methods=["GET", "POST"])
def index():
    table_html = None
    selected_index = ""
    custom_name = "Market Data"
    custom_details = "Fetched Live from NSE"

    presets = {
         "NIFTY 50": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050",
        "NIFTY NEXT 50": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20NEXT%2050",
        "NIFTY BANK": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20BANK",
        "NIFTY IT": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20IT",
        "NIFTY AUTO": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20AUTO",
        "NIFTY FMCG": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20FMCG",
        "NIFTY PHARMA": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20PHARMA",
        "NIFTY ENERGY": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20ENERGY",
        "NIFTY METAL": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20METAL",
        "PERMITTED TO TRADE": "https://www.nseindia.com/api/equity-stockIndices?index=PERMITTED%20TO%20TRADE",
        "SECURITIES IN F&O": "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O",
        "NIFTY500 MOMENTUM 50": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY500%20MOMENTUM%2050",
        "NIFTY SMALLCAP250 MOMENTUM QUALITY 100": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20SMALLCAP250%20MOMENTUM%20QUALITY%20100"
    }

    if request.method == "POST":
        selected_index = request.form.get("api_url")
        custom_name = request.form.get("name") or custom_name
        custom_details = request.form.get("details") or custom_details
        action = request.form.get("action", "load")
        best_count_raw = request.form.get("best_count", "").strip() if request.form.get("best_count") is not None else ""
        df = pd.DataFrame()
        try:
            data = fetch_json_from_nse(selected_index)
            rows = data.get("data") or data.get("value") or []
            df = pd.DataFrame(rows)
            if not df.empty:
                df = df.dropna(axis=1, how="all")
                try: df.to_csv(CSV_FILE, index=False)
                except: pass
        except Exception as e:
            print("[index] error:", e)

        if action == "top10" and not df.empty:
            df = df.head(10)

        if action == "best" and not df.empty:
            try:
                best_count = int(best_count_raw) if best_count_raw != "" else None
            except: best_count = None
            
            p_candidates = ["pChange", "pchange", "%change", "%Change", "PChange", "pchg", "pctChange"]
            def find_col(df, candidates):
                cols = list(df.columns)
                lower_map = {c.lower(): c for c in cols}
                for cand in candidates:
                    lc = cand.lower()
                    if lc in lower_map: return lower_map[lc]
                for cand in candidates:
                    for c in cols:
                        if cand.lower().replace("%","") in c.lower(): return c
                return None
            
            pcol = find_col(df, p_candidates) or find_col(df, ["change", "Change"])
            
            def numeric_from_cell(x):
                if pd.isna(x): return 0.0
                try:
                    s = str(x).strip().replace(",", "")
                    m = re.search(r"[-+]?\d+(\.\d+)?", s)
                    return float(m.group(0)) if m else 0.0
                except: return 0.0
            
            if pcol:
                try:
                    df["_pnum"] = df[pcol].apply(numeric_from_cell).astype(float)
                    df = df.sort_values("_pnum", ascending=False).reset_index(drop=True)
                    if best_count is not None and best_count > 0:
                        df = df.head(best_count)
                except: pass

        if not df.empty:
            if "symbol" in df.columns:
                df["symbol"] = df["symbol"].apply(lambda s: f'<a href="{url_for("details", symbol=s)}">{s}</a>')
            
            # Arrows logic
            def find_col(df, candidates):
                cols = list(df.columns)
                lower_map = {c.lower(): c for c in cols}
                for cand in candidates:
                    lc = cand.lower()
                    if lc in lower_map: return lower_map[lc]
                return None

            change_col = find_col(df, ["change", "Change", "chg"])
            pcol = find_col(df, ["pChange", "pchange", "%change", "pctChange"])

            def make_arrow_html(val):
                orig = "" if pd.isna(val) else str(val)
                num = 0.0
                try:
                    s = orig.replace(",", "").strip()
                    m = re.search(r"[-+]?\d+(\.\d+)?", s)
                    if m: num = float(m.group(0))
                except: pass
                
                arrow, cls = ("", "")
                if num > 0: arrow, cls = (" ▲", "positive")
                elif num < 0: arrow, cls = (" ▼", "negative")
                return f'{orig}<span class="{cls}">{arrow}</span>' if arrow else orig

            if change_col and change_col in df.columns:
                df[change_col] = df[change_col].apply(make_arrow_html)
            if pcol and pcol in df.columns:
                df[pcol] = df[pcol].apply(make_arrow_html)
            if "_pnum" in df.columns:
                df = df.drop(columns=["_pnum"])

            try:
                table_html = df.to_html(classes="table table-dark table-hover", index=False, escape=False)
            except: table_html = None

    return render_template("index.html", presets=presets, table_html=table_html,
                           selected_index_param=selected_index, custom_name=custom_name, custom_details=custom_details)

@app.route("/details/<symbol>")
def details(symbol):
    return render_template("details.html", symbol=symbol.upper())

@app.route("/download")
def download():
    if os.path.exists(CSV_FILE):
        return send_file(CSV_FILE, as_attachment=True)
    return "CSV missing", 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
