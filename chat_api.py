# chat_api.py
"""
Upgraded Brocode AI — robust chat backend that combines:
 - yfinance live + history for indicators
 - local market_data.csv as authoritative fallback / knowledge base
 - friendly, confident, contextual replies (no cryptic errors)
 - blueprint '/api/chat' route (POST JSON {"q": "..."})
"""
import os
import time
import math
import traceback
import difflib
from datetime import datetime, timedelta

from flask import Blueprint, request, jsonify
import pandas as pd
import yfinance as yf

bp = Blueprint("chat_api", __name__, url_prefix="/api")

# -------------------- Config --------------------
CANDIDATES = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "KOTAKBANK.NS","LT.NS","SBIN.NS","ITC.NS","HINDUNILVR.NS",
    "BHARTIARTL.NS","BAJFINANCE.NS","AXISBANK.NS","WIPRO.NS",
    "SUNPHARMA.NS","HCLTECH.NS","TITAN.NS","MARUTI.NS",
    "ULTRACEMCO.NS","NESTLEIND.NS","POWERGRID.NS","ADANIPORTS.NS"
]

CACHE_TTL = 60          # seconds for in-memory cache
HIST_DAYS_MIN = 260     # gather ~1 year history by default
MAX_RETRIES = 2         # yfinance retries

# PATH to CSV (auto-detect)
MARKET_CSV_PATH = os.environ.get("MARKET_CSV_PATH", "market_data.csv")

# -------------------- In-memory stores --------------------
_cache = {}             # simple TTL cache
_last_success = {}      # last successful analysis per symbol (fallback)

# -------------------- CSV loader --------------------
def load_market_csv(path=MARKET_CSV_PATH):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, dtype=str)  # load as string to avoid parse problems
        # normalize expected columns if present
        cols = {c.strip(): c for c in df.columns}
        # Common column name mapping
        mappings = {}
        for key in ("Symbol","SYMBOL","symbol"):
            if key in cols:
                mappings['Symbol'] = cols[key]
                break
        for key in ("Company Name","Company","Name","NAME"):
            if key in cols:
                mappings['Company Name'] = cols[key]
                break
        for key in ("LTP","Price","ltp","price"):
            if key in cols:
                mappings['LTP'] = cols[key]
                break
        for key in ("Change %","Change%","Change","change"):
            if key in cols:
                mappings['Change %'] = cols[key]
                break
        for key in ("High","high"):
            if key in cols:
                mappings['High'] = cols[key]
                break
        for key in ("Low","low"):
            if key in cols:
                mappings['Low'] = cols[key]
                break
        # Rename columns to canonical names when detected
        df2 = df.rename(columns={v:k for k,v in mappings.items()})
        # ensure canonical fields exist
        for c in ("Symbol","Company Name","LTP","Change %","High","Low"):
            if c not in df2.columns:
                df2[c] = None
        # add searchable lowercase fields
        df2['symbol_lower'] = df2['Symbol'].fillna("").astype(str).str.lower()
        df2['name_lower'] = df2['Company Name'].fillna("").astype(str).str.lower()
        return df2
    except Exception:
        traceback.print_exc()
        return None

MARKET_DF = load_market_csv()

# -------------------- Cache helpers --------------------
def cache_get(key):
    rec = _cache.get(key)
    if not rec:
        return None
    ts, value, ttl = rec
    if time.time() - ts > (ttl or CACHE_TTL):
        del _cache[key]
        return None
    return value

def cache_set(key, value, ttl=CACHE_TTL):
    _cache[key] = (time.time(), value, ttl)

# -------------------- Utilities --------------------
def normalize_symbol(sym):
    if not sym:
        return None
    s = sym.strip().upper()
    if s.endswith(".NS"):
        return s
    if s.isalpha() and len(s) <= 6:
        return s + ".NS"
    return s

def safe_div(a, b):
    try:
        return a / b
    except Exception:
        return None

def pretty_money(x):
    try:
        if x is None:
            return "N/A"
        return f"₹{float(x):,.2f}"
    except Exception:
        return str(x)

# -------------------- Technical helpers --------------------
def compute_sma(series, window):
    try:
        if series is None or len(series) < window:
            return None
        return float(series.rolling(window=window).mean().iloc[-1])
    except Exception:
        return None

def compute_rsi(series, period=14):
    try:
        if series is None or len(series) < period + 2:
            return None
        delta = series.diff().dropna()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.ewm(alpha=1/period, adjust=False).mean()
        roll_down = down.ewm(alpha=1/period, adjust=False).mean()
        rs = safe_div(roll_up, roll_down)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])
    except Exception:
        return None

def compute_volatility(series, window=20):
    try:
        if series is None or len(series) < window:
            return None
        returns = series.pct_change().dropna()
        vol = returns.rolling(window=window).std().iloc[-1]
        if pd.isna(vol):
            return None
        return float(vol * (252 ** 0.5))
    except Exception:
        return None

# -------------------- yfinance wrappers (robust) --------------------
def yf_history_with_retries(symbol, days=HIST_DAYS_MIN, retries=MAX_RETRIES):
    key = f"hist:{symbol}:{days}"
    cached = cache_get(key)
    if cached is not None:
        return cached
    exc = None
    for attempt in range(retries):
        try:
            tk = yf.Ticker(symbol)
            hist = tk.history(period=f"{days}d", interval="1d", auto_adjust=False)
            if hist is None or hist.empty:
                exc = None
                continue
            hist = hist.dropna(subset=["Close"])
            cache_set(key, hist, ttl=60*5)
            return hist
        except Exception as e:
            exc = e
            time.sleep(0.5)
    # final fallback: None (caller must use CSV or last success)
    if exc:
        # log quietly
        traceback.print_exc()
    return None

def yf_live_basic(symbol, retries=MAX_RETRIES):
    key = f"live:{symbol}"
    cached = cache_get(key)
    if cached is not None:
        return cached
    exc = None
    for attempt in range(retries):
        try:
            tk = yf.Ticker(symbol)
            # fetch recent 5 days to get last_close reliably
            hist = tk.history(period="5d", interval="1d")
            if hist is None or hist.empty:
                continue
            last_close = float(hist["Close"].iloc[-1])
            prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else None
            price = None
            try:
                # fast_info may exist
                fi = getattr(tk, "fast_info", None)
                if fi and "last_price" in fi:
                    price = fi["last_price"]
            except Exception:
                price = None
            if price is None:
                price = last_close
            volume = int(hist["Volume"].iloc[-1]) if "Volume" in hist.columns and not pd.isna(hist["Volume"].iloc[-1]) else None
            out = {"symbol": symbol, "price": price, "last_close": last_close, "prev_close": prev_close, "volume": volume}
            cache_set(key, out, ttl=30)
            return out
        except Exception as e:
            exc = e
            time.sleep(0.5)
    if exc:
        traceback.print_exc()
    return None

# -------------------- CSV lookup helpers --------------------
def csv_lookup_by_symbol(sym):
    if MARKET_DF is None:
        return None
    s = sym.lower()
    match = MARKET_DF[MARKET_DF['symbol_lower'] == s]
    if not match.empty:
        return match.iloc[0].to_dict()
    return None

def csv_lookup_by_name(name):
    if MARKET_DF is None:
        return None
    s = name.lower()
    match = MARKET_DF[MARKET_DF['name_lower'] == s]
    if not match.empty:
        return match.iloc[0].to_dict()
    return None

def csv_fuzzy_lookup(query, cutoff=0.6):
    if MARKET_DF is None:
        return None
    possibilities = list(MARKET_DF['symbol_lower'].astype(str).tolist()) + list(MARKET_DF['name_lower'].astype(str).tolist())
    close = difflib.get_close_matches(query.lower(), possibilities, n=1, cutoff=cutoff)
    if not close:
        return None
    val = close[0]
    match = MARKET_DF[(MARKET_DF['symbol_lower'] == val) | (MARKET_DF['name_lower'] == val)]
    if not match.empty:
        return match.iloc[0].to_dict()
    return None

# -------------------- Analysis and scoring --------------------
def analyze_symbol(symbol):
    """
    Tries to build a rich analysis dict:
     - prefer live yfinance + history
     - if some pieces fail, use CSV fallback (MARKET_DF) + last known
     - always return a 'confidence' field (0..1)
    """
    sym = normalize_symbol(symbol)
    if not sym:
        return None

    # Use cache for analysis
    key = f"analysis:{sym}"
    cached = cache_get(key)
    if cached:
        return cached

    # default structure
    analysis = {
        "symbol": sym,
        "price": None,
        "last_close": None,
        "week_change_pct": None,
        "sma50": None,
        "sma200": None,
        "rsi14": None,
        "volatility_annual": None,
        "avg_volume_20": None,
        "marketCap": None,
        "trailingPE": None,
        "dividendYield": None,
        "longName": None,
        "score": None,
        "confidence": 0.0,   # 0..1
        "source": []
    }

    # 1) attempt yfinance live & history
    hist = yf_history_with_retries(sym, days=HIST_DAYS_MIN)
    live = yf_live_basic(sym)
    used_yf = False
    if live:
        used_yf = True
        analysis['price'] = float(live.get("price")) if live.get("price") is not None else None
        analysis['last_close'] = float(live.get("last_close")) if live.get("last_close") is not None else None
        analysis['source'].append("yfinance:live")
    if hist is not None and not hist.empty:
        used_yf = True
        closes = hist["Close"]
        try:
            if len(closes) >= 6:
                recent = float(closes.iloc[-1])
                week_old = float(closes.iloc[-6])
                analysis['week_change_pct'] = safe_div((recent - week_old) * 100, week_old)
        except Exception:
            analysis['week_change_pct'] = None
        analysis['sma50'] = compute_sma(closes, 50)
        analysis['sma200'] = compute_sma(closes, 200)
        analysis['rsi14'] = compute_rsi(closes, 14)
        analysis['volatility_annual'] = compute_volatility(closes, window=20)
        try:
            if "Volume" in hist.columns and len(hist) >= 20:
                analysis['avg_volume_20'] = float(hist["Volume"].rolling(20).mean().iloc[-1])
        except Exception:
            analysis['avg_volume_20'] = None
        analysis['source'].append("yfinance:history")

    # 2) Try CSV fallback to fill missing fundamental/live fields
    csv_rec = csv_lookup_by_symbol(sym) or csv_fuzzy_lookup(sym.replace(".NS",""))
    if csv_rec:
        # set only missing fields with the CSV's values (best-effort parsing)
        try:
            if analysis['longName'] is None:
                analysis['longName'] = csv_rec.get("Company Name") or csv_rec.get("Company") or csv_rec.get("Name")
            if analysis['price'] is None:
                # try multiple possible column names
                for col in ("LTP","Price","ltp","price"):
                    if col in csv_rec and csv_rec.get(col) not in (None,"", "NaN"):
                        try:
                            analysis['price'] = float(str(csv_rec.get(col)).replace(",",""))
                        except Exception:
                            pass
            if analysis['week_change_pct'] is None:
                for col in ("Change %","Change%","Change"):
                    if col in csv_rec and csv_rec.get(col) not in (None,"", "NaN"):
                        try:
                            v = str(csv_rec.get(col)).replace("%","").replace(",","")
                            analysis['week_change_pct'] = float(v)
                        except Exception:
                            pass
            # High/Low as fallback into sma maybe not ideal so we skip
            analysis['source'].append("market_data.csv")
        except Exception:
            pass

    # 3) If no yfinance at all, but we have last_success fallback, use it
    if not used_yf and sym in _last_success:
        last = _last_success[sym]
        # merge fields that are missing
        for k,v in last.items():
            if analysis.get(k) in (None, ""):
                analysis[k] = v
        analysis['source'].append("last_success_cache")

    # 4) Compute composite score if possible
    try:
        momentum = (analysis.get('week_change_pct') or 0.0)
        vol = analysis.get('volatility_annual')
        stability = None if vol is None else (1 / (vol + 1e-9))
        avg_vol = analysis.get('avg_volume_20')
        liquidity = None if avg_vol is None or avg_vol <= 0 else math.log(avg_vol + 1)
        momentum_score = max(-10, min(10, momentum)) * 5
        stability_score = (stability or 0.0) * 50
        liquidity_score = (liquidity or 0.0) * 3
        composite = momentum_score * 0.6 + stability_score * 0.3 + liquidity_score * 0.1
        analysis['score'] = composite
    except Exception:
        analysis['score'] = None

    # 5) Confidence scoring (simple heuristic)
    conf = 0.0
    if used_yf:
        conf += 0.6
    if csv_rec is not None:
        conf += 0.35
    if analysis.get('price') is not None:
        conf += 0.05
    # clamp 0..1
    analysis['confidence'] = min(1.0, conf)
    # store last success if confidence decent
    if analysis['confidence'] >= 0.35:
        _last_success[sym] = analysis.copy()

    cache_set(key, analysis, ttl=30)
    return analysis

def rank_investment_candidates(limit=5):
    analyses = []
    for s in CANDIDATES:
        try:
            a = analyze_symbol(s)
            if a:
                analyses.append(a)
        except Exception:
            continue
    if not analyses:
        return []
    analyses.sort(key=lambda x: x.get("score", -9999), reverse=True)
    return analyses[:limit]

# -------------------- Natural language helpers --------------------
def classify(q):
    ql = q.lower()
    if any(x in ql for x in ["hi","hello","hey","namaste"]):
        return "greet"
    if any(x in ql for x in ["how are", "how's", "how is"]):
        return "howare"
    if any(x in ql for x in ["top", "best", "recommend", "picks", "suggest"]):
        return "recommend"
    if " vs " in ql or " vs. " in ql:
        return "compare"
    if any(x in ql for x in ["price", "rate", "value", "trading at", "current price", "ltp", "last price"]):
        return "price"
    if any(x in ql for x in ["rsi", "sma", "moving average", "pe ratio", "pe", "market cap", "dividend", "explain", "what is"]):
        return "explain"
    # single token like TCS
    tokens = q.strip().split()
    if len(tokens) == 1 and tokens[0].isalpha() and len(tokens[0]) <= 6:
        return "price"
    return "generic"

def extract_symbols_from_text(q):
    tokens = [t.strip(",.?!()").upper() for t in q.split()]
    syms = []
    for t in tokens:
        if t.endswith(".NS"):
            syms.append(normalize_symbol(t))
        elif t.isalpha() and len(t) <= 6:
            syms.append(normalize_symbol(t))
        if len(syms) >= 2:
            break
    return syms

def fuzzy_symbol_or_name_match(q):
    # try exact CSV symbol, exact CSV name, fuzzy CSV name, or normalize token
    if not q:
        return None
    # tokens attempt
    tokens = q.split()
    # check for tokens like 'price of TCS' -> pick TCS
    for tok in tokens[::-1]:
        tok_clean = tok.strip(",.?!").upper()
        if tok_clean.endswith(".NS") or (tok_clean.isalpha() and len(tok_clean) <= 6):
            return normalize_symbol(tok_clean)
    # exact CSV match
    if MARKET_DF is not None:
        by_sym = csv_lookup_by_symbol(q)
        if by_sym:
            return normalize_symbol(by_sym.get("Symbol") or q)
        by_name = csv_lookup_by_name(q)
        if by_name:
            return normalize_symbol(by_name.get("Symbol") or q)
        fuzzy = csv_fuzzy_lookup(q)
        if fuzzy:
            return normalize_symbol(fuzzy.get("Symbol") or q)
    return None

# -------------------- Human-friendly response builders --------------------
_EXPLAINERS = {
    "rsi": "RSI (Relative Strength Index) is a momentum indicator from 0–100. Above 70 often signals overbought; below 30 may indicate oversold — use with trend context.",
    "sma": "SMA (Simple Moving Average) smooths price over a period. SMA50 is medium-term, SMA200 long-term. Crossovers can signal trend shifts.",
    "pe": "P/E = Price ÷ Earnings-per-share. It shows how much investors pay for each unit of earnings; compare within sector.",
    "marketcap": "Market cap = current price × outstanding shares; a quick gauge of company size.",
    "dividend": "Dividend yield = annual dividend ÷ current price; it shows cash return relative to price."
}

def quick_verdict(analysis):
    # built-in conservative, explainable suggestion
    score = analysis.get("score") or 0
    rsi = analysis.get("rsi14")
    conf = analysis.get("confidence", 0.0)
    if conf < 0.35:
        return "I have limited data for this symbol — I recommend verifying with your broker or another data source."
    if score > 25 and (rsi is None or rsi < 75):
        return "Bullish momentum with relative stability — consider further research for a buy (risk-managed)."
    if score > 5:
        return "Positive momentum but check valuation and volumes — consider hold or a small watchlist buy."
    if score > -10:
        return "Mixed signals — holding or caution advised; wait for clearer trend or improved fundamentals."
    return "Weak momentum and/or high volatility — probably not a favorable entry now."

def build_price_reply(analysis):
    # produce a confident, friendly paragraph reply
    if not analysis:
        return "I couldn't find useful data for that symbol."
    name = analysis.get("longName") or analysis.get("symbol")
    symbol = analysis.get("symbol")
    price = analysis.get("price")
    week = analysis.get("week_change_pct")
    rsi = analysis.get("rsi14")
    sma50 = analysis.get("sma50")
    sma200 = analysis.get("sma200")
    marketCap = analysis.get("marketCap")
    pe = analysis.get("trailingPE")
    conf = analysis.get("confidence", 0.0)
    src = ", ".join(analysis.get("source", [])) or "unknown"

    lines = []
    # headline
    if price is not None:
        lines.append(f"🔎 {name} ({symbol}) — {pretty_money(price)} (confidence {conf*100:.0f}%)")
    else:
        lines.append(f"🔎 {name} ({symbol}) — price not available (confidence {conf*100:.0f}%)")

    # quick stats
    if week is not None:
        lines.append(f"• 1W change: {week:+.2f}%")
    if rsi is not None:
        rsi_tag = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
        lines.append(f"• RSI(14): {rsi:.1f} ({rsi_tag})")
    if sma50 is not None and sma200 is not None:
        lines.append(f"• SMA50 / SMA200: {pretty_money(sma50)} / {pretty_money(sma200)}")
    if marketCap:
        try:
            mc = int(marketCap)
            # pretty print (crore/ lakh? we'll show raw with commas)
            lines.append(f"• Market Cap: {mc:,}")
        except Exception:
            lines.append(f"• Market Cap: {marketCap}")
    if pe:
        lines.append(f"• P/E (trailing): {pe}")

    # short interpretation
    verdict = quick_verdict(analysis)
    lines.append("")
    lines.append("📝 Quick view: " + verdict)

    # sources & rationale
    lines.append(f"\nData sources: {src}. I combine historical indicators and available CSV metadata.")
    lines.append("Not financial advice — do your own research and consider risks.")

    return "\n".join(lines)

# -------------------- Flask route --------------------
@bp.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        q = (data.get("q","") or "").strip()
        if not q:
            return jsonify({"answer": "Hi — ask me about any NSE stock (e.g., 'TCS price', 'Top picks', 'RELIANCE vs TCS', 'Explain RSI')."})

        intent = classify(q)

        # greetings
        if intent == "greet":
            return jsonify({"answer": "Hey — I'am Brocode AI. Ask about any NSE stock or request top picks. I'll give a clear, confidence-rated reply."})
        if intent == "howare":
            return jsonify({"answer": "Doing great — scanning markets and your CSV knowledge base. How can I help?"})

        # recommend / top picks
        if intent == "recommend":
            picks = rank_investment_candidates(limit=5)
            if not picks:
                # fallback: if MARKET_DF present, propose top by Change %
                if MARKET_DF is not None and "Change %" in MARKET_DF.columns:
                    try:
                        tmp = MARKET_DF.copy()
                        tmp["ChangeNum"] = tmp["Change %"].astype(str).str.replace("%","").str.replace(",","")
                        tmp["ChangeNum"] = pd.to_numeric(tmp["ChangeNum"], errors='coerce').fillna(0)
                        top = tmp.sort_values("ChangeNum", ascending=False).head(5)
                        lines = ["Top movers from CSV (fallback):"]
                        for _, r in top.iterrows():
                            lines.append(f"{r.get('Symbol','?')} — {r.get('Company Name','')} | {r.get('LTP','N/A')} | {r.get('Change %','N/A')}")
                        return jsonify({"answer":"\n".join(lines)})
                    except Exception:
                        pass
                return jsonify({"answer": "I couldn't compute top picks right now — but I can still answer single-stock queries from your CSV."})
            lines = ["Top picks (momentum + stability score):"]
            for p in picks:
                lines.append(f"{p['symbol']} — {pretty_money(p.get('price'))} | 1W {p.get('week_change_pct', 0):+,.2f}% | Score {p.get('score',0):.1f} | Confidence {p.get('confidence',0)*100:.0f}%")
            lines.append("\nRationale: score blends recent momentum, volatility (stability), and liquidity. Not financial advice.")
            return jsonify({"answer": "\n".join(lines)})

        # compare two symbols
        if intent == "compare":
            syms = extract_symbols_from_text(q)
            if len(syms) < 2:
                # try fuzzy extraction
                # split by ' vs '
                if " vs " in q.lower():
                    parts = q.lower().split(" vs ")
                    left = fuzzy_symbol_or_name_match(parts[0])
                    right = fuzzy_symbol_or_name_match(parts[1])
                    if left and right:
                        syms = [left, right]
                if len(syms) < 2:
                    return jsonify({"answer": "Please give two tickers to compare, e.g. 'RELIANCE vs TCS'."})
            left = analyze_symbol(syms[0])
            right = analyze_symbol(syms[1])
            if not left and not right:
                return jsonify({"answer": "I couldn't get data for either symbol. Check spelling or ensure CSV has them."})
            # craft comparison
            lines = [f"Comparing {syms[0]} vs {syms[1]}:"]
            if left:
                lines.append(f"{left.get('symbol')} — {pretty_money(left.get('price'))} | 1W {left.get('week_change_pct',0):+,.2f}% | RSI {left.get('rsi14') or 'N/A'} | Conf {left.get('confidence',0)*100:.0f}%")
            else:
                lines.append(f"{syms[0]} — data not available.")
            if right:
                lines.append(f"{right.get('symbol')} — {pretty_money(right.get('price'))} | 1W {right.get('week_change_pct',0):+,.2f}% | RSI {right.get('rsi14') or 'N/A'} | Conf {right.get('confidence',0)*100:.0f}%")
            else:
                lines.append(f"{syms[1]} — data not available.")
            # quick verdict
            if left and right and left.get('score') is not None and right.get('score') is not None:
                if left['score'] > right['score'] + 5:
                    lines.append(f"\nVerdict: {left['symbol']} shows stronger momentum/stability currently.")
                elif right['score'] > left['score'] + 5:
                    lines.append(f"\nVerdict: {right['symbol']} shows stronger momentum/stability currently.")
                else:
                    lines.append("\nVerdict: Similar scores — choose using valuation, sector, or risk appetite.")
            lines.append("\nNot financial advice.")
            return jsonify({"answer": "\n".join(lines)})

        # explainers
        if intent == "explain":
            ql = q.lower()
            if "rsi" in ql:
                return jsonify({"answer": _EXPLAINERS["rsi"]})
            if "sma" in ql or "moving average" in ql:
                return jsonify({"answer": _EXPLAINERS["sma"]})
            if "pe" in ql:
                return jsonify({"answer": _EXPLAINERS["pe"]})
            if "market cap" in ql:
                return jsonify({"answer": _EXPLAINERS["marketcap"]})
            if "dividend" in ql:
                return jsonify({"answer": _EXPLAINERS["dividend"]})
            return jsonify({"answer": "Ask something like: 'Explain RSI', 'What is SMA50?', or 'What is P/E?'"})

        # price / single symbol
        if intent == "price" or intent == "generic":
            # extract best possible symbol
            sym = fuzzy_symbol_or_name_match(q) or (extract_symbols_from_text(q) or [None])[0]
            if not sym:
                return jsonify({"answer": f"Tell me a stock symbol or company name (e.g., 'TCS', 'Reliance') — I can answer using your CSV or live data."})
            analysis = analyze_symbol(sym)
            if not analysis:
                # CSV-only fallback message
                csv_rec = csv_lookup_by_symbol(sym) or csv_fuzzy_lookup(sym.replace(".NS",""))
                if csv_rec:
                    # craft CSV-based reply with friendly tone
                    comp = csv_rec.get("Company Name") or sym
                    price = csv_rec.get("LTP") or csv_rec.get("Price")
                    change = csv_rec.get("Change %")
                    reply = f"📌 {comp} ({sym}) — Price from CSV: {price or 'N/A'} | Change: {change or 'N/A'}\nNote: live indicators not available; consider verifying with a live feed."
                    return jsonify({"answer": reply})
                return jsonify({"answer": "I couldn't find data for that symbol. Check spelling or your CSV."})
            # build human reply
            answer = build_price_reply(analysis)
            return jsonify({"answer": answer})

        # fallback
        return jsonify({"answer": ("I can answer live prices, indicators, comparisons, and top picks.\n"
                                   "Try: 'TCS price', 'Top picks', 'RELIANCE vs TCS', or 'Explain RSI'.")})

    except Exception:
        traceback.print_exc()
        return jsonify({"answer": "I hit an unexpected error while processing your request — please try again or simplify the query."})