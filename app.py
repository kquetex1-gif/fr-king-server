import os
import requests
import pandas as pd
import numpy as np

from flask import Flask, jsonify, request

app = Flask(__name__)

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")

# =========================
# FR-KING SETTINGS
# =========================

INTERVAL = "1min"
OUTPUTSIZE = 120

ST1_PERIOD = 10
ST1_MULTIPLIER = 3.0

ST2_PERIOD = 10
ST2_MULTIPLIER = 1.0

# Auto MTG intentionally OFF
AUTO_MTG = False


# =========================
# TWELVE DATA
# =========================

def get_candles(symbol):
    if not TWELVE_DATA_API_KEY:
        raise RuntimeError("TWELVE_DATA_API_KEY is missing")

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "outputsize": OUTPUTSIZE,
        "apikey": TWELVE_DATA_API_KEY,
        "format": "JSON"
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()

    data = response.json()

    if data.get("status") == "error":
        raise RuntimeError(data.get("message", "Twelve Data error"))

    values = data.get("values")

    if not values:
        raise RuntimeError("No candle data received")

    df = pd.DataFrame(values)

    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["datetime"] = pd.to_datetime(df["datetime"])

    # Twelve Data sends newest candle first
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


# =========================
# ATR
# =========================

def atr(df, period=10):
    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - previous_close).abs()
    tr3 = (df["low"] - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    # Wilder-style ATR
    return true_range.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# =========================
# SUPERTREND
# =========================

def supertrend(df, period, multiplier):
    result = df.copy()

    atr_value = atr(result, period)

    hl2 = (result["high"] + result["low"]) / 2

    basic_upper = hl2 + multiplier * atr_value
    basic_lower = hl2 - multiplier * atr_value

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    direction = pd.Series(
        1,
        index=result.index,
        dtype=int
    )

    st = pd.Series(
        np.nan,
        index=result.index,
        dtype=float
    )

    for i in range(1, len(result)):

        if (
            basic_upper.iloc[i] < final_upper.iloc[i - 1]
            or result["close"].iloc[i - 1] > final_upper.iloc[i - 1]
        ):
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if (
            basic_lower.iloc[i] > final_lower.iloc[i - 1]
            or result["close"].iloc[i - 1] < final_lower.iloc[i - 1]
        ):
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        if direction.iloc[i - 1] == 1:

            if result["close"].iloc[i] < final_lower.iloc[i]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = 1

        else:

            if result["close"].iloc[i] > final_upper.iloc[i]:
                direction.iloc[i] = 1
            else:
                direction.iloc[i] = -1

        if direction.iloc[i] == 1:
            st.iloc[i] = final_lower.iloc[i]
        else:
            st.iloc[i] = final_upper.iloc[i]

    return st, direction


# =========================
# FRACTAL PERIOD 2
# =========================

def add_fractals(df):
    df = df.copy()

    df["up_fractal"] = False
    df["down_fractal"] = False

    # Period 2:
    # fractal candle needs 2 candles left + 2 candles right

    for i in range(2, len(df) - 2):

        high = df["high"].iloc[i]
        low = df["low"].iloc[i]

        # Upper / bearish fractal
        if (
            high > df["high"].iloc[i - 1]
            and high > df["high"].iloc[i - 2]
            and high > df["high"].iloc[i + 1]
            and high > df["high"].iloc[i + 2]
        ):
            df.loc[df.index[i], "up_fractal"] = True

        # Lower / bullish fractal
        if (
            low < df["low"].iloc[i - 1]
            and low < df["low"].iloc[i - 2]
            and low < df["low"].iloc[i + 1]
            and low < df["low"].iloc[i + 2]
        ):
            df.loc[df.index[i], "down_fractal"] = True

    return df


# =========================
# STRATEGY
# =========================

def analyze(symbol):
    df = get_candles(symbol)

    df["st_10_3"], df["dir_10_3"] = supertrend(
        df,
        ST1_PERIOD,
        ST1_MULTIPLIER
    )

    df["st_10_1"], df["dir_10_1"] = supertrend(
        df,
        ST2_PERIOD,
        ST2_MULTIPLIER
    )

    df = add_fractals(df)

    # Ignore newest possibly still-forming candle.
    closed_last = len(df) - 2

    candidates = []

    for fractal_index in range(2, closed_last - 2):

        bullish_fractal = bool(
            df["down_fractal"].iloc[fractal_index]
        )

        bearish_fractal = bool(
            df["up_fractal"].iloc[fractal_index]
        )

        if not bullish_fractal and not bearish_fractal:
            continue

        # Fractal period 2 confirms after two right candles.
        confirmation_index = fractal_index + 2

        # Count AFTER confirmation:
        # #1 = confirmation + 1
        # #2 = confirmation + 2
        # #3 = confirmation + 3
        # #4 = confirmation + 4 (ENTRY)
        entry_index = confirmation_index + 4

        if entry_index >= len(df):
            continue

        candidates.append(
            (
                fractal_index,
                confirmation_index,
                entry_index,
                bullish_fractal,
                bearish_fractal
            )
        )

    if not candidates:
        return {
            "symbol": symbol,
            "signal": "WAIT",
            "reason": "No confirmed Fractal-2 setup",
            "auto_mtg": False
        }

    (
        fractal_index,
        confirmation_index,
        entry_index,
        bullish_fractal,
        bearish_fractal
    ) = candidates[-1]

    # LIVE MODE: only act when Candle #4 is the latest CLOSED candle.
    # This prevents an old historical fractal setup from being returned
    # repeatedly as the current signal.
    if entry_index != closed_last:
        return {
            "system": "FR-KING",
            "symbol": symbol,
            "timeframe": "1 minute",
            "signal": "WAIT",
            "reason": "No current Candle #4 setup",
            "fractal_period": 2,
            "auto_mtg": False,
            "mtg": "MANUAL ONLY"
        }

    entry = df.iloc[entry_index]

    st1 = float(entry["st_10_3"])
    st2 = float(entry["st_10_1"])

    low_zone = min(st1, st2)
    high_zone = max(st1, st2)

    # Candle #4 must interact with / sit inside ST zone.
    valid_area = (
        entry["high"] >= low_zone
        and entry["low"] <= high_zone
    )

    if not valid_area:
        signal = "AVOID"
        reason = "Candle #4 outside Supertrend zone"

    elif bullish_fractal:
        signal = "BUY"
        reason = "Bullish Fractal 2 + valid ST zone"

    elif bearish_fractal:
        signal = "SELL"
        reason = "Bearish Fractal 2 + valid ST zone"

    else:
        signal = "WAIT"
        reason = "No setup"

    return {
        "system": "FR-KING",
        "symbol": symbol,
        "timeframe": "1 minute",

        "signal": signal,
        "reason": reason,

        "fractal_period": 2,

        "fractal_time":
            str(df["datetime"].iloc[fractal_index]),

        "confirmation_time":
            str(df["datetime"].iloc[confirmation_index]),

        "candle_4_entry_time":
            str(df["datetime"].iloc[entry_index]),

        "candle_4_open":
            float(entry["open"]),

        "supertrend_10_3":
            st1,

        "supertrend_10_1":
            st2,

        "valid_area":
            bool(valid_area),

        "auto_mtg":
            False,

        "mtg":
            "MANUAL ONLY"
    }


# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return jsonify({
        "status": "ONLINE",
        "system": "FR-KING",
        "strategy": "Fractal 2 + ST 10x3 + ST 10x1",
        "timeframe": "1 minute",
        "entry": "Candle #4",
        "auto_mtg": False
    })


@app.route("/signal")
def signal():
    symbol = request.args.get(
        "symbol",
        "EUR/USD"
    ).upper()

    try:
        result = analyze(symbol)
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "system": "FR-KING",
            "symbol": symbol,
            "signal": "ERROR",
            "error": str(e)
        }), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "OK",
        "api_key_loaded":
            bool(TWELVE_DATA_API_KEY)
    })


if __name__ == "__main__":
    port = int(
        os.environ.get("PORT", 10000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
