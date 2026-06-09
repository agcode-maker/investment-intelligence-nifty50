"""
indicators.py — Technical indicator computation for NIFTY-50 stocks.

All functions take a DataFrame with at least a 'Close' column
(and optionally 'High', 'Low', 'Volume') and return the same
DataFrame with new indicator columns added.
"""

import pandas as pd
import numpy as np


# ── Moving Averages ──────────────────────────────────────────────────────────

def add_sma(df: pd.DataFrame, windows: list = [20, 50, 200]) -> pd.DataFrame:
    """Simple Moving Averages."""
    df = df.copy()
    for w in windows:
        df[f"SMA_{w}"] = df["Close"].rolling(window=w).mean()
    return df

def add_ema(df: pd.DataFrame, windows: list = [12, 26, 50]) -> pd.DataFrame:
    """Exponential Moving Averages."""
    df = df.copy()
    for w in windows:
        df[f"EMA_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()
    return df


# ── Momentum Indicators ──────────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index (RSI)."""
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD, Signal Line, and Histogram."""
    df = df.copy()
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df

def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Stochastic Oscillator (%K and %D)."""
    df = df.copy()
    low_min = df["Low"].rolling(window=k_period).min()
    high_max = df["High"].rolling(window=k_period).max()
    df["Stoch_K"] = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    df["Stoch_D"] = df["Stoch_K"].rolling(window=d_period).mean()
    return df

def add_roc(df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
    """Rate of Change (momentum)."""
    df = df.copy()
    df[f"ROC_{period}"] = df["Close"].pct_change(periods=period) * 100
    return df

def add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Williams %R."""
    df = df.copy()
    high_max = df["High"].rolling(window=period).max()
    low_min = df["Low"].rolling(window=period).min()
    df["Williams_R"] = -100 * (high_max - df["Close"]) / (high_max - low_min).replace(0, np.nan)
    return df


# ── Volatility Indicators ────────────────────────────────────────────────────

def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: upper, middle (SMA), lower, and %B."""
    df = df.copy()
    sma = df["Close"].rolling(window=window).mean()
    std = df["Close"].rolling(window=window).std()
    df["BB_Upper"] = sma + num_std * std
    df["BB_Middle"] = sma
    df["BB_Lower"] = sma - num_std * std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    df["BB_Pct"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)
    return df

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range (ATR)."""
    df = df.copy()
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.ewm(com=period - 1, min_periods=period).mean()
    return df

def add_historical_volatility(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Annualised historical volatility (std of log returns)."""
    df = df.copy()
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    df[f"HV_{window}"] = log_ret.rolling(window=window).std() * np.sqrt(252) * 100
    return df


# ── Volume Indicators ────────────────────────────────────────────────────────

def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """On-Balance Volume."""
    df = df.copy()
    direction = np.sign(df["Close"].diff()).fillna(0)
    df["OBV"] = (direction * df["Volume"]).cumsum()
    return df

def add_vwap_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """Deviation of Close from VWAP (if VWAP column available)."""
    df = df.copy()
    if "VWAP" in df.columns:
        df["VWAP_Dev"] = (df["Close"] - df["VWAP"]) / df["VWAP"] * 100
    return df

def add_volume_sma(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Volume relative to its moving average."""
    df = df.copy()
    df["Volume_SMA"] = df["Volume"].rolling(window=window).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"]
    return df


# ── Trend Indicators ─────────────────────────────────────────────────────────

def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index (ADX) — trend strength."""
    df = df.copy()
    high, low, close = df["High"], df["Low"], df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(com=period - 1, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(com=period - 1, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(com=period - 1, min_periods=period).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX"] = dx.ewm(com=period - 1, min_periods=period).mean()
    df["Plus_DI"] = plus_di
    df["Minus_DI"] = minus_di
    return df

def add_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """Ichimoku Cloud components."""
    df = df.copy()
    nine_high = df["High"].rolling(9).max()
    nine_low = df["Low"].rolling(9).min()
    df["Ichimoku_Conv"] = (nine_high + nine_low) / 2

    twenty_six_high = df["High"].rolling(26).max()
    twenty_six_low = df["Low"].rolling(26).min()
    df["Ichimoku_Base"] = (twenty_six_high + twenty_six_low) / 2

    df["Ichimoku_SpanA"] = ((df["Ichimoku_Conv"] + df["Ichimoku_Base"]) / 2).shift(26)

    fifty_two_high = df["High"].rolling(52).max()
    fifty_two_low = df["Low"].rolling(52).min()
    df["Ichimoku_SpanB"] = ((fifty_two_high + fifty_two_low) / 2).shift(26)
    return df


# ── Feature Engineering for ML ───────────────────────────────────────────────

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag returns, rolling stats — useful ML features."""
    df = df.copy()
    df["Return_1d"] = df["Close"].pct_change(1)
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)
    df["Return_20d"] = df["Close"].pct_change(20)

    df["High_Low_Range"] = (df["High"] - df["Low"]) / df["Close"]
    df["Open_Close_Range"] = (df["Close"] - df["Open"]) / df["Open"]

    for w in [5, 10, 20]:
        df[f"Rolling_Std_{w}"] = df["Return_1d"].rolling(w).std()
        df[f"Rolling_Mean_{w}"] = df["Return_1d"].rolling(w).mean()

    df["Gap_Up"] = (df["Open"] > df["Close"].shift(1)).astype(int)
    df["Gap_Down"] = (df["Open"] < df["Close"].shift(1)).astype(int)
    return df

def add_target(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Add target columns for ML:
    - Future_Return: % return over next `horizon` days
    - Direction: 1 if price goes up, 0 if down
    """
    df = df.copy()
    future_price = df["Close"].shift(-horizon)
    df[f"Future_Return_{horizon}d"] = (future_price - df["Close"]) / df["Close"] * 100
    df[f"Direction_{horizon}d"] = (future_price > df["Close"]).astype(int)
    return df


# ── Master function ───────────────────────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame, target_horizon: int = 5) -> pd.DataFrame:
    """
    Apply all indicators and ML features in one call.
    Drops rows with NaN in key indicator columns.
    """
    df = add_sma(df)
    df = add_ema(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_historical_volatility(df)
    df = add_stochastic(df)
    df = add_roc(df)
    df = add_williams_r(df)
    df = add_obv(df)
    df = add_vwap_deviation(df)
    df = add_volume_sma(df)
    df = add_adx(df)
    df = add_price_features(df)
    df = add_target(df, horizon=target_horizon)
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from src.data_loader import load_stock

    df = load_stock("RELIANCE")
    df = add_all_indicators(df)
    print(df.shape)
    print(df.columns.tolist())
    print(df.tail(3))