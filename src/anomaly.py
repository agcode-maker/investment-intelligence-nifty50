"""
anomaly.py — Market Anomaly Detection Module
Optional Task C: Identify unusual patterns in historical market data.

Three detection methods:
  1. Volatility spikes     — z-score on rolling realised volatility
  2. Extreme drawdowns     — single-day returns below threshold
  3. Volume anomalies      — z-score on daily trading volume
  4. Isolation Forest      — multivariate unsupervised detection

All functions accept a standard stock DataFrame (output of load_stock)
and return the same DataFrame with new anomaly flag columns appended.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ── Statistical detectors ─────────────────────────────────────────────────────

def detect_volatility_spikes(df: pd.DataFrame,
                              window: int = 21,
                              threshold_sigma: float = 2.5) -> pd.DataFrame:
    """
    Flag days where realised volatility is >threshold_sigma std devs
    above its rolling mean (z-score based detection).

    New columns: RollingVol, Vol_Zscore, VolSpike (0/1)
    """
    df = df.copy()
    ret = df['Close'].pct_change()
    df['RollingVol'] = ret.rolling(window).std() * np.sqrt(252) * 100
    roll_mean = df['RollingVol'].rolling(window * 3).mean()
    roll_std  = df['RollingVol'].rolling(window * 3).std()
    df['Vol_Zscore'] = (df['RollingVol'] - roll_mean) / roll_std.replace(0, np.nan)
    df['VolSpike']   = (df['Vol_Zscore'] > threshold_sigma).astype(int)
    return df


def detect_extreme_drawdowns(df: pd.DataFrame,
                               daily_threshold_pct: float = -5.0,
                               sustained_threshold_pct: float = -20.0) -> pd.DataFrame:
    """
    Flag two types of drawdown anomalies:
      - ExtremeDrawdown: single-day return below daily_threshold_pct
      - DeepDrawdown:    cumulative drawdown from peak below sustained_threshold_pct

    New columns: DailyReturn, ExtremeDrawdown (0/1), Drawdown_pct, DeepDrawdown (0/1)
    """
    df = df.copy()
    df['DailyReturn']    = df['Close'].pct_change() * 100
    df['ExtremeDrawdown'] = (df['DailyReturn'] < daily_threshold_pct).astype(int)

    roll_max = df['Close'].cummax()
    df['Drawdown_pct'] = (df['Close'] - roll_max) / roll_max * 100
    df['DeepDrawdown'] = (df['Drawdown_pct'] < sustained_threshold_pct).astype(int)
    return df


def detect_volume_anomalies(df: pd.DataFrame,
                              window: int = 20,
                              threshold_sigma: float = 2.5) -> pd.DataFrame:
    """
    Flag days where trading volume is unusually high vs rolling baseline.

    New columns: VolumeSMA, Volume_Zscore, VolumeSpike (0/1)
    """
    df = df.copy()
    df['VolumeSMA']      = df['Volume'].rolling(window).mean()
    vol_std              = df['Volume'].rolling(window).std()
    df['Volume_Zscore']  = (df['Volume'] - df['VolumeSMA']) / vol_std.replace(0, np.nan)
    df['VolumeSpike']    = (df['Volume_Zscore'] > threshold_sigma).astype(int)
    return df


# ── Multivariate: Isolation Forest ───────────────────────────────────────────

def isolation_forest_anomalies(df: pd.DataFrame,
                                 contamination: float = 0.02) -> pd.DataFrame:
    """
    Multivariate anomaly detection using Isolation Forest.
    Each trading day is a point in 7-dimensional feature space.
    The most isolated points (contamination fraction) are flagged as anomalies.

    contamination: expected fraction of anomalous days (default 2%)

    New columns: IF_Anomaly (0/1), IF_Score (more negative = more anomalous)
    """
    from src.indicators import add_all_indicators

    df = df.copy()
    df = add_all_indicators(df)

    # Compute auxiliary features needed
    df['DailyReturn']   = df['Close'].pct_change() * 100
    df['VolumeSMA']     = df['Volume'].rolling(20).mean()
    df['VolumeStd']     = df['Volume'].rolling(20).std()
    df['Volume_Zscore'] = (df['Volume'] - df['VolumeSMA']) / df['VolumeStd'].replace(0, np.nan)
    df['RollingVol']    = df['DailyReturn'].rolling(21).std() * np.sqrt(252)

    feature_cols = [
        'DailyReturn', 'Volume_Zscore', 'RollingVol',
        'High_Low_Range', 'BB_Width', 'RSI', 'ATR'
    ]
    feat = [c for c in feature_cols if c in df.columns]
    X    = df[feat].dropna()

    sc      = StandardScaler()
    X_sc    = sc.fit_transform(X)

    iso     = IsolationForest(contamination=contamination,
                               random_state=42, n_estimators=200, n_jobs=-1)
    preds   = iso.fit_predict(X_sc)   # -1 = anomaly, 1 = normal
    scores  = iso.score_samples(X_sc)

    df.loc[X.index, 'IF_Anomaly'] = (preds == -1).astype(int)
    df.loc[X.index, 'IF_Score']   = scores
    df['IF_Anomaly'] = df['IF_Anomaly'].fillna(0).astype(int)
    df['IF_Score']   = df['IF_Score'].fillna(0)
    return df


# ── Master detector ───────────────────────────────────────────────────────────

def detect_all_anomalies(df: pd.DataFrame,
                          vol_sigma: float = 2.5,
                          drop_threshold: float = -5.0,
                          volume_sigma: float = 2.5) -> pd.DataFrame:
    """
    Apply all three statistical detectors and combine into one DataFrame.
    Also adds AnyAnomaly flag (1 if any detector fires).

    Does NOT include Isolation Forest (call separately — it's slower).
    """
    df = detect_volatility_spikes(df, threshold_sigma=vol_sigma)
    df = detect_extreme_drawdowns(df, daily_threshold_pct=drop_threshold)
    df = detect_volume_anomalies(df, threshold_sigma=volume_sigma)
    df['AnyAnomaly'] = (
        (df['VolSpike'] == 1) |
        (df['ExtremeDrawdown'] == 1) |
        (df['VolumeSpike'] == 1)
    ).astype(int)
    return df


# ── Anomaly summary ───────────────────────────────────────────────────────────

def anomaly_summary(df: pd.DataFrame, symbol: str = "") -> dict:
    """
    Return a summary dict of anomaly statistics for a stock.
    df must already have detect_all_anomalies applied.
    """
    total   = len(df)
    by_year = {}
    if 'AnyAnomaly' in df.columns and 'Date' in df.columns:
        df['_year'] = pd.to_datetime(df['Date']).dt.year
        by_year = df[df['AnyAnomaly'] == 1].groupby('_year').size().to_dict()
        df.drop(columns=['_year'], inplace=True)

    return {
        "symbol":              symbol,
        "total_days":          total,
        "vol_spikes":          int(df.get('VolSpike',   pd.Series([0])).sum()),
        "extreme_drops":       int(df.get('ExtremeDrawdown', pd.Series([0])).sum()),
        "volume_spikes":       int(df.get('VolumeSpike', pd.Series([0])).sum()),
        "any_anomaly":         int(df.get('AnyAnomaly', pd.Series([0])).sum()),
        "anomaly_rate_pct":    round(df.get('AnyAnomaly', pd.Series([0])).sum() / total * 100, 2),
        "anomalies_by_year":   by_year,
    }


# ── Convenience: universe-wide scan ──────────────────────────────────────────

def scan_universe(symbols: list = None) -> pd.DataFrame:
    """
    Run detect_all_anomalies on every stock and return a summary DataFrame.
    """
    from src.data_loader import load_stock, get_symbol_list, get_sector_map

    if symbols is None:
        symbols = get_symbol_list()
    sector_map = get_sector_map()

    rows = []
    for sym in symbols:
        try:
            df = load_stock(sym)
            df = detect_all_anomalies(df)
            s  = anomaly_summary(df, sym)
            s['sector'] = sector_map.get(sym, 'Unknown')
            rows.append(s)
        except Exception as e:
            rows.append({"symbol": sym, "error": str(e)})

    return pd.DataFrame(rows).set_index("symbol")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from src.data_loader import load_stock

    sym = "RELIANCE"
    df  = load_stock(sym)
    df  = detect_all_anomalies(df)
    s   = anomaly_summary(df, sym)

    print(f"Anomaly summary for {sym}:")
    for k, v in s.items():
        print(f"  {k}: {v}")

    df_if = isolation_forest_anomalies(load_stock(sym))
    print(f"\nIsolation Forest anomalies: {df_if['IF_Anomaly'].sum()}")
