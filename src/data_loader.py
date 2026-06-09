import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

METADATA_FILE = os.path.join(DATA_DIR, "stock_metadata.csv")

def load_metadata():
    """Load stock metadata (company name, industry, symbol)."""
    df = pd.read_csv(METADATA_FILE)
    df = df.drop_duplicates(subset="Symbol")
    df.columns = df.columns.str.strip()
    return df

def load_stock(symbol: str) -> pd.DataFrame:
    """Load a single stock CSV by symbol. Returns cleaned DataFrame."""
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No data file found for symbol: {symbol}")

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Keep only essential columns; rename for clarity
    keep = ["Date", "Symbol", "Open", "High", "Low", "Close", "VWAP", "Volume", "Turnover"]
    available = [c for c in keep if c in df.columns]
    df = df[available].copy()

    # Clean numeric columns
    for col in ["Open", "High", "Low", "Close", "VWAP", "Volume", "Turnover"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"])
    df = df.reset_index(drop=True)
    return df

def load_all_stocks(min_rows: int = 500) -> dict:
    """
    Load all valid stock CSVs. Skip files with too few rows (e.g. INFRATEL).
    Returns a dict: {symbol: DataFrame}
    """
    metadata = load_metadata()
    symbols = metadata["Symbol"].tolist()
    stocks = {}

    for symbol in symbols:
        try:
            df = load_stock(symbol)
            if len(df) >= min_rows:
                stocks[symbol] = df
        except FileNotFoundError:
            continue

    print(f"Loaded {len(stocks)} stocks successfully.")
    return stocks

def load_combined() -> pd.DataFrame:
    """Load the NIFTY50_all.csv combined file."""
    path = os.path.join(DATA_DIR, "NIFTY50_all.csv")
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    for col in ["Open", "High", "Low", "Close", "VWAP", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"])
    return df

def get_symbol_list() -> list:
    """Return list of all valid symbols (those with a data file and enough rows)."""
    metadata = load_metadata()
    valid = []
    for sym in metadata["Symbol"].tolist():
        path = os.path.join(DATA_DIR, f"{sym}.csv")
        if os.path.exists(path):
            valid.append(sym)
    return valid

def get_sector_map() -> dict:
    """Return {symbol: industry} mapping."""
    metadata = load_metadata()
    return dict(zip(metadata["Symbol"], metadata["Industry"]))

def get_close_price_matrix(symbols: list = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Build a wide DataFrame of close prices: rows=dates, columns=symbols.
    Useful for portfolio and correlation analysis.
    """
    if symbols is None:
        symbols = get_symbol_list()

    frames = {}
    for sym in symbols:
        try:
            df = load_stock(sym)
            if start_date:
                df = df[df["Date"] >= start_date]
            if end_date:
                df = df[df["Date"] <= end_date]
            frames[sym] = df.set_index("Date")["Close"]
        except Exception:
            continue

    price_matrix = pd.DataFrame(frames)
    price_matrix = price_matrix.sort_index()
    return price_matrix

def get_daily_returns(symbols: list = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Return daily percentage returns for all symbols."""
    prices = get_close_price_matrix(symbols, start_date, end_date)
    returns = prices.pct_change().dropna(how="all")
    return returns


if __name__ == "__main__":
    meta = load_metadata()
    print(meta.head())
    print(f"\nTotal companies: {len(meta)}")

    df = load_stock("RELIANCE")
    print(f"\nRELIANCE shape: {df.shape}")
    print(df.tail(3))

    price_matrix = get_close_price_matrix(["RELIANCE", "TCS", "INFY", "HDFCBANK"])
    print(f"\nPrice matrix shape: {price_matrix.shape}")
    print(price_matrix.tail(3))