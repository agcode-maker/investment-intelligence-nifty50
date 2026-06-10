"""
portfolio.py — Portfolio Construction Module
Mandatory Task B: Generate portfolios for Conservative, Balanced, Aggressive investors.

Methods used:
  - Mean-Variance Optimization (Markowitz)
  - Maximum Sharpe Ratio portfolio
  - Minimum Volatility portfolio
  - Risk-Parity portfolio
  - Equal-Weight baseline

All portfolios are built on historical returns using PyPortfolioOpt.
"""

import os, warnings, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

from pypfopt import expected_returns, risk_models, EfficientFrontier, plotting
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# ── Helpers ───────────────────────────────────────────────────────────────────

INVESTOR_PROFILES = {
    "conservative": {
        "label":       "Conservative Investor",
        "description": "Capital preservation, low risk, steady income",
        "target_return": None,   # uses min-volatility
        "weight_bounds": (0.0, 0.10),  # max 10% per stock → forced diversification
        "risk_free_rate": 0.05,
        "color": "#2ecc71",
    },
    "balanced": {
        "label":       "Balanced Investor",
        "description": "Growth + stability, moderate risk",
        "target_return": None,   # uses max Sharpe
        "weight_bounds": (0.0, 0.15),
        "risk_free_rate": 0.05,
        "color": "#3498db",
    },
    "aggressive": {
        "label":       "Aggressive Investor",
        "description": "Maximum growth, high risk tolerance",
        "target_return": None,   # uses max return subject to volatility cap
        "weight_bounds": (0.0, 0.30),  # allows concentration
        "risk_free_rate": 0.05,
        "color": "#e74c3c",
    },
}


def get_price_data(start_date: str = "2015-01-01", end_date: str = "2021-04-30",
                   min_coverage: float = 0.85) -> pd.DataFrame:
    """
    Return clean wide price DataFrame (dates × symbols).
    Filters to stocks with sufficient coverage in the period.
    """
    from src.data_loader import get_close_price_matrix
    prices = get_close_price_matrix()
    prices = prices[prices.index >= pd.Timestamp(start_date)]
    prices = prices[prices.index <= pd.Timestamp(end_date)]

    # Keep stocks with enough non-null values
    coverage = prices.notna().sum() / len(prices)
    good_cols = coverage[coverage >= min_coverage].index.tolist()
    prices = prices[good_cols]

    # Forward-fill small gaps (weekends/holidays already excluded, but some stocks have odd gaps)
    prices = prices.ffill().dropna(axis=1, thresh=int(0.9 * len(prices)))
    return prices.dropna()


def compute_expected_returns_and_cov(prices: pd.DataFrame):
    """Return annualised expected returns and sample covariance matrix."""
    mu = expected_returns.mean_historical_return(prices)
    S  = risk_models.sample_cov(prices)
    return mu, S


# ── Core Portfolio Optimisers ─────────────────────────────────────────────────

def build_min_volatility_portfolio(prices: pd.DataFrame, weight_bounds=(0, 0.10)) -> dict:
    """Minimum variance portfolio — for Conservative investor."""
    mu, S = compute_expected_returns_and_cov(prices)
    ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    ef.min_volatility()
    weights = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False, risk_free_rate=0.05)
    return {
        "strategy": "Minimum Volatility",
        "weights": {k: round(v, 4) for k, v in weights.items() if v > 0.001},
        "expected_annual_return": round(perf[0], 4),
        "annual_volatility":      round(perf[1], 4),
        "sharpe_ratio":           round(perf[2], 4),
    }


def build_max_sharpe_portfolio(prices: pd.DataFrame, weight_bounds=(0, 0.15)) -> dict:
    """Maximum Sharpe Ratio portfolio — for Balanced investor."""
    mu, S = compute_expected_returns_and_cov(prices)
    ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    ef.max_sharpe(risk_free_rate=0.05)
    weights = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False, risk_free_rate=0.05)
    return {
        "strategy": "Maximum Sharpe Ratio",
        "weights": {k: round(v, 4) for k, v in weights.items() if v > 0.001},
        "expected_annual_return": round(perf[0], 4),
        "annual_volatility":      round(perf[1], 4),
        "sharpe_ratio":           round(perf[2], 4),
    }


def build_max_return_portfolio(prices: pd.DataFrame, weight_bounds=(0, 0.30),
                                target_volatility: float = 0.35) -> dict:
    """Max return under volatility cap — for Aggressive investor."""
    mu, S = compute_expected_returns_and_cov(prices)
    ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    try:
        ef.efficient_risk(target_volatility=target_volatility)
    except Exception:
        ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
        ef.max_sharpe(risk_free_rate=0.05)
    weights = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False, risk_free_rate=0.05)
    return {
        "strategy": "Maximum Return (Risk-Capped)",
        "weights": {k: round(v, 4) for k, v in weights.items() if v > 0.001},
        "expected_annual_return": round(perf[0], 4),
        "annual_volatility":      round(perf[1], 4),
        "sharpe_ratio":           round(perf[2], 4),
    }


def build_risk_parity_portfolio(prices: pd.DataFrame) -> dict:
    """
    Risk Parity: weight each stock inversely proportional to its volatility.
    Every stock contributes equal risk.
    """
    returns = prices.pct_change().dropna()
    vols = returns.std() * np.sqrt(252)
    inv_vol = 1.0 / vols
    weights = (inv_vol / inv_vol.sum()).round(4)

    port_ret_series = (returns * weights).sum(axis=1)
    ann_return = port_ret_series.mean() * 252
    ann_vol    = port_ret_series.std()  * np.sqrt(252)
    sharpe     = (ann_return - 0.05) / ann_vol if ann_vol > 0 else 0

    return {
        "strategy": "Risk Parity",
        "weights": weights[weights > 0.001].round(4).to_dict(),
        "expected_annual_return": round(float(ann_return), 4),
        "annual_volatility":      round(float(ann_vol),    4),
        "sharpe_ratio":           round(float(sharpe),     4),
    }


def build_equal_weight_portfolio(prices: pd.DataFrame) -> dict:
    """Equal-weight benchmark."""
    n = prices.shape[1]
    w = 1.0 / n
    returns = prices.pct_change().dropna()
    port_ret = returns.mean(axis=1)
    ann_ret  = port_ret.mean() * 252
    ann_vol  = port_ret.std()  * np.sqrt(252)
    sharpe   = (ann_ret - 0.05) / ann_vol if ann_vol > 0 else 0
    return {
        "strategy": "Equal Weight (Benchmark)",
        "weights": {c: round(w, 4) for c in prices.columns},
        "expected_annual_return": round(float(ann_ret), 4),
        "annual_volatility":      round(float(ann_vol), 4),
        "sharpe_ratio":           round(float(sharpe),  4),
    }


# ── Investor Profile Portfolios ───────────────────────────────────────────────

def build_all_portfolios(prices: pd.DataFrame) -> dict:
    """
    Build all three investor profile portfolios + risk parity + equal weight.
    Returns dict with full specs for each.
    """
    portfolios = {}

    print("Building Conservative portfolio (Min Volatility)...")
    portfolios["conservative"] = {
        **INVESTOR_PROFILES["conservative"],
        **build_min_volatility_portfolio(prices, weight_bounds=(0.0, 0.10)),
    }

    print("Building Balanced portfolio (Max Sharpe)...")
    portfolios["balanced"] = {
        **INVESTOR_PROFILES["balanced"],
        **build_max_sharpe_portfolio(prices, weight_bounds=(0.0, 0.15)),
    }

    print("Building Aggressive portfolio (Max Return)...")
    portfolios["aggressive"] = {
        **INVESTOR_PROFILES["aggressive"],
        **build_max_return_portfolio(prices, weight_bounds=(0.0, 0.30)),
    }

    print("Building Risk Parity portfolio...")
    portfolios["risk_parity"] = {
        "label": "Risk Parity",
        "description": "Equal risk contribution from each stock",
        "color": "#9b59b6",
        **build_risk_parity_portfolio(prices),
    }

    print("Building Equal Weight benchmark...")
    portfolios["equal_weight"] = {
        "label": "Equal Weight",
        "description": "Naive 1/N baseline benchmark",
        "color": "#95a5a6",
        **build_equal_weight_portfolio(prices),
    }

    return portfolios


# ── Discrete Allocation ───────────────────────────────────────────────────────

def get_discrete_allocation(weights: dict, prices: pd.DataFrame,
                             total_portfolio_value: float = 1_000_000) -> dict:
    """
    Convert fractional weights into actual share counts for a given budget.
    Returns {symbol: shares} and leftover cash.
    """
    latest_prices = prices.iloc[-1]
    # Only use symbols in weights
    w_series = pd.Series(weights)
    latest_prices = latest_prices[w_series.index]

    da = DiscreteAllocation(w_series.to_dict(), latest_prices, total_portfolio_value=total_portfolio_value)
    allocation, leftover = da.greedy_portfolio()
    return {"shares": allocation, "leftover_cash": round(leftover, 2)}


# ── Historical Performance Simulation ────────────────────────────────────────

def simulate_portfolio_performance(weights: dict, prices: pd.DataFrame) -> pd.Series:
    """
    Simulate daily portfolio value (normalised to 100 at start).
    weights: {symbol: weight}
    """
    w = pd.Series(weights)
    common = [c for c in w.index if c in prices.columns]
    w = w[common] / w[common].sum()   # renormalise in case some symbols missing
    p = prices[common].dropna()
    daily_ret = p.pct_change().dropna()
    port_ret  = (daily_ret * w).sum(axis=1)
    port_value = (1 + port_ret).cumprod() * 100
    return port_value


# ── Efficient Frontier ────────────────────────────────────────────────────────

def compute_efficient_frontier(prices: pd.DataFrame, n_portfolios: int = 5000) -> pd.DataFrame:
    """
    Monte Carlo simulation of random portfolios to plot efficient frontier.
    Returns DataFrame with columns: Return, Volatility, Sharpe.
    """
    returns = prices.pct_change().dropna()
    n = prices.shape[1]
    results = []

    np.random.seed(42)
    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n))
        r = (returns * w).sum(axis=1)
        ann_r = r.mean() * 252
        ann_v = r.std()  * np.sqrt(252)
        sharpe = (ann_r - 0.05) / ann_v if ann_v > 0 else 0
        results.append({"Return": ann_r, "Volatility": ann_v, "Sharpe": sharpe})

    return pd.DataFrame(results)


# ── Sector Exposure ───────────────────────────────────────────────────────────

def get_sector_exposure(weights: dict, sector_map: dict) -> dict:
    """Aggregate portfolio weights by sector."""
    exposure = {}
    for sym, w in weights.items():
        sector = sector_map.get(sym, "Unknown")
        exposure[sector] = exposure.get(sector, 0) + w
    return {k: round(v, 4) for k, v in sorted(exposure.items(), key=lambda x: -x[1])}


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_portfolios(portfolios: dict, path: str = None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "models", "portfolios.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convert any non-serialisable types
    clean = json.loads(json.dumps(portfolios, default=str))
    with open(path, 'w') as f:
        json.dump(clean, f, indent=2)
    print(f"Portfolios saved to {path}")


def load_portfolios(path: str = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "models", "portfolios.json")
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    prices = get_price_data(start_date="2015-01-01")
    print(f"Price matrix: {prices.shape}")

    portfolios = build_all_portfolios(prices)

    for name, p in portfolios.items():
        print(f"\n{'='*50}")
        print(f"{p['label']} — {p['strategy']}")
        print(f"  Return: {p['expected_annual_return']*100:.2f}%  "
              f"Vol: {p['annual_volatility']*100:.2f}%  "
              f"Sharpe: {p['sharpe_ratio']:.3f}")
        top = sorted(p['weights'].items(), key=lambda x: -x[1])[:5]
        print(f"  Top holdings: {top}")

    save_portfolios(portfolios)
