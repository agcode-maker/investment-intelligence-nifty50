"""
risk.py — Risk Assessment Module
Mandatory Task C: Evaluate historical risk of stocks and portfolios.

Metrics computed:
  - Annualised Volatility
  - Sharpe Ratio
  - Sortino Ratio
  - Maximum Drawdown
  - Calmar Ratio
  - VaR (Historical & Parametric)
  - CVaR / Expected Shortfall
  - Beta (vs market proxy)
  - Information Ratio
  - Risk-Adjusted Return (RAR)

Also includes:
  - Portfolio-level risk breakdown
  - Stress testing (2008 GFC, 2020 COVID)
  - Rolling risk metrics
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

RISK_FREE_RATE_ANNUAL = 0.05        # 5% annual (approx Indian T-bill)
RISK_FREE_DAILY       = RISK_FREE_RATE_ANNUAL / 252


# ── Single-stock metrics ──────────────────────────────────────────────────────

def compute_stock_metrics(prices: pd.Series, symbol: str = "") -> dict:
    """
    Full risk profile for a single stock price series.
    prices: pd.Series with DatetimeIndex.
    """
    returns = prices.pct_change().dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # ── Returns ────────────────────────────────────────────────────────────────
    total_return     = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
    n_years          = (prices.index[-1] - prices.index[0]).days / 365.25
    cagr             = ((prices.iloc[-1] / prices.iloc[0]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    # ── Volatility ─────────────────────────────────────────────────────────────
    ann_vol          = returns.std() * np.sqrt(252) * 100
    downside_ret     = returns[returns < RISK_FREE_DAILY]
    downside_vol     = downside_ret.std() * np.sqrt(252) * 100 if len(downside_ret) > 5 else ann_vol

    # ── Risk-adjusted returns ──────────────────────────────────────────────────
    excess_return    = cagr / 100 - RISK_FREE_RATE_ANNUAL
    sharpe           = excess_return / (ann_vol / 100) if ann_vol > 0 else 0

    downside_std     = downside_vol / 100
    sortino          = excess_return / downside_std if downside_std > 0 else 0

    # ── Drawdown ───────────────────────────────────────────────────────────────
    roll_max         = prices.cummax()
    drawdown_series  = (prices - roll_max) / roll_max * 100
    max_drawdown     = drawdown_series.min()

    # Max drawdown duration
    in_drawdown      = drawdown_series < 0
    dd_groups        = (in_drawdown != in_drawdown.shift()).cumsum()
    dd_lengths       = in_drawdown.groupby(dd_groups).sum()
    max_dd_duration  = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0

    calmar           = (cagr / 100) / abs(max_drawdown / 100) if max_drawdown < 0 else 0

    # ── VaR & CVaR (Historical) ────────────────────────────────────────────────
    var_95_hist      = float(np.percentile(returns, 5) * 100)    # 5th percentile daily loss
    var_99_hist      = float(np.percentile(returns, 1) * 100)
    cvar_95          = float(returns[returns <= np.percentile(returns, 5)].mean() * 100)
    cvar_99          = float(returns[returns <= np.percentile(returns, 1)].mean() * 100)

    # ── Parametric VaR (Normal assumption) ────────────────────────────────────
    mu_d, sigma_d    = returns.mean(), returns.std()
    var_95_param     = float(stats.norm.ppf(0.05, mu_d, sigma_d) * 100)

    # ── Distribution shape ─────────────────────────────────────────────────────
    skewness         = float(returns.skew())
    kurtosis         = float(returns.kurt())

    # ── Positive days ─────────────────────────────────────────────────────────
    win_rate         = float((returns > 0).mean() * 100)
    avg_gain         = float(returns[returns > 0].mean() * 100) if (returns > 0).any() else 0
    avg_loss         = float(returns[returns < 0].mean() * 100) if (returns < 0).any() else 0
    gain_loss_ratio  = abs(avg_gain / avg_loss) if avg_loss != 0 else 0

    return {
        "symbol":                 symbol,
        "period_start":           str(prices.index[0].date()),
        "period_end":             str(prices.index[-1].date()),
        "n_trading_days":         len(prices),

        # Returns
        "total_return_pct":       round(total_return,    2),
        "cagr_pct":               round(cagr,            2),

        # Volatility
        "annual_volatility_pct":  round(ann_vol,         2),
        "downside_volatility_pct": round(downside_vol,   2),

        # Risk-adjusted
        "sharpe_ratio":           round(sharpe,          3),
        "sortino_ratio":          round(sortino,         3),
        "calmar_ratio":           round(calmar,          3),

        # Drawdown
        "max_drawdown_pct":       round(max_drawdown,    2),
        "max_drawdown_duration_days": max_dd_duration,

        # VaR / CVaR
        "var_95_daily_pct":       round(var_95_hist,     3),
        "var_99_daily_pct":       round(var_99_hist,     3),
        "cvar_95_daily_pct":      round(cvar_95,         3),
        "cvar_99_daily_pct":      round(cvar_99,         3),
        "var_95_param_pct":       round(var_95_param,    3),

        # Distribution
        "skewness":               round(skewness,        4),
        "excess_kurtosis":        round(kurtosis,        4),

        # Trading stats
        "win_rate_pct":           round(win_rate,        2),
        "avg_gain_pct":           round(avg_gain,        4),
        "avg_loss_pct":           round(avg_loss,        4),
        "gain_loss_ratio":        round(gain_loss_ratio, 3),
    }


def compute_beta(stock_returns: pd.Series, market_returns: pd.Series) -> dict:
    """
    Compute beta, alpha, R² of stock vs market proxy.
    Both series should be daily returns aligned on the same index.
    """
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    aligned.columns = ['stock', 'market']
    if len(aligned) < 30:
        return {"beta": None, "alpha_annual": None, "r_squared": None, "correlation": None}

    cov = np.cov(aligned['stock'], aligned['market'])
    beta = cov[0, 1] / cov[1, 1]
    alpha_daily = aligned['stock'].mean() - beta * aligned['market'].mean()
    alpha_annual = alpha_daily * 252 * 100
    corr = aligned.corr().iloc[0, 1]
    r_sq = corr ** 2

    return {
        "beta":         round(float(beta), 3),
        "alpha_annual": round(float(alpha_annual), 3),
        "r_squared":    round(float(r_sq), 4),
        "correlation":  round(float(corr), 4),
    }


# ── Risk classification ───────────────────────────────────────────────────────

def classify_risk(metrics: dict) -> str:
    """Return LOW / MEDIUM / HIGH / VERY HIGH based on vol + max drawdown."""
    vol = metrics.get("annual_volatility_pct", 0)
    dd  = abs(metrics.get("max_drawdown_pct", 0))
    score = 0
    if vol > 40:   score += 3
    elif vol > 25: score += 2
    elif vol > 15: score += 1
    if dd > 60:    score += 3
    elif dd > 40:  score += 2
    elif dd > 20:  score += 1
    if score >= 5: return "VERY HIGH"
    if score >= 3: return "HIGH"
    if score >= 2: return "MEDIUM"
    return "LOW"


# ── Portfolio-level risk ──────────────────────────────────────────────────────

def compute_portfolio_metrics(weights: dict, prices: pd.DataFrame) -> dict:
    """
    Full risk profile for a portfolio defined by {symbol: weight}.
    """
    w = pd.Series(weights)
    common = [c for c in w.index if c in prices.columns]
    if not common:
        raise ValueError("No overlapping symbols between weights and prices.")
    w = w[common] / w[common].sum()
    p = prices[common].dropna()

    daily_ret = p.pct_change().dropna()
    port_ret  = (daily_ret * w).sum(axis=1)

    # Reconstruct portfolio price series from returns
    port_prices = (1 + port_ret).cumprod()
    port_prices.iloc[0] = 1.0
    port_prices = port_prices * 100   # index base 100

    metrics = compute_stock_metrics(port_prices, symbol="Portfolio")

    # Add covariance-based metrics
    cov_matrix = daily_ret.cov() * 252
    port_var   = float(w @ cov_matrix @ w)
    port_vol_cov = np.sqrt(port_var) * 100

    # Marginal contribution to risk
    marginal_contrib = (cov_matrix @ w) / np.sqrt(port_var)
    risk_contrib     = w * marginal_contrib
    pct_risk_contrib = (risk_contrib / risk_contrib.sum() * 100).round(2)

    metrics["portfolio_vol_cov_pct"]     = round(port_vol_cov, 2)
    metrics["risk_contributions"]         = pct_risk_contrib.to_dict()
    metrics["concentration_herfindahl"]   = round(float((w ** 2).sum()), 4)
    metrics["effective_n_stocks"]         = round(float(1 / (w ** 2).sum()), 1)

    return metrics


# ── Stress Testing ────────────────────────────────────────────────────────────

STRESS_PERIODS = {
    "2008_GFC":            ("2008-01-01", "2009-03-31"),
    "2011_EuroDebt":       ("2011-06-01", "2011-12-31"),
    "2015_ChinaSlowdown":  ("2015-06-01", "2015-12-31"),
    "2020_COVID":          ("2020-01-01", "2020-06-30"),
    "2020_Recovery":       ("2020-04-01", "2021-04-30"),
}


def stress_test_portfolio(weights: dict, prices: pd.DataFrame) -> dict:
    """
    Compute portfolio return during each stress period.
    Returns dict: {period_name: {"return_pct": ..., "max_drawdown_pct": ...}}
    """
    w = pd.Series(weights)
    common = [c for c in w.index if c in prices.columns]
    w = w[common] / w[common].sum()

    results = {}
    for period_name, (start, end) in STRESS_PERIODS.items():
        p_slice = prices[common]
        p_slice = p_slice[
            (p_slice.index >= pd.Timestamp(start)) &
            (p_slice.index <= pd.Timestamp(end))
        ].dropna()

        if len(p_slice) < 10:
            results[period_name] = {"return_pct": None, "max_drawdown_pct": None}
            continue

        daily_ret  = p_slice.pct_change().dropna()
        port_ret   = (daily_ret * w).sum(axis=1)
        total_ret  = ((1 + port_ret).prod() - 1) * 100
        port_price = (1 + port_ret).cumprod()
        roll_max   = port_price.cummax()
        max_dd     = ((port_price - roll_max) / roll_max * 100).min()

        results[period_name] = {
            "return_pct":       round(float(total_ret), 2),
            "max_drawdown_pct": round(float(max_dd),    2),
            "start":            start,
            "end":              end,
        }
    return results


# ── Rolling Risk ──────────────────────────────────────────────────────────────

def compute_rolling_risk(prices: pd.Series, window: int = 252) -> pd.DataFrame:
    """
    Compute rolling Sharpe, Volatility, and Max Drawdown for a price series.
    """
    returns = prices.pct_change().dropna()
    roll_vol    = returns.rolling(window).std() * np.sqrt(252) * 100
    roll_mean   = returns.rolling(window).mean() * 252
    roll_sharpe = (roll_mean - RISK_FREE_RATE_ANNUAL) / (roll_vol / 100)

    roll_max = prices.rolling(window, min_periods=1).max()
    roll_dd  = (prices - roll_max) / roll_max * 100

    return pd.DataFrame({
        "Rolling_Vol_pct":    roll_vol,
        "Rolling_Sharpe":     roll_sharpe,
        "Rolling_Drawdown":   roll_dd,
    }, index=returns.index)


# ── Full universe risk table ──────────────────────────────────────────────────

def build_risk_table(prices: pd.DataFrame, start_date: str = "2015-01-01") -> pd.DataFrame:
    """
    Compute risk metrics for all stocks. Returns a ranked DataFrame.
    """
    from src.data_loader import get_sector_map
    sector_map = get_sector_map()

    prices_filtered = prices[prices.index >= pd.Timestamp(start_date)].dropna(how='all')
    market_proxy = prices_filtered.mean(axis=1).pct_change().dropna()

    rows = []
    for sym in prices_filtered.columns:
        series = prices_filtered[sym].dropna()
        if len(series) < 200:
            continue
        m = compute_stock_metrics(series, sym)
        rets = series.pct_change().dropna()
        beta_info = compute_beta(rets, market_proxy)
        m.update(beta_info)
        m['sector']     = sector_map.get(sym, 'Unknown')
        m['risk_level'] = classify_risk(m)
        rows.append(m)

    df = pd.DataFrame(rows)
    return df.sort_values('sharpe_ratio', ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from src.data_loader import load_stock
    from src.portfolio import get_price_data, build_all_portfolios, load_portfolios

    # Test single stock
    df = load_stock("RELIANCE")
    m = compute_stock_metrics(df.set_index('Date')['Close'], "RELIANCE")
    print("RELIANCE Risk Metrics:")
    for k, v in m.items():
        print(f"  {k}: {v}")

    # Test portfolio risk
    prices = get_price_data('2015-01-01')
    portfolios = load_portfolios()
    bal_weights = portfolios['balanced']['weights']
    pm = compute_portfolio_metrics(bal_weights, prices)
    print(f"\nBalanced Portfolio Sharpe: {pm['sharpe_ratio']}")
    print(f"Max Drawdown: {pm['max_drawdown_pct']}%")

    # Stress test
    st = stress_test_portfolio(bal_weights, prices)
    print("\nStress Test Results:")
    for period, result in st.items():
        print(f"  {period}: {result}")
