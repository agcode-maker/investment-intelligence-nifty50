"""
predictor.py — Stock Predictor Engine
Mandatory Task A: Price direction classification + return forecasting.

Models:
  1. XGBoostClassifier  — direction prediction (up/down over N days)
  2. XGBoostRegressor   — future return forecasting
  3. RandomForest       — ensemble direction prediction

All models expose a unified interface:
  .train(df)     → fits on historical data
  .predict(df)   → returns predictions
  .evaluate(df)  → returns metrics dict
"""

import os, warnings, json
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_absolute_error, mean_squared_error, r2_score)
import xgboost as xgb

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

BASE_FEATURES = [
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Width', 'BB_Pct', 'ATR',
    'Price_vs_SMA20', 'Price_vs_SMA50', 'Price_vs_SMA200',
    'Return_1d', 'Return_5d', 'Return_10d', 'Return_20d',
    'Rolling_Std_5', 'Rolling_Std_10', 'Rolling_Std_20',
    'Rolling_Mean_5', 'Rolling_Mean_10',
    'Volume_Ratio', 'HV_21',
    'ADX', 'Stoch_K', 'Stoch_D',
    'Williams_R', 'ROC_12',
    'High_Low_Range', 'Open_Close_Range',
    'Bull_Trend', 'RSI_Overbought', 'RSI_Oversold',
    'DayOfWeek', 'Month', 'Quarter',
]


def _get_features(df, feature_cols=None):
    cols = feature_cols or BASE_FEATURES
    return [c for c in cols if c in df.columns]


def _prepare_xy(df, target, feature_cols=None):
    feat = _get_features(df, feature_cols)
    clean = df[feat + [target, 'Date']].dropna()
    return clean[feat].values, clean[target].values, clean['Date'].values, feat


def _align_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Ensure df has exactly feature_names columns in the right order.
    Missing columns are filled with 0. Prevents feature-count mismatch at predict time.
    """
    df = df.copy()
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    return df[feature_names]


def _add_extra_features(df):
    """Add features not in indicators.py that predictors need."""
    df = df.copy()
    for col, base in [('Price_vs_SMA20','SMA_20'),
                       ('Price_vs_SMA50','SMA_50'),
                       ('Price_vs_SMA200','SMA_200')]:
        if base in df.columns:
            df[col] = (df['Close'] - df[base]) / df[base] * 100
    if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
        df['Bull_Trend']      = (df['SMA_50'] > df['SMA_200']).astype(int)
    if 'RSI' in df.columns:
        df['RSI_Overbought']  = (df['RSI'] > 70).astype(int)
        df['RSI_Oversold']    = (df['RSI'] < 30).astype(int)
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month']     = df['Date'].dt.month
    df['Quarter']   = df['Date'].dt.quarter
    return df


# ── 1. Direction Classifier (XGBoost) ────────────────────────────────────────

class DirectionClassifier:
    def __init__(self, horizon=5, feature_cols=None):
        self.horizon = horizon
        self.feature_cols = feature_cols
        self.target = f"Direction_{horizon}d"
        self.model = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', random_state=42, n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.feature_names_ = None
        self.is_fitted = False

    def train(self, df):
        X, y, _, feat = _prepare_xy(df, self.target, self.feature_cols)
        self.feature_names_ = feat
        self.model.fit(self.scaler.fit_transform(X), y)
        self.is_fitted = True
        return self

    def predict(self, df):
        X = _align_features(df, self.feature_names_).fillna(0).values
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, df):
        X = _align_features(df, self.feature_names_).fillna(0).values
        return self.model.predict_proba(self.scaler.transform(X))

    def evaluate(self, df):
        X, y, _, feat = _prepare_xy(df, self.target, self.feature_cols)
        tscv = TimeSeriesSplit(n_splits=5)
        accs, preds_all, true_all = [], [], []
        for tr_idx, te_idx in tscv.split(X):
            sc = StandardScaler()
            m = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                   eval_metric='logloss', random_state=42, n_jobs=-1)
            m.fit(sc.fit_transform(X[tr_idx]), y[tr_idx])
            ypred = m.predict(sc.transform(X[te_idx]))
            accs.append(accuracy_score(y[te_idx], ypred))
            preds_all.extend(ypred); true_all.extend(y[te_idx])
        report = classification_report(true_all, preds_all, output_dict=True)
        return {
            "model": "XGBoost Direction Classifier",
            "horizon_days": self.horizon, "cv_folds": 5,
            "mean_accuracy": round(float(np.mean(accs)), 4),
            "std_accuracy":  round(float(np.std(accs)),  4),
            "overall_accuracy": round(accuracy_score(true_all, preds_all), 4),
            "precision_up": round(report.get('1',{}).get('precision', 0), 4),
            "recall_up":    round(report.get('1',{}).get('recall',    0), 4),
            "f1_up":        round(report.get('1',{}).get('f1-score',  0), 4),
            "directional_accuracy": round(accuracy_score(true_all, preds_all), 4),
        }

    def save(self, symbol):
        joblib.dump({'model': self.model, 'scaler': self.scaler,
                     'features': self.feature_names_, 'horizon': self.horizon},
                    os.path.join(MODELS_DIR, f"{symbol}_direction_clf.pkl"))

    def load(self, symbol):
        obj = joblib.load(os.path.join(MODELS_DIR, f"{symbol}_direction_clf.pkl"))
        self.model = obj['model']; self.scaler = obj['scaler']
        self.feature_names_ = obj['features']; self.horizon = obj['horizon']
        self.is_fitted = True; return self


# ── 2. Return Forecaster (XGBoost Regressor) ─────────────────────────────────

class ReturnForecaster:
    def __init__(self, horizon=5, feature_cols=None):
        self.horizon = horizon
        self.feature_cols = feature_cols
        self.target = f"Future_Return_{horizon}d"
        self.model = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.feature_names_ = None
        self.is_fitted = False

    def train(self, df):
        X, y, _, feat = _prepare_xy(df, self.target, self.feature_cols)
        self.feature_names_ = feat
        self.model.fit(self.scaler.fit_transform(X), y)
        self.is_fitted = True; return self

    def predict(self, df):
        X = _align_features(df, self.feature_names_).fillna(0).values
        return self.model.predict(self.scaler.transform(X))

    def evaluate(self, df):
        X, y, _, feat = _prepare_xy(df, self.target, self.feature_cols)
        tscv = TimeSeriesSplit(n_splits=5)
        maes, rmses, r2s = [], [], []
        for tr_idx, te_idx in tscv.split(X):
            sc = StandardScaler()
            m = xgb.XGBRegressor(n_estimators=200, max_depth=4,
                                  learning_rate=0.05, random_state=42, n_jobs=-1)
            m.fit(sc.fit_transform(X[tr_idx]), y[tr_idx])
            ypred = m.predict(sc.transform(X[te_idx]))
            maes.append(mean_absolute_error(y[te_idx], ypred))
            rmses.append(np.sqrt(mean_squared_error(y[te_idx], ypred)))
            r2s.append(r2_score(y[te_idx], ypred))
        return {
            "model": "XGBoost Return Forecaster",
            "horizon_days": self.horizon, "cv_folds": 5,
            "mean_MAE":  round(float(np.mean(maes)),  4),
            "mean_RMSE": round(float(np.mean(rmses)), 4),
            "mean_R2":   round(float(np.mean(r2s)),   4),
            "std_MAE":   round(float(np.std(maes)),   4),
        }

    def save(self, symbol):
        joblib.dump({'model': self.model, 'scaler': self.scaler,
                     'features': self.feature_names_, 'horizon': self.horizon},
                    os.path.join(MODELS_DIR, f"{symbol}_return_reg.pkl"))

    def load(self, symbol):
        obj = joblib.load(os.path.join(MODELS_DIR, f"{symbol}_return_reg.pkl"))
        self.model = obj['model']; self.scaler = obj['scaler']
        self.feature_names_ = obj['features']; self.horizon = obj['horizon']
        self.is_fitted = True; return self


# ── 3. Random Forest Classifier ───────────────────────────────────────────────

class RFDirectionClassifier:
    def __init__(self, horizon=5, feature_cols=None):
        self.horizon = horizon
        self.feature_cols = feature_cols
        self.target = f"Direction_{horizon}d"
        self.model = RandomForestClassifier(n_estimators=300, max_depth=8,
                                            min_samples_leaf=10, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.feature_names_ = None

    def train(self, df):
        X, y, _, feat = _prepare_xy(df, self.target, self.feature_cols)
        self.feature_names_ = feat
        self.model.fit(self.scaler.fit_transform(X), y); return self

    def predict(self, df):
        X = _align_features(df, self.feature_names_).fillna(0).values
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, df):
        X = _align_features(df, self.feature_names_).fillna(0).values
        return self.model.predict_proba(self.scaler.transform(X))


# ── 4. Ensemble Predictor ─────────────────────────────────────────────────────

class EnsemblePredictor:
    """Combines XGBoost + RandomForest by averaging probabilities."""

    def __init__(self, horizon=5):
        self.horizon = horizon
        self.xgb_clf = DirectionClassifier(horizon)
        self.rf_clf  = RFDirectionClassifier(horizon)
        self.ret_reg = ReturnForecaster(horizon)
        self.is_fitted = False

    def train(self, df):
        print("  Training XGBoost classifier...")
        self.xgb_clf.train(df)
        print("  Training Random Forest classifier...")
        self.rf_clf.train(df)
        print("  Training return forecaster...")
        self.ret_reg.train(df)
        self.is_fitted = True; return self

    def predict(self, df):
        feat = _get_features(df, BASE_FEATURES)
        clean = df[feat + ['Date', 'Close']].dropna()
        xgb_p = self.xgb_clf.predict_proba(clean)[:, 1]
        rf_p  = self.rf_clf.predict_proba(clean)[:, 1]
        avg_p = (xgb_p + rf_p) / 2
        exp_r = self.ret_reg.predict(clean)
        return pd.DataFrame({
            'Date': clean['Date'].values,
            'Close': clean['Close'].values,
            'Predicted_Direction': (avg_p >= 0.5).astype(int),
            'Up_Probability': np.round(avg_p, 4),
            'Expected_Return_Pct': np.round(exp_r, 3),
            'Signal': np.where(avg_p > 0.60, 'STRONG BUY',
                      np.where(avg_p > 0.50, 'BUY',
                      np.where(avg_p < 0.40, 'STRONG SELL', 'SELL'))),
        })

    def evaluate(self, df):
        return {
            "classifier": self.xgb_clf.evaluate(df),
            "regressor":  self.ret_reg.evaluate(df),
        }

    def save(self, symbol):
        self.xgb_clf.save(symbol)
        self.ret_reg.save(symbol)
        joblib.dump({'model': self.rf_clf.model, 'scaler': self.rf_clf.scaler,
                     'features': self.rf_clf.feature_names_},
                    os.path.join(MODELS_DIR, f"{symbol}_rf_clf.pkl"))

    def load(self, symbol):
        self.xgb_clf.load(symbol)
        self.ret_reg.load(symbol)
        obj = joblib.load(os.path.join(MODELS_DIR, f"{symbol}_rf_clf.pkl"))
        self.rf_clf.model = obj['model']; self.rf_clf.scaler = obj['scaler']
        self.rf_clf.feature_names_ = obj['features']
        self.is_fitted = True; return self


# ── 5. Convenience functions ──────────────────────────────────────────────────

def prepare_stock_df(symbol: str, horizon: int = 5):
    """Load, add all indicators + extra features. Returns ready-to-use df."""
    from src.data_loader import load_stock
    from src.indicators import add_all_indicators
    df = load_stock(symbol)
    df = add_all_indicators(df, target_horizon=horizon)
    df = _add_extra_features(df)
    return df


def train_all_models(symbols: list, horizon: int = 5, verbose: bool = True) -> dict:
    """Train and save EnsemblePredictor for every symbol. Returns metrics dict."""
    results = {}
    for i, sym in enumerate(symbols, 1):
        if verbose: print(f"\n[{i}/{len(symbols)}] Training {sym}...")
        try:
            df = prepare_stock_df(sym, horizon)
            p = EnsemblePredictor(horizon=horizon)
            p.train(df)
            metrics = p.evaluate(df)
            p.save(sym)
            results[sym] = metrics
            if verbose:
                acc = metrics['classifier']['mean_accuracy']
                mae = metrics['regressor']['mean_MAE']
                print(f"  Accuracy: {acc:.4f} | MAE: {mae:.4f}%")
        except Exception as e:
            if verbose: print(f"  ERROR: {e}")
            results[sym] = {"error": str(e)}
    with open(os.path.join(MODELS_DIR, "training_summary.json"), 'w') as f:
        json.dump(results, f, indent=2)
    return results


def get_latest_signal(symbol: str, horizon: int = 5) -> dict:
    """Return most recent trading signal for a symbol."""
    df = prepare_stock_df(symbol, horizon)
    model_path = os.path.join(MODELS_DIR, f"{symbol}_direction_clf.pkl")
    p = EnsemblePredictor(horizon=horizon)
    if os.path.exists(model_path):
        p.load(symbol)
    else:
        p.train(df); p.save(symbol)
    preds = p.predict(df)
    latest = preds.iloc[-1]
    return {
        "symbol": symbol,
        "date": str(latest['Date'])[:10],
        "current_price": round(float(latest['Close']), 2),
        "signal": latest['Signal'],
        "up_probability": float(latest['Up_Probability']),
        "expected_return_pct": float(latest['Expected_Return_Pct']),
        "horizon_days": horizon,
    }


if __name__ == "__main__":
    sym = "RELIANCE"
    print(f"Testing EnsemblePredictor on {sym}...")
    df = prepare_stock_df(sym)
    p = EnsemblePredictor(horizon=5)
    p.train(df)
    metrics = p.evaluate(df)
    print(f"\nMetrics:\n{json.dumps(metrics, indent=2)}")
    preds = p.predict(df)
    print(f"\nLast 5 predictions:\n{preds.tail(5).to_string(index=False)}")
    p.save(sym)
    print("\nModel saved successfully.")
