# _________________________  model.py  _________________________
"""This is the model layer for the CLI portfolio tracker.

*  Auto-fetches sector, asset-class and P/E from Yahoo Finance (fallback: Alpha-Vantage).
*  Computes alpha & beta versus the S&P 500 (SPY proxy)
   - per asset, per sector and for the total portfolio.
*  Fetches all relevant data for the portfolio cmd
*   
*  Monte-Carlo simulation: GARCH(1,1) with a drift cap of 30% p.a.
"""
import json, math, secrets, warnings, multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
import yfinance as yf
from tqdm import tqdm

# ________ Constants & global set‑up _____________________________________________
_RET_SCALE:   int   = 100                     # decimal to percentage conversion
_YF_THREADS:  bool  = False                  # yfinance threading flag
_MAX_ANN_DRIFT     = 0.30                   # cap drift for GARCH model at 30 % p.a.
_SPY_TICKER        = "SPY"                  # proxy for S&P 500

# Alpha‑Vantage key (leave blank to disable)
_AV_KEY     = ""  # insert API key here or leave blank to only use yf
_USE_AV     = bool(_AV_KEY)
# _______________________________________________________________________________

# _________ Optional Alpha‑Vantage helpers ______________________________________
if _USE_AV:
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.fundamentaldata import FundamentalData

    _TS = TimeSeries(key=_AV_KEY, output_format="pandas", indexing_type="date")
    _FD = FundamentalData(key=_AV_KEY, output_format="pandas")

    def _av_quote_close(tkr: str) -> float | None:
        try:
            df, _ = _TS.get_quote_endpoint(symbol=tkr)
            return float(df["05. price"].iloc[0]) if not df.empty else None
        except Exception as e:
            warnings.warn(f"failed to use Alpha Vantage Falling back to Yahoo finance")
            return None

    def _av_overview(tkr: str) -> Dict[str, str] | None:
        try:
            data, _ = _FD.get_company_overview(tkr)
            return data.iloc[0].to_dict() if not data.empty else None
        except Exception as e:
            warnings.warn(f"[AV overview] {tkr}: {e}")
            return None
else:
    _av_quote_close = lambda *_: None  # type: ignore
    _av_overview    = lambda *_: None  # type: ignore


# _________ yfinance helpers ____________________________________________________

def _yf_quote_close(tkr: str) -> float | None:
    try:
        yf_tkr = yf.Ticker(tkr)
        price  = yf_tkr.info.get("regularMarketPrice")
        if price is None:
            hist = yf_tkr.history(period="5d", interval="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        return float(price) if price is not None else None
    except Exception as e:
        warnings.warn(f"[yfinance] {tkr}: {e}")
        return None


def _yf_daily_series(tkr: str, period: str = "max") -> pd.Series | None:
    try:
        df = yf.download(
            tkr, period=period, interval="1d", progress=False,
            threads=_YF_THREADS, auto_adjust=False,
        )
        if "Adj Close" not in df.columns or df.empty:
            return None
        return df["Adj Close"].sort_index()
    except Exception as e:
        warnings.warn(f"[yfinance] {tkr}: {e}")
        return None


def _yf_metadata(tkr: str) -> Tuple[str, str, float | None]:
    # Return (sector, asset_class, pe) from Yahoo Finance
    try:
        info = yf.Ticker(tkr).info
        sector = info.get("sector", "UNKNOWN")
        quote_type = info.get("quoteType", "").upper()
        asset_class = {
            "ETF": "ETF",
            "MUTUALFUND": "FUND",
            "EQUITY": "STOCK",
        }.get(quote_type, "OTHER")
        pe = info.get("trailingPE")
        return sector, asset_class, float(pe) if pe is not None else None
    except Exception as e:
        warnings.warn(f"[yfinance] metadata {tkr}: {e}")
        return "UNKNOWN", "OTHER", None


# _________ GARCH helper ______________________________________________________

def _fit_garch_params(log_ret: pd.Series):
    ret_pct = log_ret * _RET_SCALE
    mdl = arch_model(ret_pct, p=1, q=1, mean="Constant", rescale=False)
    res = mdl.fit(disp="off")

    mu_dec     = float(res.params["mu"]) / _RET_SCALE
    omega_dec  = float(res.params["omega"]) / (_RET_SCALE ** 2)
    alpha      = float(res.params["alpha[1]"])
    beta       = float(res.params["beta[1]"])

    # stationarity / exponential check
    if alpha + beta >= 1:
        scale       = 0.999 / (alpha + beta)
        alpha      *= scale
        beta       *= scale
        warnings.warn(f"GARCH params rescaled to keep stationarity ({log_ret.name})")

    sigma0_dec = math.sqrt(omega_dec / (1 - alpha - beta)) if alpha + beta < 1 else math.sqrt(omega_dec)
    return mu_dec, omega_dec, alpha, beta, sigma0_dec


def _simulate_garch_chunk(mu: float, omega: float, alpha: float, beta: float, sigma0: float,
                          horizon: int, n_paths: int, seed: int):
    rng       = np.random.default_rng(seed)
    sigma_t   = np.full(n_paths, sigma0, dtype=np.float64)
    cum_ret   = np.zeros(n_paths, dtype=np.float64)

    for _ in range(horizon):
        z         = rng.standard_normal(n_paths)
        epsilon   = sigma_t * z
        cum_ret  += mu + epsilon
        sigma_t   = np.sqrt(omega + alpha * epsilon ** 2 + beta * sigma_t ** 2)

    return np.exp(cum_ret)


# _________ data classes _______________________________________________________

@dataclass
class Asset:
    ticker:         str
    quantity:       float
    purchase_price: float
    # fetched / calculated __
    sector:         str  = "UNKNOWN"
    asset_class:    str  = "OTHER"
    current_price:  float | None = None
    pe_ratio:       float | None = None
    beta:           float | None = None
    alpha:          float | None = None

    def market_value(self) -> float:
        return (self.current_price or 0) * self.quantity


class Portfolio:
    """Main portfolio model."""
    def __init__(self):
        self.assets: List[Asset] = []
        self._sector_metrics: Dict[str, Dict[str, float]] = {}
        self._portfolio_metrics: Dict[str, float] = {}

    # __ persistence __
    def load(self, fname: str = "portfolio.json") -> bool:
        try:
            with open(fname) as f:
                data = json.load(f).get("assets", [])
            self.assets = [Asset(
                ticker=item["ticker"],
                quantity=float(item["quantity"]),
                purchase_price=float(item["purchase_price"]),
                sector=item.get("sector", "UNKNOWN"),
                asset_class=item.get("asset_class", "OTHER"),
            ) for item in data]
            return bool(self.assets)
        except FileNotFoundError:
            return False

    def save(self, fname: str = "portfolio.json"):
        with open(fname, "w") as f:
            json.dump({"assets": [a.__dict__ for a in self.assets]}, f, indent=4)

    # __ Altrations __
    def add_asset(self, ticker: str, quantity: float, purchase_price: float,
                  sector: str | None = None, asset_class: str | None = None):
        ticker = ticker.upper()
        if any(a.ticker == ticker for a in self.assets):
            return False
        # request fundamentals if not provided
        sector_f, cls_f, pe_f = _yf_metadata(ticker)
        sector      = sector or sector_f
        asset_class = asset_class or cls_f
        new_asset   = Asset(ticker, quantity, purchase_price, sector, asset_class, pe_ratio=pe_f)
        self.assets.append(new_asset)
        return True

    def remove_asset(self, ticker: str) -> bool:
        ticker = ticker.upper()
        for i, a in enumerate(self.assets):
            if a.ticker == ticker:
                self.assets.pop(i)
                return True
        return False

    # __ Pricing __
    def update_prices(self):
        for a in self.assets:
            price = _av_quote_close(a.ticker) or _yf_quote_close(a.ticker)
            a.current_price = price

    def total_value(self) -> float:
        return sum(a.market_value() for a in self.assets)

    def asset_weights(self) -> Dict[str, float]:
        tot = self.total_value()
        return {a.ticker: a.market_value() / tot for a in self.assets if tot}

    def class_weights(self) -> Dict[str, float]:
        tot = self.total_value(); out: Dict[str, float] = {}
        for a in self.assets:
            out[a.asset_class] = out.get(a.asset_class, 0) + a.market_value()
        return {k: v / tot for k, v in out.items()} if tot else {}

    def sector_weights(self) -> Dict[str, float]:
        tot = self.total_value(); out: Dict[str, float] = {}
        for a in self.assets:
            out[a.sector] = out.get(a.sector, 0) + a.market_value()
        return {k: v / tot for k, v in out.items()} if tot else {}

    # __ Fundamentals __
    def refresh_fundamentals(self):
        """Fetch sector, asset class & P/E for all tickers."""
        for a in self.assets:
            sec, cls, pe = _yf_metadata(a.ticker)
            a.sector       = a.sector or sec
            a.asset_class  = a.asset_class or cls
            a.pe_ratio     = pe

    # __ Alpha and Beta calculations __
    def _calc_alpha_beta(self, ret_asset: pd.Series, ret_mkt: pd.Series):
        """Return (alpha, beta) via OLS of asset on market."""
        slope, intercept, _, _, _ = stats.linregress(ret_mkt.values, ret_asset.values)
        return intercept, slope

    def update_risk_metrics(self, years: int = 5):
        """Compute alpha & beta for every asset + aggregate (sector & portfolio)."""
        # Download price data (one bulk call) _________________________________
        tickers = [a.ticker for a in self.assets] + [_SPY_TICKER]
        prices  = yf.download(tickers, period=f"{years}y", interval="1d", progress=False,
                              threads=_YF_THREADS, auto_adjust=False)["Adj Close"].dropna()
        returns = prices.pct_change().dropna()
        mkt_ret = returns[_SPY_TICKER]

        # ___________Per‑asset ________________________________________________
        for a in self.assets:
            if a.ticker not in returns.columns:
                continue
            a.alpha, a.beta = self._calc_alpha_beta(returns[a.ticker], mkt_ret)

        # Sector‑level (value‑weighted) _______________________________________
        weights = self.asset_weights()
        sector_ret: Dict[str, pd.Series] = {}
        for sector in set(a.sector for a in self.assets):
            cols = [a.ticker for a in self.assets if a.sector == sector and a.ticker in returns.columns]
            if not cols:
                continue
            w = np.array([weights[c] for c in cols])
            w = w / w.sum()
            sector_ret[sector] = (returns[cols] * w).sum(axis=1)
        self._sector_metrics.clear()
        for sector, r in sector_ret.items():
            alpha, beta = self._calc_alpha_beta(r, mkt_ret)
            self._sector_metrics[sector] = {"alpha": alpha, "beta": beta}

        # Portfolio‑level _____________________________________________
        port_ret = pd.Series(0, index=returns.index, dtype=float)
        for a in self.assets:
            if a.ticker in returns.columns:
                port_ret += returns[a.ticker] * weights[a.ticker]
        a_p, b_p = self._calc_alpha_beta(port_ret, mkt_ret)
        self._portfolio_metrics = {"alpha": a_p, "beta": b_p}

    def portfolio_metrics(self) -> Dict[str, float]:
        return self._portfolio_metrics

    def sector_metrics(self) -> Dict[str, Dict[str, float]]:
        return self._sector_metrics

    # __ Monte‑Carlo simulation __
    def simulate(self, years: int = 15, num_paths: int = 100_000, steps_per_year: int = 252,
                 chunk_size: int | None = None):
        if not self.assets:
            return None
        self.update_prices()
        start_val = self.total_value()
        if start_val <= 0:
            return None

        horizon   = years * steps_per_year
        cpu_count = max(1, mp.cpu_count())
        chunk_size = chunk_size or max(1, num_paths // cpu_count)
        rng = np.random.default_rng(secrets.randbits(32))
        final_vals = np.zeros(num_paths)

        for a in tqdm(self.assets, desc="Simulating assets"):
            if (a.current_price or 0) <= 0 or a.quantity <= 0:
                continue

            series = _yf_daily_series(a.ticker, period="10y")
            if series is None or len(series) < 252:
                warnings.warn(f"{a.ticker}: insufficient data - value held constant in sim")
                final_vals += a.market_value()
                continue

            log_ret = np.log(series / series.shift(1)).dropna()
            mu, omega, alpha, beta, sigma0 = _fit_garch_params(log_ret)

            # __ cap drift at _MAX_ANN_DRIFT (for realism) ________
            ann_mu = mu * steps_per_year
            if ann_mu > _MAX_ANN_DRIFT:
                mu = _MAX_ANN_DRIFT / steps_per_year

            # __ build jobs ________________________________
            jobs, seeds = [], []
            left = num_paths
            while left > 0:
                n = min(chunk_size, left)
                jobs.append(n)
                seeds.append(int(rng.integers(0, 2**32 - 1)))
                left -= n

            with mp.Pool() as pool:
                results = pool.starmap(
                    _simulate_garch_chunk,
                    [(mu, omega, alpha, beta, sigma0, horizon, n, s)
                     for n, s in zip(jobs, seeds)]
                )
            factors = np.concatenate(results)
            final_vals += a.market_value() * factors

        return {
            "distribution":   final_vals,
            "mean":           float(np.mean(final_vals)),
            "median":         float(np.median(final_vals)),
            "p5":             float(np.percentile(final_vals, 5)),
            "p95":            float(np.percentile(final_vals, 95)),
            "prob_loss":      float((final_vals < start_val).mean()),
            "current_value":  start_val,
        }

    # __ Historical data for graphing __
    def get_historical_data(self, tickers: List[str], period: str = "1y") -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()
        intraday = {"1d": "5m", "5d": "30m"}
        interval = intraday.get(period, "1d")
        try:
            df = yf.download(tickers, period=period, interval=interval,
                             progress=False, threads=_YF_THREADS, auto_adjust=False)
            adj = df.get("Adj Close")
            if adj is None or adj.empty:
                return pd.DataFrame()
            if isinstance(adj, pd.Series):
                adj = adj.to_frame(name=tickers[0])
            adj.index = adj.index.tz_localize(None)
            return adj.dropna(axis=1, how="all")
        except Exception as e:
            warnings.warn(f"Historical download failed {tickers}: {e}")
            return pd.DataFrame()

# _________________________  end model.py  _________________________
