from itertools import combinations
import pandas as pd
import yfinance as yf
import numpy as np
from yfinance.utils import auto_adjust
from Utils import ticks
import statsmodels.api as sm 
from statsmodels.tsa.stattools import coint


def data_retireve(ticks:list =[], start:str ="", end:str =""): 
    raw_data = yf.download(ticks, start=start, end=end, auto_adjust= True)['Close']
    return raw_data

def hurst_exponent(s: pd.Series,
                   min_block: int = 16,
                   min_segments: int = 10,
                   max_blocks: int = 4) -> float:
    """
    Improved R/S Hurst:
    - z-score the series for stability
    - use powers-of-two lags up to n//max_blocks
    - require enough non-overlapping segments per lag
    """
    x = pd.Series(s).dropna().astype(float).values
    n = x.size
    if n < 2 * min_block * max_blocks:
        return np.nan

    # z-score (doesn't change H theoretically, stabilizes numerics)
    y = (x - x.mean()) / (x.std(ddof=1) + 1e-12)

    # powers-of-two lags: 16, 32, 64, ..., n//max_blocks
    max_lag = n // max_blocks
    start_pow = int(np.log2(min_block))
    stop_pow  = int(np.floor(np.log2(max_lag)))
    if stop_pow <= start_pow:
        return np.nan
    cand = 2 ** np.arange(start_pow, stop_pow + 1)

    rs_vals, kept = [], []
    for lag in cand:
        m = n // lag
        if m < min_segments:
            continue
        trimmed = y[: m * lag].reshape(m, lag)

        ratios = []
        for row in trimmed:
            r = row - row.mean()
            z = np.cumsum(r)
            R = z.max() - z.min()
            S = r.std(ddof=1)
            if S > 0:
                ratios.append(R / S)
        if ratios:
            rs_vals.append(float(np.mean(ratios)))
            kept.append(int(lag))

    if len(rs_vals) < 2:
        return np.nan

    H = np.polyfit(np.log(kept), np.log(rs_vals), 1)[0]
    return float(H)

def estimate_half_life(series: pd.Series) -> float:
    x = series.dropna().values
    if len(x) < 3:
      return np.nan

    # lag the series
    x_lag = x[:-1]
    x_now = x[1:]

    # dependent variable: Δx_t
    delta_x = x_now - x_lag

    # add constant
    X = np.column_stack([np.ones_like(x_lag), x_lag])

    # OLS estimation: Δx_t = α + β * x_{t-1} + ε_t
    beta = np.linalg.lstsq(X, delta_x, rcond=None)[0][1]

    # speed of mean reversion (kappa)
    kappa = -np.log(1 + beta)

    # valid only if kappa > 0
    if not np.isfinite(kappa) or kappa <= 0:
        return np.nan

    # half-life = ln(2) / kappa
    half_life = np.log(2) / kappa
    return float(half_life)


def pair_selection(prices: pd.DataFrame, 
                   min_obs: int = 252, 
                   use_log: bool = True, 
                   pval_tresh: float = 0.05, 
                   top_n: int = 20, 
                   hurst_tresh: float = 0.5, 
                   half_life_tresh: float = 120.0,
                   ) -> pd.DataFrame:
    P = prices.copy() 
    P = P.sort_index() 
    if use_log: 
        P = np.log(P.clip(lower=1e-12))
        P = P.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    tickers = [c for c in P.columns if P[c].dropna().size >= min_obs]
    results = [] 

    for y_tkr, x_tkr in combinations(tickers, 2): 
        df = P[[y_tkr, x_tkr]].dropna() 
        if df.shape[0] < min_obs: 
            continue

        y = df[y_tkr].to_numpy()   # NumPy arrays for OLS
        x = df[x_tkr].to_numpy()

        stat, pval, _= coint(y ,x, trend = "c", autolag="aic") 
        if not np.isfinite(pval) or pval > pval_tresh: 
            continue 

        X = sm.add_constant(x) 
        model = sm.OLS(y, X, missing="drop").fit() 
        alpha = float(model.params[0]) 
        beta = float(model.params[1]) 

        spread = pd.Series(y - (alpha + beta * x), index = df.index) 
        spread_z = (spread - spread.mean()) / (spread.std(ddof=1) + 1e-12)


        

        H = hurst_exponent(spread_z) 
        HL = estimate_half_life(spread_z) 
        print(f"{y_tkr}  {x_tkr}  {pval} {H} {HL}")

        if (
                np.isfinite(H) and np.isfinite(HL) 
                and H <= hurst_tresh 
                and HL <= half_life_tresh
                ):
            results.append(
                    {
                        "y": y_tkr, 
                        "x": x_tkr, 
                        "pval_coint": float(pval), 
                        "alpha": alpha, 
                        "beta": beta, 
                        "hurst": float(H), 
                        "half_life": float(HL)
                        }
                    )
            print(f"Collected suitable pair {y_tkr} and {x_tkr}")

    if not results:  
        print("No suitable pairs")
        return pd.DataFrame()
    else: 
        out = pd.DataFrame(results).sort_values(
                by = ['pval_coint', "hurst", "half_life"], ascending=[True, True, True] 
                )
        return out


def _debug_hurst():
    np.random.seed(0)
    wn = pd.Series(np.random.randn(5000))              # stationary
    rw = pd.Series(np.random.randn(5000)).cumsum()     # random walk

    print("H (R/S) white noise ~0.5 ->", hurst_exponent(wn))
    print("H (R/S) random walk ~1.0 ->", hurst_exponent(rw))


################
################ Buying Logic
################



def main(): 



    price_data = data_retireve(ticks, start="2018-01-01", end="2020-01-01")
    screened_pairs = pair_selection(price_data)
    print(screened_pairs)









if __name__ == "__main__": 
    main()
