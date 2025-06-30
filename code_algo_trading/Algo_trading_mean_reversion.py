import yfinance as yf
import pandas as pd
import time
from Utilities import test_stocks, all_sector_stocks
from itertools import combinations, product
from statsmodels.tsa.stattools import coint
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import timedelta

#--------------------------------
# Global parameters
#--------------------------------
start_date    = '2020-01-01' 
end_date       = '2025-01-01'           # ← defined here
train_duration = timedelta(days=252 * 2)   # initial 2-year window
min_data_thresh = 0.9

#--------------------------------
# 1) Data fetching & screening
#--------------------------------
def fetch_price_data(stock_dict, start_date, end_date, min_data_thresh):
    records = pd.DataFrame()
    for sector in stock_dict:
        time.sleep(1)
        for ticker in stock_dict[sector]:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=False
            )['Close']
            records = pd.concat([records, df], axis=1)
    thresh = int(min_data_thresh * len(records))
    return records.dropna(axis=1, thresh=thresh)

def screen_cointegrated_pairs(prices, pval_thresh=0.05):
    rows = []
    for s1, s2 in combinations(prices.columns, 2):
        pair = prices[[s1, s2]].dropna()
        if len(pair) < 100:
            continue
        _, pval, _ = coint(pair[s1], pair[s2])
        if pval < pval_thresh:
            rows.append({'Stock 1': s1, 'Stock 2': s2, 'pval': pval})
    return pd.DataFrame(rows)

def filter_pairs(prices, coint_df, hurst_thresh=0.41, hl_bounds=(5,40)):
    out = []
    for _, r in coint_df.iterrows():
        s1, s2 = r['Stock 1'], r['Stock 2']
        data    = prices[[s1, s2]].dropna()
        beta    = sm.OLS(data[s2], sm.add_constant(data[s1])).fit().params.iloc[1]
        spread  = data[s2] - beta * data[s1]
        h       = hurst_exponent(spread.values)
        hl      = half_life(spread)
        if h < hurst_thresh and hl_bounds[0] < hl < hl_bounds[1]:
            out.append({'Stock 1': s1, 'Stock 2': s2, 'Beta': beta})
    return pd.DataFrame(out)

#--------------------------------
# 2) Metrics
#--------------------------------
def hurst_exponent(ts):
    lags = range(2,100)
    tau  = [np.std(ts[lag:]-ts[:-lag]) for lag in lags]
    m, _ = np.polyfit(np.log(lags), np.log(tau), 1)
    return m

def half_life(spread):
    ret = spread.diff().dropna()
    lag = spread.shift(1).dropna().loc[ret.index]
    rho = sm.OLS(ret, sm.add_constant(lag)).fit().params.iloc[1]
    return -np.log(2)/rho if rho<0 else np.inf

#--------------------------------
# 3) Simulation & evaluation
#--------------------------------
def simulate_pair(s1, s2, prices, z_entry, z_exit):
    df   = prices[[s1,s2]].dropna().copy()
    beta = sm.OLS(df[s2], sm.add_constant(df[s1])).fit().params.iloc[1]
    spread = df[s2] - beta * df[s1]
    zscore = (spread - spread.mean()) / spread.std()

    df['position'] = 0
    df.loc[zscore >  z_entry, 'position'] = -1
    df.loc[zscore < -z_entry, 'position'] =  1
    df.loc[abs(zscore) < z_exit,  'position'] =  0
    df['position'] = df['position'].ffill().fillna(0)

    df['returns']     = df['position'].shift(1) * (
                          df[s2].pct_change() - beta * df[s1].pct_change()
                        )
    df['cum_returns'] = (1 + df['returns']).cumprod()
    df['zscore']      = zscore
    return df

def evaluate(df):
    rets = df['returns'].dropna()
    if len(rets) < 2:
        return np.nan, np.nan, np.nan
    sharpe   = np.sqrt(252) * rets.mean() / rets.std()
    dd       = (df['cum_returns'] - df['cum_returns'].cummax()) / df['cum_returns'].cummax()
    duration = (df['position'] != 0).sum()
    return sharpe, dd.min(), duration

#--------------------------------
# 4) Sequential feed‐forward
#--------------------------------
def run_sequential(prices, train_start, train_end,
                   pval_thresh=0.05, hurst_thresh=0.41, hl_bounds=(5,40),
                   z_entry_grid=(0.8,1.0,1.2), z_exit_grid=(0.0,0.2)):
    # initial train slice
    train = prices.loc[train_start:train_end]
    # screen & filter once
    coint_df = screen_cointegrated_pairs(train, pval_thresh)
    filt_df  = filter_pairs(train, coint_df, hurst_thresh, hl_bounds)

    # optimize thresholds in-sample for each pair
    bests = []
    for _, r in filt_df.iterrows():
        best_sh, best_thr = -np.inf, None
        for ze, zx in product(z_entry_grid, z_exit_grid):
            df_is = simulate_pair(r['Stock 1'], r['Stock 2'], train, ze, zx)
            sh, _, _ = evaluate(df_is)
            if sh > best_sh:
                best_sh, best_thr = sh, (ze, zx)
        bests.append({'Stock 1': r['Stock 1'],
                      'Stock 2': r['Stock 2'],
                      'Thresholds': best_thr,
                      'IS Sharpe': best_sh})

    bests_df = pd.DataFrame(bests).sort_values('IS Sharpe', ascending=False)
    # pick the single best pair + thresholds
    best_pair = bests_df.iloc[0]

    # now simulate sequentially on the remaining data
    results = []
    test_start = train_end
    end_date   = prices.index.max()

    while test_start < end_date:
        # simulate live until closure
        df_oos = simulate_pair(best_pair['Stock 1'],
                               best_pair['Stock 2'],
                               prices.loc[test_start:],
                               *best_pair['Thresholds'])
        # find first return to flat after entry
        pts = df_oos.index[df_oos['position'].diff() != 0]
        if len(pts) >= 2:
            close_dt = pts[1]
            df_segment = df_oos.loc[:close_dt]
        else:
            df_segment = df_oos

        sh, dd, dur = evaluate(df_segment)
        results.append({
            'Start':       test_start,
            'End':         df_segment.index[-1],
            'Stock 1':     best_pair['Stock 1'],
            'Stock 2':     best_pair['Stock 2'],
            'Thresholds':  best_pair['Thresholds'],
            'OOS Sharpe':  sh,
            'OOS DD':      dd,
            'OOS Return':  df_segment['cum_returns'].iloc[-1],
            'Duration':    dur,
            'History':     df_segment.copy()
        })

        # advance to next
        test_start = df_segment.index[-1]

    return results

def plot_equity(results):
    # stitch all daily returns
    all_rets = pd.concat([r['History']['returns'] for r in results]).sort_index()
    eq = (1 + all_rets).cumprod()
    eq.plot(title='Strategy Equity Curve', figsize=(10,4))
    plt.show()

#--------------------------------
# 5) Main
#--------------------------------
if __name__ == "__main__":
    # 1) download
    prices = fetch_price_data(all_sector_stocks, start_date, 
                              (pd.to_datetime(start_date) + train_duration).strftime('%Y-%m-%d'),
                              min_data_thresh)

    # 2) full data for live sim
    all_prices = fetch_price_data(all_sector_stocks, start_date, end_date, min_data_thresh)

    # 3) train slice boundaries
    train_start = pd.to_datetime(start_date)
    train_end   = train_start + train_duration

    # 4) run sequential feed-forward
    results = run_sequential(all_prices, train_start, train_end)

    # 5) summary
    summary = pd.DataFrame([{
        k:v for k,v in r.items() if k not in ('History',)
    } for r in results])
    print(summary.to_string(index=False))

    # 6) overall cumulative return
    overall = np.prod([r['OOS Return'] for r in results])
    print(f"\nOverall Cumulative Return: {overall:.2f}×")

    # 7) equity curve
    plot_equity(results)
