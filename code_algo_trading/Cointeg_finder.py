import yfinance as yf
import pandas as pd
import time
from Utilities import all_sector_stocks, test_stocks 
from itertools import combinations 
from statsmodels.tsa.stattools import coint
import numpy as np
import statsmodels.api as sm 
import matplotlib.pyplot as plt 

start_date = '2020-01-01'
end_date   = '2025-01-01'

records = pd.DataFrame()
for sector in test_stocks: 
    time.sleep(1)
    for ticker in all_sector_stocks[sector]: 
        df = yf.download(tickers=ticker, start=start_date, end=end_date)['Close'] 
        records = pd.concat([records, df], axis=1)

records = records.dropna(axis=1, thresh=int(0.9* len(records))) 
cointegration_results = [] 

for stock1, stock2 in combinations(records.columns, 2): 
    series1 = records[stock1]
    series2 = records[stock2] 
    combined = pd.concat([series1, series2], axis=1).dropna() 
    if len(combined)> 100:
        score, pvalue, _= coint(combined.iloc[:, 0], combined.iloc[:, 1]) 
        cointegration_results.append({
            "Stock 1": stock1, 
            "Stock 2": stock2, 
            "p-value": pvalue
        })

cointegration_results_df = pd.DataFrame(cointegration_results) 
cointegration_results_df = cointegration_results_df.sort_values(by="p-value") 
print(cointegration_results_df) 

def hurst_exponent(ts): 
    lags = range(2, 100) 
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1) 
    return poly[0] 

def calculate_metrics(df): 
    daily_returns = df['returns'].dropna() 
    if len(daily_returns) <2:
        return np.nan, np.nan, np.nan
    
    sharpe_ratio = np.sqrt(252) * daily_returns.mean()/ daily_returns.std() 

    cum_returns = df['cum_returns'] 
    rolling_max = cum_returns.cummax() 
    drawdown = (cum_returns - rolling_max)/ rolling_max 
    max_drawdown = drawdown.min() 

    trade_duration = (df['position'] !=0).sum() 

    return sharpe_ratio, max_drawdown, trade_duration

cointegration_pairs = cointegration_results_df[cointegration_results_df['p-value'] < 0.05].copy() 
filter_results =[] 

for _, row in cointegration_pairs.iterrows(): 
    stock1 = row['Stock 1'] 
    stock2 = row['Stock 2'] 
    series1 = records[stock1] 
    series2 = records[stock2] 
    combined = pd.concat([series1, series2], axis=1).dropna()

    X = sm.add_constant(combined.iloc[:,0]) 
    y = combined.iloc[:,1] 
    model = sm.OLS(y,X).fit() 
    beta = model.params.iloc[1]

    spread = combined.iloc[:,1] - beta * combined.iloc[:,0] 
    hurst = hurst_exponent(spread.values)  

    spread_ret = spread.diff().dropna() 
    spread_lag = spread.shift(1).dropna() 
    spread_lag = spread_lag.loc[spread_ret.index] 
    model_hl = sm.OLS(spread_ret, sm.add_constant(spread_lag)).fit() 
    rho = model_hl.params.iloc[1] 
    half_life = -np.log(2) / rho if rho < 0 else np.inf

    filter_results.append({ 
        'Stock 1': stock1, 
        'Stock 2': stock2, 
        'p-value': row['p-value'], 
        'Beta': beta, 
        'Hurst exponent': hurst,
        'Half-life': half_life
    })

filter_df = pd.DataFrame(filter_results) 
filter_df = filter_df.sort_values(by= 'Hurst exponent') 
filter_df = filter_df[filter_df['Hurst exponent']<0.41].copy()
filter_df = filter_df[(filter_df['Half-life'] > 5) & (filter_df['Half-life'] < 40)]
print(filter_df) 

filtered_pairs = filter_df[['Stock 1', 'Stock 2']].to_dict(orient='records')


def simulate_pair_trading(stock1, stock2, prices, z_entry=1.0, z_exit=0.0): 
    df = pd.DataFrame() 
    df[stock1] = prices[stock1] 
    df[stock2] = prices[stock2] 
    df = df.dropna() 

    X = sm.add_constant(df[stock1]) 
    y = df[stock2] 
    model = sm.OLS(y, X).fit() 
    beta = model.params[1] 

    spread = df[stock2] - beta * df[stock1]
    zscore = (spread - spread.mean()) /spread.std() 

    df['spread'] = spread
    df['zscore'] = zscore 

    df['position'] = 0
    df.loc[zscore > z_entry, 'position'] = -1
    df.loc[zscore < -z_entry, 'position'] = 1
    df.loc[abs(zscore) < z_exit, 'position'] = 0
    df['position'] = df['position'].ffill().fillna(0)

    # Daily returns from the spread
    df['returns'] = df['position'].shift(1) * (df[stock2].pct_change() - beta * df[stock1].pct_change())
    df['cum_returns'] = (1 + df['returns']).cumprod()

    return df

performance = []
for pair in filtered_pairs:
    try:
        df_trading = simulate_pair_trading(pair['Stock 1'], pair['Stock 2'], records)
        final_return = df_trading['cum_returns'].iloc[-1]
        sharpe, max_dd, duration = calculate_metrics(df_trading)
        performance.append({
            'Stock 1': pair['Stock 1'],
            'Stock 2': pair['Stock 2'],
            'Final Return': final_return,
            'Sharpe Ratio': sharpe, 
            'Max drawdown': max_dd, 
            'Trade duration': duration,
            'Data': df_trading
        })
    except Exception as e: 
        print(f"Failed for {pair['Stock 1']} & {pair['Stock 2']}: {e}") 

# After performance is collected
if performance:
    summary = pd.DataFrame([{
        'Stock 1': p['Stock 1'],
        'Stock 2': p['Stock 2'],
        'Final Return': round(p['Final Return'], 2),
        'Sharpe Ratio': round(p['Sharpe Ratio'], 2),
        'Max Drawdown': round(p['Max drawdown'], 2),
        'Trade Duration': p['Trade duration']
    } for p in performance])

    summary = summary.sort_values(by='Sharpe Ratio', ascending=False)

    print("\n=== Backtest Summary ===")
    print(summary.to_string(index=False))

    # pick best based on Sharpe ratio or your preferred metric
    best = performance[summary.index[0]]
else:
    print("No valid trading pairs passed filtering and backtesting.")
    exit()
 
# Step 4: Pick the best performing one 


best = sorted(performance, key=lambda x: -x['Final Return'])[0]

# Step 5: Plot
best_data = best['Data']
fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax[0].plot(best_data.index, best_data['cum_returns'], label='Cumulative Return')
ax[0].set_title(f"Cumulative Return: {best['Stock 1']} vs {best['Stock 2']}")
ax[0].legend()

ax[1].plot(best_data.index, best_data['zscore'], label='Z-score')
ax[1].axhline(1.0, color='red', linestyle='--', label='Entry Threshold')
ax[1].axhline(-1.0, color='red', linestyle='--')
ax[1].axhline(0.0, color='black', linestyle='-')
ax[1].set_title("Z-score of Spread")
ax[1].legend()

plt.tight_layout()
plt.show()