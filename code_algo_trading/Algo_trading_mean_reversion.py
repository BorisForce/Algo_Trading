import yfinance as yf 
import pandas as pd 
import time 
from Utilities import test_stocks
from itertools import combinations 
from statsmodels.tsa.stattools import coint
import numpy as np 
import statsmodels.tsa.stattools as sm

start_date = '2020-01-01'
end_date   = '2025-01-01'

def fetch_price_data(stock_dict, start_date, end_date, min_data_treshold= 0.9): 
    records = pd.DataFrame() 
    for sector in stock_dict: 
        time.sleep(1) 
        for ticker in stock_dict[sector]: 
            df = yf.download(ticker, start=start_date, end=end_date)['Close'] 
            records = pd.concat([records, df], axis= 1)
    records = records.dropna(axis=1, thresh=int(min_data_treshold * len(records)))
    return records



def screen_cointegrated_pair(price_df, pval_treshold = 0.05): 
    results = [] 
    for stock1, stock2 in combinations(price_df.columns, 2): 
        series1 = price_df[stock1] 
        series2 = price_df[stock2] 
        combined = pd.concat([series1, series2], axis=1).dropna() 
        score, p_val, _=coint(combined.iloc[:, 0], combined.iloc[:, 1]) 
        if p_val < pval_treshold: 
            results.append({
                'Stock 1': stock1, 
                'Stock 2': stock2, 
                'p_value': p_val
            }) 
    return pd.DataFrame(results)

def hurst_exponent(ts): 
    lags = range(2,100) 
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1) 
    return poly[0]

def half_life(spread):
    spread_ret = spread.diff().dropna() 
    spread_lag = spread.shift(1).dropna().loc[spread_ret.index] 
    model = sm.OLS(spread_ret, sm.add_constant(spread_lag)).fit() 
    rho = model.params.iloc[1] 
    return -np.log(2) / rho if rho < 0 else np.inf

def filtered_pairs(price_df, cointegration_df, hurst_treshold = 0.41, hl_bounds=(5,40)):
    results = [] 
    for _, row in cointegration_df.iterrows(): 
        stock1, stock2 = row['Stock 1'], row['Stock 2'] 
        combined = pd.concat([price_df[stock1], price_df[stock2]], axis=1).dropna() 
        beta = sm.OLS(combined[stock2], sm.add_constant(combined[stock1])).fit().params[1]
        spread = combined[stock2] - beta * combined[stock1]  
        hurst = hurst_exponent(spread.values) 
        hl = half_life(spread) 
        if hurst < hurst_treshold and hl_bounds[0] < hl < hl_bounds[1]: 
            results.append({
                'Stock 1': stock1, 
                'Stock 2': stock2, 
                'p-value': row['p_value'], 
                'Beta': beta, 
                'Hurst': hurst, 
                'Half-life': hl
            })
    return pd.DataFrame(results)


data = fetch_price_data(test_stocks, start_date=start_date, end_date=end_date) 

print(filtered_pairs(data, screen_cointegrated_pair(data)))