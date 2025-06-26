import yfinance as yf 
import pandas as pd 
import time 
from Utilities import test_stocks
from itertools import combinations 
from statsmodels.tsa.stattools import coint

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
    return pd.DataFrame(results).sort_values(by='p_value') 

data = fetch_price_data(test_stocks, start_date=start_date, end_date=end_date) 
print(screen_cointegrated_pair(data))