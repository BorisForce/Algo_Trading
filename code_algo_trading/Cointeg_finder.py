import yfinance as yf
import pandas as pd
import time
from Utilities import all_sector_stocks
from itertools import combinations 
from statsmodels.tsa.stattools import coint
start_date = '2020-01-01'
end_date   = '2025-01-01'

records = pd.DataFrame()
for sector in all_sector_stocks: 
    time.sleep(2)
    for ticker in all_sector_stocks[sector]: 
        df = yf.download(tickers=ticker, start=start_date, end=end_date)['Close'] 
        records = pd.concat([records, df], axis=1)

records = records.dropna(axis=1, thresh=int(0.9* len(records))) 
cointegration_results = [] 

for stock1, stock2 in combinations(records.columns, 2): 
    series1 = records[stock1].dropna() 
    series2 = records[stock2].dropna() 
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