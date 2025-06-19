import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt 

tickers = ['KO', 'PEP'] 
start_date = '2020-01-01' 
end_date = '2024-12-31' 
data = yf.download(tickers, start= start_date, end=end_date)
data_close = data[['Close']] 
print(data_close)