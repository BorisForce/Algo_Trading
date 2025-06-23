import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint

# 1. Data Download
tickers = ['KO', 'PEP']
start_date = '2020-01-01'
end_date = '2024-12-31'
data = yf.download(tickers, start=start_date, end=end_date)
print(data)
data_close = data['Close'].dropna()
print(data_close)

# 2. Cointegration Test
score, pvalue, _ = coint(data_close['KO'], data_close['PEP'])
print(f'Cointegration p-value: {pvalue:.4f}')
if pvalue > 0.05:
    print("Warning: Series may not be cointegrated")

# 3. Rolling Hedge Ratio
window = 60
rolling_hedge = data_close['KO'].rolling(window).apply(
    lambda x: sm.OLS(x, sm.add_constant(data_close['PEP'].loc[x.index])).fit().params[1],
    raw=False
) 
rolling_hedge = rolling_hedge.shift(1)

# 4. Spread & Z-score
spread = data_close['KO'] - rolling_hedge * data_close['PEP']
spread = spread.dropna() 
zscore = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()

# 5. Trading Logic
entry_threshold = 1.0
exit_threshold = 0.0
stop_loss_threshold = 2.5

signals = pd.DataFrame(index=spread.index)
signals['zscore'] = zscore
signals['long'] = zscore < -entry_threshold
signals['short'] = zscore > entry_threshold
signals['exit'] = abs(zscore) < exit_threshold

# 6. Position with Stop-Loss
position = 0
positions = []
for i in range(len(signals)):
    z = signals['zscore'].iloc[i]
    if signals['long'].iloc[i] and position == 0:
        position = 1
    elif signals['short'].iloc[i] and position == 0:
        position = -1
    elif signals['exit'].iloc[i]:
        position = 0
    elif position == 1 and z > stop_loss_threshold:
        position = 0
    elif position == -1 and z < -stop_loss_threshold:
        position = 0
    positions.append(position)

signals['position'] = positions
signals['spread'] = spread
signals['returns'] = signals['position'].shift(1) * spread.diff()
signals['cum_returns'] = signals['returns'].cumsum()

# 7. Investment Constraints & Transaction Costs
starting_capital = 100
capital_per_trade = starting_capital * 0.40
initial_spread_value = abs(spread.iloc[0])
spread_units = capital_per_trade / initial_spread_value

# Calculate gross and net returns
signals['returns_$'] = signals['position'].shift(1) * spread.diff() * spread_units
position_diff = np.abs(pd.Series(positions).diff()).fillna(0)
signals['transaction_costs'] = position_diff.values * capital_per_trade * 0.001  # 0.1%
signals['net_returns_$'] = signals['returns_$'] - signals['transaction_costs']
signals['equity_curve'] = signals['net_returns_$'].cumsum() + starting_capital
capital_per_trade = signals['equity_curve'].shift(1).fillna(starting_capital) * 0.4


# 8. Plot Price Series and Z-Score
plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(data_close['KO'], label='KO')
plt.plot(data_close['PEP'], label='PEP')
plt.title('Price Series')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(signals['zscore'], label='Z-score')
plt.axhline(entry_threshold, color='red', linestyle='--', label='Entry Threshold')
plt.axhline(-entry_threshold, color='green', linestyle='--')
plt.axhline(0, color='black', linestyle='-')
plt.title('Z-score of Spread')
plt.legend()
plt.tight_layout()
plt.show()

# 9. Plot Equity Curve (Net of Costs)
plt.figure(figsize=(10, 4))
plt.plot(signals['equity_curve'], label='Equity Curve ($, net of costs)')
plt.title('Backtest: Pairs Trading Strategy (20% Capital, Alpaca Costs)')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.show()

# 10. Performance Summary
num_trades = position_diff.sum()
avg_duration = signals['position'].ne(0).astype(int).groupby((signals['position'] != 0).astype(int).diff().ne(0).cumsum()).sum().mean()
total_costs = signals['transaction_costs'].sum()
daily_returns = signals['net_returns_$']
sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else np.nan

print(f"\n=== Performance Summary ===")
print(f"Total trades executed     : {int(num_trades)}")
print(f"Average holding duration : {avg_duration:.2f} days")
print(f"Total transaction costs  : ${total_costs:.2f}")
print(f"Final portfolio value     : ${signals['equity_curve'].iloc[-1]:.2f}")
print(f"Sharpe Ratio              : {sharpe_ratio:.2f}")
