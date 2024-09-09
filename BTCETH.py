import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf
from scipy import stats
from fredapi import Fred

fred = Fred(api_key='c6a3529843d2d1e6ff7ea9b09fdb27ea')

# Download SPY ETF which tracks S&P500 index
SPY = yf.download("SPY", start = '2022-01-01', end = '2024-08-31')
BTC = yf.download("BTC-USD", start = '2022-01-01', end = '2023-12-31')

# Takes only the last adj closing price of each month and calculates monthly returns
SPY_Monthly_Return = (SPY['Adj Close'].resample("M").last().pct_change().dropna())
BTC_Monthly_Return = (BTC['Adj Close'].resample("M").last().pct_change().dropna())

# Calculate monthly returns as percentage change
print(SPY_Monthly_Return.head())
print(BTC_Monthly_Return.head())

# Plot a histogram of the monthly returns using only matplotlib
plt.figure(figsize=(10, 6))
plt.hist(SPY_Monthly_Return, bins=20, edgecolor='black')
# Add titles and labels
plt.title('Histogram of SPY Monthly Returns (2008 - 2023)')
plt.xlabel('Monthly Return')
plt.ylabel('Frequency')
# Show the plot
plt.show()

# Plot a histogram of the monthly returns using only matplotlib
plt.figure(figsize=(10, 6))
plt.hist(BTC_Monthly_Return, bins=20, edgecolor='black')
# Add titles and labels
plt.title('Histogram of BTC Monthly Returns (2008 - 2023)')
plt.xlabel('Monthly Return')
plt.ylabel('Frequency')
# Show the plot
plt.show()

# Print Summary Statistic - displayed in percentages for easier interpretation
print(SPY_Monthly_Return.describe()*100)
print(BTC_Monthly_Return.describe()*100)

Stocks = pd.DataFrame({'SPY': SPY_Monthly_Return, 'BTC': BTC_Monthly_Return })
print(Stocks.corr())

# Get 3-month Treasury Bill (risk-free rate)
tbill = fred.get_series('TB3MS', start='2008-01-01', end='2023-12-31').resample("M").last().dropna()
print(tbill.tail())
# Align tbill with stock returns (convert tbill to monthly percentages for excess returns)
tbill = tbill / 100 / 12  # Convert annualized rate to monthly risk-free rate
print(tbill.tail())

# Align data
data = pd.DataFrame({'BTC_Return': BTC_Monthly_Return, 'SPY_Return': SPY_Monthly_Return, 'TBILL': tbill}).dropna()

# Calculate excess returns
data['Excess_BTC'] = data['BTC_Return'] - data['TBILL']
data['Excess_SPY'] = data['SPY_Return'] - data['TBILL']

# Perform regression (CAPM Model)
X = sm.add_constant(data['Excess_SPY'])  # Add constant for intercept
model = sm.OLS(data['Excess_BTC'], X).fit()

# Print the regression results
print(model.summary())

# Extract Beta from the model (slope of the regression)
beta = model.params['Excess_SPY']
print(f"Beta for BTC: {beta}")

"""
BTC's Beta or Coefficient is calculated based on the slope of the regression.
Since Pvalue is less than 0.05, we reject the null hypothesis and 
conclude that BTC’s returns are significantly related to the market (SPY), 
meaning beta is statistically significant and is a nonzero number.
"""

#Calulate Beta Alternative way
print(Stocks.cov())
Cov = Stocks.cov()
print(SPY_Monthly_Return.var())
Var = SPY_Monthly_Return.var()
beta2 = (Cov/Var)
print(beta2)
# BTC's beta is calculated based on covariance of the stock return with market return and variance of market return

"""
Interpretation:

The beta coefficient in the regression output represents the sensitivity of BTC's excess returns to SPY's excess returns (which represents the market in this case).
There is a positive relationship between SPY (market) and BTC returns
Specifically, for every 1% increase in the excess return of SPY, BTC's excess return is expected to increase by the calculated beta.
"""

# Scatter plot of Excess_BTC vs. Excess_SPY
plt.scatter(data['Excess_SPY'], data['Excess_BTC'], color='blue', label='Data Points')
# Generate predicted values based on the regression model
predicted_values = model.predict(X)
# Plot the regression line
plt.plot(data['Excess_SPY'], predicted_values, color='red', label='Fitted Line')
# Add labels and title
plt.xlabel('Excess SPY Returns')
plt.ylabel('Excess BTC Returns')
plt.title('Excess BTC Returns vs. Excess SPY Returns')
# Show the legend
plt.legend()
# Display the plot
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf
from scipy import stats
from fredapi import Fred
import pandas_datareader.data as web

# Get Fama-French Factors
ff_factors = web.DataReader("F-F_Research_Data_Factors", "famafrench", start="2008-01-01")[0]
ff_factors.index = pd.to_datetime(ff_factors.index, format='%Y%m')  # Convert index to datetime
ff_factors = ff_factors.resample("M").last() / 100  # Convert to percentage
ff_factors = ff_factors.loc[ff_factors.index >= '2008-01-01']  # Aligning the start date

# Get 3-month Treasury Bill (risk-free rate)
tbill = fred.get_series('TB3MS', start='2008-01-01', end='2023-12-31')
tbill.index = pd.to_datetime(tbill.index)
tbill = tbill.resample("M").last().dropna()
tbill = tbill / 100 / 12  # Convert annualized rate to monthly risk-free rate

# Align returns and factors data
data = pd.DataFrame({
    'BTC_Return': BTC_Monthly_Return,
    'SPY_Return': SPY_Monthly_Return,
    'TBILL': tbill
}).dropna()

# Calculate excess returns
data['Excess_BTC'] = data['BTC_Return'] - data['TBILL']
data['Excess_SPY'] = data['SPY_Return'] - data['TBILL']

# Merge Fama-French factors with returns data
data_ff = pd.merge(data, ff_factors, left_index=True, right_index=True).dropna()

# CAPM Model with Fama-French factors
X = sm.add_constant(data_ff[['Mkt-RF', 'SMB', 'HML']])  # Market return, SMB, HML
model_ff = sm.OLS(data_ff['Excess_BTC'], X).fit()

# Print the regression results
print(model_ff.summary())

# If needed: Scatter plot of Excess_BTC vs. Mkt-RF
plt.scatter(data_ff['Mkt-RF'], data_ff['Excess_BTC'], color='blue', label='Data Points')
predicted_values_ff = model_ff.predict(X)
plt.plot(data_ff['Mkt-RF'], predicted_values_ff, color='red', label='Fitted Line')
plt.xlabel('Excess Market Returns (Mkt-RF)')
plt.ylabel('Excess BTC Returns')
plt.title('Excess BTC Returns vs. Excess Market Returns (Mkt-RF)')
plt.legend()
plt.show()