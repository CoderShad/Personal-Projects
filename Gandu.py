import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf
from fredapi import Fred

# Your CoinMarketCap API Key
cmc_api_key = '27958b18-8e2e-4a33-8155-376c3cdd0406'

# Function to get data from CoinMarketCap API
def get_coinmarketcap_data(url, params):
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': cmc_api_key,
    }
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    return data

# Get Bitcoin market cap and total market cap data
def get_btc_market_cap():
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
    params = {
        'symbol': 'BTC',
        'convert': 'USD'
    }
    data = get_coinmarketcap_data(url, params)
    btc_market_cap = data['data']['BTC']['quote']['USD']['market_cap']
    return btc_market_cap

def get_total_market_cap():
    url = 'https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest'
    params = {
        'convert': 'USD'
    }
    data = get_coinmarketcap_data(url, params)
    total_market_cap = data['data']['quote']['USD']['total_market_cap']
    return total_market_cap

# Example usage to fetch Bitcoin dominance
def get_btc_dominance():
    btc_market_cap = get_btc_market_cap()
    total_market_cap = get_total_market_cap()
    btc_dominance = btc_market_cap / total_market_cap
    return btc_dominance

# Get Bitcoin dominance over time (placeholder for historical data, you need to fetch daily and resample)
btc_dominance = get_btc_dominance()
print(f"Bitcoin Dominance: {btc_dominance * 100:.2f}%")

# Get SPY ETF and BTC data from Yahoo Finance
SPY = yf.download("SPY", start='2008-01-01', end='2023-12-31')
BTC = yf.download("BTC-USD", start='2008-01-01', end='2023-12-31')

# Ensure the index is a DatetimeIndex before resampling
SPY.index = pd.to_datetime(SPY.index)
BTC.index = pd.to_datetime(BTC.index)

# Resample using 'ME' for month-end frequency
SPY_Monthly_Return = (SPY['Adj Close'].resample("ME").last().pct_change().dropna())
BTC_Monthly_Return = (BTC['Adj Close'].resample("ME").last().pct_change().dropna())

# Fetch USDT supply from CoinMarketCap
def get_usdt_supply():
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
    params = {
        'symbol': 'USDT',
        'convert': 'USD'
    }
    data = get_coinmarketcap_data(url, params)
    usdt_market_cap = data['data']['USDT']['quote']['USD']['market_cap']
    return usdt_market_cap

usdt_supply = get_usdt_supply()
print(f"USDT Supply: {usdt_supply}")

# Get 3-month Treasury Bill (risk-free rate) from FRED
fred = Fred(api_key='c6a3529843d2d1e6ff7ea9b09fdb27ea')
tbill = fred.get_series('TB3MS', start='2008-01-01', end='2023-12-31')
tbill.index = pd.to_datetime(tbill.index)
tbill = tbill.resample("ME").last().dropna()
tbill = tbill / 100 / 12  # Convert annualized rate to monthly risk-free rate

# Align returns and factors data
data = pd.DataFrame({
    'BTC_Return': BTC_Monthly_Return,
    'SPY_Return': SPY_Monthly_Return,
    'BTC_Dominance': btc_dominance,  # If you have time-series data, adjust accordingly
    'USDT_Supply': usdt_supply,  # Static value for now; adjust if you can get historical data
    'TBILL': tbill
}).dropna()

# Calculate excess returns
data['Excess_BTC'] = data['BTC_Return'] - data['TBILL']
data['Excess_SPY'] = data['SPY_Return'] - data['TBILL']

# Print the data to check if it's properly aligned and populated
print("Data after alignment and excess return calculation:")
print(data.head())

# Ensure data isn't empty after transformations
if data.empty:
    print("Data is empty after alignment and cleaning")
else:
    # Crypto factor model: BTC returns regressed on SPY returns, BTC dominance, and stablecoin supply
    X = sm.add_constant(data[['Excess_SPY', 'BTC_Dominance', 'USDT_Supply']])
    model_crypto = sm.OLS(data['Excess_BTC'], X).fit()

    # Print the regression results
    print(model_crypto.summary())

    # Plot Scatter plot of Excess_BTC vs. Excess_SPY with other factors
    plt.scatter(data['Excess_SPY'], data['Excess_BTC'], color='blue', label='Data Points')
    predicted_values_crypto = model_crypto.predict(X)
    plt.plot(data['Excess_SPY'], predicted_values_crypto, color='red', label='Fitted Line')
    plt.xlabel('Excess SPY Returns')
    plt.ylabel('Excess BTC Returns')
    plt.title('Excess BTC Returns vs. Excess SPY Returns (With Crypto Factors)')
    plt.legend()
    plt.show()
