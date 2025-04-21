#test
import yfinance as yf
import pandas as pd
from datetime import datetime
import os


data_dir = 'data_backup/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


core_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'V', 'PG', 'NVDA']  # Example tickers

# Economic indicators available on Yahoo Finance
economic_indicators = {
    'US GDP': '^GSPC',           # S&P 500 index as a proxy for overall economy
    'Inflation': '^TNX',         # U.S. 10-year Treasury yield as a proxy for inflation expectations
    'Unemployment': '^DJI',      # Dow Jones Industrial Average as a proxy for economic sentiment
}


def enhance_market_data():
    # Add VIX data for volatility regime
    vix_data = yf.download('^VIX', start="2015-01-01")
    
    # Add market breadth indicators
    spy_data = yf.download('SPY', start="2015-01-01")
    
    # Add sector rotation data
    sector_etfs = ['XLF', 'XLK', 'XLE', 'XLV', 'XLI']
    sector_data = pd.DataFrame()
    for etf in sector_etfs:
        sector_data[etf] = yf.download(etf, start="2015-01-01")['Close']
    
    return vix_data, spy_data, sector_data

def download_stock_data(ticker):
    """Download historical stock data for a given ticker."""
    stock_data = yf.download(ticker, start="2015-01-01", end=datetime.today().strftime('%Y-%m-%d'))
    stock_data.reset_index(inplace=True)
    stock_data['Ticker'] = ticker
    return stock_data

def download_economic_data():
    """Download economic indicators data from Yahoo Finance."""
    economic_data = pd.DataFrame()
    for name, ticker in economic_indicators.items():
        data = yf.download(ticker, start="2015-01-01", end=datetime.today().strftime('%Y-%m-%d'))
        data = data[['Close']].rename(columns={'Close': name})
        data['Date'] = data.index
        data.reset_index(drop=True, inplace=True)
        if economic_data.empty:
            economic_data = data
        else:
            economic_data = pd.merge(economic_data, data, on='Date', how='outer')
    return economic_data

def save_to_csv(df, filename):
    """Save a DataFrame to a CSV file."""
    df.to_csv(data_dir + filename, index=False)


all_stock_data = pd.DataFrame()
for ticker in core_stocks:
    stock_data = download_stock_data(ticker)
    save_to_csv(stock_data, f"{ticker}_stock_data.csv")  # Save to CSV
    all_stock_data = pd.concat([all_stock_data, stock_data], axis=0)

economic_data = download_economic_data()
save_to_csv(economic_data, "economic_data.csv")

print("Data download and storage complete.")
