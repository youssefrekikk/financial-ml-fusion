import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
from sklearn.preprocessing import StandardScaler

def load_data(ticker, start_date="2018-01-01", end_date=date.today().strftime('%Y-%m-%d'), use_local=True):
    """
    Load financial data for a given ticker with option to use local files.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    use_local : bool
        Whether to try loading from local files first
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the stock data
    """
    # Try loading from local CSV first if use_local is True
    if use_local:
        data_dir = "data"
        file_path = os.path.join(data_dir, f"{ticker.replace('^', '')}.csv")
        
        if os.path.exists(file_path):
            try:
                # Explicitly set numeric columns to float
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Convert price and volume columns to numeric
                numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                for col in numeric_columns:
                    if col in data.columns:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                
                # Filter by date range
                data = data[(data.index >= start_date) & (data.index <= end_date)]
                
                # Check if we have valid data
                if len(data) > 0 and not data['Close'].isna().all():
                    print(f"Successfully loaded {len(data)} rows from local file for {ticker}")
                    return data
                else:
                    print(f"Local file exists but no valid data in specified date range")
            except Exception as e:
                print(f"Error loading local file for {ticker}: {e}")
    
   
