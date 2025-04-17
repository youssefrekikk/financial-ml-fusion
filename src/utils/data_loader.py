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
                # Custom loading for the unusual CSV format
                # Skip the first 3 rows and use the 1st row as header
                data = pd.read_csv(file_path, skiprows=3, header=None)
                
                # Assign column names from the first row of the file
                with open(file_path, 'r') as f:
                    header = f.readline().strip().split(',')
                
                data.columns = header
                
                # Convert the first column to datetime and set as index
                data[header[0]] = pd.to_datetime(data[header[0]])
                data.set_index(header[0], inplace=True)
                
                # Convert numeric columns
                for col in header[1:]:  # Skip date column
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
    
   
