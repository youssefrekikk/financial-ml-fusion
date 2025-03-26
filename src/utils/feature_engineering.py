import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def add_features(data):
    """
    Add technical indicators with careful handling of lookback periods.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing OHLCV data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added technical indicators
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Ensure numeric data types
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Price-based features
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Return_Volatility'] = df['Log_Return'].rolling(window=21).std()
    
    # Range-based features
    df['True_Range'] = np.maximum(
        df['High'] - df['Low'],
        np.abs(df['High'] - df['Close'].shift(1)),
        np.abs(df['Low'] - df['Close'].shift(1))
    )
    df['ATR_14'] = df['True_Range'].rolling(window=14).mean()
    df['ATR_Pct'] = df['ATR_14'] / df['Close']
    
    # Price relationship features
    df['Open_Close_Diff'] = (df['Close'] - df['Open']) / df['Open']
    df['High_Low_Diff'] = (df['High'] - df['Low']) / df['Open']
    df['Close_Prev_Close_Diff'] = df['Close'].pct_change()
    
    # Moving averages - use shorter lookback periods for testing
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Only calculate SMA_200 if we have enough data
    if len(df) >= 200:
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
    else:
        # Use a shorter period as fallback
        lookback = min(len(df) // 2, 100)
        if lookback > 0:
            df['SMA_200'] = df['Close'].rolling(window=lookback).mean()
        else:
            df['SMA_200'] = df['Close']  # Fallback to price itself
    
    # Moving average relationships - handle potential division by zero
    df['SMA_20_50_Ratio'] = np.where(df['SMA_50'] > 0, df['SMA_20'] / df['SMA_50'], 1)
    df['SMA_20_200_Ratio'] = np.where(df['SMA_200'] > 0, df['SMA_20'] / df['SMA_200'], 1)
    df['Price_SMA_20_Ratio'] = np.where(df['SMA_20'] > 0, df['Close'] / df['SMA_20'], 1)
    
    # RSI (Relative Strength Index) - handle division by zero
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    # Avoid division by zero
    rs = np.where(loss > 0, gain / loss, 0)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands - handle potential division by zero
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    # Avoid division by zero
    df['BB_Width'] = np.where(df['BB_Middle'] > 0,
                            (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'],
                            0)
    # Handle division by zero and potential NaN
    bb_range = df['BB_Upper'] - df['BB_Lower']
    df['BB_Pct'] = np.where(bb_range > 0,
                        (df['Close'] - df['BB_Lower']) / bb_range,
                        0.5)  # Default to middle
    
    # Momentum indicators
    df['Momentum_14'] = df['Close'].pct_change(periods=14)
    df['ROC_14'] = df['Close'].pct_change(periods=14) * 100
    
    # Volume indicators
    if 'Volume' in df.columns:
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        # Avoid division by zero
        df['Relative_Volume'] = np.where(df['Volume_MA_20'] > 0,
                                        df['Volume'] / df['Volume_MA_20'],
                                        1)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # CRITICAL CHANGE: Instead of dropping NaN rows, fill them with appropriate values
    # This ensures we don't lose data points
    
    # For momentum/trend indicators, forward fill is appropriate
    momentum_cols = ['SMA_20', 'SMA_50', 'SMA_200', 'SMA_20_50_Ratio',
                    'SMA_20_200_Ratio', 'Price_SMA_20_Ratio', 'MACD',
                    'MACD_Signal', 'MACD_Hist', 'BB_Middle', 'BB_Upper',
                    'BB_Lower', 'BB_Width', 'BB_Pct']
    df[momentum_cols] = df[momentum_cols].fillna(method='ffill')
    
    # For oscillators, fill with neutral values
    df['RSI'] = df['RSI'].fillna(50)  # Neutral RSI
    
    # For volatility indicators, fill with median or mean
    volatility_cols = ['Return_Volatility', 'ATR_14', 'ATR_Pct']
    for col in volatility_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
    
    # For return-based features, fill with 0 (no change)
    return_cols = ['Log_Return', 'Open_Close_Diff', 'High_Low_Diff',
                'Close_Prev_Close_Diff', 'Momentum_14', 'ROC_14']
    df[return_cols] = df[return_cols].fillna(0)
    
    # For volume indicators, fill with 1 (neutral)
    if 'Relative_Volume' in df.columns:
        df['Relative_Volume'] = df['Relative_Volume'].fillna(1)
    
    # Only drop rows where Close price is NaN, as this is essential
    df = df.dropna(subset=['Close'])
    
    # Print data shape after feature engineering
    print(f"Data shape after feature engineering: {df.shape}")
    
    return df



# Create a global scaler that can be used across function calls
_scaler = None

def normalize_features(features, fit=False):
    """
    Normalize features using StandardScaler.
    
    Parameters:
    -----------
    features : pd.DataFrame or np.ndarray
        Features to normalize
    fit : bool
        Whether to fit the scaler on this data
        
    Returns:
    --------
    np.ndarray
        Normalized features
    """
    global _scaler
    
    if fit or _scaler is None:
        _scaler = StandardScaler()
        normalized_data = _scaler.fit_transform(features)
    else:
        normalized_data = _scaler.transform(features)
    
    return normalized_data, _scaler



