#this is old implementation of a hidden markov model not in use anymore 

import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import time
from datetime import date
from sklearn.preprocessing import StandardScaler

def load_data(ticker,start_date="2018-01-01",end_date=date.today().strftime('%Y-%m-%d')):
    data = yf.download(ticker, start=start_date,end=end_date)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    
    
    return data







def add_features(data):
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    
    
    data['True_Range'] = np.maximum(
        data['High'] - data['Low'], 
        np.abs(data['High'] - data['Close'].shift(1)),
        np.abs(data['Low'] - data['Close'].shift(1))
    )
    
    # Average True Range (ATR)
    data['ATR_14'] = data['True_Range'].rolling(window=14).mean()
    
    # Open-Close Relationship
    data['Open_Close_Diff'] = data['Close'] - data['Open']
    
   
    data['Open_Close_Percentage'] = np.zeros(len(data))  # Initialize with zeros
    mask = data['Open'] != 0  # Create a mask for non-zero 'Open' values
    data.loc[mask, 'Open_Close_Percentage'] = (data.loc[mask, 'Open_Close_Diff'] / data.loc[mask, 'Open']) * 100
    
    # Volume-based Features (if available)
    if 'Volume' in data.columns:
        data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()
        data['Relative_Volume'] = data['Volume'] / data['Volume_MA_20']
    
    # RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema_12 - ema_26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    data['20d_MA'] = data['Close'].rolling(window=20).mean()
    data['Upper_Band'] = data['20d_MA'] + (data['Close'].rolling(window=20).std() * 2)
    data['Lower_Band'] = data['20d_MA'] - (data['Close'].rolling(window=20).std() * 2)
    data['Bollinger_Width'] = (data['Upper_Band'] - data['Lower_Band']) / data['20d_MA']
    
    # Momentum Indicator
    data['Momentum_14'] = data['Close'].pct_change(periods=14)
    
    
    data = data.dropna()
    return data


# Prepare data for HMM model
def prepare_hmm_data(data):
    hmm_features = [
        'Log_Return',
        'True_Range',
        'ATR_14',
        'Open_Close_Percentage',
        'RSI',
        'MACD',
        'MACD_Signal',
        'Bollinger_Width',
        'Momentum_14'
    ]
    
    # Add volume feature if available
    if 'Relative_Volume' in data.columns:
        hmm_features.append('Relative_Volume')
    
    return data[hmm_features]

def normalize_hmm_feautures(data):
    normalized_data = data.copy()
    
    scaler = StandardScaler()
    
    normalized_values = scaler.fit_transform(data)
    
    # Replace data with normalized values
    normalized_data.loc[:] = normalized_values
    
    return normalized_data, scaler
    

# Optimize HMM parameters for best separation
def optimize_hmm_parameters(hmm_data):
    best_model = None
    best_score = -np.inf
    best_params = {}
    
    param_grid = {
        "n_components": [2, 3, 4],
        "covariance_type": ["full", "diag", "tied", "spherical"]
    }
    
    for params in ParameterGrid(param_grid):
        model = GaussianHMM(n_components=params['n_components'], covariance_type=params['covariance_type'], n_iter=1000)
        model.fit(hmm_data)
        
        # Predict hidden states and calculate silhouette score
        hidden_states = model.predict(hmm_data)
        score = silhouette_score(hmm_data, hidden_states)
        
        if score > best_score:
            best_score = score
            best_model = model
            best_params = params
    
    print("Best HMM Parameters:", best_params)
    return best_model



# Predict the current state using a rolling window (e.g., last 30 days)
def predict_current_state(model, recent_data, scaler, window_size=30):
    
    hmm_features = [
        'Log_Return',
        'True_Range',
        'ATR_14',
        'Open_Close_Percentage',
        'RSI',
        'MACD',
        'MACD_Signal',
        'Bollinger_Width',
        'Momentum_14'
    ]
    
    if 'Relative_Volume' in recent_data.columns:
        hmm_features.append('Relative_Volume')
    
    # Ensure we have the correct features
    hmm_recent_data = recent_data[-window_size:][hmm_features]
    
    # Normalize the recent data using the same scaler used during training
    normalized_hmm_recent_data = scaler.transform(hmm_recent_data)
    
    # Predict the state based on the recent rolling window
    state = model.predict(normalized_hmm_recent_data)[-1]
    return state




def plot_states(data,ticker):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the Closing Price
    ax.plot(data.index, data['Close'], color='black', label='Closing Price', linewidth=1.5, alpha=0.7)
    
    # Plot each state with a different color
    for state in range(data['State'].nunique()):
        state_data = data[data['State'] == state]
        ax.scatter(state_data.index, state_data['Close'], label=f'State {state}', alpha=0.6)
    
    # Set plot title and labels
    ax.set(title=f'{ticker} Market Regimes with Closing Price', xlabel='Date', ylabel='Price')
    ax.legend()
    plt.show()



# Improved rolling window implementation
def rolling_window_hmm(data, window_size=90, step_size=1):
    states = []
    n_rows = data.shape[0]
    
    for start in range(0, n_rows - window_size, step_size):
        end = start + window_size
        
        window_data = data.iloc[start:end]
        hmm_data = prepare_hmm_data(window_data)
        
        normalized_hmm_data, _ = normalize_hmm_feautures(hmm_data)
        
        hmm_model = optimize_hmm_parameters(normalized_hmm_data)
        
        window_states = hmm_model.predict(normalized_hmm_data)
        
        window_timestamps = window_data.index
        states.extend(list(zip(window_timestamps, window_states)))
    
    return states

# Consistently map HMM states using clustering
def remap_states(states, n_clusters=3):
    from sklearn.cluster import KMeans
    
    # Cluster state sequences into a consistent set of labels
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    remapped_states = kmeans.fit_predict(states.reshape(-1, 1))
    
    return remapped_states






def main():
    
   

    # Load data for a tech stock (e.g., AAPL)
    ticker = "TSLA"
    print(f"Loading data for {ticker}...")
    data = load_data(ticker,"2020-01-01","2021-01-01")
    
    # Step 1: Feature Engineering
    print("Adding technical indicators...")
    data = add_features(data)
    
    # Step 2: Prepare HMM Features
    print("Preparing HMM data...")
    hmm_data = prepare_hmm_data(data)
    
    # Normalize the data
    print("Normalizing data...")
    normalized_hmm_data, scaler = normalize_hmm_feautures(hmm_data)
    
    # Step 3: Train HMM Model
    print("Optimizing and training HMM model...")
    hmm_model = optimize_hmm_parameters(normalized_hmm_data)
    
    # Predict states for the entire dataset
    print("Predicting states for the dataset...")
    data['State'] = hmm_model.predict(normalized_hmm_data)
    
    # Step 4: Predict Current Market State
    print("Predicting the current market state...")
    current_state = predict_current_state(hmm_model, data, scaler, window_size=30)
    print(f"The predicted current market state for {ticker}: State {current_state}")
    
    # Step 5: Apply Rolling Window HMM for Historical Analysis
    print("Applying rolling window HMM...")
    predicted_states = rolling_window_hmm(data, window_size=60, step_size=10)
    
    # Create a state series for visualization
    state_series = pd.Series(
        dict(predicted_states),
        name="State"
    )
    
    # Ensure the index is datetime and matches the original data
    state_series.index = pd.to_datetime(state_series.index)
    data_with_states = data.join(state_series, how="left", rsuffix="_rolling")
    
    # Step 6: Visualize the Results
    print("Visualizing market regimes...")
    plot_states(data_with_states,ticker)
    print("Visualization complete!")

if __name__ == "__main__":
    main()
