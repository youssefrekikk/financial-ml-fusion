#unfinished/ still expperimenting  to try and run move to src
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import date, timedelta
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils import load_data,add_features

# Import custom modules
from time_series_transformer import (TimeSeriesTransformer, 
                                    generate_signal_with_confidence, 
                                    StockDataset)
from trade_simulator import TradeSimulator

def custom_loss(outputs, targets):
    """
    Custom loss function that adds a penalty for predictions close to zero
    to encourage the model to make stronger predictions.
    
    Args:
        outputs: Model predictions
        targets: Target values
    
    Returns:
        Combined loss value
    """
    # Standard MSE loss
    mse_loss = torch.nn.MSELoss()(outputs, targets)
    
    # Add penalty for predictions close to zero to encourage stronger signals
    # The closer to zero, the higher the penalty
    zero_penalty = 0.1 * torch.mean(1.0 / (torch.abs(outputs) + 0.05))
    
    return mse_loss + zero_penalty

def train_transformer_model(csv_file, seq_len=10, epochs=20, batch_size=32, lr=1e-3, 
                           model_dim=64, num_heads=4, num_layers=2, dropout=0.2,
                           k_best_features=10):
    """
    Train the transformer model on a single stock with feature selection.
    
    Args:
        csv_file (str): Path to CSV file with stock data
        seq_len (int): Sequence length for the model
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        lr (float): Learning rate
        model_dim (int): Model dimension
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer layers
        dropout (float): Dropout rate
        k_best_features (int): Number of best features to select
    
    Returns:
        tuple: Trained model and feature list
    """
    print(f"Training model on {csv_file}...")
    
    # Create dataset with feature selection
    dataset = StockDataset(csv_file, seq_len=seq_len, k_best_features=k_best_features)
    
    # Split into train and validation sets (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Get input dimension from dataset
    input_dim = len(dataset.feature_list)
    feature_list = dataset.feature_list
    
    # Create model with updated architecture
    model = TimeSeriesTransformer(
        input_dim=input_dim, 
        model_dim=model_dim, 
        num_heads=num_heads, 
        num_layers=num_layers, 
        dropout=dropout
    )
    
    # Train model with validation
    criterion = custom_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
            print(f"New best model saved with validation loss: {best_val_loss:.6f}")
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # Test the model on a few samples to verify it's producing varied outputs
    model.eval()
    with torch.no_grad():
        # Test on training sample
        sample_x, sample_y = train_dataset[0]
        pred = model(sample_x.unsqueeze(0))
        print(f"Training sample prediction: {pred.item():.6f}, Target: {sample_y.item():.6f}")
        
        # Test on validation sample
        sample_x, sample_y = val_dataset[0]
        pred = model(sample_x.unsqueeze(0))
        print(f"Validation sample prediction: {pred.item():.6f}, Target: {sample_y.item():.6f}")
    
    print("Training complete.")
    return model, feature_list


def generate_predictions_with_dynamic_threshold(model, df, feature_list, seq_len=10):
    """
    Generate predictions for a DataFrame using the trained model with dynamic threshold.
    
    Args:
        model (TimeSeriesTransformer): Trained model
        df (DataFrame): Stock data
        feature_list (list): List of features to use
        seq_len (int): Sequence length for the model
    
    Returns:
        DataFrame: Original DataFrame with predictions added
        float: Dynamic threshold calculated from prediction distribution
    """
    model.eval()
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Apply feature engineering to the test data
    result_df = add_features(result_df)
    
    print(f"Data shape after feature engineering: {result_df.shape}")
    
    # Initialize columns for predictions
    result_df['Prediction'] = np.nan
    result_df['Signal'] = np.nan
    result_df['Confidence'] = np.nan
    
    # We need at least seq_len days to make predictions
    if len(result_df) <= seq_len:
        print("Not enough data for predictions")
        return result_df, 0.03
    
    # Check if all required features exist in the dataframe
    missing_features = [f for f in feature_list if f not in result_df.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        print(f"Available columns: {result_df.columns.tolist()}")
        # Use only available features
        feature_list = [f for f in feature_list if f in result_df.columns]
        if not feature_list:
            raise ValueError("No usable features found")
    
    # First pass: collect all predictions to calculate dynamic threshold
    all_predictions = []
    
    for i in range(seq_len, len(result_df)):
        # Get sequence data
        sequence = result_df.iloc[i-seq_len:i][feature_list].values.astype(np.float32)
        sequence_tensor = torch.tensor(sequence).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            prediction = model(sequence_tensor)
        
        pred_value = prediction.item()
        all_predictions.append(pred_value)
        
        # Store prediction (without signal yet)
        result_df.iloc[i, result_df.columns.get_loc('Prediction')] = pred_value
    
    # Calculate dynamic threshold based on prediction distribution
    # Use a fraction of the standard deviation of predictions
    if len(all_predictions) > 0:
        pred_std = np.std(all_predictions)
        dynamic_threshold = 0.5 * pred_std  # Adjust this multiplier as needed
        print(f"Prediction std: {pred_std:.6f}, Dynamic threshold: {dynamic_threshold:.6f}")
    else:
        dynamic_threshold = 0.03  # Fallback to default
    
    # Second pass: generate signals using the dynamic threshold
    for i in range(seq_len, len(result_df)):
        pred_value = result_df.iloc[i, result_df.columns.get_loc('Prediction')]
        if pd.isna(pred_value):
            continue
            
        signal, confidence = generate_signal_with_confidence(pred_value, dynamic_threshold)
        
        # Store results
        result_df.iloc[i, result_df.columns.get_loc('Signal')] = signal
        result_df.iloc[i, result_df.columns.get_loc('Confidence')] = confidence
        
        # Debug output for first few predictions
        if i < seq_len + 20 or i % 50 == 0:
            print(f"Debug: prediction={pred_value}, threshold={dynamic_threshold:.6f}, signal={'BUY' if signal == 1 else 'SELL' if signal == -1 else 'HOLD'}, confidence={confidence:.4f}")
    
    # Count signal types
    buy_signals = (result_df['Signal'] == 1).sum()
    sell_signals = (result_df['Signal'] == -1).sum()
    hold_signals = (result_df['Signal'] == 0).sum()
    
    print(f"Signal distribution: BUY: {buy_signals}, SELL: {sell_signals}, HOLD: {hold_signals}")
    
    return result_df, dynamic_threshold

def evaluate_predictions(df):
    """
    Evaluate prediction accuracy.
    
    Args:
        df (DataFrame): DataFrame with predictions and actual returns
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Filter out rows without predictions
    eval_df = df.dropna(subset=['Prediction'])
    
    if len(eval_df) == 0:
        return {"error": "No valid predictions to evaluate"}
    
    # Calculate actual next-day returns
    eval_df['Actual_Return'] = eval_df['Close'].pct_change(1).shift(-1)
    
    # Remove the last row (no actual return available)
    eval_df = eval_df[:-1]
    
    # Calculate metrics
    mse = np.mean((eval_df['Prediction'] - eval_df['Actual_Return']) ** 2)
    mae = np.mean(np.abs(eval_df['Prediction'] - eval_df['Actual_Return']))
    
    # Direction accuracy (did we predict the sign correctly?)
    direction_correct = np.sign(eval_df['Prediction']) == np.sign(eval_df['Actual_Return'])
    direction_accuracy = np.mean(direction_correct)
    
    # Signal accuracy
    signal_correct = (
        (eval_df['Signal'] == 1) & (eval_df['Actual_Return'] > 0) |
        (eval_df['Signal'] == -1) & (eval_df['Actual_Return'] < 0) |
        (eval_df['Signal'] == 0) & (abs(eval_df['Actual_Return']) < 0.005)  # Small threshold for "no change"
    )
    signal_accuracy = np.mean(signal_correct)
    
    # Count signals by type
    buy_signals = sum(eval_df['Signal'] == 1)
    sell_signals = sum(eval_df['Signal'] == -1)
    hold_signals = sum(eval_df['Signal'] == 0)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'Direction_Accuracy': direction_accuracy,
        'Signal_Accuracy': signal_accuracy,
        'Total_Predictions': len(eval_df),
        'Buy_Signals': buy_signals,
        'Sell_Signals': sell_signals,
        'Hold_Signals': hold_signals,
        'Signal_Rate': (buy_signals + sell_signals) / len(eval_df) if len(eval_df) > 0 else 0
    }

def run_trade_simulation(stock_data, threshold):
    """
    Run a trading simulation using the TradeSimulator.
    
    Args:
        stock_data (dict): Dictionary of stock DataFrames with predictions
        threshold (float): Threshold used for signal generation
    
    Returns:
        dict: Dictionary of simulation results
    """
    # Initialize the trade simulator
    simulator = TradeSimulator(
        initial_capital=100000,
        commission=0.001,           # 0.1% commission per trade
        slippage=0.0005,            # 0.05% slippage
        stop_loss_pct=0.02,         # 2% stop loss
        trailing_stop_pct=0.03,     # 3% trailing stop
        take_profit_pct=0.05,       # 5% take profit
        max_position_size=0.2,      # Max 20% of portfolio per position
        risk_per_trade=0.01         # Risk 1% of portfolio per trade
    )
    
    # Prepare data for simulation
    data = {}  # Dictionary of price data for simulator
    signals = {}  # Dictionary of signals for simulator
    
    for ticker, df in stock_data.items():
        # Filter out rows without predictions
        pred_df = df.dropna(subset=['Prediction'])
        
        # Add to data dictionary
        data[ticker] = pred_df
        
        # Create signals dictionary
        for idx, row in pred_df.iterrows():
            if pd.isna(row['Signal']):
                continue
                
            timestamp = idx
            if timestamp not in signals:
                signals[timestamp] = {}
            
            signals[timestamp][ticker] = {
                'signal': int(row['Signal']),
                'confidence': float(row['Confidence'])
            }
    
    print(f"Running simulation with {len(signals)} trading days and threshold {threshold:.6f}")
    
    # Run simulation
    simulation_results = simulator.run_simulation(data, signals)
    
    # Add simulator instance to results for plotting
    simulation_results['simulator'] = simulator
    
    return simulation_results

def plot_simulation_results(simulator, title="Trading Simulation Results"):
    """
    Plot the results of the trading simulation.
    
    Args:
        simulator (TradeSimulator): The trade simulator with results
        title (str): Title for the plots
    """
    # Create a directory for plots if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Get equity curve
    equity_curve = simulator.get_equity_curve()
    
    if equity_curve.empty:
        print("No equity data to plot.")
        return
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve.index, equity_curve['portfolio_value'])
    plt.title(f'{title} - Portfolio Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.savefig(f"plots/equity_curve.png")
    plt.close()
    
    # Plot drawdown curve
    running_max = equity_curve['portfolio_value'].cummax()
    drawdown = (running_max - equity_curve['portfolio_value']) / running_max * 100
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve.index, drawdown)
    plt.fill_between(equity_curve.index, drawdown, 0, alpha=0.3)
    plt.title(f'{title} - Portfolio Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.savefig(f"plots/drawdown_curve.png")
    plt.close()
    
    # Get trade log
    trade_log = simulator.get_trade_log()
    
    # Plot trade distribution
    if not trade_log.empty:
        plt.figure(figsize=(12, 6))
        
                # Group trades by symbol
        trade_counts = trade_log['symbol'].value_counts()
        
        plt.bar(trade_counts.index, trade_counts.values)
        plt.title(f'{title} - Trade Distribution by Symbol')
        plt.xlabel('Symbol')
        plt.ylabel('Number of Trades')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"plots/trade_distribution.png")
        plt.close()
        
        # Plot trade P&L distribution if available
        if 'pnl' in trade_log.columns:
            plt.figure(figsize=(12, 6))
            plt.hist(trade_log['pnl'], bins=20)
            plt.title(f'{title} - Trade P&L Distribution')
            plt.xlabel('P&L ($)')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(f"plots/pnl_distribution.png")
            plt.close()
    
    
def plot_predictions(df, ticker):
    """
    Plot the predictions against actual returns.
    
    Args:
        df (DataFrame): DataFrame with predictions and actual returns
        ticker (str): Ticker symbol for the title
    """
    # Create a directory for plots if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Filter out rows without predictions
    plot_df = df.dropna(subset=['Prediction'])
    
    if len(plot_df) == 0:
        print("No predictions to plot.")
        return
    
    # Calculate actual returns if not already present
    if 'Actual_Return' not in plot_df.columns:
        plot_df['Actual_Return'] = plot_df['Close'].pct_change(1).shift(-1)
        plot_df = plot_df[:-1]  # Remove last row (no actual return)
    
    # Plot predictions vs actual returns
    plt.figure(figsize=(14, 7))
    plt.plot(plot_df.index, plot_df['Actual_Return'], label='Actual Return', alpha=0.7)
    plt.plot(plot_df.index, plot_df['Prediction'], label='Predicted Return', alpha=0.7)
    
    # Add buy/sell signals
    buy_signals = plot_df[plot_df['Signal'] == 1]
    sell_signals = plot_df[plot_df['Signal'] == -1]
    
    plt.scatter(buy_signals.index, buy_signals['Prediction'], 
                color='green', label='Buy Signal', marker='^', s=100)
    plt.scatter(sell_signals.index, sell_signals['Prediction'], 
                color='red', label='Sell Signal', marker='v', s=100)
    
    plt.title(f'{ticker} - Predicted vs Actual Returns')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{ticker}_predictions.png")
    plt.close()
    
    # Plot signal confidence
    plt.figure(figsize=(14, 7))
    
    # Create a colormap for the confidence
    colors = []
    for signal, conf in zip(plot_df['Signal'], plot_df['Confidence']):
        if signal == 1:  # Buy
            colors.append((0, conf, 0))  # Green with intensity based on confidence
        elif signal == -1:  # Sell
            colors.append((conf, 0, 0))  # Red with intensity based on confidence
        else:  # Hold
            colors.append((0.7, 0.7, 0.7))  # Gray
    
    plt.scatter(plot_df.index, plot_df['Prediction'], c=colors, s=50)
    plt.title(f'{ticker} - Signal Confidence')
    plt.xlabel('Date')
    plt.ylabel('Predicted Return')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{ticker}_signal_confidence.png")
    plt.close()    

def main():
    # Create necessary directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Define parameters
    ticker ="AAPL"
    data_dir = "data"
    csv_file = os.path.join(data_dir, f"{ticker}.csv")
    test_start_date = "2018-01-01"
    seq_len = 10
    k_best_features = 10  # Number of features to select
    
    if not os.path.exists(csv_file):
        print(f"Error: Training file '{csv_file}' not found.")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in data directory: {os.listdir(data_dir) if os.path.exists(data_dir) else 'data directory not found'}")
        return
    
    # Train the model with feature selection
    model, feature_list = train_transformer_model(
        csv_file=csv_file,
        seq_len=seq_len,
        epochs=30,
        batch_size=32,
        lr=1e-3,
        model_dim=64,
        num_heads=4,
        num_layers=2,  # Reduced from 3 to 2
        dropout=0.2,
        k_best_features=k_best_features
    )
    
    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_list': feature_list,
    }, "models/transformer_model.pt")
    
    # Load test data for each ticker individually
    stock_data = load_data(ticker, start_date=test_start_date)
    # If load_data returns a dictionary, extract the ticker's data
    if isinstance(stock_data, dict):
        df = stock_data.get(ticker)
    # If load_data returns a DataFrame directly
    else:
        df = stock_data
    
    if df is None or len(df) == 0:
        print(f"Error: No data loaded for {ticker}")
        return
    
    print(f"Loaded {len(df)} rows of test data for {ticker}")
    
    # Generate predictions with dynamic threshold
    print(f"Generating predictions for {ticker}...")
    pred_df, dynamic_threshold = generate_predictions_with_dynamic_threshold(
        model, df, feature_list, seq_len
    )
    
    # Save predictions to CSV
    pred_df.to_csv(f"results/{ticker}_predictions.csv")
    
    # Evaluate predictions
    metrics = evaluate_predictions(pred_df)
    
    # Print evaluation results
    print("\nPrediction Evaluation Results:")
    print("==============================")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    
    # Plot predictions
    plot_predictions(pred_df, ticker)
    
    # Run trade simulation
    print("\nRunning Trade Simulation...")
    simulation_results = run_trade_simulation({ticker: pred_df}, dynamic_threshold)
    
    # Print simulation results
    print("\nTrade Simulation Results:")
    print("========================")
    for metric, value in simulation_results.items():
        if metric not in ['portfolio_history', 'trade_history', 'simulator']:
            print(f"  {metric}: {value}")
    
    # Plot simulation results
    plot_simulation_results(simulation_results['simulator'] if 'simulator' in simulation_results else None)
    
    # Generate summary report
    generate_summary_report({ticker: metrics}, simulation_results)
    
    print("\nEvaluation complete. Results saved to 'results' directory.")

def generate_summary_report(evaluation_results, simulation_results):
    """
    Generate a summary report of all results.
    
    Args:
        evaluation_results (dict): Dictionary of evaluation metrics by ticker
        simulation_results (dict): Dictionary of simulation results
    """
    # Create a summary DataFrame for evaluation metrics
    eval_summary = pd.DataFrame(evaluation_results).T
    
    # Save to CSV
    eval_summary.to_csv("results/evaluation_summary.csv")
    
    # Create a summary of simulation results
    sim_summary = pd.Series({k: v for k, v in simulation_results.items() 
                            if k not in ['portfolio_history', 'trade_history', 'simulator']})
    
    # Save to CSV
    sim_summary.to_frame('Value').to_csv("results/simulation_summary.csv")
    
    # Print summary to console
    print("\nEvaluation Summary:")
    print("===================")
    for ticker, metrics in evaluation_results.items():
        print(f"\n{ticker}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    print("\nSimulation Summary:")
    print("==================")
    for metric, value in sim_summary.items():
        print(f"  {metric}: {value}")

if __name__ == "__main__":
    main()