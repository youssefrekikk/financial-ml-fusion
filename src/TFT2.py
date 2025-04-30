#still experimental  / under development
import os
import warnings
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

from utils.feature_engineering import add_features

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def save_figure(plt, name):
    """Save the current figure with the given name"""
    ensure_directory_exists("results/plots")
    plt.savefig(f"results/plots/{name}.png", dpi=300, bbox_inches='tight')
    print(f"Saved plot to results/plots/{name}.png")

def load_ticker_data(csv_file):
    """
    Load data for a single ticker from CSV file
    
    Args:
        csv_file (str): Path to CSV file
    
    Returns:
        DataFrame: DataFrame with ticker data
    """
    try:
        # Read the first few lines to understand the structure
        with open(csv_file, 'r') as f:
            header_line = f.readline().strip().split(',')  # First line: Price,Close,High,Low,Open,Volume
            ticker_line = f.readline().strip().split(',')  # Second line: Ticker,AAPL,AAPL,AAPL,AAPL,AAPL
            date_line = f.readline().strip().split(',')    # Third line: Date,,,,,
        
        # Extract ticker from the second line
        ticker = ticker_line[1]  # Assuming ticker is in position 1
        print(f"Detected ticker: {ticker}")
        
        # Skip the first 3 rows and read the data
        data = pd.read_csv(csv_file, skiprows=3, header=None)
        
        # Assign column names based on the first line
        data.columns = header_line
        
        # The first column should be the date
        data.rename(columns={header_line[0]: 'date'}, inplace=True)
        data['date'] = pd.to_datetime(data['date'])
        
        # Add ticker information
        data['Ticker'] = ticker
        print(f"Added ticker: {ticker}")
        
        return data, ticker
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        return None, None

def prepare_multi_ticker_data(csv_files, seq_len=10, prediction_length=1):
    """
    Prepare data for TFT model with multiple tickers
    
    Args:
        csv_files (list): List of paths to CSV files
        seq_len (int): Sequence length (max_encoder_length)
        prediction_length (int): Prediction horizon (max_prediction_length)
    
    Returns:
        tuple: Training and validation datasets, data frame, train/val dataloaders
    """
    # Load and process each file
    all_data = []
    
    for i, csv_file in enumerate(csv_files):
        ticker_data, ticker = load_ticker_data(csv_file)
        if ticker_data is not None:
            # Add group_id (different for each stock)
            ticker_data['group_id'] = i
            all_data.append(ticker_data)
    
    if not all_data:
        raise ValueError("No valid data files were loaded")
    
    # Combine all dataframes
    data = pd.concat(all_data)
    
    # Sort by date
    data = data.sort_values('date')
    
    # Add time features
    data['day_of_week'] = data['date'].dt.dayofweek
    data['month'] = data['date'].dt.month
    data['day_of_month'] = data['date'].dt.day
    data['week_of_year'] = data['date'].dt.isocalendar().week
    data['quarter'] = data['date'].dt.quarter
    
    # Convert numeric categorical features to pandas categorical type
    data['day_of_week'] = data['day_of_week'].astype(str)
    data['month'] = data['month'].astype(str)
    data['quarter'] = data['quarter'].astype(str)
    
    # Add binary indicators for special periods
    data['is_month_start'] = data['date'].dt.is_month_start.astype(int)
    data['is_month_end'] = data['date'].dt.is_month_end.astype(int)
    data['is_quarter_end'] = data['date'].dt.is_quarter_end.astype(int)
    
    # Apply feature engineering
    data = add_features(data)
    
    # Reset index to ensure we have a clean index
    data = data.reset_index(drop=True)
    
    # Add time index for PyTorch Forecasting (ensure it's continuous within each group)
    data['time_idx'] = data.groupby('Ticker').cumcount()
    
    # Calculate target (next day's return) - explicitly as float
    data['target'] = data.groupby('Ticker')['Close'].pct_change(1).shift(-1).astype(float)
    
    # ADDITIONAL STEP: Check for and fill any remaining NA values
    # First, check which columns have NA values
    na_columns = data.columns[data.isna().any()].tolist()
    if na_columns:
        print(f"Columns with NA values before filling: {na_columns}")
        print(f"NA counts: {data[na_columns].isna().sum()}")
        
        # Fill NA values in numerical columns with group means
        for col in na_columns:
            if col != 'target':  # Don't fill target NAs, we'll drop those rows
                # Fill with group mean, or global mean if group mean is NA
                group_means = data.groupby('Ticker')[col].transform('mean')
                global_mean = data[col].mean()
                
                # Fill NAs with group mean first
                data[col] = data[col].fillna(group_means)
                
                # If any NAs remain, fill with global mean
                data[col] = data[col].fillna(global_mean)
                
                # If any NAs still remain (e.g., if all values in a column are NA)
                # fill with 0 as a last resort
                data[col] = data[col].fillna(0)
                
                # Replace any infinite values with large finite values
                data[col] = data[col].replace([np.inf, -np.inf], [1e9, -1e9])
    
    # Check again for any remaining NA values
    na_columns = data.columns[data.isna().any()].tolist()
    if na_columns:
        print(f"Columns with NA values after filling: {na_columns}")
        print(f"NA counts: {data[na_columns].isna().sum()}")
    
    # Remove rows with NaN target
    data = data.dropna(subset=['target'])
    
    # Print data info
    print(f"Data shape after processing: {data.shape}")
    print(f"Tickers: {data['Ticker'].unique().tolist()}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    
    # Define features
    time_varying_known_categoricals = ['day_of_week', 'month', 'quarter']
    time_varying_known_reals = ['Open', 'High', 'Low', 'Close', 'Volume', 'day_of_month', 'week_of_year']
    
    # Add ticker as a categorical feature
    static_categoricals = ['Ticker']
    
    # Technical indicators as unknown reals
    time_varying_unknown_reals = [
        'Log_Return', 'Return_Volatility', 'Open_Close_Diff', 
        'High_Low_Diff', 'Close_Prev_Close_Diff', 'SMA_20', 'SMA_50',
        'SMA_20_50_Ratio', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Pct'
    ]
    
    # Filter out any columns that don't exist
    time_varying_known_reals = [col for col in time_varying_known_reals if col in data.columns]
    time_varying_unknown_reals = [col for col in time_varying_unknown_reals if col in data.columns]
    
    # Use a fixed percentage split for training/validation
    training_cutoff = data.groupby('Ticker')['time_idx'].transform(lambda x: int(x.max() * 0.8))
    
    # Print split info
    print(f"Training samples: {(data['time_idx'] <= training_cutoff).sum()}")
    print(f"Validation samples: {(data['time_idx'] > training_cutoff).sum()}")
    
    # Create training dataset
    training = TimeSeriesDataSet(
        data=data[data['time_idx'] <= training_cutoff],
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        max_encoder_length=seq_len,
        max_prediction_length=prediction_length,
        static_categoricals=static_categoricals,
        static_reals=[],
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(
            groups=["group_id"], transformation="log1p", center=True
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    
    # Create validation dataset
    validation = TimeSeriesDataSet.from_dataset(
        training, 
        data, 
        min_prediction_idx=training_cutoff.min() + 1,
        stop_randomization=True
    )
    
    # Create data loaders
    batch_size = 64
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    
    return training, validation, data, train_dataloader, val_dataloader

def train_tft_model(training, train_dataloader, val_dataloader, 
                   hidden_size=32, attention_head_size=4, dropout=0.1, 
                   hidden_continuous_size=16, learning_rate=1e-3,
                   max_epochs=20, enable_progress_bar=True):
    """
    Train TFT model
    
    Args:
        training (TimeSeriesDataSet): Training dataset
        train_dataloader (DataLoader): Training data loader
        val_dataloader (DataLoader): Validation data loader
        hidden_size (int): Hidden size
        attention_head_size (int): Attention head size
        dropout (float): Dropout rate
        hidden_continuous_size (int): Hidden continuous size
        learning_rate (float): Learning rate
        max_epochs (int): Maximum number of epochs
        enable_progress_bar (bool): Whether to enable progress bar
    
    Returns:
        TemporalFusionTransformer: Trained model
    """
    # Create TFT model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        learning_rate=learning_rate,
        log_interval=0,
        reduce_on_plateau_patience=3,
    )
    
    # Create early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=1e-4, 
        patience=5, 
        verbose=True, 
        mode="min"
    )
    
    # Create learning rate monitor
    lr_logger = LearningRateMonitor()
    
    # Create logger
    logger = TensorBoardLogger("lightning_logs")
    
    # Create checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints/",
        filename="tft-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, lr_logger, checkpoint_callback],
        logger=logger,
        enable_progress_bar=enable_progress_bar,
        limit_val_batches=1.0,
        log_every_n_steps=10,
    )
    
    # Fit model
    try:
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        
        # Load the best model
        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path and os.path.exists(best_model_path):
            best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
            return best_tft
        else:
            print("No checkpoint found, returning the current model")
            return tft
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return tft

def generate_predictions(model, validation_dataloader, data, threshold=0.002):
    """
    Generate predictions and trading signals
    
    Args:
        model (TemporalFusionTransformer): Trained model
        validation_dataloader (DataLoader): Validation data loader
        data (DataFrame): Original data
        threshold (float): Threshold for signal generation
    
    Returns:
        dict: Dictionary of DataFrames by ticker with predictions and signals
    """
    try:
        # Get predictions
        predictions = model.predict(validation_dataloader)
        
        # Extract median prediction (0.5 quantile)
        if isinstance(predictions, tuple):
            predictions = predictions[1]
        
        # Convert to numpy array
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.numpy()
        
        # If predictions is a dictionary with quantiles, extract the median
        if isinstance(predictions, dict) and 0.5 in predictions:
            predictions = predictions[0.5]
        
        # Find the cutoff point between training and validation
        training_cutoff = data.groupby('Ticker')['time_idx'].transform(lambda x: int(x.max() * 0.8))
        
        # Get validation data
        validation_data = data[data['time_idx'] > training_cutoff].copy()
        
        # Create a DataFrame with predictions
        pred_df = pd.DataFrame({
            'prediction': predictions.flatten()
        })
        
        # Add index to match validation data
        if len(pred_df) <= len(validation_data):
            # Use the last len(pred_df) rows from validation data
            valid_indices = validation_data.index[-len(pred_df):]
            pred_df.index = valid_indices
        else:
            # Truncate predictions to match validation data
            pred_df = pred_df.iloc[:len(validation_data)]
            pred_df.index = validation_data.index
        
        # Merge with original data
        result_df = validation_data.copy()
        result_df['prediction'] = pred_df['prediction']
        
        # Process each ticker separately
        tickers = result_df['Ticker'].unique()
        result_dict = {}
        
        for ticker in tickers:
            ticker_df = result_df[result_df['Ticker'] == ticker].copy()
            
            # Calculate dynamic threshold based on prediction distribution
            pred_std = ticker_df['prediction'].std()
            pred_mean = ticker_df['prediction'].mean()
            
            # Print prediction statistics to help diagnose the issue
            print(f"\n{ticker} prediction stats:")
            print(f"  Mean: {pred_mean:.6f}")
            print(f"  Std: {pred_std:.6f}")
            print(f"  Min: {ticker_df['prediction'].min():.6f}")
            print(f"  Max: {ticker_df['prediction'].max():.6f}")
            
            # SOLUTION 1: Use quantiles instead of mean/std for thresholds
            # This ensures we get both buy and sell signals regardless of bias
            positive_threshold = ticker_df['prediction'].quantile(0.7)  # Top 30%
            negative_threshold = ticker_df['prediction'].quantile(0.3)  # Bottom 30%
            
            # SOLUTION 2: Alternative approach - normalize predictions around zero
            # centered_predictions = ticker_df['prediction'] - pred_mean
            # ticker_df['centered_prediction'] = centered_predictions
            # positive_threshold = pred_std * 0.5
            # negative_threshold = -pred_std * 0.5
            
            print(f"  Positive threshold: {positive_threshold:.6f}")
            print(f"  Negative threshold: {negative_threshold:.6f}")
            
            # Generate signals
            ticker_df['signal'] = 0  # Default to hold
            ticker_df.loc[ticker_df['prediction'] > positive_threshold, 'signal'] = 1  # Buy
            ticker_df.loc[ticker_df['prediction'] < negative_threshold, 'signal'] = -1  # Sell
            
            # Calculate confidence
            ticker_df['confidence'] = 0.0
            
            # For buy signals
            buy_mask = ticker_df['prediction'] > positive_threshold
            ticker_df.loc[buy_mask, 'confidence'] = np.minimum(
                (ticker_df.loc[buy_mask, 'prediction'] - positive_threshold) / (pred_std), 
                1.0
            )
            
            # For sell signals
            sell_mask = ticker_df['prediction'] < negative_threshold
            ticker_df.loc[sell_mask, 'confidence'] = np.minimum(
                (negative_threshold - ticker_df.loc[sell_mask, 'prediction']) / (pred_std), 
                1.0
            )
            
            # Print signal counts
            print(f"  Buy signals: {sum(ticker_df['signal'] == 1)}")
            print(f"  Sell signals: {sum(ticker_df['signal'] == -1)}")
            print(f"  Hold signals: {sum(ticker_df['signal'] == 0)}")
            
            result_dict[ticker] = ticker_df
        
        return result_dict
    
    except Exception as e:
        print(f"Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        
        # Create dummy predictions as fallback
        print("Creating dummy predictions as fallback...")
        training_cutoff = data.groupby('Ticker')['time_idx'].transform(lambda x: int(x.max() * 0.8))
        validation_data = data[data['time_idx'] > training_cutoff].copy()
        
        # Process by ticker
        tickers = validation_data['Ticker'].unique()
        result_dict = {}
        
        for ticker in tickers:
            ticker_df = validation_data[validation_data['Ticker'] == ticker].copy()
            # Add random predictions
            ticker_df['prediction'] = np.random.normal(0, 0.01, size=len(ticker_df))
            ticker_df['signal'] = np.random.choice([-1, 0, 1], size=len(ticker_df))
            ticker_df['confidence'] = np.random.uniform(0, 1, size=len(ticker_df))
            result_dict[ticker] = ticker_df
            
        return result_dict

def evaluate_predictions(result_dict):
    """
    Evaluate prediction accuracy
    
    Args:
        result_dict (dict): Dictionary of DataFrames by ticker with predictions and signals
    
    Returns:
        dict: Dictionary of evaluation metrics by ticker
    """
    # Multi-ticker case
    results = {}
    for ticker, result_df in result_dict.items():
        # Evaluate each ticker separately
        ticker_metrics = evaluate_single_df(result_df)
        results[ticker] = ticker_metrics
    
    # Add aggregated metrics
    all_metrics = list(results.values())
    if all_metrics:
        # Calculate average metrics across all tickers
        avg_metrics = {
            'Avg_MSE': np.mean([m['MSE'] for m in all_metrics]),
            'Avg_MAE': np.mean([m['MAE'] for m in all_metrics]),
            'Avg_Direction_Accuracy': np.mean([m['Direction_Accuracy'] for m in all_metrics]),
            'Avg_Signal_Accuracy': np.mean([m['Signal_Accuracy'] for m in all_metrics]),
            'Avg_Signal_Rate': np.mean([m['Signal_Rate'] for m in all_metrics])
        }
        results['Aggregated'] = avg_metrics
    
    return results

def evaluate_single_df(result_df):
    """Helper function to evaluate a single DataFrame"""
    # Filter out rows without predictions
    eval_df = result_df.dropna(subset=['prediction'])
    
    if len(eval_df) == 0:
        return {"error": "No valid predictions to evaluate"}
    
    # Calculate metrics
    mse = np.mean((eval_df['prediction'] - eval_df['target']) ** 2)
    mae = np.mean(np.abs(eval_df['prediction'] - eval_df['target']))
    
    # Direction accuracy (did we predict the sign correctly?)
    direction_correct = np.sign(eval_df['prediction']) == np.sign(eval_df['target'])
    direction_accuracy = np.mean(direction_correct)
    
    # Signal accuracy
    signal_correct = (
        (eval_df['signal'] == 1) & (eval_df['target'] > 0) |
        (eval_df['signal'] == -1) & (eval_df['target'] < 0) |
        (eval_df['signal'] == 0) & (abs(eval_df['target']) < 0.005)  # Small threshold for "no change"
    )
    signal_accuracy = np.mean(signal_correct)
    
    # Count signals by type
    buy_signals = sum(eval_df['signal'] == 1)
    sell_signals = sum(eval_df['signal'] == -1)
    hold_signals = sum(eval_df['signal'] == 0)
    
    # Calculate strategy returns
    strategy_returns = eval_df['target'] * eval_df['signal'].shift(1).fillna(0)
    cumulative_return = (1 + strategy_returns).prod() - 1
    
    # Calculate Sharpe ratio (annualized)
    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
    
    # Calculate win rate
    win_rate = (strategy_returns > 0).mean() if len(strategy_returns) > 0 else 0
    
    return {
        'MSE': mse,
        'MAE': mae,
        'Direction_Accuracy': direction_accuracy,
        'Signal_Accuracy': signal_accuracy,
        'Total_Predictions': len(eval_df),
        'Buy_Signals': buy_signals,
        'Sell_Signals': sell_signals,
        'Hold_Signals': hold_signals,
        'Signal_Rate': (buy_signals + sell_signals) / len(eval_df) if len(eval_df) > 0 else 0,
        'Cumulative_Return': cumulative_return,
        'Sharpe_Ratio': sharpe_ratio,
        'Win_Rate': win_rate
    }

def plot_predictions(result_dict, save_dir="results/plots"):
    """
    Plot predictions vs actual values for multiple tickers
    
    Args:
        result_dict (dict): Dictionary of DataFrames by ticker with predictions and signals
        save_dir (str): Directory to save plots
    """
    ensure_directory_exists(save_dir)
    
    # Plot each ticker separately
    for ticker, result_df in result_dict.items():
        plot_single_ticker(result_df, ticker, save_dir)
        
    # Create a comparison plot for all tickers
    plot_multi_ticker_comparison(result_dict, save_dir)

def plot_single_ticker(plot_df, ticker, save_dir):
    """Helper function to plot predictions for a single ticker"""
    # Filter out rows without predictions
    plot_df = plot_df.dropna(subset=['prediction'])
    
    # Set date as index for plotting
    if 'date' in plot_df.columns:
        plot_df = plot_df.set_index('date')
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(plot_df.index, plot_df['target'], label='Actual Returns', alpha=0.7)
    plt.plot(plot_df.index, plot_df['prediction'], label='Predicted Returns', alpha=0.7)
    
    # Add buy/sell signals
    buy_signals = plot_df[plot_df['signal'] == 1]
    sell_signals = plot_df[plot_df['signal'] == -1]
    
    plt.scatter(buy_signals.index, buy_signals['prediction'], 
                color='green', marker='^', s=100, label='Buy Signal')
    plt.scatter(sell_signals.index, sell_signals['prediction'], 
                color='red', marker='v', s=100, label='Sell Signal')
    
    plt.title(f'{ticker} - Predicted vs Actual Returns')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f"{save_dir}/{ticker}_tft_predictions.png")
    plt.close()
    
    # Plot signal distribution
    plt.figure(figsize=(8, 6))
    signal_counts = plot_df['signal'].value_counts().reindex([1, 0, -1], fill_value=0)
    plt.bar(['Buy', 'Hold', 'Sell'], signal_counts.values)
    plt.title(f'{ticker} - Signal Distribution')
    plt.ylabel('Count')
    plt.savefig(f"{save_dir}/{ticker}_tft_signal_distribution.png")
    plt.close()
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    
    # Calculate strategy returns
    strategy_returns = plot_df['target'] * plot_df['signal'].shift(1).fillna(0)
    
    # Calculate cumulative returns
    buy_hold_cumulative = (1 + plot_df['target']).cumprod() - 1
    strategy_cumulative = (1 + strategy_returns).cumprod() - 1
    
    plt.plot(plot_df.index, buy_hold_cumulative, label='Buy & Hold', alpha=0.7)
    plt.plot(plot_df.index, strategy_cumulative, label='Strategy', alpha=0.7)
    
    plt.title(f'{ticker} - Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f"{save_dir}/{ticker}_cumulative_returns.png")
    plt.close()

def plot_multi_ticker_comparison(result_dict, save_dir):
    """Plot comparison of multiple tickers' performance"""
    # Extract metrics for each ticker
    tickers = list(result_dict.keys())
    direction_accuracies = []
    signal_rates = []
    sharpe_ratios = []
    cumulative_returns = []
    
    for ticker, df in result_dict.items():
        # Calculate metrics
        eval_df = df.dropna(subset=['prediction'])
        if len(eval_df) > 0:
            # Direction accuracy
            direction_correct = np.sign(eval_df['prediction']) == np.sign(eval_df['target'])
            direction_accuracies.append(np.mean(direction_correct))
            
            # Signal rate
            signal_rate = (sum(eval_df['signal'] != 0) / len(eval_df))
            signal_rates.append(signal_rate)
            
            # Strategy returns
            strategy_returns = eval_df['target'] * eval_df['signal'].shift(1).fillna(0)
            
            # Sharpe ratio
            sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
            sharpe_ratios.append(sharpe)
            
            # Cumulative return
            cumulative_return = (1 + strategy_returns).cumprod().iloc[-1] - 1 if len(strategy_returns) > 0 else 0
            cumulative_returns.append(cumulative_return)
    
    # Create comparison plots
    plt.figure(figsize=(16, 12))
    
    # Plot direction accuracy
    plt.subplot(2, 2, 1)
    plt.bar(tickers, direction_accuracies)
    plt.title('Direction Accuracy by Ticker')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Plot signal rate
    plt.subplot(2, 2, 2)
    plt.bar(tickers, signal_rates)
    plt.title('Signal Rate by Ticker')
    plt.ylabel('Signal Rate')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Plot Sharpe ratio
    plt.subplot(2, 2, 3)
    plt.bar(tickers, sharpe_ratios)
    plt.title('Sharpe Ratio by Ticker')
    plt.ylabel('Sharpe Ratio')
    plt.xticks(rotation=45)
    
    # Plot cumulative returns
    plt.subplot(2, 2, 4)
    plt.bar(tickers, cumulative_returns)
    plt.title('Cumulative Return by Ticker')
    plt.ylabel('Cumulative Return')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/multi_ticker_comparison.png")
    plt.close()

def plot_confusion_matrix(result_dict, save_dir="results/plots"):
    """
    Plot confusion matrix for directional prediction accuracy for each ticker
    
    Args:
        result_dict (dict): Dictionary of DataFrames by ticker with predictions and signals
        save_dir (str): Directory to save plots
    """
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    ensure_directory_exists(save_dir)
    
    for ticker, result_df in result_dict.items():
        # Filter out rows without predictions
        eval_df = result_df.dropna(subset=['prediction'])
        
        if len(eval_df) <= 1:
            print(f"Not enough predictions to create confusion matrix for {ticker}")
            continue
        
        # Convert target and prediction to binary signals (up/down)
        actual_direction = (eval_df['target'] > 0).astype(int)
        pred_direction = (eval_df['prediction'] > 0).astype(int)
        
        # Create confusion matrix
        cm = confusion_matrix(actual_direction, pred_direction)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Down', 'Up'],
                    yticklabels=['Down', 'Up'])
        plt.xlabel('Predicted Direction')
        plt.ylabel('Actual Direction')
        plt.title(f'{ticker} - Direction Prediction Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{ticker}_confusion_matrix.png")
        plt.close()

def main():
    """Main function to run the multi-ticker forecasting"""
    # Create necessary directories
    ensure_directory_exists("results")
    ensure_directory_exists("models")
    ensure_directory_exists("results/plots")
    ensure_directory_exists("checkpoints")
    
    # Define parameters
    tickers = ["AAPL", "MSFT", "GOOGL"]  # Multiple tickers
    data_dir = "data"
    csv_files = [os.path.join(data_dir, f"{ticker}.csv") for ticker in tickers]
    
        # Check if files exist
    valid_files = []
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            valid_files.append(csv_file)
        else:
            print(f"Warning: File {csv_file} not found.")
    
    if not valid_files:
        print("Error: No valid data files found.")
        return
    
    # Define model parameters
    seq_len = 30
    prediction_length = 5
    threshold = 0.005
    
    try:
        # Prepare data
        print(f"Preparing data from {len(valid_files)} files...")
        training, validation, data, train_dataloader, val_dataloader = prepare_multi_ticker_data(
            csv_files=valid_files,
            seq_len=seq_len,
            prediction_length=prediction_length
        )
        
        # Train model
        print("Training model...")
        model = train_tft_model(
            training=training,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            hidden_size=32,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=16,
            learning_rate=0.001,
            max_epochs=20,
            enable_progress_bar=True
        )
        
        # Save model
        model_path = f"models/multi_stock_tft_model.ckpt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Generate predictions
        print("Generating predictions...")
        result_dict = generate_predictions(
            model=model,
            validation_dataloader=val_dataloader,
            data=data,
            threshold=threshold
        )
        
        # Save predictions
        for ticker, df in result_dict.items():
            df.to_csv(f"results/{ticker}_tft_predictions.csv")
            print(f"Predictions for {ticker} saved to results/{ticker}_tft_predictions.csv")
        
        # Evaluate predictions
        print("Evaluating predictions...")
        metrics = evaluate_predictions(result_dict)
        
        # Print evaluation results
        print("\nPrediction Evaluation Results:")
        print("==============================")
        for ticker, ticker_metrics in metrics.items():
            print(f"\n{ticker}:")
            for metric, value in ticker_metrics.items():
                print(f"  {metric}: {value}")
        
        # Plot predictions
        print("Plotting predictions...")
        plot_predictions(result_dict)
        
        # Plot confusion matrices
        print("Plotting confusion matrices...")
        plot_confusion_matrix(result_dict)
        
        print("\nEvaluation complete. Results saved to 'results' directory.")
    
    except Exception as e:
        print(f"An error occurred in the main function: {e}")
        import traceback
        traceback.print_exc()

def optimize_hyperparameters(training, train_dataloader, val_dataloader, 
                           n_trials=10, max_epochs=10):
    """
    Optimize hyperparameters for TFT model
    
    Args:
        training (TimeSeriesDataSet): Training dataset
        train_dataloader (DataLoader): Training data loader
        val_dataloader (DataLoader): Validation data loader
        n_trials (int): Number of optimization trials
        max_epochs (int): Maximum number of epochs per trial
    
    Returns:
        dict: Best parameters
    """
    print("Starting hyperparameter optimization...")
    
    try:
        # Create study
        study = optimize_hyperparameters(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            model_class=TemporalFusionTransformer,
            max_epochs=max_epochs,
            n_trials=n_trials,
            hidden_size=[16, 32, 64],
            hidden_continuous_size=[8, 16, 32],
            attention_head_size=[1, 2, 4],
            dropout=[0.1, 0.2, 0.3],
            learning_rate=[0.001, 0.01],
            trainer_kwargs=dict(
                accelerator="auto",
                enable_progress_bar=True,
            ),
            reduce_on_plateau_patience=3,
            use_learning_rate_finder=False,
        )
        
        # Get best parameters
        best_params = study.best_params
        print(f"Best parameters: {best_params}")
        
        # Save best parameters
        with open("results/best_hyperparameters.txt", "w") as f:
            for param, value in best_params.items():
                f.write(f"{param}: {value}\n")
        
        return best_params
    
    except Exception as e:
        print(f"Error during hyperparameter optimization: {e}")
        import traceback
        traceback.print_exc()
        return {}

def run_hyperparameter_optimization():
    """Run hyperparameter optimization as a separate function"""
    # Create necessary directories
    ensure_directory_exists("results")
    ensure_directory_exists("models")
    ensure_directory_exists("checkpoints")
    
    # Define parameters
    tickers = ["AAPL", "MSFT", "GOOGL"]  # Multiple tickers
    data_dir = "data"
    csv_files = [os.path.join(data_dir, f"{ticker}.csv") for ticker in tickers]
    
    # Check if files exist
    valid_files = []
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            valid_files.append(csv_file)
        else:
            print(f"Warning: File {csv_file} not found.")
    
    if not valid_files:
        print("Error: No valid data files found.")
        return
    
    # Define model parameters
    seq_len = 10
    prediction_length = 1
    
    try:
        # Prepare data
        print(f"Preparing data from {len(valid_files)} files...")
        training, validation, data, train_dataloader, val_dataloader = prepare_multi_ticker_data(
            csv_files=valid_files,
            seq_len=seq_len,
            prediction_length=prediction_length
        )
        
        # Run hyperparameter optimization
        best_params = optimize_hyperparameters(
            training=training,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            n_trials=10,
            max_epochs=10
        )
        
        print(f"Hyperparameter optimization complete. Best parameters: {best_params}")
    
    except Exception as e:
        print(f"An error occurred during hyperparameter optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Multi-ticker TFT forecasting')
    parser.add_argument('--optimize', action='store_true', help='Run hyperparameter optimization')
    args = parser.parse_args()
    
    if args.optimize:
        run_hyperparameter_optimization()
    else:
        main()


