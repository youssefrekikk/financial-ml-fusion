#unfinished/ still expperimenting
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

from utils import add_features

def prepare_data(csv_file, seq_len=10, prediction_length=1):
    """
    Prepare data for TFT model
    
    Args:
        csv_file (str): Path to CSV file
        seq_len (int): Sequence length (max_encoder_length)
        prediction_length (int): Prediction horizon (max_prediction_length)
    
    Returns:
        tuple: Training and validation datasets, data frame
    """
    # Load data
    try:
        # Try to load with date index
        data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        print(f"Data loaded with DatetimeIndex: {data.index[0]} to {data.index[-1]}")
    except:
        # Fall back to regular loading
        data = pd.read_csv(csv_file)
        # Check for date column
        date_column = None
        for col in ['Date', 'date', 'datetime', 'Datetime']:
            if col in data.columns:
                date_column = col
                break
        
        if date_column:
            data[date_column] = pd.to_datetime(data[date_column])
            data.set_index(date_column, inplace=True)
            print(f"Data loaded with date column: {data.index[0]} to {data.index[-1]}")
        else:
            print(f"Warning: No date column found. Available columns: {data.columns.tolist()}")
    
    # Sort by date
    data = data.sort_index()
    
    # Apply feature engineering
    data = add_features(data)
    print(f"Data shape after feature engineering: {data.shape}")
    
    # Reset index to have date as a column
    data = data.reset_index()
    data.rename(columns={data.columns[0]: 'date'}, inplace=True)
    
    # Add time index for PyTorch Forecasting
    data['time_idx'] = np.arange(len(data))
    
    # Add group ID (we only have one time series)
    data['group_id'] = 0
    
    # Calculate target (next day's return)
    data['target'] = data['Close'].pct_change(1).shift(-1)
    
    # Remove rows with NaN target
    data = data.dropna(subset=['target'])
    
    # Define static and time-varying features
    time_varying_known_categoricals = []
    time_varying_known_reals = ['Open', 'High', 'Low', 'Close', 'Volume']
    time_varying_unknown_categoricals = []
    time_varying_unknown_reals = [
        'Return', 'Log_Return', 'Return_Volatility', 'Open_Close_Diff', 
        'High_Low_Diff', 'Close_Prev_Close_Diff', 'SMA_5', 'SMA_10', 'SMA_20',
        'SMA_50', 'SMA_20_50_Ratio', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Pct', 'Relative_Volume'
    ]
    
    # Filter out any columns that don't exist
    time_varying_known_reals = [col for col in time_varying_known_reals if col in data.columns]
    time_varying_unknown_reals = [col for col in time_varying_unknown_reals if col in data.columns]
    
    # Create training dataset
    max_encoder_length = seq_len
    max_prediction_length = prediction_length
    
    training_cutoff = data['time_idx'].max() - max_prediction_length
    
    training = TimeSeriesDataSet(
        data=data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_categoricals=time_varying_unknown_categoricals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(
            groups=["group_id"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # Create validation dataset
    validation = TimeSeriesDataSet.from_dataset(
        training, data, min_prediction_idx=training_cutoff + 1, stop_randomization=True
    )
    
    # Create data loaders
    batch_size = 128
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
        loss=QuantileLoss(),
        learning_rate=learning_rate,
        log_interval=10,
        reduce_on_plateau_patience=3,
    )
    
    # Create logger
    logger = TensorBoardLogger("lightning_logs")
    
    # Create early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min"
    )
    
    # Create learning rate monitor
    lr_logger = LearningRateMonitor()
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, lr_logger],
        logger=logger,
        enable_progress_bar=enable_progress_bar,
    )
    
    # Fit model
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    # Load the best model
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    
    return best_tft

def generate_predictions(model, validation_dataloader, data, threshold=0.002):
    """
    Generate predictions and trading signals
    
    Args:
        model (TemporalFusionTransformer): Trained model
        validation_dataloader (DataLoader): Validation data loader
        data (DataFrame): Original data
        threshold (float): Threshold for signal generation
    
    Returns:
        DataFrame: DataFrame with predictions and signals
    """
    # Get predictions
    predictions = model.predict(validation_dataloader)
    
    # Extract median prediction (0.5 quantile)
    if isinstance(predictions, tuple):
        # If predictions is a tuple of (x, y), extract y
        predictions = predictions[1]
    
    # Convert to numpy array
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    
    # If predictions is a dictionary with quantiles, extract the median
    if isinstance(predictions, dict) and 0.5 in predictions:
        predictions = predictions[0.5]
    
    # Get validation indices
    validation_data = validation_dataloader.dataset.data
    
    # Create DataFrame with predictions
    pred_df = pd.DataFrame({
        'time_idx': validation_data['time_idx'].values,
        'prediction': predictions.flatten()
    })
    
    # Merge with original data
    result_df = pd.merge(data, pred_df, on='time_idx', how='left')
    
    # Generate signals
    result_df['signal'] = 0  # Default to hold
    result_df.loc[result_df['prediction'] > threshold, 'signal'] = 1  # Buy
    result_df.loc[result_df['prediction'] < -threshold, 'signal'] = -1  # Sell
    
    # Calculate confidence
    result_df['confidence'] = 0.0
    mask = abs(result_df['prediction']) > threshold
    result_df.loc[mask, 'confidence'] = np.minimum(
        (abs(result_df.loc[mask, 'prediction']) - threshold) / (0.01 - threshold), 
        1.0
    )
    
    # Set date as index
    result_df.set_index('date', inplace=True)
    
    return result_df

def evaluate_predictions(result_df):
    """
    Evaluate prediction accuracy
    
    Args:
        result_df (DataFrame): DataFrame with predictions and signals
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
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

def plot_predictions(result_df, ticker, save_dir="plots"):
    """Plot predictions vs actual values"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter out rows without predictions
    plot_df = result_df.dropna(subset=['prediction'])
    
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
    
    print(f"Prediction plots saved to {save_dir}/{ticker}_tft_predictions.png and {save_dir}/{ticker}_tft_signal_distribution.png")

def optimize_model_hyperparameters(training, train_dataloader, val_dataloader, 
                                  n_trials=20, max_epochs=20, enable_progress_bar=False):
    """
    Optimize hyperparameters for TFT model
    
    Args:
        training (TimeSeriesDataSet): Training dataset
        train_dataloader (DataLoader): Training data loader
        val_dataloader (DataLoader): Validation data loader
        n_trials (int): Number of optimization trials
        max_epochs (int): Maximum number of epochs per trial
        enable_progress_bar (bool): Whether to enable progress bar
    
    Returns:
        dict: Best parameters
    """
    # Create study
    study = optimize_hyperparameters(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model_class=TemporalFusionTransformer,
        max_epochs=max_epochs,
        n_trials=n_trials,
        hidden_size=[16, 32, 64, 128],
        hidden_continuous_size=[8, 16, 32, 64],
        attention_head_size=[1, 2, 4, 8],
        dropout=[0.1, 0.2, 0.3],
        learning_rate=[0.001, 0.01],
        trainer_kwargs=dict(
            accelerator="auto",
            enable_progress_bar=enable_progress_bar,
        ),
        reduce_on_plateau_patience=3,
        use_learning_rate_finder=False,
    )
    
    # Get best parameters
    best_params = study.best_params
    print(f"Best parameters: {best_params}")
    
    return best_params

def main():
    # Create necessary directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Define parameters
    ticker = "AAPL"
    data_dir = "data"
    csv_file = os.path.join(data_dir, f"{ticker}.csv")
    seq_len = 10
    prediction_length = 1
    threshold = 0.002  # Threshold for signal generation
    optimize_hyperparams = False  # Set to True to run hyperparameter optimization
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found.")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in data directory: {os.listdir(data_dir) if os.path.exists(data_dir) else 'data directory not found'}")
        return
    
    # Prepare data
    print(f"Preparing data from {csv_file}...")
    training, validation, data, train_dataloader, val_dataloader = prepare_data(
        csv_file=csv_file,
        seq_len=seq_len,
        prediction_length=prediction_length
    )
    
    # Optimize hyperparameters if requested
    if optimize_hyperparams:
        print("Optimizing hyperparameters...")
        best_params = optimize_model_hyperparameters(
            training=training,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            n_trials=20,
            max_epochs=20,
            enable_progress_bar=False
        )
        
        # Train model with best parameters
        print("Training model with optimized hyperparameters...")
        model = train_tft_model(
            training=training,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            hidden_size=best_params.get('hidden_size', 32),
            attention_head_size=best_params.get('attention_head_size', 4),
            dropout=best_params.get('dropout', 0.1),
            hidden_continuous_size=best_params.get('hidden_continuous_size', 16),
            learning_rate=best_params.get('learning_rate', 0.001),
            max_epochs=30,
            enable_progress_bar=True
        )
    else:
        # Train model with default parameters
        print("Training model with default parameters...")
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
    model_path = f"models/{ticker}_tft_model.ckpt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Generate predictions
    print("Generating predictions...")
    result_df = generate_predictions(
        model=model,
        validation_dataloader=val_dataloader,
        data=data,
        threshold=threshold
    )
    
    # Save predictions to CSV
    result_df.to_csv(f"results/{ticker}_tft_predictions.csv")
    print(f"Predictions saved to results/{ticker}_tft_predictions.csv")
    
    # Evaluate predictions
    print("Evaluating predictions...")
    metrics = evaluate_predictions(result_df)
    
    # Print evaluation results
    print("\nPrediction Evaluation Results:")
    print("==============================")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    
    # Plot predictions
    print("Plotting predictions...")
    plot_predictions(result_df, ticker)
    
    # Analyze feature importance
    print("Analyzing feature importance...")
    interpretation = model.interpret_output(
        val_dataloader,
        reduction="sum"
    )
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    order = interpretation["variable_importance"].mean(0).sort_values(ascending=False).index
    interpretation["variable_importance"].mean(0)[order].plot.barh()
    plt.title(f"{ticker} - Feature Importance")
    plt.tight_layout()
    plt.savefig(f"plots/{ticker}_tft_feature_importance.png")
    plt.close()
    print(f"Feature importance plot saved to plots/{ticker}_tft_feature_importance.png")
    
    # Save metrics to CSV
    pd.DataFrame([metrics]).to_csv(f"results/{ticker}_tft_metrics.csv", index=False)
    print(f"Metrics saved to results/{ticker}_tft_metrics.csv")
    
    print("\nEvaluation complete. Results saved to 'results' directory.")

if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings("ignore")
    main()

