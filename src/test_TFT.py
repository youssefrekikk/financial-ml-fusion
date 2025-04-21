import os
import warnings
import itertools
from pathlib import Path
import json
from datetime import datetime

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from tqdm import tqdm

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

# Import the prepare_data and other functions from TFT_forcasting.py
from TFT_forcasting import prepare_data, train_tft_model, generate_predictions, evaluate_predictions, plot_predictions

def run_grid_search(tickers, seq_lengths, pred_lengths, thresholds, data_dir="data", results_dir="grid_search_results"):
    """
    Run grid search over multiple tickers and hyperparameters.
    
    Args:
        tickers (list): List of ticker symbols
        seq_lengths (list): List of sequence lengths to try
        pred_lengths (list): List of prediction lengths to try
        thresholds (list): List of signal thresholds to try
        data_dir (str): Directory containing data files
        results_dir (str): Directory to save results
    
    Returns:
        pd.DataFrame: Results of grid search
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, data_dir)
    results_dir = os.path.join(root_dir, results_dir)
    
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize results list
    results = []
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a progress bar for the grid search
    total_combinations = len(tickers) * len(seq_lengths) * len(pred_lengths) * len(thresholds)
    pbar = tqdm(total=total_combinations, desc="Grid Search Progress")
    
    # Loop through all combinations
    for ticker in tickers:
        csv_file = os.path.join(data_dir, f"{ticker}.csv")
        
        if not os.path.exists(csv_file):
            print(f"Warning: File {csv_file} not found. Skipping {ticker}.")
            continue
        
        for seq_len in seq_lengths:
            for pred_length in pred_lengths:
                # Prepare data only once for each seq_len/pred_length combination
                try:
                    training, validation, data, train_dataloader, val_dataloader = prepare_data(
                        csv_file=csv_file,
                        seq_len=seq_len,
                        prediction_length=pred_length
                    )
                    
                    # Train model
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
                        enable_progress_bar=False  # Disable progress bar for cleaner output
                    )
                    
                    # Test different thresholds
                    for threshold in thresholds:
                        # Generate predictions with current threshold
                        result_df = generate_predictions(
                            model=model,
                            validation_dataloader=val_dataloader,
                            data=data,
                            threshold=threshold
                        )
                        
                        # Evaluate predictions
                        metrics = evaluate_predictions(result_df)
                        
                        # Add additional evaluation metrics
                        additional_metrics = evaluate_trading_strategy(result_df)
                        metrics.update(additional_metrics)
                        
                        # Add parameters to metrics
                        metrics['ticker'] = ticker
                        metrics['seq_len'] = seq_len
                        metrics['pred_length'] = pred_length
                        metrics['threshold'] = threshold
                        
                        # Add to results
                        results.append(metrics)
                        
                        # Save individual result
                        result_file = f"{results_dir}/{ticker}_seq{seq_len}_pred{pred_length}_thresh{threshold}_{timestamp}.json"
                        with open(result_file, 'w') as f:
                            json.dump(metrics, f, indent=4)
                        
                        # Update progress bar
                        pbar.update(1)
                
                except Exception as e:
                    print(f"Error processing {ticker} with seq_len={seq_len}, pred_length={pred_length}: {e}")
                    # Update progress bar for skipped combinations
                    pbar.update(len(thresholds))
    
    pbar.close()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save complete results
    results_df.to_csv(f"{results_dir}/grid_search_results_{timestamp}.csv", index=False)
    
    return results_df

def evaluate_trading_strategy(result_df):
    """
    Evaluate the trading strategy performance.
    
    Args:
        result_df (DataFrame): DataFrame with predictions and signals
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Filter out rows without predictions
    eval_df = result_df.dropna(subset=['prediction'])
    
    if len(eval_df) <= 1:
        return {
            "strategy_return": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0
        }
    
    # Calculate strategy returns (assuming next-day execution)
    eval_df['strategy_return'] = eval_df['signal'].shift(1) * eval_df['target']
    eval_df['cumulative_return'] = (1 + eval_df['strategy_return']).cumprod() - 1
    
    # Calculate buy-and-hold returns
    eval_df['bh_cumulative_return'] = (1 + eval_df['target']).cumprod() - 1
    
    # Calculate metrics
    total_return = eval_df['cumulative_return'].iloc[-1] if len(eval_df) > 0 else 0
    
    # Sharpe ratio (annualized)
    daily_returns = eval_df['strategy_return'].dropna()
    sharpe_ratio = 0
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    
    # Maximum drawdown
    cumulative_returns = (1 + daily_returns).cumprod()
    max_drawdown = 0
    if len(cumulative_returns) > 0:
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
    
    # Win rate
    winning_trades = (eval_df['strategy_return'] > 0).sum()
    total_trades = (~eval_df['strategy_return'].isna() & (eval_df['signal'].shift(1) != 0)).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Profit factor
    gross_profits = eval_df.loc[eval_df['strategy_return'] > 0, 'strategy_return'].sum()
    gross_losses = abs(eval_df.loc[eval_df['strategy_return'] < 0, 'strategy_return'].sum())
    profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
    
    # Classification metrics for directional accuracy
    # Convert target and prediction to binary signals (up/down)
    actual_direction = (eval_df['target'] > 0).astype(int)
    pred_direction = (eval_df['prediction'] > 0).astype(int)
    
    precision = precision_score(actual_direction, pred_direction, zero_division=0)
    recall = recall_score(actual_direction, pred_direction, zero_division=0)
    f1 = f1_score(actual_direction, pred_direction, zero_division=0)
    
    return {
        "strategy_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def plot_confusion_matrix(result_df, ticker, save_dir="plots"):
    """
    Plot confusion matrix for directional prediction accuracy.
    
    Args:
        result_df (DataFrame): DataFrame with predictions and signals
        ticker (str): Ticker symbol
        save_dir (str): Directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter out rows without predictions
    eval_df = result_df.dropna(subset=['prediction'])
    
    if len(eval_df) <= 1:
        print(f"Not enough predictions to create confusion matrix for {ticker}")
        return
    
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

def plot_equity_curve(result_df, ticker, save_dir="plots"):
    """
    Plot equity curve for the trading strategy.
    
    Args:
        result_df (DataFrame): DataFrame with predictions and signals
        ticker (str): Ticker symbol
        save_dir (str): Directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter out rows without predictions
    eval_df = result_df.dropna(subset=['prediction'])
    
    if len(eval_df) <= 1:
        print(f"Not enough predictions to create equity curve for {ticker}")
        return
    
    # Calculate strategy returns (assuming next-day execution)
    eval_df['strategy_return'] = eval_df['signal'].shift(1) * eval_df['target']
    eval_df['strategy_cumulative'] = (1 + eval_df['strategy_return']).cumprod()
    
    # Calculate buy-and-hold returns
    eval_df['bh_cumulative'] = (1 + eval_df['target']).cumprod()
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(eval_df.index, eval_df['strategy_cumulative'], label='Strategy', color='blue')
    plt.plot(eval_df.index, eval_df['bh_cumulative'], label='Buy & Hold', color='gray', alpha=0.7)
    
    # Add buy/sell markers
    buy_signals = eval_df[eval_df['signal'].shift(1) == 1]
    sell_signals = eval_df[eval_df['signal'].shift(1) == -1]
    
    if len(buy_signals) > 0:
        plt.scatter(buy_signals.index, buy_signals['strategy_cumulative'], 
                    color='green', marker='^', s=50, label='Buy Signal')
    if len(sell_signals) > 0:
        plt.scatter(sell_signals.index, sell_signals['strategy_cumulative'], 
                    color='red', marker='v', s=50, label='Sell Signal')
    
    plt.title(f'{ticker} - Strategy Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (1.0 = 100%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{ticker}_equity_curve.png")
    plt.close()

def analyze_best_parameters(results_df, metric='sharpe_ratio'):
    """
    Analyze the best parameters from grid search results.
    
    Args:
        results_df (DataFrame): Grid search results
        metric (str): Metric to optimize
    
    Returns:
        dict: Best parameters for each ticker
    """
    best_params = {}
    
    # Group by ticker and find best parameters
    for ticker, group in results_df.groupby('ticker'):
        # Find row with best metric
        if metric in ['MSE', 'MAE', 'max_drawdown']:
            # Lower is better
            best_row = group.loc[group[metric].idxmin()]
        else:
            # Higher is better
            best_row = group.loc[group[metric].idxmax()]
        
        best_params[ticker] = {
            'seq_len': int(best_row['seq_len']),
            'pred_length': int(best_row['pred_length']),
            'threshold': float(best_row['threshold']),
            metric: float(best_row[metric]),
            'direction_accuracy': float(best_row['Direction_Accuracy']),
            'strategy_return': float(best_row['strategy_return']),
            'sharpe_ratio': float(best_row['sharpe_ratio']),
            'win_rate': float(best_row['win_rate'])
        }
    
    return best_params

def run_detailed_evaluation(ticker, seq_len, pred_length, threshold, data_dir="data", results_dir="detailed_results"):
    """
    Run detailed evaluation for a specific ticker with best parameters.
    
    Args:
        ticker (str): Ticker symbol
        seq_len (int): Sequence length
        pred_length (int): Prediction length
        threshold (float): Signal threshold
        data_dir (str): Directory containing data files
        results_dir (str): Directory to save results
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, data_dir)
    results_dir = os.path.join(root_dir, results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    csv_file = os.path.join(data_dir, f"{ticker}.csv")
    
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found.")
        return
    
    # Prepare data
    print(f"Preparing data for {ticker} with seq_len={seq_len}, pred_length={pred_length}...")
    training, validation, data, train_dataloader, val_dataloader = prepare_data(
        csv_file=csv_file,
        seq_len=seq_len,
        prediction_length=pred_length
    )
    
    # Train model
    print(f"Training model for {ticker}...")
    model = train_tft_model(
        training=training,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        learning_rate=0.001,
        max_epochs=30,  # More epochs for final model
        enable_progress_bar=True
    )
    
    # Save model
    model_path = f"{results_dir}/{ticker}_tft_model.ckpt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Generate predictions
    print(f"Generating predictions for {ticker}...")
    result_df = generate_predictions(
        model=model,
        validation_dataloader=val_dataloader,
        data=data,
        threshold=threshold
    )
    
    # Save predictions to CSV
    result_df.to_csv(f"{results_dir}/{ticker}_tft_predictions.csv")
    print(f"Predictions saved to {results_dir}/{ticker}_tft_predictions.csv")
    
    # Evaluate predictions
    print(f"Evaluating predictions for {ticker}...")
    metrics = evaluate_predictions(result_df)
    
    # Add trading strategy evaluation
    additional_metrics = evaluate_trading_strategy(result_df)
    metrics.update(additional_metrics)
    
    # Print evaluation results
    print("\nPrediction Evaluation Results:")
    print("==============================")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    
    # Plot predictions
    print(f"Plotting predictions for {ticker}...")
    plot_predictions(result_df, ticker, save_dir=results_dir)
    
    # Plot confusion matrix
    plot_confusion_matrix(result_df, ticker, save_dir=results_dir)
    
    # Plot equity curve
    plot_equity_curve(result_df, ticker, save_dir=results_dir)
    
    # Try to analyze feature importance
    try:
        print(f"Analyzing feature importance for {ticker}...")
        batch = next(iter(val_dataloader))
        interpretation = model.interpret_output(batch)
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        order = interpretation["variable_importance"].mean(0).sort_values(ascending=False).index
        interpretation["variable_importance"].mean(0)[order].plot.barh()
        plt.title(f"{ticker} - Feature Importance")
        plt.tight_layout()
        plt.savefig(f"{results_dir}/{ticker}_tft_feature_importance.png")
        plt.close()
        print(f"Feature importance plot saved to {results_dir}/{ticker}_tft_feature_importance.png")
    except Exception as e:
        print(f"Error generating feature importance: {e}")
        print("Skipping feature importance analysis")
    
    # Save metrics to CSV
    pd.DataFrame([metrics]).to_csv(f"{results_dir}/{ticker}_tft_metrics.csv", index=False)
    print(f"Metrics saved to {results_dir}/{ticker}_tft_metrics.csv")
    
    return metrics

def plot_parameter_heatmap(results_df, ticker, metric='sharpe_ratio', save_dir="plots"):
    """
    Plot heatmap of parameter performance.
    
    Args:
        results_df (DataFrame): Grid search results
        ticker (str): Ticker symbol
        metric (str): Metric to visualize
        save_dir (str): Directory to save plot
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(root_dir, save_dir)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter results for the specified ticker
    ticker_results = results_df[results_df['ticker'] == ticker]
    
    if len(ticker_results) == 0:
        print(f"No results found for ticker {ticker}")
        return
    
    # Get unique values for seq_len and pred_length
    seq_lengths = sorted(ticker_results['seq_len'].unique())
    pred_lengths = sorted(ticker_results['pred_length'].unique())
    
    # For each threshold, create a heatmap
    for threshold in ticker_results['threshold'].unique():
        # Filter results for this threshold
        threshold_results = ticker_results[ticker_results['threshold'] == threshold]
        
        # Create a pivot table
        pivot_data = threshold_results.pivot_table(
            index='seq_len', 
            columns='pred_length',
            values=metric,
            aggfunc='mean'
        )
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis')
        plt.title(f'{ticker} - {metric} (threshold={threshold})')
        plt.xlabel('Prediction Length')
        plt.ylabel('Sequence Length')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{ticker}_{metric}_heatmap_thresh{threshold}.png")
        plt.close()

def main():
    # Get the root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create necessary directories with absolute paths
    os.makedirs(os.path.join(root_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "grid_search_results"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "detailed_results"), exist_ok=True)
    
    # Define tickers to test
    tickers = ["AAPL", "MSFT"]
    
    # Define parameter ranges for grid search
    seq_lengths = [10, 20, 30, 60]
    pred_lengths = [1, 3, 5]
    thresholds = [0.001, 0.002, 0.005]
    
    # Run grid search
    print("Starting grid search...")
    results_df = run_grid_search(
        tickers=tickers,
        seq_lengths=seq_lengths,
        pred_lengths=pred_lengths,
        thresholds=thresholds,
        data_dir="data",
        results_dir="grid_search_results"
    )
    
    # Analyze best parameters
    print("\nAnalyzing best parameters...")
    best_params = analyze_best_parameters(results_df, metric='sharpe_ratio')
    
    # Save best parameters
    with open("grid_search_results/best_parameters.json", 'w') as f:
        json.dump(best_params, f, indent=4)
    
    print("\nBest parameters for each ticker:")
    for ticker, params in best_params.items():
        print(f"  {ticker}: seq_len={params['seq_len']}, pred_length={params['pred_length']}, threshold={params['threshold']}")
        print(f"    Sharpe Ratio: {params['sharpe_ratio']:.4f}, Direction Accuracy: {params['direction_accuracy']:.4f}")
        print(f"    Strategy Return: {params['strategy_return']:.4f}, Win Rate: {params['win_rate']:.4f}")
    
    # Plot parameter heatmaps
    print("\nCreating parameter heatmaps...")
    for ticker in tickers:
        plot_parameter_heatmap(results_df, ticker, metric='sharpe_ratio')
        plot_parameter_heatmap(results_df, ticker, metric='Direction_Accuracy')
    
    # Run detailed evaluation with best parameters
    print("\nRunning detailed evaluation with best parameters...")
    for ticker, params in best_params.items():
        print(f"\nDetailed evaluation for {ticker}...")
        run_detailed_evaluation(
            ticker=ticker,
            seq_len=params['seq_len'],
            pred_length=params['pred_length'],
            threshold=params['threshold'],
            data_dir="data",
            results_dir="detailed_results"
        )
    
    print("\nEvaluation complete. Results saved to 'grid_search_results' and 'detailed_results' directories.")

if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings("ignore")
    main()