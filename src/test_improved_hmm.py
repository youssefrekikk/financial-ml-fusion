import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from functools import wraps
import seaborn as sns
from improved_hmm_model import ImprovedHMMModel
from utils import load_data,compare_monte_carlo_to_original

from trade_simulator import TradeSimulator
from sklearn.metrics import classification_report, confusion_matrix
import yfinance as yf
import os


os.makedirs("results/plots", exist_ok=True)

figure_count = 0

original_show = plt.show

# Define a wrapper for plt.show
@wraps(plt.show)
def show_and_save(*args, **kwargs):
    global figure_count
    plt.savefig(f"results/plots/figure_{figure_count}.png", dpi=300, bbox_inches='tight')
    print(f"Saved figure to results/plots/figure_{figure_count}.png")
    figure_count += 1
    # Call the original function
    return original_show(*args, **kwargs)

# Replace plt.show with our wrapper
plt.show = show_and_save

def save_figure(plt, name):
    """Save the current figure with the given name"""
    plt.savefig(f"results/plots/{name}.png", dpi=300, bbox_inches='tight')
    print(f"Saved plot to results/plots/{name}.png")



def split_data(data, train_size=0.7):
    """Split data into training and testing sets."""
    split_idx = int(len(data) * train_size)
    train_data = data[:split_idx].copy()
    test_data = data[split_idx:].copy()
    return train_data, test_data

def evaluate_model(model, test_data):
    """Evaluate model performance on test data."""
    # Predict states
    predictions = model.predict(test_data)
    
    # Calculate strategy returns
    predictions['Strategy_Return'] = predictions['Signal'].shift(1) * predictions['Log_Return']
    predictions['BuyHold_Return'] = predictions['Log_Return']
    
    # Calculate cumulative returns
    predictions['Strategy_Cumulative'] = np.exp(predictions['Strategy_Return'].cumsum())
    predictions['BuyHold_Cumulative'] = np.exp(predictions['BuyHold_Return'].cumsum())
    
    # Calculate performance metrics
    strategy_return = predictions['Strategy_Cumulative'].iloc[-1] - 1
    buyhold_return = predictions['BuyHold_Cumulative'].iloc[-1] - 1
    
    strategy_sharpe = (predictions['Strategy_Return'].mean() / predictions['Strategy_Return'].std() * np.sqrt(252)) if predictions['Strategy_Return'].std() > 0 else 0
    buyhold_sharpe = (predictions['BuyHold_Return'].mean() / predictions['BuyHold_Return'].std() * np.sqrt(252)) if predictions['BuyHold_Return'].std() > 0 else 0
    
    # Print results
    print("\nPerformance Metrics:")
    print(f"Strategy Total Return: {strategy_return*100:.2f}%")
    print(f"Buy & Hold Total Return: {buyhold_return*100:.2f}%")
    print(f"Strategy Sharpe Ratio: {strategy_sharpe:.2f}")
    print(f"Buy & Hold Sharpe Ratio: {buyhold_sharpe:.2f}")
    print(f"Strategy Win Rate: {(predictions['Strategy_Return'] > 0).mean()*100:.2f}%")
    print(f"Buy & Hold Win Rate: {(predictions['BuyHold_Return'] > 0).mean()*100:.2f}%")
    
    # Create directional accuracy confusion matrix
    true_direction = np.sign(predictions['Log_Return'])
    pred_direction = np.sign(predictions['Signal'].shift(1))
    
    # Remove NaNs
    mask = ~(np.isnan(true_direction) | np.isnan(pred_direction))
    true_direction = true_direction[mask]
    pred_direction = pred_direction[mask]
    
    # Print classification report
    print("\nDirectional Accuracy:")
    print(classification_report(true_direction, pred_direction))
    
    # Plot confusion matrix
    cm = confusion_matrix(true_direction, pred_direction)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Down', 'Flat', 'Up'],
                yticklabels=['Down', 'Flat', 'Up'])
    plt.title('Confusion Matrix for Directional Prediction')
    plt.xlabel('Predicted Direction')
    plt.ylabel('True Direction')
    plt.title('Confusion Matrix for Directional Prediction')
    save_figure(plt, "confusion_matrix")
    plt.show()
    
    return predictions

def test_with_simulator(model, test_data, ticker):
    """Test the model using the TradeSimulator with improved parameters."""
    # Generate signals for simulator
    signals = model.generate_signals_for_simulator(test_data,symbol=ticker)
    
    # Check if we have any signals
    if not signals:
        print("No signals generated for simulator. Skipping simulation.")
        return None
    
    # Initialize simulator with more conservative parameters
    simulator = TradeSimulator(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
        stop_loss_pct=0.03,  # Increased from 0.02
        trailing_stop_pct=0.04,  # Increased from 0.03
        take_profit_pct=0.09,  # Increased from 0.05
        max_position_size=0.15,  # Reduced from 0.2
        risk_per_trade=0.01
    )
    
    # Prepare data for simulator
    data = {ticker: test_data}
    
    # Print signal summary before simulation
    signal_counts = {-1: 0, 0: 0, 1: 0}
    for ts, sig_dict in signals.items():
        if ticker in sig_dict:
            signal_counts[sig_dict[ticker]['signal']] += 1
    
    print(f"\nSignal Summary for Simulator:")
    print(f"Buy signals (1): {signal_counts[1]}")
    print(f"Sell signals (-1): {signal_counts[-1]}")
    print(f"Neutral signals (0): {signal_counts[0]}")
    
    # Run simulation
    performance = simulator.run_simulation(data, signals)
    
    # Print performance metrics
    print("\nTradeSimulator Performance:")
    print(f"Total Return: {performance['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {performance['max_drawdown']*100:.2f}%")
    print(f"Win Rate: {performance['win_rate']*100:.2f}%")
    print(f"Profit Factor: {performance['profit_factor']:.2f}")
    print(f"Average Trade: ${performance['avg_trade']:.2f}")
    
    # Get trade log
    trade_log = simulator.get_trade_log()
    if not trade_log.empty:
        print(f"\nTotal trades executed: {len(trade_log)}")
        print(f"First few trades:")
        print(trade_log.head())
    else:
        print("\nNo trades were executed during simulation.")
    
    # Plot equity curve
    simulator.plot_equity_curve()
    
    # Plot drawdown curve
    simulator.plot_drawdown_curve()
    
    return simulator


def run_monte_carlo_analysis(model, test_data, ticker):
    """
    Run Monte Carlo analysis on the model to assess robustness.
    
    Parameters:
    -----------
    model : ImprovedHMMModel
        A fitted HMM model
    test_data : pd.DataFrame
        Test data with OHLCV columns
    ticker : str
        Ticker symbol for display purposes
    """
    print(f"\n=== Running Monte Carlo Analysis for {ticker} ===")
    print("This will generate synthetic price series and test model robustness...")
    
    # Run Monte Carlo simulation with 1000 simulations and 95% confidence level
    mc_results, original_metrics, comparison_df = compare_monte_carlo_to_original(
        model, test_data, n_simulations=1000, confidence_level=0.95
    )
    
    # Print comparison results
    print("\nMonte Carlo vs Original Performance:")
    print(comparison_df.to_string(index=False))
    
    # Interpret results
    print("\nInterpretation:")
    for _, row in comparison_df.iterrows():
        metric = row['Metric']
        percentile = row['Percentile']
        within_ci = row['Within 95% CI']
        
        if within_ci:
            print(f"- {metric}: Original performance is within the 95% confidence interval (percentile: {percentile:.1f})")
        else:
            if percentile < 2.5:
                print(f"- {metric}: Original performance is WORSE than expected (percentile: {percentile:.1f})")
            elif percentile > 97.5:
                print(f"- {metric}: Original performance is BETTER than expected (percentile: {percentile:.1f})")
    
    return mc_results, original_metrics, comparison_df


def main():
    # Set parameters
    ticker = "AAPL"  # Changed from SPY to AAPL
    start_date = "2014-01-01"
    end_date = "2024-01-01"
    
    # Ensure we have the required imports
    import os
    from utils import run_walk_forward_permutation_test, plot_walk_forward_permutation_results
    # Load data
    data = load_data(ticker, start_date, end_date)
    if data is None or len(data) == 0:
        print("Failed to load any data. Exiting.")
        return None, None, None
    
    # Split data
    train_data, test_data = split_data(data, train_size=0.7)
    print(f"Training data: {len(train_data)} rows")
    print(f"Testing data: {len(test_data)} rows")
    
    # Initialize and train model
    model = ImprovedHMMModel(
        n_components_range=(2, 3),  # Reduced range to focus on more stable models
        covariance_types=["diag", "spherical"],  # Focus on more stable covariance types
        n_splits=3,  # Fewer splits for more stable cross-validation
        feature_selection=True,
        k_features=6  # Fewer features to reduce overfitting
    )
    
    # Fit model
    model.fit(train_data)
    
    
    # Evaluate on test data
    predictions = evaluate_model(model, test_data)
    
    # Plot states
    model.plot_states(predictions, title=f"{ticker} Market Regimes")
    
    # Plot state distributions
    model.plot_state_distributions(predictions)
    
    # Compare to benchmark
    # Load benchmark data (e.g., S&P 500 for individual stocks)
    benchmark_ticker = "^GSPC" if ticker != "SPY" else "QQQ"
    benchmark_data = load_data(benchmark_ticker, start_date, end_date)
    
    comparison = model.compare_to_benchmark(test_data, benchmark_data)
    print("\nPerformance Comparison:")
    print(comparison)
    
    # Plot comparison
    model.plot_comparison(test_data, benchmark_data)
    
    # Perform walk-forward validation
    wf_results = model.walk_forward_validation(data, initial_train_size=0.5, step_size=30)
    print("\nWalk-Forward Validation Results:")
    print(wf_results.describe())
    
    # Plot walk-forward results
    plt.figure(figsize=(12, 6))
    plt.plot(wf_results['test_end'], wf_results['sharpe'], marker='o')
    plt.title('Walk-Forward Validation: Sharpe Ratio')
    plt.xlabel('Test End Date')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    save_figure(plt, "Walk-Forward Validation: Sharpe Ratio")
    plt.show()
    
    # Test with simulator
    simulator = test_with_simulator(model, test_data, ticker)
    
    
    
    
    #monte carlo simulation (generate sythetic data price series with the same statistical proprities as the real data than test the robustness of the model)
    mc_results, original_metrics, mc_comparison = run_monte_carlo_analysis(model, test_data, ticker)

    
    
    #  Run permutation tests
    print("\n=== Running Permutation Tests ===")
    
    # Run standard permutation tests
    perm_results = model.run_permutation_tests(test_data, n_permutations=1000)
    
    # Run walk-forward permutation test
    print("\n=== Running Walk-Forward Permutation Test ===")
    wf_perm_results = run_walk_forward_permutation_test(
        data, 
        model, 
        window_size=252*2,  # 1 year of trading days
        step_size=63,     # ~3 months
        n_permutations=1000,
        metric='total_return',
        optimize_once=True
    )
    plot_walk_forward_permutation_results(wf_perm_results, "total_return")
    
    return model, predictions, simulator

if __name__ == "__main__":
    model, predictions, simulator = main()
