def run_monte_carlo_simulation(model, test_data, n_simulations=1000, confidence_level=0.95, 
                              metrics=['total_return', 'sharpe', 'max_drawdown', 'win_rate']):
    """
    Run Monte Carlo simulation to assess the robustness of a trading model.
    
    Parameters:
    -----------
    model : object
        A fitted model with a predict() method that returns a DataFrame with 'Signal' and 'Log_Return' columns
    test_data : pd.DataFrame
        Historical price data with OHLCV columns as returned by load_data()
    n_simulations : int
        Number of Monte Carlo simulations to run
    confidence_level : float
        Confidence level for calculating confidence intervals (0-1)
    metrics : list
        List of performance metrics to calculate
        
    Returns:
    --------
    dict
        Dictionary containing simulation results for each metric
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from utils import add_features  # Import your feature engineering function
    
    # First, add features to the original data to ensure it has 'Log_Return'
    original_data_with_features = add_features(test_data)
    
    # Extract log returns from the original data
    original_returns = original_data_with_features['Log_Return'].dropna()
    
    # Calculate statistical properties of returns
    mean_return = original_returns.mean()
    std_return = original_returns.std()
    
    # Store simulation results
    simulation_results = {metric: [] for metric in metrics}
    
    # Run simulations
    print(f"Running {n_simulations} Monte Carlo simulations...")
    for i in tqdm(range(n_simulations)):
        # Generate synthetic returns using a normal distribution
        synthetic_returns = np.random.normal(mean_return, std_return, len(original_returns))
        
        # Create synthetic price series
        start_price = test_data['Close'].iloc[0]
        synthetic_prices = start_price * np.exp(np.cumsum(synthetic_returns))
        
        # Create synthetic OHLCV data
        synthetic_data = test_data.copy()
        synthetic_data['Close'] = synthetic_prices
        
        # Adjust other price columns based on their typical relationship to Close
        # Calculate average ratios from original data
        high_close_ratio = (test_data['High'] / test_data['Close']).mean()
        low_close_ratio = (test_data['Low'] / test_data['Close']).mean()
        
        synthetic_data['High'] = synthetic_data['Close'] * high_close_ratio
        synthetic_data['Low'] = synthetic_data['Close'] * low_close_ratio
        synthetic_data['Open'] = synthetic_data['Close'].shift(1)
        synthetic_data.loc[synthetic_data.index[0], 'Open'] = synthetic_data['Close'].iloc[0]
        
        # Keep original volume data
        # Volume could also be simulated if needed
        
        try:
            # Add features to synthetic data using your feature engineering function
            synthetic_data_with_features = add_features(synthetic_data)
            
            # Run the model on synthetic data
            predictions = model.predict(synthetic_data_with_features)
            
            # Calculate strategy returns
            predictions['Strategy_Return'] = predictions['Signal'].shift(1) * predictions['Log_Return']
            
            # Calculate cumulative returns
            predictions['Strategy_Cumulative'] = np.exp(predictions['Strategy_Return'].cumsum())
            
            # Calculate metrics
            if 'total_return' in metrics:
                total_return = predictions['Strategy_Cumulative'].iloc[-1] - 1
                simulation_results['total_return'].append(total_return)
                
            if 'sharpe' in metrics:
                sharpe = (predictions['Strategy_Return'].mean() / predictions['Strategy_Return'].std() * np.sqrt(252)) if predictions['Strategy_Return'].std() > 0 else 0
                simulation_results['sharpe'].append(sharpe)
                
            if 'max_drawdown' in metrics:
                # Calculate running maximum
                running_max = predictions['Strategy_Cumulative'].cummax()
                # Calculate drawdown
                drawdown = (running_max - predictions['Strategy_Cumulative']) / running_max
                # Get maximum drawdown
                max_drawdown = drawdown.max()
                simulation_results['max_drawdown'].append(max_drawdown)
                
            if 'win_rate' in metrics:
                win_rate = (predictions['Strategy_Return'] > 0).mean()
                simulation_results['win_rate'].append(win_rate)
                
        except Exception as e:
            print(f"Simulation {i} failed: {e}")
            # Continue with next simulation
            continue
    
    # Calculate statistics for each metric
    results = {}
    for metric in metrics:
        if len(simulation_results[metric]) > 0:
            values = np.array(simulation_results[metric])
            
            # Calculate confidence intervals
            lower_percentile = (1 - confidence_level) / 2 * 100
            upper_percentile = (1 + confidence_level) / 2 * 100
            
            results[metric] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                f'lower_{confidence_level*100:.0f}': np.percentile(values, lower_percentile),
                f'upper_{confidence_level*100:.0f}': np.percentile(values, upper_percentile),
                'values': values  # Store all values for plotting
            }
    
    return results

def plot_monte_carlo_results(mc_results, metrics=None, original_metrics=None):
    """
    Plot the results of Monte Carlo simulations.
    
    Parameters:
    -----------
    mc_results : dict
        Results from run_monte_carlo_simulation
    metrics : list, optional
        List of metrics to plot. If None, plot all metrics in mc_results.
    original_metrics : dict, optional
        Dictionary of original model metrics for comparison
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    if metrics is None:
        metrics = [m for m in mc_results.keys() if m != 'values']
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 5*n_metrics))
    
    # Ensure axes is always a list
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        if metric not in mc_results:
            continue
            
        ax = axes[i]
        
        # Plot histogram with KDE
        sns.histplot(mc_results[metric]['values'], kde=True, ax=ax)
        
        # Add vertical lines for confidence intervals
        conf_level = [k for k in mc_results[metric].keys() if 'lower_' in k][0]
        conf_pct = conf_level.split('_')[1]
        
        lower_bound = mc_results[metric][conf_level]
        upper_bound = mc_results[metric][f'upper_{conf_pct}']
        
        ax.axvline(lower_bound, color='r', linestyle='--', 
                   label=f'Lower {conf_pct}% CI: {lower_bound:.4f}')
        ax.axvline(upper_bound, color='r', linestyle='--',
                   label=f'Upper {conf_pct}% CI: {upper_bound:.4f}')
        
        # Add mean line
        ax.axvline(mc_results[metric]['mean'], color='g', linestyle='-',
                   label=f'Mean: {mc_results[metric]["mean"]:.4f}')
        
        # Add original metric if provided
        if original_metrics is not None and metric in original_metrics:
            ax.axvline(original_metrics[metric], color='b', linestyle='-',
                       label=f'Original: {original_metrics[metric]:.4f}')
        
        # Format plot
        ax.set_title(f'Monte Carlo Distribution: {metric.replace("_", " ").title()}')
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_ylabel('Frequency')
        ax.legend()
    
    plt.tight_layout()
    plt.show()




def compare_monte_carlo_to_original(model, test_data, n_simulations=1000, confidence_level=0.95):
    """
    Run Monte Carlo simulation and compare results to the original model performance.
    
    Parameters:
    -----------
    model : object
        A fitted model with a predict() method
    test_data : pd.DataFrame
        Historical price data with OHLCV columns as returned by load_data()
    n_simulations : int
        Number of Monte Carlo simulations to run
    confidence_level : float
        Confidence level for calculating confidence intervals
        
    Returns:
    --------
    tuple
        (mc_results, original_metrics, comparison_df)
    """
    import pandas as pd
    import numpy as np
    from utils import add_features  # Import your feature engineering function
    
    # Define metrics to calculate
    metrics = ['total_return', 'sharpe', 'max_drawdown', 'win_rate']
    
    # Add features to the original data
    data_with_features = add_features(test_data)
    
    # Calculate original model metrics
    predictions = model.predict(data_with_features)
    predictions['Strategy_Return'] = predictions['Signal'].shift(1) * predictions['Log_Return']
    predictions['Strategy_Cumulative'] = np.exp(predictions['Strategy_Return'].cumsum())
    
    original_metrics = {}
    
    # Total return
    original_metrics['total_return'] = predictions['Strategy_Cumulative'].iloc[-1] - 1
    
    # Sharpe ratio
    original_metrics['sharpe'] = (predictions['Strategy_Return'].mean() / 
                                 predictions['Strategy_Return'].std() * 
                                 np.sqrt(252)) if predictions['Strategy_Return'].std() > 0 else 0
    
    # Max drawdown
    running_max = predictions['Strategy_Cumulative'].cummax()
    drawdown = (running_max - predictions['Strategy_Cumulative']) / running_max
    original_metrics['max_drawdown'] = drawdown.max()
    
    # Win rate
    original_metrics['win_rate'] = (predictions['Strategy_Return'] > 0).mean()
    
    # Run Monte Carlo simulation
    mc_results = run_monte_carlo_simulation(model, test_data, n_simulations, 
                                           confidence_level, metrics)
    
    # Create comparison DataFrame
    comparison_data = []
    for metric in metrics:
        if metric in mc_results:
            result = mc_results[metric]
            
            # Calculate percentile of original metric in MC distribution
            percentile = sum(result['values'] <= original_metrics[metric]) / len(result['values']) * 100
            
            # Determine if original is within confidence interval
            lower_bound = result[f'lower_{confidence_level*100:.0f}']
            upper_bound = result[f'upper_{confidence_level*100:.0f}']
            within_ci = lower_bound <= original_metrics[metric] <= upper_bound
            
            comparison_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Original': original_metrics[metric],
                'MC Mean': result['mean'],
                'MC Median': result['median'],
                f'MC {confidence_level*100:.0f}% Lower': lower_bound,
                f'MC {confidence_level*100:.0f}% Upper': upper_bound,
                'Percentile': percentile,
                f'Within {confidence_level*100:.0f}% CI': within_ci
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Plot results
    plot_monte_carlo_results(mc_results, metrics, original_metrics)
    
    return mc_results, original_metrics, comparison_df

