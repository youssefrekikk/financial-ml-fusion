import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def run_permutation_test(returns, signals, n_permutations=1000, metric='sharpe'):
    """
    Run a permutation test to assess the statistical significance of a trading strategy.
    
    Parameters:
    -----------
    returns : pd.Series
        The asset returns
    signals : pd.Series
        The trading signals (-1, 0, 1)
    n_permutations : int
        Number of permutations to run
    metric : str
        Performance metric to use ('sharpe', 'total_return', 'win_rate')
        
    Returns:
    --------
    dict
        Dictionary containing test results and p-value
    """
    
    # Calculate actual strategy performance
    strategy_returns = returns * signals.shift(1).fillna(0)
    
    if metric == 'sharpe':
        actual_metric = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    elif metric == 'total_return':
        actual_metric = (1 + strategy_returns).prod() - 1
    elif metric == 'win_rate':
        actual_metric = (strategy_returns > 0).mean()
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Run permutations
    permuted_metrics = []
    for _ in tqdm(range(n_permutations), desc="Running permutation test"):
        # Shuffle the signals
        permuted_signals = signals.sample(frac=1).reset_index(drop=True)
        permuted_signals.index = signals.index
        
        # Calculate strategy returns with permuted signals
        permuted_strategy_returns = returns * permuted_signals.shift(1).fillna(0)
        
        # Calculate metric
        if metric == 'sharpe':
            permuted_metric = permuted_strategy_returns.mean() / permuted_strategy_returns.std() * np.sqrt(252)
        elif metric == 'total_return':
            permuted_metric = (1 + permuted_strategy_returns).prod() - 1
        elif metric == 'win_rate':
            permuted_metric = (permuted_strategy_returns > 0).mean()
            
        permuted_metrics.append(permuted_metric)
    
    # Calculate p-value (one-sided test)
    p_value = np.mean([m >= actual_metric for m in permuted_metrics])
    
    return {
        'actual_metric': actual_metric,
        'permuted_metrics': permuted_metrics,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def plot_permutation_test_results(results, metric_name):
    """
    Plot the results of a permutation test.
    
    Parameters:
    -----------
    results : dict
        Results from run_permutation_test
    metric_name : str
        Name of the metric for plot labels
    """
    plt.figure(figsize=(12, 6))
    
    # Plot histogram of permuted metrics
    sns.histplot(results['permuted_metrics'], kde=True, color='skyblue')
    
    # Plot vertical line for actual metric
    plt.axvline(results['actual_metric'], color='red', linestyle='--', 
                label=f'Actual {metric_name}: {results["actual_metric"]:.4f}')
    
    # Add p-value annotation
    significance = "Significant" if results['p_value'] < 0.05 else "Not Significant"
    plt.text(0.05, 0.95, f'p-value: {results["p_value"]:.4f} ({significance})', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title(f'Permutation Test Results for {metric_name}')
    plt.xlabel(metric_name)
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def run_walk_forward_permutation_test(data, model, window_size=252, step_size=21, 
                                     n_permutations=100, metric='sharpe', optimize_once=True):
    """
    Run a walk-forward permutation test to assess strategy significance over time.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The full dataset
    model : ImprovedHMMModel
        The model to test
    window_size : int
        Size of each walk-forward window in days
    step_size : int
        Number of days to step forward
    n_permutations : int
        Number of permutations per window
    metric : str
        Performance metric to use
    optimize_once : bool
        Whether to optimize parameters only on the first window
        
    Returns:
    --------
    pd.DataFrame
        Results of walk-forward permutation test
    """
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    
    results = []
    
    # Ensure we have enough data
    if len(data) <= window_size:
        raise ValueError("Data length must be greater than window size")
    
    # Get the first window for initial optimization
    first_window = data.iloc[:window_size].copy()
    train_size = int(len(first_window) * 0.7)
    initial_train_data = first_window.iloc[:train_size]
    
    # Fit model on initial training data with optimization
    if optimize_once:
        model.fit(initial_train_data, optimize=True)
        # Store optimized parameters
        optimized_params = model.best_params.copy() if model.best_params else None
    
    # Run walk-forward analysis
    for start_idx in tqdm(range(0, len(data) - window_size, step_size), 
                          desc="Running walk-forward permutation test"):
        end_idx = start_idx + window_size
        window_data = data.iloc[start_idx:end_idx].copy()
        
        # Split window into train/test
        train_size = int(len(window_data) * 0.7)
        train_data = window_data.iloc[:train_size]
        test_data = window_data.iloc[train_size:]
        
        try:
            # Fit model on training data
            if optimize_once and start_idx > 0:
                # Use the parameters optimized on initial training set
                model.best_params = optimized_params
                model.fit(train_data, optimize=False)
            else:
                # Optimize for this window
                model.fit(train_data, optimize=True)
            
            # Generate predictions on test data
            predictions = model.predict(test_data)
            
            # Extract returns and signals
            returns = predictions['Log_Return']
            signals = predictions['Signal']
            
            # Check if we have valid signals
            if signals.isna().all() or len(signals) == 0:
                print(f"Skipping window {start_idx}-{end_idx}: No valid signals")
                continue
                
            # Run permutation test
            perm_results = run_permutation_test(
                returns, signals, n_permutations=n_permutations, metric=metric
            )
            
            # Store results
            results.append({
                'start_date': window_data.index[0],
                'end_date': window_data.index[-1],
                'actual_metric': perm_results['actual_metric'],
                'p_value': perm_results['p_value'],
                'significant': perm_results['p_value'] < 0.05
            })
        except Exception as e:
            print(f"Error in window {start_idx}-{end_idx}: {e}")
            continue
    
    # Check if we have any results
    if not results:
        print("No valid results from walk-forward permutation test")
        return pd.DataFrame()
        
    return pd.DataFrame(results)

def plot_walk_forward_permutation_results(wf_results, metric_name):
    """
    Plot the results of a walk-forward permutation test.
    
    Parameters:
    -----------
    wf_results : pd.DataFrame
        Results from run_walk_forward_permutation_test
    metric_name : str
        Name of the metric for plot labels
    """
    plt.figure(figsize=(14, 8))
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot actual metric values
    ax1.plot(wf_results['end_date'], wf_results['actual_metric'], 
             marker='o', linestyle='-', color='blue')
    ax1.set_title(f'Walk-Forward {metric_name} Values')
    ax1.set_ylabel(metric_name)
    ax1.grid(True, alpha=0.3)
    
    # Plot p-values
    scatter = ax2.scatter(wf_results['end_date'], wf_results['p_value'], 
                         c=wf_results['significant'].map({True: 'green', False: 'red'}),
                         marker='o', s=50, alpha=0.7)
    
    # Add significance threshold line
    ax2.axhline(0.05, color='black', linestyle='--', alpha=0.7, label='Significance threshold (p=0.05)')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Significant (p<0.05)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Not significant (p≥0.05)')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    ax2.set_title('Statistical Significance (p-values)')
    ax2.set_xlabel('Test End Date')
    ax2.set_ylabel('p-value')
    ax2.grid(True, alpha=0.3)
    
    # Format the plot
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    significant_pct = wf_results['significant'].mean() * 100
    print(f"Percentage of significant periods: {significant_pct:.2f}%")
    print(f"Average {metric_name}: {wf_results['actual_metric'].mean():.4f}")
    print(f"Average p-value: {wf_results['p_value'].mean():.4f}")
