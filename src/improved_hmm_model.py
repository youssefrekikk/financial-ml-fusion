import pandas as pd
import numpy as np
import yfinance as yf
from utils import  add_features, normalize_features

from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import silhouette_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
from datetime import date
import warnings
warnings.filterwarnings('ignore')

class ImprovedHMMModel:
    """
    My improved HMM implementation for market regime detection.
    
    After struggling with the limitations of my first HMM model (overfitting and poor
    regime identification), I rebuilt it with better feature selection and more robust
    state detection. This version has been much more reliable in testing.
    """
    
    def __init__(self, 
                 n_components_range=(2, 5),
                 covariance_types=["full", "diag", "tied", "spherical"],
                 n_splits=5,
                 feature_selection=True,
                 k_features=8,
                 random_state=42):
        """
        Initialize the improved HMM model.
        Spent a lot of time experimenting with these defaults - they seem to work well
        across most stocks I've tested. Diag and spherical covariances tend to be more
        stable than full covariance matrices.
        
        Args:
            n_components_range: Range of states to test
            covariance_types: List of covariance types to test
            n_splits: Number of splits for time series cross-validation
            feature_selection: Whether to perform feature selection
            k_features: Number of features to select if feature_selection is True
            random_state: Random seed for reproducibility
        """
        self.n_components_range = range(n_components_range[0], n_components_range[1] + 1)
        self.covariance_types = covariance_types
        self.n_splits = n_splits
        self.feature_selection = feature_selection
        self.k_features = k_features
        self.random_state = random_state
        
        # Model storage
        self.best_model = None
        self.best_params = None
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        self.bull_state = None
        self.bear_state = None
        self.neutral_state = None
        
    

    
    def prepare_hmm_features(self, data):
        """
        Prepare features for HMM model with option for feature selection.
        """
        # Define candidate features
        feature_candidates = [
            'Log_Return', 'Return_Volatility', 'ATR_Pct',
            'Open_Close_Diff', 'High_Low_Diff', 'Close_Prev_Close_Diff',
            'SMA_20_50_Ratio', 'SMA_20_200_Ratio', 'Price_SMA_20_Ratio',
            'RSI', 'MACD', 'MACD_Hist', 'BB_Width', 'BB_Pct',
            'Momentum_14', 'ROC_14'
        ]
        
        # Add volume features if available
        if 'Relative_Volume' in data.columns:
            feature_candidates.extend(['Relative_Volume'])
        
        # Select features that exist in the data
        available_features = [f for f in feature_candidates if f in data.columns]
        
        self.available_features = available_features
        
        if not available_features:
            print("Warning: No features available for HMM model")
            return pd.DataFrame()  
        
        # Return the selected features
        return data[available_features]

    
    
    
    def select_features(self, X, y, fit=False):
        """
        Perform feature selection using SelectKBest.
        
        Feature selection made a huge difference in model stability.
        I tried more complex methods (RFE, LASSO) but they were too slow
        and didn't perform much better than this simple approach.
        """
        if not self.feature_selection:
            return X
            
        if fit or self.feature_selector is None:
            self.feature_selector = SelectKBest(f_regression, k=min(self.k_features, X.shape[1]))
            X_selected = self.feature_selector.fit_transform(X, y)
            
            # Store selected feature names
            mask = self.feature_selector.get_support()
            self.selected_features = [self.available_features[i] for i in range(len(mask)) if mask[i]]
            print(f"Selected features: {self.selected_features}")
        else:
            X_selected = self.feature_selector.transform(X)
            
        return X_selected
    
    def silhouette_scorer(self, estimator, X):
        """
        Custom scorer for GridSearchCV using silhouette score.
        """
        labels = estimator.predict(X)
        score = silhouette_score(X, labels)
        return score
    
    def optimize_hmm_parameters(self, hmm_data, y=None):
        """
        Optimize HMM parameters using time series cross-validation.
        This is the slowest part of the model fitting process.
        GridSearchCV doesn't work well with HMM so I had to implement
        my own cross-validation loop
        to do : Parallelize this to speed it up 
        """
        print("Optimizing HMM parameters...")
        
        # Define parameter grid
        param_grid = {
            "n_components": self.n_components_range,
            "covariance_type": self.covariance_types
        }
        
        # Create time series cross-validation splits
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        # Initialize best score and parameters
        best_score = -np.inf
        best_params = None
        best_model = None
        
        # Manual cross-validation (GridSearchCV doesn't work well with HMM)
        for n_components in param_grid["n_components"]:
            for covariance_type in param_grid["covariance_type"]:
                cv_scores = []
                
                for train_idx, val_idx in tscv.split(hmm_data):
                    # Split data
                    train_data = hmm_data[train_idx]
                    val_data = hmm_data[val_idx]
                    
                    # Train model
                    model = GaussianHMM(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        n_iter=1000,
                        random_state=self.random_state
                    )
                    
                    try:
                        model.fit(train_data)
                        
                        # Predict states
                        val_states = model.predict(val_data)
                        
                        # Calculate silhouette score
                        score = silhouette_score(val_data, val_states)
                        cv_scores.append(score)
                    except Exception as e:
                        print(f"Error fitting model with n_components={n_components}, covariance_type={covariance_type}: {e}")
                        cv_scores.append(-np.inf)
                
                # Calculate mean score across CV splits
                mean_score = np.mean(cv_scores) if cv_scores else -np.inf
                
                print(f"n_components={n_components}, covariance_type={covariance_type}, score={mean_score:.4f}")
                
                # Update best parameters if better score
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {
                        "n_components": n_components,
                        "covariance_type": covariance_type
                    }
        
        print(f"Best parameters: {best_params}, score: {best_score:.4f}")
        
        # Train final model with best parameters
        self.best_params = best_params
        self.best_model = GaussianHMM(
            n_components=best_params["n_components"],
            covariance_type=best_params["covariance_type"],
            n_iter=1000,
            random_state=self.random_state
        )
        
        self.best_model.fit(hmm_data)
        
        return self.best_model
    
    def identify_market_states(self, data, states):
        """
        Identify which states correspond to bull, bear, and neutral markets
        with improved logic.
        Had to be careful with the neutral state - sometimes it's actually the most
        profitable but with higher volatility.
        """
        # Add states to the data
        df = data.copy()
        df['State'] = states
        
        # Calculate forward returns for state analysis
        df['Forward_Return_5d'] = df['Close'].pct_change(periods=5).shift(-5)
        
        # Analyze each state
        state_analysis = {}
        
        for state in range(self.best_params["n_components"]):
            state_data = df[df['State'] == state]
            
            if len(state_data) > 0:
                # Calculate metrics for this state
                mean_return = state_data['Forward_Return_5d'].mean()
                volatility = state_data['Forward_Return_5d'].std()
                win_rate = (state_data['Forward_Return_5d'] > 0).mean()
                
                # Calculate additional metrics
                mean_daily_return = state_data['Log_Return'].mean()
                daily_win_rate = (state_data['Log_Return'] > 0).mean()
                
                state_analysis[state] = {
                    'mean_return': mean_return,
                    'volatility': volatility,
                    'win_rate': win_rate,
                    'sharpe': mean_return / volatility if volatility > 0 else 0,
                    'frequency': len(state_data) / len(df),
                    'mean_daily_return': mean_daily_return,
                    'daily_win_rate': daily_win_rate
                }
        
        # Identify bull, bear, and neutral states using multiple metrics
        # Use a weighted score of forward returns and daily returns
        state_scores = {}
        for state, metrics in state_analysis.items():
            # Weight forward returns more heavily but consider daily returns too
            score = (0.7 * metrics['mean_return']) + (0.3 * metrics['mean_daily_return'])
            state_scores[state] = score
        
        states_by_score = sorted(state_scores.items(), key=lambda x: x[1], reverse=True)
        
        if len(states_by_score) >= 3:
            self.bull_state = states_by_score[0][0]  # Highest score state
            self.neutral_state = states_by_score[1][0]  # Middle score state
            self.bear_state = states_by_score[-1][0]  # Lowest score state
        elif len(states_by_score) == 2:
            self.bull_state = states_by_score[0][0]  # Highest score state
            self.bear_state = states_by_score[-1][0]  # Lowest score state
            self.neutral_state = None
        elif len(states_by_score) == 1:
            self.bull_state = states_by_score[0][0]
            self.bear_state = self.neutral_state = None
        
        print(f"Bull state: {self.bull_state}, Neutral state: {self.neutral_state}, Bear state: {self.bear_state}")
        
        # Print detailed state analysis
        print("\nDetailed State Analysis:")
        for state, metrics in state_analysis.items():
            state_type = "Bull" if state == self.bull_state else "Bear" if state == self.bear_state else "Neutral"
            print(f"State {state} ({state_type}):")
            print(f"  Forward Return: {metrics['mean_return']*100:.2f}%")
            print(f"  Daily Return: {metrics['mean_daily_return']*100:.2f}%")
            print(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
            print(f"  Sharpe: {metrics['sharpe']:.2f}")
            print(f"  Frequency: {metrics['frequency']*100:.2f}%")
        
        return state_analysis

    
    def fit(self, data, target_column='Close', optimize=True):
        """
        Fit the HMM model to the data with improved data handling.
        This is the main entry point for training the model.
        Handles all the preprocessing, feature selection, and parameter optimization.
        
        Had some issues with NaN values in the early versions - this implementation
        is much more robust to missing data.
        """
        print("Fitting HMM model...")
        
        # Add features
        data_with_features = add_features(data)
        
        if len(data_with_features) <= 5:
            print("Warning: Not enough data for feature selection. Disabling feature selection.")
            self.feature_selection = False
        
        hmm_features = self.prepare_hmm_features(data_with_features)
        
        if len(hmm_features) == 0:
            print("Error: No data available after feature preparation")
            raise ValueError("No data available after feature preparation")
        
        # Create target for feature selection (future returns)
        if self.feature_selection:
            # Calculate future returns but don't trim the data yet
            y = data_with_features[target_column].pct_change(5).shift(-5)
            
            # Find where both X and y are valid (not NaN)
            valid_indices = ~y.isna()
            
            # Only keep rows where both X and y are valid
            valid_hmm_features = hmm_features.loc[valid_indices]
            valid_y = y.loc[valid_indices]
            
            # Check if we still have enough data
            if len(valid_hmm_features) > 0:
                hmm_features_selected = self.select_features(valid_hmm_features, valid_y, fit=True)
            else:
                print("Warning: No valid data for feature selection. Disabling feature selection.")
                self.feature_selection = False
                hmm_features_selected = hmm_features.values
        else:
            hmm_features_selected = hmm_features.values
        
        # Normalize features
        normalized_features,scaler = normalize_features(hmm_features_selected, fit=True)
        
        # Optimize HMM parameters
        if optimize or self.best_model is None:
            self.optimize_hmm_parameters(normalized_features)
        else:
            print("Using existing model parameters:", self.best_params)
            # Just fit the model with existing parameters
            if self.best_model is not None:
                try:
                    self.best_model.fit(normalized_features)
                except Exception as e:
                    print(f"Error fitting model with existing parameters: {e}")
                    print("Re-optimizing parameters...")
                    self.optimize_hmm_parameters(normalized_features)
        # Predict states
        states = self.best_model.predict(normalized_features)
        
        # Identify market states (only if we have enough data)
        if len(data_with_features) > 5:
            # For state analysis, we need to align the states with the original data
            # that has forward returns calculated
            analysis_data = data_with_features.copy()
            analysis_data['Forward_Return_5d'] = analysis_data['Close'].pct_change(5).shift(-5)
            
            # Only use the portion of data that has valid states and forward returns
            valid_analysis_data = analysis_data.iloc[:len(states)].copy()
            valid_analysis_data = valid_analysis_data[~valid_analysis_data['Forward_Return_5d'].isna()]
            
            # Only use the corresponding states
            valid_states = states[:len(valid_analysis_data)]
            
            self.state_analysis = self.identify_market_states(valid_analysis_data, valid_states)
        else:
            print("Warning: Not enough data for state analysis.")
            # Set default state mappings
            self.bull_state = 0
            self.bear_state = 1 if self.best_params["n_components"] > 1 else 0
            self.neutral_state = 2 if self.best_params["n_components"] > 2 else None
        
        return self


    
    def predict(self, data):
        """
        Predict hidden states for new data with improved signal generation.
        to do: Add more sophisticated entry/exit rules based on state transitions
        """
        if self.best_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Add features
        data_with_features = add_features(data)
        
        # Prepare HMM features
        hmm_features = self.prepare_hmm_features(data_with_features)
        
        # Check if we have data after feature preparation
        if len(hmm_features) == 0:
            print("Warning: No data available after feature preparation")
            # Return original data with default signals
            result = data_with_features.copy()
            result['Predicted_State'] = np.nan
            result['Market_Regime'] = 'Unknown'
            result['Signal'] = 0
            result['Signal_Confidence'] = 0
            return result
        
        # Select features if enabled
        if self.feature_selection and self.feature_selector is not None:
            try:
                hmm_features_selected = self.feature_selector.transform(hmm_features)
            except Exception as e:
                print(f"Feature selection failed: {e}. Using all features.")
                hmm_features_selected = hmm_features.values
        else:
            hmm_features_selected = hmm_features.values
        
        # Normalize features
        normalized_features,scaler = normalize_features(hmm_features_selected, fit=False)
        
        # Predict states
        states = self.best_model.predict(normalized_features)
        
        # Get state probabilities for confidence
        state_probs = self.best_model.predict_proba(normalized_features)
        
        # Add states to the data
        result = data_with_features.copy()
        result['Predicted_State'] = states
        
        # Add state probabilities for each state
        for i in range(self.best_params["n_components"]):
            result[f'State_{i}_Prob'] = [probs[i] for probs in state_probs]
        
        # Add state labels
        result['Market_Regime'] = result['Predicted_State'].map(
            lambda x: 'Bull' if x == self.bull_state else 
                    ('Bear' if x == self.bear_state else 'Neutral')
        )
        
        # Add trading signals with improved logic
        result['Signal'] = 0
        result['Signal_Confidence'] = 0
        
        # Only generate bull signals when confidence is high and trend is favorable
        bull_mask = (result['Predicted_State'] == self.bull_state) & \
                    (result[f'State_{self.bull_state}_Prob'] > 0.7) & \
                    (result['Close'] > result['SMA_20'])
        
        # Only generate bear signals when confidence is high and trend is unfavorable
        bear_mask = (result['Predicted_State'] == self.bear_state) & \
                    (result[f'State_{self.bear_state}_Prob'] > 0.7) & \
                    (result['Close'] < result['SMA_20'])
        
        result.loc[bull_mask, 'Signal'] = 1
        result.loc[bull_mask, 'Signal_Confidence'] = result.loc[bull_mask, f'State_{self.bull_state}_Prob']
        
        result.loc[bear_mask, 'Signal'] = -1
        result.loc[bear_mask, 'Signal_Confidence'] = result.loc[bear_mask, f'State_{self.bear_state}_Prob']
        
        return result


    
    def predict_current_state(self, recent_data, window_size=30):
        """
        Predict the current market state using recent data.
        
        Useful for real-time applications where you want to know
        the current regime without running the full backtest.
        Window size of 30 days seems to be a good balance between stability
        and responsiveness.        
        Args:
            recent_data: Recent OHLCV data
            window_size: Number of recent days to use
            
        Returns:
            current_state: Current market state
            confidence: Confidence in the prediction
            regime: Market regime label (Bull, Bear, Neutral)
        """
        if self.best_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Use the most recent window_size days
        recent_window = recent_data.tail(window_size)
        
        # Add features
        data_with_features = add_features(recent_window)
        
        # Prepare HMM features
        hmm_features = self.prepare_hmm_features(data_with_features)
        
        # Select features if enabled
        if self.feature_selection and self.feature_selector is not None:
            hmm_features_selected = self.feature_selector.transform(hmm_features)
        else:
            hmm_features_selected = hmm_features.values
        
        # Normalize features
        normalized_features,scaler = normalize_features(hmm_features_selected, fit=False)
        
        # Predict states
        states = self.best_model.predict(normalized_features)
        
        # Get the most recent state
        current_state = states[-1]
        
        # Get state probabilities for confidence
        state_probs = self.best_model.predict_proba(normalized_features)
        confidence = state_probs[-1, current_state]
        
        # Map to regime
        if current_state == self.bull_state:
            regime = 'Bull'
        elif current_state == self.bear_state:
            regime = 'Bear'
        else:
            regime = 'Neutral'
        
        return current_state, confidence, regime
    
    def walk_forward_validation(self, data, initial_train_size=0.5, step_size=30, optimize_once=True):
        """
        Perform walk-forward validation to test model robustness.
        This was a game-changer for evaluating the model properly.
        The step size is a trade-off - smaller steps give more data points
        but take longer to run.
        
        Args:
            data: DataFrame with OHLCV data
            initial_train_size: Initial training set size as fraction of data
            step_size: Number of days to step forward in each iteration
            optimize_once: Whether to optimize parameters only on initial training set
            
        Returns:
            results: DataFrame with validation results
        """
        print("Performing walk-forward validation...")
        
        # Initialize results
        results = []
        
        # Calculate initial train size
        n_samples = len(data)
        initial_train_end = int(n_samples * initial_train_size)
        
        # Initial training with optimization
        initial_train_data = data.iloc[:initial_train_end]
        self.fit(initial_train_data, optimize=True)
        
        # Store the optimized parameters
        optimized_params = self.best_params.copy() if self.best_params else None
        
        # Walk forward
        for train_end in range(initial_train_end, n_samples - step_size, step_size):
            try:
                # Split data
                train_data = data.iloc[:train_end]
                test_data = data.iloc[train_end:train_end + step_size]
                
                # Fit model on training data (with or without optimization)
                if optimize_once:
                    # Use the parameters optimized on initial training set
                    self.best_params = optimized_params
                    self.fit(train_data, optimize=False)
                else:
                    # Re-optimize for each window
                    self.fit(train_data, optimize=True)
                
                # Predict on test data
                predictions = self.predict(test_data)
                
                # Calculate performance
                predictions['Forward_Return_5d'] = predictions['Close'].pct_change(5).shift(-5)
                
                # Calculate strategy returns
                predictions['Strategy_Return'] = predictions['Signal'].shift(1) * predictions['Log_Return']
                
                # Store results
                results.append({
                    'train_end': train_end,
                    'test_start': train_end,
                    'test_end': train_end + step_size,
                    'n_states': self.best_params['n_components'],
                    'covariance_type': self.best_params['covariance_type'],
                    'bull_state': self.bull_state,
                    'bear_state': self.bear_state,
                    'avg_return': predictions['Strategy_Return'].mean(),
                    'sharpe': (predictions['Strategy_Return'].mean() / predictions['Strategy_Return'].std() * np.sqrt(252)) 
                            if predictions['Strategy_Return'].std() > 0 else 0,
                    'accuracy': np.mean((predictions['Signal'].shift(1) > 0) == (predictions['Log_Return'] > 0))
                })
            except Exception as e:
                print(f"Error in walk-forward window ending at {train_end}: {e}")
                # Continue with the next window instead of stopping
                continue
        
        # Check if we have any results
        if not results:
            print("No valid results from walk-forward validation")
            return pd.DataFrame()
                
        return pd.DataFrame(results)

    
    def plot_states(self, data_with_states, title=None):
        """
        Plot the price series with colored market regimes.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Ensure index is datetime
            if not isinstance(data_with_states.index, pd.DatetimeIndex):
                print("Warning: Converting index to datetime")
                data_with_states.index = pd.to_datetime(data_with_states.index)
            
            # Plot the closing price
            ax.plot(data_with_states.index, data_with_states['Close'], color='black', linewidth=1.5, label='Close Price')
            
            # Color the background based on market regime
            unique_states = data_with_states['Predicted_State'].dropna().unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_states)))
            
            for i, state in enumerate(unique_states):
                mask = data_with_states['Predicted_State'] == state
                
                # Find contiguous regions
                regions = []
                start_idx = None
                
                for idx, val in enumerate(mask):
                    if val and start_idx is None:
                        start_idx = idx
                    elif not val and start_idx is not None:
                        regions.append((start_idx, idx - 1))
                        start_idx = None
                
                # Add the last region if it extends to the end
                if start_idx is not None:
                    regions.append((start_idx, len(mask) - 1))
                
                # Shade each region
                for start, end in regions:
                    ax.axvspan(data_with_states.index[start], data_with_states.index[end], 
                            alpha=0.3, color=colors[i], label=f'State {state}' if start == regions[0][0] else None)
            
            # Add labels for bull and bear states
            handles, labels = ax.get_legend_handles_labels()
            for i, label in enumerate(labels):
                if label.startswith('State'):
                    state = int(float(label.split(' ')[1]))  # Handle potential float states
                    if state == self.bull_state:
                        labels[i] = f'Bull Market (State {state})'
                    elif state == self.bear_state:
                        labels[i] = f'Bear Market (State {state})'
                    elif state == self.neutral_state:
                        labels[i] = f'Neutral Market (State {state})'
            
            # Format the plot
            ax.set_title(title or 'Market Regimes Detected by HMM')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend(handles, labels)
            ax.grid(True, alpha=0.3)
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib is required for plotting.")
        except Exception as e:
            print(f"Error plotting states: {e}")

    
    def plot_state_distributions(self, data_with_states):
        """
        Plot the distribution of returns in each state.
        
        Args:
            data_with_states: DataFrame with price data and predicted states
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Calculate returns
            data = data_with_states.copy()
            data['Return'] = data['Close'].pct_change()
            
            # Create figure
            fig, axes = plt.subplots(1, len(set(data['Predicted_State'])), figsize=(15, 5))
            
            # Ensure axes is always a list
            if len(set(data['Predicted_State'])) == 1:
                axes = [axes]
            
            # Plot distribution for each state
            for i, state in enumerate(sorted(set(data['Predicted_State']))):
                state_data = data[data['Predicted_State'] == state]['Return'].dropna()
                
                if len(state_data) > 0:
                    # Determine state type
                    state_label = 'Bull' if state == self.bull_state else 'Bear' if state == self.bear_state else 'Neutral'
                    color = 'green' if state == self.bull_state else 'red' if state == self.bear_state else 'blue'
                    
                    # Plot distribution
                    sns.histplot(state_data, kde=True, ax=axes[i], color=color)
                    
                    # Add statistics
                    mean = state_data.mean()
                    std = state_data.std()
                    sharpe = mean / std * np.sqrt(252) if std > 0 else 0
                    
                    axes[i].set_title(f'{state_label} Market (State {state})\nMean: {mean:.4f}, Sharpe: {sharpe:.2f}')
                    axes[i].axvline(0, color='black', linestyle='--')
                    axes[i].axvline(mean, color='red', linestyle='-')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib and seaborn are required for plotting.")
    
    def compare_to_benchmark(self, data, benchmark_data=None):
        """
        Compare HMM strategy to buy-and-hold and other benchmarks.
        
        Args:
            data: DataFrame with OHLCV data
            benchmark_data: Optional DataFrame with benchmark data
            
        Returns:
            comparison: DataFrame with performance comparison
        """
        # Predict states and generate signals
        predictions = self.predict(data)
        
        # Calculate strategy returns
        predictions['Strategy_Return'] = predictions['Signal'].shift(1) * predictions['Log_Return']
        
        # Calculate buy-and-hold returns
        predictions['BuyHold_Return'] = predictions['Log_Return']
        
        # Calculate cumulative returns
        predictions['Strategy_Cumulative'] = np.exp(predictions['Strategy_Return'].cumsum())
        predictions['BuyHold_Cumulative'] = np.exp(predictions['BuyHold_Return'].cumsum())
        
        # Calculate performance metrics
        strategy_sharpe = (predictions['Strategy_Return'].mean() / predictions['Strategy_Return'].std() * np.sqrt(252)) if predictions['Strategy_Return'].std() > 0 else 0
        buyhold_sharpe = (predictions['BuyHold_Return'].mean() / predictions['BuyHold_Return'].std() * np.sqrt(252)) if predictions['BuyHold_Return'].std() > 0 else 0
        
        strategy_drawdown = self._calculate_max_drawdown(predictions['Strategy_Cumulative'])
        buyhold_drawdown = self._calculate_max_drawdown(predictions['BuyHold_Cumulative'])
        
        # Prepare comparison DataFrame
        comparison = pd.DataFrame({
            'Metric': ['Total Return', 'Annualized Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'],
            'HMM Strategy': [
                f"{(predictions['Strategy_Cumulative'].iloc[-1] - 1) * 100:.2f}%",
                f"{(predictions['Strategy_Cumulative'].iloc[-1] ** (252/len(predictions)) - 1) * 100:.2f}%",
                f"{strategy_sharpe:.2f}",
                f"{strategy_drawdown * 100:.2f}%",
                 f"{(predictions['Strategy_Return'] > 0).mean() * 100:.2f}%"
            ],
            'Buy & Hold': [
                f"{(predictions['BuyHold_Cumulative'].iloc[-1] - 1) * 100:.2f}%",
                f"{(predictions['BuyHold_Cumulative'].iloc[-1] ** (252/len(predictions)) - 1) * 100:.2f}%",
                f"{buyhold_sharpe:.2f}",
                f"{buyhold_drawdown * 100:.2f}%",
                f"{(predictions['BuyHold_Return'] > 0).mean() * 100:.2f}%"
            ]
        })
        
        # Add benchmark if provided
        if benchmark_data is not None:
            # Align benchmark data with predictions
            aligned_benchmark = benchmark_data.reindex(predictions.index, method='ffill')
            
            # Calculate benchmark returns
            aligned_benchmark['Benchmark_Return'] = aligned_benchmark['Close'].pct_change()
            aligned_benchmark['Benchmark_Cumulative'] = np.exp(aligned_benchmark['Benchmark_Return'].cumsum())
            
            # Calculate benchmark metrics
            benchmark_sharpe = (aligned_benchmark['Benchmark_Return'].mean() / aligned_benchmark['Benchmark_Return'].std() * np.sqrt(252)) if aligned_benchmark['Benchmark_Return'].std() > 0 else 0
            benchmark_drawdown = self._calculate_max_drawdown(aligned_benchmark['Benchmark_Cumulative'])
            
            # Add benchmark to comparison
            comparison['Benchmark'] = [
                f"{(aligned_benchmark['Benchmark_Cumulative'].iloc[-1] - 1) * 100:.2f}%",
                f"{(aligned_benchmark['Benchmark_Cumulative'].iloc[-1] ** (252/len(aligned_benchmark)) - 1) * 100:.2f}%",
                f"{benchmark_sharpe:.2f}",
                f"{benchmark_drawdown * 100:.2f}%",
                f"{(aligned_benchmark['Benchmark_Return'] > 0).mean() * 100:.2f}%"
            ]
        
        return comparison
    
    def _calculate_max_drawdown(self, cumulative_returns):
        """
        Calculate maximum drawdown from peak to trough.
        Drawdown is one of the most important metrics to mesure risk and downside volatility 
        
        Args:
            cumulative_returns: Series of cumulative returns
            
        Returns:
            max_drawdown: Maximum drawdown as a decimal
        """
        # Calculate running maximum
        running_max = cumulative_returns.cummax()
        
        # Calculate drawdown
        drawdown = (running_max - cumulative_returns) / running_max
        
        # Return maximum drawdown
        return drawdown.max()
    
    def plot_comparison(self, data, benchmark_data=None):
        """
        Plot cumulative returns comparison between strategy, buy-hold, and benchmark.
        
        Args:
            data: DataFrame with OHLCV data
            benchmark_data: Optional DataFrame with benchmark data
        """
        try:
            import matplotlib.pyplot as plt
            
            # Predict states and generate signals
            predictions = self.predict(data)
            
            # Calculate strategy returns
            predictions['Strategy_Return'] = predictions['Signal'].shift(1) * predictions['Log_Return']
            
            # Calculate buy-and-hold returns
            predictions['BuyHold_Return'] = predictions['Log_Return']
            
            # Calculate cumulative returns
            predictions['Strategy_Cumulative'] = np.exp(predictions['Strategy_Return'].cumsum())
            predictions['BuyHold_Cumulative'] = np.exp(predictions['BuyHold_Return'].cumsum())
            
            # Create plot
            plt.figure(figsize=(15, 8))
            
            # Plot strategy and buy-hold
            plt.plot(predictions.index, predictions['Strategy_Cumulative'], label='HMM Strategy', linewidth=2)
            plt.plot(predictions.index, predictions['BuyHold_Cumulative'], label='Buy & Hold', linewidth=2, alpha=0.7)
            
            # Add benchmark if provided
            if benchmark_data is not None:
                # Align benchmark data with predictions
                aligned_benchmark = benchmark_data.reindex(predictions.index, method='ffill')
                
                # Calculate benchmark returns
                aligned_benchmark['Benchmark_Return'] = aligned_benchmark['Close'].pct_change()
                aligned_benchmark['Benchmark_Cumulative'] = np.exp(aligned_benchmark['Benchmark_Return'].cumsum())
                
                # Plot benchmark
                plt.plot(predictions.index, aligned_benchmark['Benchmark_Cumulative'], label='Benchmark', linewidth=2, alpha=0.7)
            
            # Format plot
            plt.title('Cumulative Returns Comparison')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib is required for plotting.")
    
    def generate_signals_for_simulator(self, data, symbol='SYMBOL'):
        """
        Generate signals in the format expected by the TradeSimulator.
        This bridges the gap between the HMM model and the trade simulator.
        The simulator needs signals in a specific format with confidence values.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: The ticker symbol to use for signals
            
        Returns:
            signals: Dictionary of signals for each timestamp and symbol
        """
        # Predict states
        predictions = self.predict(data)
        
        # Generate signals dictionary
        signals = {}
        
        # Check if we have valid predictions
        if 'Predicted_State' not in predictions.columns or predictions['Predicted_State'].isna().all():
            print("Warning: No valid predictions for signal generation")
            # Return empty signals
            return signals
        
        for timestamp, row in predictions.iterrows():
            if pd.isna(row['Predicted_State']):
                continue
                
            state = row['Predicted_State']
            
            # Get signal and confidence
            if 'Signal' in row and 'Signal_Confidence' in row:
                signal = row['Signal']
                confidence = row['Signal_Confidence']
            else:
                # Fallback to basic signal generation
                if state == self.bull_state:
                    signal = 1
                    confidence = 0.8
                elif state == self.bear_state:
                    signal = -1
                    confidence = 0.8
                else:
                    signal = 0
                    confidence = 0.5
            
            # Add to signals dictionary
            signals[timestamp] = {
                symbol: {
                    'signal': signal,
                    'confidence': confidence
                }
            }
        
        # Print signal summary
        bull_signals = sum(1 for ts, sig_dict in signals.items() 
                        if symbol in sig_dict and sig_dict[symbol]['signal'] == 1)
        bear_signals = sum(1 for ts, sig_dict in signals.items() 
                        if symbol in sig_dict and sig_dict[symbol]['signal'] == -1)
        neutral_signals = sum(1 for ts, sig_dict in signals.items() 
                            if symbol in sig_dict and sig_dict[symbol]['signal'] == 0)
        
        print(f"\nSignal Summary:")
        print(f"Bull signals: {bull_signals}")
        print(f"Bear signals: {bear_signals}")
        print(f"Neutral signals: {neutral_signals}")
        print(f"Total signals: {len(signals)}")
        
        return signals

    def run_permutation_tests(self, test_data, n_permutations=1000):
        """
        Run permutation tests to assess the statistical significance of the model's predictions.
        This was a game-changer for me , so many strategies look good until
        you test if they're actually better than random Permutation tests
        help separate skill from luck.
        
        1000 permutations gives a good balance between accuracy and runtime.
        Parameters:
        -----------
        test_data : pd.DataFrame
            Test data to evaluate
        n_permutations : int
            Number of permutations to run
            
        Returns:
        --------
        dict
            Dictionary containing test results for different metrics
        """
        from utils import run_permutation_test, plot_permutation_test_results
        
        # Generate predictions
        predictions = self.predict(test_data)
        
        # Run permutation tests for different metrics
        results = {}
        
        print("Running permutation test for Sharpe ratio...")
        sharpe_results = run_permutation_test(
            predictions['Log_Return'], 
            predictions['Signal'], 
            n_permutations=n_permutations, 
            metric='sharpe'
        )
        results['sharpe'] = sharpe_results
        plot_permutation_test_results(sharpe_results, "Sharpe Ratio")
        
        print("Running permutation test for total return...")
        return_results = run_permutation_test(
            predictions['Log_Return'], 
            predictions['Signal'], 
            n_permutations=n_permutations, 
            metric='total_return'
        )
        results['total_return'] = return_results
        plot_permutation_test_results(return_results, "Total Return")
        
        print("Running permutation test for win rate...")
        win_rate_results = run_permutation_test(
            predictions['Log_Return'], 
            predictions['Signal'], 
            n_permutations=n_permutations, 
            metric='win_rate'
        )
        results['win_rate'] = win_rate_results
        plot_permutation_test_results(win_rate_results, "Win Rate")
        
        return results


