#this is the test script for the old HMM to try and run move to src
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    silhouette_score,
    accuracy_score,
    precision_recall_fscore_support
)
from scipy import stats
from hmm_model import (
    load_data,
    add_features,
    prepare_hmm_data,
    normalize_hmm_feautures,
    optimize_hmm_parameters,
    predict_current_state,
    rolling_window_hmm,
    plot_states
)

def split_data(data, train_size=0.8):
    split_idx = int(len(data) * train_size)
    train_data = data[:split_idx].copy()
    test_data = data[split_idx:].copy()
    return train_data, test_data

def backtest_strategy(data, bull_state):
    portfolio_value = [1000]  # Initial capital
    returns = []
    
    
    for i in range(1, len(data)):
        daily_return = data['Log_Return'].iloc[i]
        trend_up = data['Close'].iloc[i] > data['20d_MA'].iloc[i]
        if data['Predicted_State'].iloc[i-1] == bull_state and trend_up :
            portfolio_value.append(portfolio_value[-1] * (1 + daily_return))
            returns.append(daily_return)
            
        else:
            portfolio_value.append(portfolio_value[-1])  # Hold cash
            returns.append(0)
            
    
    returns = np.array(returns)
    # Calculate Sharpe Ratio
    excess_returns = returns - 0.02/252  # Risk-free rate of 2% annually
    sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
    
    return {
        'Portfolio_Value': portfolio_value,
        'Returns': returns,
        'Total_Return': (portfolio_value[-1] / portfolio_value[0] - 1) * 100,
        'Win_Rate': len([r for r in returns if r > 0]) / len(returns),
        'Max_Drawdown': calculate_max_drawdown(portfolio_value),
        'Sharpe_Ratio': sharpe  # Added this key
    }


def calculate_max_drawdown(portfolio_values):
    peak = portfolio_values[0]
    max_drawdown = 0
    
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)
    
    return max_drawdown * 100


class AdvancedModelTester:
    def __init__(self, tickers=None):
        self.tickers = tickers or ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'XOM']

    def comprehensive_model_evaluation(self):
        results = {}
        for ticker in self.tickers:
            print(f"Evaluating model for {ticker}")
            try:
                ticker_results = self.evaluate_single_stock(ticker)
                results[ticker] = ticker_results
            except Exception as e:
                print(f"Error evaluating {ticker}: {e}")
        
        self.aggregate_results(results)
        return results

    def evaluate_single_stock(self, ticker):
        # Load and preprocess data
        data = load_data(ticker)
        data = add_features(data)

        # Split data
        train_data, test_data = split_data(data, train_size=0.8)

        # Prepare and normalize training data
        hmm_train_data = prepare_hmm_data(train_data)
        normalized_train_data, scaler = normalize_hmm_feautures(hmm_train_data)

        # Train HMM model
        hmm_model = optimize_hmm_parameters(normalized_train_data)

        # Evaluate model
        test_results = self.advanced_model_evaluation(
            hmm_model, 
            test_data, 
            scaler
        )

        return test_results

    def advanced_model_evaluation(self, hmm_model, test_data, scaler):
        # Prepare test data
        hmm_test_data = prepare_hmm_data(test_data)
        normalized_test_data = scaler.transform(hmm_test_data)
        rolling_states = rolling_window_hmm(test_data, window_size=90, step_size=1)
        current_state = predict_current_state(
            hmm_model, 
            test_data.tail(30),  # Last 30 days
            scaler
        )

        # Predict states
        predicted_states = hmm_model.predict(normalized_test_data)
        test_data['Predicted_State'] = predicted_states

        # Calculate state characteristics
        state_validator = HMMStateValidator(hmm_model, test_data)
        state_analysis = state_validator.analyze_states()

        # Performance Metrics
        metrics = {
            'silhouette_score': silhouette_score(
                normalized_test_data, 
                predicted_states
            ),
            'state_distribution': state_analysis,
            'state_transitions': state_validator.analyze_transitions(),
            'rolling_states': rolling_states,
            'current_state': current_state
        }

        # Backtest strategy
        backtest_results = backtest_strategy(
            test_data, 
            state_analysis['bull_state']
        )

        return {
            'metrics': metrics,
            'backtest_results': backtest_results,
            'state_analysis': state_analysis
        }

    def aggregate_results(self, results):
        aggregated_metrics = {
            'silhouette_scores': [],
            'returns': [],
            'sharpe_ratios': []
        }

        for ticker, result in results.items():
            aggregated_metrics['silhouette_scores'].append(
                result['metrics']['silhouette_score']
            )
            aggregated_metrics['returns'].append(
                result['backtest_results']['Total_Return']
            )
            aggregated_metrics['sharpe_ratios'].append(
                result['backtest_results']['Sharpe_Ratio']
            )

        print("\nAggregated Model Performance:")
        print(f"Average Silhouette Score: {np.mean(aggregated_metrics['silhouette_scores']):.3f}")
        print(f"Average Return: {np.mean(aggregated_metrics['returns']):.2f}%")
        print(f"Average Sharpe Ratio: {np.mean(aggregated_metrics['sharpe_ratios']):.2f}")

class HMMStateValidator:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def analyze_states(self):
        states = self.data['Predicted_State']
        state_analysis = {}

        for state in range(self.model.n_components):
            mask = states == state
            state_returns = self.data.loc[mask, 'Log_Return']
            
            state_analysis[f'State_{state}'] = {
                'mean_return': state_returns.mean(),
                'volatility': state_returns.std(),
                'frequency': np.mean(mask),
                'duration': self.calculate_duration(states, state)
            }

        # Identify bull/bear states
        returns_by_state = {
            state: analysis['mean_return'] 
            for state, analysis in state_analysis.items()
        }
        state_analysis['bull_state'] = max(returns_by_state, key=returns_by_state.get)

        return state_analysis

    def analyze_transitions(self):
        states = self.data['Predicted_State']
        transition_matrix = pd.crosstab(
            pd.Series(states[:-1]), 
            pd.Series(states[1:]), 
            normalize='index'
        )
        return transition_matrix

    def calculate_duration(self, states, target_state):
        durations = []
        current_duration = 0
        
        for state in states:
            if state == target_state:
                current_duration += 1
            elif current_duration > 0:
                durations.append(current_duration)
                current_duration = 0
        
        return np.mean(durations) if durations else 0

def main():
    tester = AdvancedModelTester()
    results = tester.comprehensive_model_evaluation()

if __name__ == "__main__":
    main()