import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from trade_simulator import TradeSimulator

def generate_test_data(symbol='TEST', days=252, start_price=100, volatility=0.015):
    """Generate synthetic price data for testing."""
    dates = [datetime.now() - timedelta(days=days-i) for i in range(days)]
    
    # Generate random price movements
    returns = np.random.normal(0.0005, volatility, days)
    prices = [start_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'Open': prices[:-1],
        'High': [p * (1 + np.random.uniform(0, 0.005)) for p in prices[:-1]],
        'Low': [p * (1 - np.random.uniform(0, 0.005)) for p in prices[:-1]],
        'Close': prices[1:],
        'Volume': [int(np.random.uniform(100000, 1000000)) for _ in range(days)]
    }, index=dates)
    
    return data

def generate_test_signals(data, symbol='TEST'):
    """Generate test trading signals."""
    signals = {}
    
    # Simple moving average crossover strategy
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    
    # Generate signals
    for i in range(50, len(data)):
        date = data.index[i]
        
        # Crossover logic
        if data['SMA20'].iloc[i-1] < data['SMA50'].iloc[i-1] and data['SMA20'].iloc[i] > data['SMA50'].iloc[i]:
            signal = 1  # Buy signal
            confidence = 0.8
        elif data['SMA20'].iloc[i-1] > data['SMA50'].iloc[i-1] and data['SMA20'].iloc[i] < data['SMA50'].iloc[i]:
            signal = -1  # Sell signal
            confidence = 0.8
        else:
            signal = 0  # No signal
            confidence = 0.0
        
        signals[date] = {symbol: {'signal': signal, 'confidence': confidence}}
    
    return signals

def test_simulator():
    # Generate test data
    symbol = 'AAPL'
    test_data = generate_test_data(symbol=symbol, days=252, start_price=150)
    
    # Generate test signals
    test_signals = generate_test_signals(test_data, symbol=symbol)
    
    # Initialize simulator
    simulator = TradeSimulator(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
        stop_loss_pct=0.02,
        trailing_stop_pct=0.03,
        take_profit_pct=0.05,
        max_position_size=0.2,
        risk_per_trade=0.01
    )
    
    # Run simulation
    data = {symbol: test_data}
    performance = simulator.run_simulation(data, test_signals)
    
    # Print performance metrics
    print("=== Performance Metrics ===")
    print(f"Total Return: {performance['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {performance['max_drawdown']*100:.2f}%")
    print(f"Win Rate: {performance['win_rate']*100:.2f}%")
    print(f"Profit Factor: {performance['profit_factor']:.2f}")
    print(f"Average Trade: ${performance['avg_trade']:.2f}")
    
    # Plot equity curve
    simulator.plot_equity_curve()
    
    # Plot drawdown curve
    simulator.plot_drawdown_curve()
    
    return simulator

if __name__ == "__main__":
    simulator = test_simulator()
