import numpy as np
import pandas as pd
from datetime import datetime

class TradeSimulator:
    """
    Centralized trade simulation system that executes trades based on signals,
    applies risk management, and tracks performance metrics.
    """
    
    def __init__(self, 
                 initial_capital=100000,
                 commission=0.001,           # 0.1% commission per trade
                 slippage=0.0005,            # 0.05% slippage
                 stop_loss_pct=0.02,         # 2% stop loss
                 trailing_stop_pct=0.03,     # 3% trailing stop
                 take_profit_pct=0.05,       # 5% take profit
                 max_position_size=0.2,      # Max 20% of portfolio per position
                 risk_per_trade=0.01):       # Risk 1% of portfolio per trade
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.take_profit_pct = take_profit_pct
        self.max_position_size = max_position_size
        self.risk_per_trade = risk_per_trade
        
        # Portfolio tracking
        self.positions = {}          # Current positions {symbol: quantity}
        self.entry_prices = {}       # Entry prices for positions {symbol: price}
        self.trailing_prices = {}    # Trailing prices for trailing stops {symbol: price}
        
        # Performance tracking
        self.portfolio_history = []  # List of portfolio values over time
        self.trade_history = []      # List of executed trades
        self.daily_returns = []      # Daily returns
        
    def calculate_position_size(self, symbol, price, signal, confidence):
        """
        Calculate the appropriate position size based on risk parameters.
        
        Args:
            symbol: The trading symbol
            price: Current price
            signal: Trading signal (-1, 0, 1)
            confidence: Signal confidence (0-1)
            
        Returns:
            quantity: Number of shares/contracts to trade
        """
        if signal == 0:
            return 0
            
        # Calculate dollar risk amount (% of portfolio at risk)
        risk_amount = self.current_capital * self.risk_per_trade
        
        # Calculate stop loss distance in price terms
        stop_distance = price * self.stop_loss_pct
        
        # Calculate position size based on risk
        position_size = risk_amount / stop_distance
        
        # Scale by confidence
        position_size *= confidence
        
        # Ensure we don't exceed maximum position size
        max_position = self.current_capital * self.max_position_size / price
        position_size = min(position_size, max_position)
        
        # Convert to integer number of shares
        quantity = int(position_size)
        
        # Adjust direction based on signal
        quantity *= signal
        
        return quantity
    
    def execute_trade(self, symbol, price, quantity, timestamp, trade_type="MARKET"):
        """
        Execute a trade and update portfolio.
        
        Args:
            symbol: Trading symbol
            price: Execution price
            quantity: Number of shares (negative for sell)
            timestamp: Trade timestamp
            trade_type: Type of trade (MARKET, LIMIT, etc.)
            
        Returns:
            executed_price: The price after slippage
            commission_paid: Commission amount paid
        """
        # Apply slippage (worse for market orders)
        slippage_factor = self.slippage if trade_type == "MARKET" else self.slippage / 2
        executed_price = price * (1 + slippage_factor * np.sign(quantity))
        
        # Calculate trade value and commission
        trade_value = executed_price * abs(quantity)
        commission_paid = trade_value * self.commission
        
        # Update capital
        self.current_capital -= trade_value * np.sign(quantity) - commission_paid
        
        # Update positions
        current_position = self.positions.get(symbol, 0)
        new_position = current_position + quantity
        
        # If position direction changed or closed, reset entry price
        if current_position * new_position <= 0:  # Direction changed or position closed
            self.entry_prices.pop(symbol, None)
            self.trailing_prices.pop(symbol, None)
        
        # If opening or adding to position, update entry price (weighted average)
        if new_position != 0:
            if symbol not in self.entry_prices:
                self.entry_prices[symbol] = executed_price
                self.trailing_prices[symbol] = executed_price
            else:
                # Weighted average for adding to position
                if np.sign(current_position) == np.sign(new_position):
                    self.entry_prices[symbol] = (
                        (abs(current_position) * self.entry_prices[symbol] + 
                         abs(quantity) * executed_price) / 
                        abs(new_position)
                    )
        
        # Update position
        if new_position == 0:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = new_position
        
        # Record the trade
        self.trade_history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'quantity': quantity,
            'price': executed_price,
            'commission': commission_paid,
            'trade_value': trade_value,
            'portfolio_value': self.current_capital + self.calculate_positions_value(price_dict={symbol: price})
        })
        
        return executed_price, commission_paid
    
    def calculate_positions_value(self, price_dict):
        """
        Calculate the current value of all positions.
        
        Args:
            price_dict: Dictionary of current prices {symbol: price}
            
        Returns:
            total_value: Total value of all positions
        """
        total_value = 0
        for symbol, quantity in self.positions.items():
            if symbol in price_dict:
                total_value += quantity * price_dict[symbol]
        return total_value
    
    def apply_risk_management(self, price_dict, timestamp):
        """
        Apply risk management rules to current positions.
        
        Args:
            price_dict: Dictionary of current prices {symbol: price}
            timestamp: Current timestamp
            
        Returns:
            orders: List of orders to execute for risk management
        """
        orders = []
        
        for symbol, quantity in list(self.positions.items()):
            if symbol not in price_dict:
                continue
                
            current_price = price_dict[symbol]
            entry_price = self.entry_prices.get(symbol)
            trailing_price = self.trailing_prices.get(symbol)
            
            if entry_price is None or trailing_price is None:
                continue
            
            # Update trailing price if price moved favorably
            if quantity > 0 and current_price > trailing_price:  # Long position
                self.trailing_prices[symbol] = current_price
            elif quantity < 0 and current_price < trailing_price:  # Short position
                self.trailing_prices[symbol] = current_price
            
            # Check stop loss
            stop_triggered = False
            
            # Regular stop loss
            if quantity > 0:  # Long position
                stop_price = entry_price * (1 - self.stop_loss_pct)
                if current_price <= stop_price:
                    stop_triggered = True
            else:  # Short position
                stop_price = entry_price * (1 + self.stop_loss_pct)
                if current_price >= stop_price:
                    stop_triggered = True
            
            # Trailing stop
            if quantity > 0:  # Long position
                trailing_stop = trailing_price * (1 - self.trailing_stop_pct)
                if current_price <= trailing_stop:
                    stop_triggered = True
            else:  # Short position
                trailing_stop = trailing_price * (1 + self.trailing_stop_pct)
                if current_price >= trailing_stop:
                    stop_triggered = True
            
            # Take profit
            take_profit_triggered = False
            if quantity > 0:  # Long position
                take_profit_price = entry_price * (1 + self.take_profit_pct)
                if current_price >= take_profit_price:
                    take_profit_triggered = True
            else:  # Short position
                take_profit_price = entry_price * (1 - self.take_profit_pct)
                if current_price <= take_profit_price:
                    take_profit_triggered = True
            
            # Create exit order if stop loss or take profit triggered
            if stop_triggered or take_profit_triggered:
                exit_reason = "STOP_LOSS" if stop_triggered else "TAKE_PROFIT"
                orders.append({
                    'symbol': symbol,
                    'quantity': -quantity,  # Close position
                    'price': current_price,
                    'timestamp': timestamp,
                    'reason': exit_reason
                })
        
        return orders
    
    def update_portfolio(self, price_dict, timestamp):
        """
        Update portfolio value and apply risk management.
        
        Args:
            price_dict: Dictionary of current prices {symbol: price}
            timestamp: Current timestamp
            
        Returns:
            portfolio_value: Updated portfolio value
        """
        # Apply risk management
        risk_orders = self.apply_risk_management(price_dict, timestamp)
        
        # Execute risk management orders
        for order in risk_orders:
            self.execute_trade(
                symbol=order['symbol'],
                price=order['price'],
                quantity=order['quantity'],
                timestamp=timestamp,
                trade_type="MARKET"
            )
        
        # Calculate current portfolio value
        positions_value = self.calculate_positions_value(price_dict)
        portfolio_value = self.current_capital + positions_value
        
        # Record portfolio history
        self.portfolio_history.append({
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'cash': self.current_capital,
            'positions_value': positions_value
        })
        
        # Calculate daily return if we have previous values
        if len(self.portfolio_history) > 1:
            prev_value = self.portfolio_history[-2]['portfolio_value']
            daily_return = (portfolio_value / prev_value) - 1
            self.daily_returns.append(daily_return)
        
        return portfolio_value
    
    def process_signals(self, signals, prices, timestamp):
        """
        Process trading signals and execute trades.
        
        Args:
            signals: Dictionary of trading signals {symbol: {'signal': -1/0/1, 'confidence': 0-1}}
            prices: Dictionary of current prices {symbol: price}
            timestamp: Current timestamp
            
        Returns:
            executed_orders: List of executed orders
        """
        executed_orders = []
        
        # Process each signal
        for symbol, signal_data in signals.items():
            if symbol not in prices:
                continue
                
            signal = signal_data.get('signal', 0)
            confidence = signal_data.get('confidence', 1.0)
            current_price = prices[symbol]
            
            # Calculate target position
            target_quantity = self.calculate_position_size(symbol, current_price, signal, confidence)
            
            # Get current position
            current_quantity = self.positions.get(symbol, 0)
            
            # Calculate quantity to trade
            trade_quantity = target_quantity - current_quantity
            
            # Execute trade if needed
            if trade_quantity != 0:
                executed_price, commission = self.execute_trade(
                    symbol=symbol,
                    price=current_price,
                    quantity=trade_quantity,
                    timestamp=timestamp
                )
                
                executed_orders.append({
                    'symbol': symbol,
                    'quantity': trade_quantity,
                    'price': executed_price,
                    'commission': commission,
                    'timestamp': timestamp
                })
        
        # Update portfolio after processing all signals
        self.update_portfolio(prices, timestamp)
        
        return executed_orders
    
    def run_simulation(self, data, signals):
        """
        Run a full backtest simulation.
        
        Args:
            data: Dictionary of price data {symbol: DataFrame}
            signals: Dictionary of signals {timestamp: {symbol: {'signal': -1/0/1, 'confidence': 0-1}}}
            
        Returns:
            performance_metrics: Dictionary of performance metrics
        """
        # Reset simulator state
        self.current_capital = self.initial_capital
        self.positions = {}
        self.entry_prices = {}
        self.trailing_prices = {}
        self.portfolio_history = []
        self.trade_history = []
        self.daily_returns = []
        
        # Get sorted timestamps from signals
        timestamps = sorted(signals.keys())
        
        # Run simulation for each timestamp
        for timestamp in timestamps:
            # Get current prices
            prices = {}
            for symbol in data:
                if timestamp in data[symbol].index:
                    prices[symbol] = data[symbol].loc[timestamp, 'Close']
            
            # Process signals for this timestamp
            self.process_signals(signals[timestamp], prices, timestamp)
        
        # Calculate performance metrics
        return self.calculate_performance()
    
    def calculate_performance(self):
        """
        Calculate performance metrics for the simulation.
        
        Returns:
            metrics: Dictionary of performance metrics
        """
        if not self.portfolio_history:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_trade': 0
            }
        
        # Extract portfolio values
        portfolio_values = [entry['portfolio_value'] for entry in self.portfolio_history]
        
        # Calculate total return
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        daily_returns = np.array(self.daily_returns)
        sharpe_ratio = 0
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        
        # Calculate max drawdown
        max_drawdown = self.calculate_max_drawdown(portfolio_values)
        
        # Calculate trade statistics
        win_rate, profit_factor, avg_trade = self.calculate_trade_stats()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'portfolio_history': self.portfolio_history,
            'trade_history': self.trade_history
        }
    
    def calculate_max_drawdown(self, portfolio_values):
        """Calculate the maximum drawdown from peak to trough."""
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    
    def calculate_trade_stats(self):
        """Calculate trade-level statistics."""
        if not self.trade_history:
            return 0, 0, 0
            
        # Group trades by symbol and calculate P&L
        trades_by_symbol = {}
        for trade in self.trade_history:
            symbol = trade['symbol']
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = []
            trades_by_symbol[symbol].append(trade)
        
        # Calculate P&L for each completed round trip
        completed_trades = []
        
        for symbol, trades in trades_by_symbol.items():
            position = 0
            cost_basis = 0
            
            for trade in trades:
                quantity = trade['quantity']
                price = trade['price']
                commission = trade['commission']
                
                # Opening or adding to position
                if position == 0 or (position > 0 and quantity > 0) or (position < 0 and quantity < 0):
                    # Update cost basis (weighted average)
                    cost_basis = ((abs(position) * cost_basis) + (abs(quantity) * price)) / (abs(position) + abs(quantity))
                    position += quantity
                
                # Reducing or closing position
                elif (position > 0 and quantity < 0) or (position < 0 and quantity > 0):
                    # Calculate P&L for the closed portion
                    closed_quantity = min(abs(position), abs(quantity))
                    if position > 0:  # Long position
                        trade_pnl = (price - cost_basis) * closed_quantity - commission
                    else:  # Short position
                        trade_pnl = (cost_basis - price) * closed_quantity - commission
                    
                    completed_trades.append(trade_pnl)
                    
                    # Update position
                    position += quantity
                    
                    # If position direction changed, reset cost basis
                    if position * (position - quantity) <= 0:
                        cost_basis = price if position != 0 else 0
        
        # Calculate statistics
        if not completed_trades:
            return 0, 0, 0
            
        wins = sum(1 for pnl in completed_trades if pnl > 0)
        losses = sum(1 for pnl in completed_trades if pnl <= 0)
        
        win_rate = wins / len(completed_trades) if len(completed_trades) > 0 else 0
        
        gross_profit = sum(pnl for pnl in completed_trades if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in completed_trades if pnl <= 0))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_trade = sum(completed_trades) / len(completed_trades) if len(completed_trades) > 0 else 0
        
        return win_rate, profit_factor, avg_trade
    
    def get_equity_curve(self):
        """
        Get the equity curve as a pandas DataFrame.
        
        Returns:
            DataFrame with portfolio value over time
        """
        if not self.portfolio_history:
            return pd.DataFrame()
            
        equity_curve = pd.DataFrame(self.portfolio_history)
        equity_curve.set_index('timestamp', inplace=True)
        return equity_curve
    
    def get_trade_log(self):
        """
        Get the trade log as a pandas DataFrame.
        
        Returns:
            DataFrame with all executed trades
        """
        if not self.trade_history:
            return pd.DataFrame()
            
        trade_log = pd.DataFrame(self.trade_history)
        return trade_log
    
    def plot_equity_curve(self):
        """Plot the equity curve."""
        try:
            import matplotlib.pyplot as plt
            
            equity_curve = self.get_equity_curve()
            if equity_curve.empty:
                print("No equity data to plot.")
                return
                
            plt.figure(figsize=(12, 6))
            plt.plot(equity_curve.index, equity_curve['portfolio_value'])
            plt.title('Portfolio Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value')
            plt.grid(True)
            plt.show()
        except ImportError:
            print("Matplotlib is required for plotting.")
    
    def plot_drawdown_curve(self):
        """Plot the drawdown curve."""
        try:
            import matplotlib.pyplot as plt
            
            equity_curve = self.get_equity_curve()
            if equity_curve.empty:
                print("No equity data to plot.")
                return
            
            # Calculate running maximum
            running_max = equity_curve['portfolio_value'].cummax()
            
            # Calculate drawdown percentage
            drawdown = (running_max - equity_curve['portfolio_value']) / running_max * 100
            
            plt.figure(figsize=(12, 6))
            plt.plot(equity_curve.index, drawdown)
            plt.fill_between(equity_curve.index, drawdown, 0, alpha=0.3)
            plt.title('Portfolio Drawdown')
            plt.xlabel('Date')
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
            plt.show()
        except ImportError:
            print("Matplotlib is required for plotting.")
