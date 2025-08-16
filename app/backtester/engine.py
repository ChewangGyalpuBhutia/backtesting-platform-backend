import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

class BacktestEngine:
    """Core backtesting engine with enhanced risk management"""
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001, 
                 max_position_size: float = 0.1, stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.15, max_risk_per_trade: float = 0.02):
        self.initial_capital = initial_capital
        self.commission = commission  # 0.1% commission per trade
        self.max_position_size = max_position_size  # Max 10% per position
        self.stop_loss_pct = stop_loss_pct  # 5% stop loss
        self.take_profit_pct = take_profit_pct  # 15% take profit
        self.max_risk_per_trade = max_risk_per_trade  # Max 2% risk per trade
        self.open_positions = []  # Track open positions for risk management
        
    def run_backtest(self, data: pd.DataFrame, strategy) -> Dict:
        """Run backtest with the given strategy"""
        
        # Generate signals using the strategy
        signal_data = strategy.generate_signals(data)
        
        # Initialize portfolio tracking
        portfolio = {
            'cash': self.initial_capital,
            'holdings': 0,
            'portfolio_value': [],
            'dates': [],
            'returns': []
        }
        
        trades = []
        equity_curve = []
        
        prev_portfolio_value = self.initial_capital
        
        for i, (date, row) in enumerate(signal_data.iterrows()):
            current_price = row['close']  # Use lowercase 'close'
            signal = row.get('signal', 0)  # Get signal from strategy
            
            # Execute trades based on signals
            if signal == 1 and portfolio['holdings'] == 0:  # Buy signal
                # Calculate how many shares we can buy
                available_cash = portfolio['cash'] * 0.95  # Keep 5% as buffer
                shares_to_buy = int(available_cash / (current_price * (1 + self.commission)))
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + self.commission)
                    portfolio['cash'] -= cost
                    portfolio['holdings'] += shares_to_buy
                    
                    trades.append({
                        'date': date,
                        'action': 'BUY',
                        'price': current_price,
                        'quantity': shares_to_buy,
                        'value': cost,
                        'commission': cost * self.commission
                    })
            
            elif signal == -1 and portfolio['holdings'] > 0:  # Sell signal
                # Sell all holdings
                shares_to_sell = portfolio['holdings']
                proceeds = shares_to_sell * current_price * (1 - self.commission)
                portfolio['cash'] += proceeds
                
                trades.append({
                    'date': date,
                    'action': 'SELL',
                    'price': current_price,
                    'quantity': shares_to_sell,
                    'value': proceeds,
                    'commission': proceeds * self.commission
                })
                
                portfolio['holdings'] = 0
            
            # Calculate current portfolio value
            current_portfolio_value = portfolio['cash'] + (portfolio['holdings'] * current_price)
            daily_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
            
            portfolio['portfolio_value'].append(current_portfolio_value)
            portfolio['dates'].append(date)
            portfolio['returns'].append(daily_return)
            
            equity_curve.append({
                'date': date,
                'portfolio_value': current_portfolio_value,
                'cash': portfolio['cash'],
                'holdings_value': portfolio['holdings'] * current_price,
                'returns': daily_return
            })
            
            prev_portfolio_value = current_portfolio_value
        
        # Create equity curve DataFrame
        equity_df = pd.DataFrame(equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # Calculate PnL for each trade
        for i in range(1, len(trades)):
            if trades[i]['action'] == 'SELL' and trades[i-1]['action'] == 'BUY':
                buy_price = trades[i-1]['price']
                sell_price = trades[i]['price']
                quantity = trades[i]['quantity']
                pnl = (sell_price - buy_price) * quantity
                trades[i]['pnl'] = pnl
        
        return {
            'equity_curve': equity_df,
            'trades': trades,
            'final_portfolio_value': current_portfolio_value,
            'initial_capital': self.initial_capital
        }
    
    def calculate_position_size(self, current_portfolio_value: float, price: float, 
                              volatility: float = None) -> int:
        """Calculate position size based on risk management rules"""
        # Method 1: Fixed percentage of portfolio
        max_dollar_amount = current_portfolio_value * self.max_position_size
        basic_shares = int(max_dollar_amount / price)
        
        # Method 2: Risk-based position sizing (if volatility provided)
        if volatility and volatility > 0:
            # Calculate position size based on maximum risk per trade
            max_risk_amount = current_portfolio_value * self.max_risk_per_trade
            # Assume stop loss will be hit at stop_loss_pct
            risk_per_share = price * self.stop_loss_pct
            risk_based_shares = int(max_risk_amount / risk_per_share) if risk_per_share > 0 else basic_shares
            
            # Use the smaller of the two methods
            shares = min(basic_shares, risk_based_shares)
        else:
            shares = basic_shares
        
        return max(0, shares)
    
    def check_stop_loss(self, entry_price: float, current_price: float, position_type: str) -> bool:
        """Check if stop loss should be triggered"""
        if position_type == 'long':
            return (entry_price - current_price) / entry_price >= self.stop_loss_pct
        else:  # short position
            return (current_price - entry_price) / entry_price >= self.stop_loss_pct
    
    def check_take_profit(self, entry_price: float, current_price: float, position_type: str) -> bool:
        """Check if take profit should be triggered"""
        if position_type == 'long':
            return (current_price - entry_price) / entry_price >= self.take_profit_pct
        else:  # short position
            return (entry_price - current_price) / entry_price >= self.take_profit_pct
    
    def calculate_portfolio_risk_metrics(self, equity_curve: pd.DataFrame) -> Dict:
        """Calculate real-time portfolio risk metrics"""
        returns = equity_curve['returns'].dropna()
        
        if len(returns) < 30:  # Need at least 30 days for meaningful metrics
            return {}
        
        # Rolling volatility (30-day)
        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
        
        # Rolling Sharpe ratio (30-day)
        rolling_sharpe = (returns.rolling(window=30).mean() * 252) / rolling_vol
        
        # Current drawdown
        portfolio_values = equity_curve['portfolio_value']
        peak = portfolio_values.expanding().max()
        current_drawdown = (portfolio_values.iloc[-1] - peak.iloc[-1]) / peak.iloc[-1] * 100
        
        return {
            'current_volatility': rolling_vol.iloc[-1] if not rolling_vol.empty else 0,
            'current_sharpe': rolling_sharpe.iloc[-1] if not rolling_sharpe.empty else 0,
            'current_drawdown': current_drawdown,
            'portfolio_value': portfolio_values.iloc[-1]
        }
