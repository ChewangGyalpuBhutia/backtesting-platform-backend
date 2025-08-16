import pandas as pd
import numpy as np
from typing import Dict

class PerformanceMetrics:
    """Calculate comprehensive performance metrics for backtesting results"""
    
    def __init__(self, backtest_results: Dict):
        self.results = backtest_results
        self.equity_curve = backtest_results['equity_curve']
        self.trades = backtest_results['trades']
        self.initial_capital = backtest_results['initial_capital']
        self.final_value = backtest_results['final_portfolio_value']
    
    def calculate_total_return(self) -> float:
        """Calculate total return in absolute terms"""
        return self.final_value - self.initial_capital
    
    def calculate_total_return_pct(self) -> float:
        """Calculate total return percentage"""
        return ((self.final_value - self.initial_capital) / self.initial_capital) * 100
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio (assuming 252 trading days per year)"""
        returns = self.equity_curve['returns'].dropna()
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() - (risk_free_rate / 252)
        return (excess_returns / returns.std()) * np.sqrt(252)
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage"""
        portfolio_values = self.equity_curve['portfolio_value']
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return abs(drawdown.min()) * 100
    
    def calculate_win_rate(self) -> float:
        """Calculate win rate percentage"""
        profitable_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        total_trades = len([t for t in self.trades if 'pnl' in t])
        
        if total_trades == 0:
            return 0.0
        
        return (len(profitable_trades) / total_trades) * 100
    
    def calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        profits = [t['pnl'] for t in self.trades if t.get('pnl', 0) > 0]
        losses = [abs(t['pnl']) for t in self.trades if t.get('pnl', 0) < 0]
        
        gross_profit = sum(profits) if profits else 0
        gross_loss = sum(losses) if losses else 0
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0
        
        return gross_profit / gross_loss
    
    def calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (Annual Return / Max Drawdown)"""
        annual_return = self.calculate_total_return_pct()
        max_dd = self.calculate_max_drawdown()
        return annual_return / max_dd if max_dd != 0 else 0
    
    def calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (uses downside deviation instead of total volatility)"""
        returns = self.equity_curve['returns'].dropna()
        
        if len(returns) == 0:
            return 0.0
        
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf') if returns.mean() > 0 else 0
        
        downside_std = negative_returns.std() * np.sqrt(252)
        excess_return = returns.mean() * 252 - risk_free_rate
        
        return excess_return / downside_std if downside_std != 0 else 0
    
    def calculate_var(self, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk at given confidence level"""
        returns = self.equity_curve['returns'].dropna()
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, confidence_level * 100)
    
    def calculate_cvar(self, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        returns = self.equity_curve['returns'].dropna()
        if len(returns) == 0:
            return 0.0
        var = self.calculate_var(confidence_level)
        tail_returns = returns[returns <= var]
        return tail_returns.mean() if len(tail_returns) > 0 else 0.0
    
    def calculate_information_ratio(self, benchmark_returns=None) -> float:
        """Calculate Information Ratio (Active Return / Tracking Error)"""
        if benchmark_returns is None:
            return 0.0
        
        returns = self.equity_curve['returns'].dropna()
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Align lengths
        min_length = min(len(returns), len(benchmark_returns))
        returns = returns.tail(min_length)
        benchmark_returns = benchmark_returns.tail(min_length)
        
        active_returns = returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(252)
        
        if tracking_error == 0:
            return 0.0
        
        return (active_returns.mean() * 252) / tracking_error
    
    def calculate_monthly_returns(self) -> Dict:
        """Calculate monthly returns"""
        equity_curve = self.equity_curve.copy()
        equity_curve['month'] = equity_curve.index.to_period('M')
        
        monthly_returns = {}
        for month in equity_curve['month'].unique():
            month_data = equity_curve[equity_curve['month'] == month]
            if len(month_data) > 1:
                start_value = month_data['portfolio_value'].iloc[0]
                end_value = month_data['portfolio_value'].iloc[-1]
                monthly_return = ((end_value - start_value) / start_value) * 100
                monthly_returns[month] = monthly_return
        
        return monthly_returns
    
    def calculate_all_metrics(self) -> Dict:
        """Calculate all performance metrics"""
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.final_value,
            'total_return': self.calculate_total_return(),
            'total_return_pct': self.calculate_total_return_pct(),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'sortino_ratio': self.calculate_sortino_ratio(),
            'calmar_ratio': self.calculate_calmar_ratio(),
            'max_drawdown': self.calculate_max_drawdown(),
            'win_rate': self.calculate_win_rate(),
            'profit_factor': self.calculate_profit_factor(),
            'var_5': self.calculate_var(),
            'cvar_5': self.calculate_cvar(),
            'information_ratio': self.calculate_information_ratio(),
            'total_trades': len([t for t in self.trades if 'pnl' in t])
        }
