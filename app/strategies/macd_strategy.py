from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np

class MACDStrategy(BaseStrategy):
    """
    MACD (Moving Average Convergence Divergence) Strategy
    
    Generates buy signals when MACD line crosses above signal line
    Generates sell signals when MACD line crosses below signal line
    """
    
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.name = "MACD Strategy"
        self.description = f"MACD({fast_period},{slow_period},{signal_period}) crossover strategy"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on MACD crossover"""
        data = data.copy()
        
        # Calculate MACD
        exp1 = data['close'].ewm(span=self.fast_period).mean()
        exp2 = data['close'].ewm(span=self.slow_period).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=self.signal_period).mean()
        histogram = macd_line - signal_line
        
        # Add technical indicators to data
        data['macd'] = macd_line
        data['macd_signal'] = signal_line
        data['macd_histogram'] = histogram
        
        # Generate signals
        data['signal'] = 0
        
        # Buy when MACD crosses above signal line
        data.loc[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1)), 'signal'] = 1
        
        # Sell when MACD crosses below signal line
        data.loc[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1)), 'signal'] = -1
        
        return data
    
    def get_parameters(self):
        """Return strategy parameters for optimization"""
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period
        }
