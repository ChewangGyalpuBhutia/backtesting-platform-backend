import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class RSIMeanReversion(BaseStrategy):
    """RSI Mean Reversion Strategy"""
    
    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__("RSI Mean Reversion")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on RSI levels"""
        data = data.copy()
        
        # Calculate RSI
        data['rsi'] = self.calculate_rsi(data['close'], self.period)
        
        # Generate signals
        data['signal'] = 0
        
        # Buy signal: RSI < oversold level
        data.loc[data['rsi'] < self.oversold, 'signal'] = 1
        
        # Sell signal: RSI > overbought level
        data.loc[data['rsi'] > self.overbought, 'signal'] = -1
        
        # Remove rows with insufficient data
        data = data.dropna()
        
        return data
