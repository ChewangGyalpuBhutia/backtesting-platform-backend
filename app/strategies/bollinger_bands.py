import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class BollingerBandStrategy(BaseStrategy):
    """Bollinger Band Strategy"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__("Bollinger Band Strategy")
        self.period = period
        self.std_dev = std_dev
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on Bollinger Bands"""
        data = data.copy()
        
        # Calculate Bollinger Bands
        data['ma'] = data['close'].rolling(window=self.period).mean()
        data['std'] = data['close'].rolling(window=self.period).std()
        data['upper_band'] = data['ma'] + (data['std'] * self.std_dev)
        data['lower_band'] = data['ma'] - (data['std'] * self.std_dev)
        
        # Generate signals
        data['signal'] = 0
        
        # Buy signal: price touches lower band
        data.loc[data['close'] <= data['lower_band'], 'signal'] = 1
        
        # Sell signal: price touches upper band
        data.loc[data['close'] >= data['upper_band'], 'signal'] = -1
        
        # Remove rows with insufficient data
        data = data.dropna()
        
        return data
