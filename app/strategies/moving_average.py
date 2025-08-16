import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class MovingAverageCrossover(BaseStrategy):
    """Moving Average Crossover Strategy"""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        super().__init__("Moving Average Crossover")
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on moving average crossover"""
        print("THIS IS THE FAST PERIOD AND SLOW PERIOS",self.fast_period, self.slow_period)
        data = data.copy()
        
        # Calculate moving averages
        data['ma_fast'] = data['close'].rolling(window=self.fast_period).mean()
        data['ma_slow'] = data['close'].rolling(window=self.slow_period).mean()
        
        # Generate signals
        data['signal'] = 0
        
        # Buy signal: fast MA crosses above slow MA
        data.loc[data['ma_fast'] > data['ma_slow'], 'signal'] = 1
        
        # Sell signal: fast MA crosses below slow MA  
        data.loc[data['ma_fast'] < data['ma_slow'], 'signal'] = -1
        
        # Remove rows with insufficient data
        data = data.dropna()
        
        return data
