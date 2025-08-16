from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np

class VolumeMomentumStrategy(BaseStrategy):
    """
    Volume Momentum Strategy
    
    Combines price momentum with volume analysis to identify strong moves
    Buy when price momentum is positive with above-average volume
    Sell when price momentum is negative with above-average volume
    """
    
    def __init__(self, lookback=20, volume_threshold=1.5, momentum_threshold=0.02):
        self.lookback = lookback
        self.volume_threshold = volume_threshold
        self.momentum_threshold = momentum_threshold
        self.name = "Volume Momentum Strategy"
        self.description = f"Volume-confirmed momentum strategy with {lookback}d lookback"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on volume-confirmed momentum"""
        data = data.copy()
        
        # Calculate volume moving average and momentum
        data['volume_ma'] = data['volume'].rolling(window=self.lookback).mean()
        data['price_momentum'] = data['close'].pct_change(self.lookback)
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        # Calculate short-term momentum for more responsive signals
        data['short_momentum'] = data['close'].pct_change(5)
        
        # Generate signals
        data['signal'] = 0
        
        # Buy conditions: positive momentum + high volume + price above MA
        data['price_ma'] = data['close'].rolling(window=self.lookback).mean()
        
        buy_condition = (
            (data['price_momentum'] > self.momentum_threshold) &
            (data['short_momentum'] > 0) &
            (data['volume_ratio'] > self.volume_threshold) &
            (data['close'] > data['price_ma'])
        )
        
        # Sell conditions: negative momentum + high volume + price below MA
        sell_condition = (
            (data['price_momentum'] < -self.momentum_threshold) &
            (data['short_momentum'] < 0) &
            (data['volume_ratio'] > self.volume_threshold) &
            (data['close'] < data['price_ma'])
        )
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        return data
    
    def get_parameters(self):
        """Return strategy parameters for optimization"""
        return {
            'lookback': self.lookback,
            'volume_threshold': self.volume_threshold,
            'momentum_threshold': self.momentum_threshold
        }
