from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on the strategy logic"""
        pass
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean the data for strategy calculation"""
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Forward fill any missing values (updated method)
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        return data
