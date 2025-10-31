import pandas as pd
import numpy as np
import ta

class FeatureEngineer:
    """Class for computing technical indicators and features."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_list = []
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators."""
        # Price action features
        df['returns'] = df['close'].pct_change()
        df['range'] = df['high'] - df['low']
        
        # Moving averages
        df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
        df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)
        
        # Momentum indicators
        df['rsi14'] = ta.momentum.rsi(df['close'], window=14)
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        
        # Store feature list
        self.feature_list = ['returns', 'range', 'ema50', 'ema200', 'rsi14', 'stoch_k']
        
        return df
    
    def get_feature_list(self) -> list:
        """Return list of feature names."""
        return self.feature_list