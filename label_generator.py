"""
Label Generator Module for MT5 Forex Bot
Generates supervised learning labels based on future price action.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LabelGenerator:
    """
    Generates labels for supervised learning based on ATR-based stop-loss and take-profit logic.
    """
    
    def __init__(self, sl_atr_mult: float = 1.0, tp_atr_mult: float = 2.0, 
                 max_bars: int = 48):
        """
        Initialize LabelGenerator.
        
        Args:
            sl_atr_mult: Stop-loss multiplier for ATR
            tp_atr_mult: Take-profit multiplier for ATR
            max_bars: Maximum bars to wait for TP/SL hit
        """
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.max_bars = max_bars
    
    def generate_labels(self, df: pd.DataFrame, 
                       label_type: str = 'classification') -> pd.DataFrame:
        """
        Generate labels for the entire dataset.
        
        Args:
            df: DataFrame with OHLC data and 'atr14' column
            label_type: 'classification' or 'regression'
            
        Returns:
            DataFrame with added label columns
        """
        logger.info(f"Generating {label_type} labels for {len(df)} bars")
        
        df = df.copy()
        
        if label_type == 'classification':
            df['label'], df['bars_to_outcome'] = self._generate_classification_labels(df)
        elif label_type == 'regression':
            df['label'], df['reward_r'] = self._generate_regression_labels(df)
        else:
            raise ValueError("label_type must be 'classification' or 'regression'")
        
        logger.info(f"Labels generated. Positive labels: {(df['label'] == 1).sum()}")
        
        return df
    
    def _generate_classification_labels(self, df: pd.DataFrame) -> tuple:
        """
        Generate binary classification labels.
        
        Label = 1 if TP hit before SL within max_bars
        Label = 0 otherwise
        
        Returns:
            Tuple of (labels Series, bars_to_outcome Series)
        """
        labels = np.zeros(len(df), dtype=int)
        bars_to_outcome = np.full(len(df), np.nan)
        
        for i in range(len(df) - self.max_bars):
            current_price = df['Close'].iloc[i]
            atr = df['atr14'].iloc[i]
            
            if pd.isna(atr) or atr <= 0:
                continue
            
            # Define TP and SL levels for long trade
            tp_long = current_price + (atr * self.tp_atr_mult)
            sl_long = current_price - (atr * self.sl_atr_mult)
            
            # Define TP and SL levels for short trade
            tp_short = current_price - (atr * self.tp_atr_mult)
            sl_short = current_price + (atr * self.sl_atr_mult)
            
            # Check future bars
            future_highs = df['High'].iloc[i+1:i+1+self.max_bars]
            future_lows = df['Low'].iloc[i+1:i+1+self.max_bars]
            
            # Check long trade outcome
            tp_hit_long = (future_highs >= tp_long).any()
            sl_hit_long = (future_lows <= sl_long).any()
            
            if tp_hit_long and sl_hit_long:
                # Both hit - check which came first
                tp_bar = (future_highs >= tp_long).idxmax()
                sl_bar = (future_lows <= sl_long).idxmax()
                
                if tp_bar <= sl_bar:
                    labels[i] = 1
                    bars_to_outcome[i] = (tp_bar - df.index[i]).total_seconds() / 60 / 15  # Assuming 15m bars
                else:
                    labels[i] = 0
                    bars_to_outcome[i] = (sl_bar - df.index[i]).total_seconds() / 60 / 15
            elif tp_hit_long:
                labels[i] = 1
                tp_bar = (future_highs >= tp_long).idxmax()
                bars_to_outcome[i] = (tp_bar - df.index[i]).total_seconds() / 60 / 15
            else:
                labels[i] = 0
                if sl_hit_long:
                    sl_bar = (future_lows <= sl_long).idxmax()
                    bars_to_outcome[i] = (sl_bar - df.index[i]).total_seconds() / 60 / 15
        
        return pd.Series(labels, index=df.index), pd.Series(bars_to_outcome, index=df.index)
    
    def _generate_regression_labels(self, df: pd.DataFrame) -> tuple:
        """
        Generate regression labels (R-multiple).
        
        R-multiple = Actual profit / Risk
        
        Returns:
            Tuple of (labels Series for direction, reward_r Series)
        """
        labels = np.zeros(len(df), dtype=int)
        reward_r = np.zeros(len(df), dtype=float)
        
        for i in range(len(df) - self.max_bars):
            current_price = df['Close'].iloc[i]
            atr = df['atr14'].iloc[i]
            
            if pd.isna(atr) or atr <= 0:
                continue
            
            # Risk per trade
            risk = atr * self.sl_atr_mult
            
            # Look at future price movement
            future_slice = df['Close'].iloc[i+1:i+1+self.max_bars]
            
            if len(future_slice) == 0:
                continue
            
            # Find maximum favorable and adverse excursions
            max_gain = (future_slice.max() - current_price)
            max_loss = (current_price - future_slice.min())
            
            # Determine if trade would be profitable
            if max_gain >= atr * self.tp_atr_mult:
                labels[i] = 1
                reward_r[i] = (atr * self.tp_atr_mult) / risk
            elif max_loss >= atr * self.sl_atr_mult:
                labels[i] = 0
                reward_r[i] = -(atr * self.sl_atr_mult) / risk
            else:
                # Neither TP nor SL hit - use actual movement
                actual_move = future_slice.iloc[-1] - current_price
                labels[i] = 1 if actual_move > 0 else 0
                reward_r[i] = actual_move / risk
        
        return pd.Series(labels, index=df.index), pd.Series(reward_r, index=df.index)
    
    def generate_directional_labels(self, df: pd.DataFrame, 
                                   direction: str = 'long') -> pd.DataFrame:
        """
        Generate labels for specific trade direction.
        
        Args:
            df: DataFrame with OHLC and ATR
            direction: 'long' or 'short'
            
        Returns:
            DataFrame with direction-specific labels
        """
        df = df.copy()
        labels = np.zeros(len(df), dtype=int)
        
        for i in range(len(df) - self.max_bars):
            current_price = df['Close'].iloc[i]
            atr = df['atr14'].iloc[i]
            
            if pd.isna(atr) or atr <= 0:
                continue
            
            if direction == 'long':
                tp = current_price + (atr * self.tp_atr_mult)
                sl = current_price - (atr * self.sl_atr_mult)
                
                future_highs = df['High'].iloc[i+1:i+1+self.max_bars]
                future_lows = df['Low'].iloc[i+1:i+1+self.max_bars]
                
                tp_hit = (future_highs >= tp).any()
                sl_hit = (future_lows <= sl).any()
                
                if tp_hit and not sl_hit:
                    labels[i] = 1
                elif tp_hit and sl_hit:
                    tp_bar = (future_highs >= tp).idxmax()
                    sl_bar = (future_lows <= sl).idxmax()
                    labels[i] = 1 if tp_bar <= sl_bar else 0
            
            elif direction == 'short':
                tp = current_price - (atr * self.tp_atr_mult)
                sl = current_price + (atr * self.sl_atr_mult)
                
                future_highs = df['High'].iloc[i+1:i+1+self.max_bars]
                future_lows = df['Low'].iloc[i+1:i+1+self.max_bars]
                
                tp_hit = (future_lows <= tp).any()
                sl_hit = (future_highs >= sl).any()
                
                if tp_hit and not sl_hit:
                    labels[i] = 1
                elif tp_hit and sl_hit:
                    tp_bar = (future_lows <= tp).idxmax()
                    sl_bar = (future_highs >= sl).idxmax()
                    labels[i] = 1 if tp_bar <= sl_bar else 0
        
        df[f'label_{direction}'] = labels
        
        return df
    
    def get_label_statistics(self, df: pd.DataFrame) -> dict:
        """
        Calculate statistics about the labels.
        
        Args:
            df: DataFrame with 'label' column
            
        Returns:
            Dictionary of statistics
        """
        if 'label' not in df.columns:
            raise ValueError("DataFrame must have 'label' column")
        
        stats = {
            'total_samples': len(df),
            'positive_labels': (df['label'] == 1).sum(),
            'negative_labels': (df['label'] == 0).sum(),
            'positive_ratio': (df['label'] == 1).mean(),
            'null_labels': df['label'].isna().sum()
        }
        
        if 'bars_to_outcome' in df.columns:
            valid_bars = df['bars_to_outcome'].dropna()
            if len(valid_bars) > 0:
                stats['avg_bars_to_outcome'] = valid_bars.mean()
                stats['median_bars_to_outcome'] = valid_bars.median()
        
        if 'reward_r' in df.columns:
            valid_rewards = df['reward_r'].dropna()
            if len(valid_rewards) > 0:
                stats['avg_reward_r'] = valid_rewards.mean()
                stats['median_reward_r'] = valid_rewards.median()
                stats['expectancy'] = valid_rewards.mean()
        
        return stats


# Example usage
if __name__ == "__main__":
    from data_handler import DataHandler
    from feature_engineering import compute_features
    
    # Load data
    handler = DataHandler(data_dir="forex_data")
    
    try:
        df = handler.load_data("EURUSD", "15m")
    except:
        print("No saved data. Please run data_handler.py first.")
        exit()
    
    if df.empty:
        print("No data available.")
        exit()
    
    # Compute features (needed for ATR)
    df = compute_features(df)
    
    # Generate labels
    label_gen = LabelGenerator(sl_atr_mult=1.0, tp_atr_mult=2.0, max_bars=48)
    
    # Classification labels
    df_labeled = label_gen.generate_labels(df, label_type='classification')
    
    print("\nClassification Labels Generated:")
    print(f"Data shape: {df_labeled.shape}")
    
    # Get statistics
    stats = label_gen.get_label_statistics(df_labeled)
    print("\nLabel Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Show sample
    print("\nSample of labeled data:")
    print(df_labeled[['Close', 'atr14', 'label', 'bars_to_outcome']].tail(10))
    
    # Generate regression labels
    df_regression = label_gen.generate_labels(df, label_type='regression')
    print("\nRegression labels sample:")
    print(df_regression[['Close', 'atr14', 'label', 'reward_r']].tail(10))