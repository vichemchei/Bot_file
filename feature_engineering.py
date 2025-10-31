"""
Feature Engineering Module for MT5 Forex Bot
Computes technical indicators and engineered features.
"""

import pandas as pd
import numpy as np
import ta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Computes technical indicators and features for trading.
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_list = []
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators and features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added features
        """
        df = df.copy()
        
        # Ensure column names are lowercase for consistency
        df.columns = df.columns.str.lower()
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return df
        
        logger.info("Computing features...")
        
        try:
            # === PRICE ACTION FEATURES ===
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['range'] = df['high'] - df['low']
            df['body'] = abs(df['close'] - df['open'])
            df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
            
            # === MOVING AVERAGES ===
            df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
            df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)
            df['sma20'] = ta.trend.sma_indicator(df['close'], window=20)
            
            # EMA spreads and slopes
            df['ema_spread'] = df['ema50'] - df['ema200']
            df['ema_spread_pct'] = (df['ema50'] / df['ema200'] - 1) * 100
            df['ema50_slope'] = df['ema50'].diff(5)
            df['ema200_slope'] = df['ema200'].diff(5)
            
            # Price to EMA distance
            df['price_to_ema50'] = (df['close'] / df['ema50'] - 1) * 100
            df['price_to_ema200'] = (df['close'] / df['ema200'] - 1) * 100
            
            # === MOMENTUM INDICATORS ===
            df['rsi14'] = ta.momentum.rsi(df['close'], window=14)
            df['rsi_oversold'] = (df['rsi14'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi14'] > 70).astype(int)
            
            # Stochastic
            stoch = ta.momentum.stoch(df['high'], df['low'], df['close'], 
                                     window=14, smooth_window=3)
            df['stoch_k'] = stoch
            df['stoch_d'] = stoch.rolling(window=3).mean()
            df['stoch_crossover'] = ((df['stoch_k'] > df['stoch_d']) & 
                                    (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))).astype(int)
            
            # MACD
            macd_obj = ta.trend.MACD(df['close'])
            df['macd'] = macd_obj.macd()
            df['macd_signal'] = macd_obj.macd_signal()
            df['macd_diff'] = macd_obj.macd_diff()
            
            # === VOLATILITY INDICATORS ===
            df['atr14'] = ta.volatility.average_true_range(df['high'], df['low'], 
                                                           df['close'], window=14)
            df['atr_pct'] = (df['atr14'] / df['close']) * 100
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # === VOLUME INDICATORS ===
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            df['obv_slope'] = df['obv'].diff(5)
            df['volume_sma20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma20']
            
            # === INSTITUTIONAL CONCEPTS ===
            # Fair Value Gaps (FVG)
            df['fvg_bullish'] = self._detect_fvg_bullish(df).astype(int)
            df['fvg_bearish'] = self._detect_fvg_bearish(df).astype(int)
            
            # Break of Structure (BOS)
            df['bos_bullish'] = self._detect_bos_bullish(df).astype(int)
            df['bos_bearish'] = self._detect_bos_bearish(df).astype(int)
            
            # === TIME-BASED FEATURES ===
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Trading session
            df['session_asian'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['session_london'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['session_ny'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
            
            # === FEATURE LIST ===
            self.feature_list = [
                'returns', 'log_returns', 'range', 'body',
                'ema50', 'ema200', 'sma20',
                'ema_spread', 'ema_spread_pct', 'ema50_slope', 'ema200_slope',
                'price_to_ema50', 'price_to_ema200',
                'rsi14', 'rsi_oversold', 'rsi_overbought',
                'stoch_k', 'stoch_d', 'stoch_crossover',
                'macd', 'macd_signal', 'macd_diff',
                'atr14', 'atr_pct',
                'bb_width', 'bb_position',
                'obv_slope', 'volume_ratio',
                'fvg_bullish', 'fvg_bearish',
                'bos_bullish', 'bos_bearish',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                'session_asian', 'session_london', 'session_ny'
            ]
            
            logger.info(f"Features computed: {len(self.feature_list)} features")
            
        except Exception as e:
            logger.error(f"Error computing features: {e}")
            import traceback
            traceback.print_exc()
        
        return df
    
    def _detect_fvg_bullish(self, df: pd.DataFrame, atr_mult: float = 0.5) -> pd.Series:
        """
        Detect bullish Fair Value Gaps.
        
        A bullish FVG occurs when:
        - Current bar's low > 2 bars ago high
        - Gap is significant (> atr_mult * ATR)
        """
        result = pd.Series(False, index=df.index)
        
        if len(df) < 3:
            return result
        
        for i in range(2, len(df)):
            if 'atr14' in df.columns and not pd.isna(df['atr14'].iloc[i]):
                gap = df['low'].iloc[i] - df['high'].iloc[i-2]
                min_gap = df['atr14'].iloc[i] * atr_mult
                
                if gap > min_gap:
                    result.iloc[i] = True
        
        return result
    
    def _detect_fvg_bearish(self, df: pd.DataFrame, atr_mult: float = 0.5) -> pd.Series:
        """
        Detect bearish Fair Value Gaps.
        
        A bearish FVG occurs when:
        - Current bar's high < 2 bars ago low
        - Gap is significant (> atr_mult * ATR)
        """
        result = pd.Series(False, index=df.index)
        
        if len(df) < 3:
            return result
        
        for i in range(2, len(df)):
            if 'atr14' in df.columns and not pd.isna(df['atr14'].iloc[i]):
                gap = df['high'].iloc[i-2] - df['low'].iloc[i]
                min_gap = df['atr14'].iloc[i] * atr_mult
                
                if gap > min_gap:
                    result.iloc[i] = True
        
        return result
    
    def _detect_bos_bullish(self, df: pd.DataFrame, lookback: int = 21) -> pd.Series:
        """
        Detect bullish Break of Structure.
        
        Occurs when price breaks above recent swing high.
        """
        result = pd.Series(False, index=df.index)
        
        if len(df) < lookback:
            return result
        
        for i in range(lookback, len(df)):
            recent_high = df['high'].iloc[i-lookback:i].max()
            current_high = df['high'].iloc[i]
            
            if current_high > recent_high:
                result.iloc[i] = True
        
        return result
    
    def _detect_bos_bearish(self, df: pd.DataFrame, lookback: int = 21) -> pd.Series:
        """
        Detect bearish Break of Structure.
        
        Occurs when price breaks below recent swing low.
        """
        result = pd.Series(False, index=df.index)
        
        if len(df) < lookback:
            return result
        
        for i in range(lookback, len(df)):
            recent_low = df['low'].iloc[i-lookback:i].min()
            current_low = df['low'].iloc[i]
            
            if current_low < recent_low:
                result.iloc[i] = True
        
        return result
    
    def get_feature_list(self) -> list:
        """
        Return list of feature names.
        
        Returns:
            List of feature column names
        """
        return self.feature_list


# Example usage
if __name__ == "__main__":
    from data_handler import DataHandler
    
    # Load data
    handler = DataHandler(data_dir="forex_data")
    df = handler.load_data("EURUSD", "15m")
    
    if df.empty:
        print("No data available. Run data download first.")
        exit()
    
    # Compute features
    engineer = FeatureEngineer()
    df = engineer.compute_features(df)
    
    print(f"\nData shape: {df.shape}")
    print(f"\nFeatures computed: {len(engineer.get_feature_list())}")
    print(f"\nFeature list:\n{engineer.get_feature_list()}")
    print(f"\nSample data:\n{df[engineer.get_feature_list()].tail()}")