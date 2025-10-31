"""
Data Handler Module for MT5 Forex Bot
Handles all data collection, cleaning, and storage operations.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataHandler:
    """
    Manages historical and live data from MetaTrader 5.
    Handles data collection, cleaning, and storage.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataHandler with storage directory.
        
        Args:
            data_dir: Directory to store historical data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.initialized = False
        
    def initialize_mt5(self, path: str = None) -> bool:
        """
        Initialize MT5 connection.
        
        Args:
            path: Path to MT5 terminal executable
            
        Returns:
            Boolean indicating success
        """
        if path:
            success = mt5.initialize(path=path)
        else:
            success = mt5.initialize()
            
        if success:
            self.initialized = True
            logger.info(f"MT5 initialized. Version: {mt5.version()}")
            logger.info(f"Terminal info: {mt5.terminal_info()}")
        else:
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            
        return success
    
    def shutdown_mt5(self):
        """Shutdown MT5 connection gracefully."""
        if self.initialized:
            mt5.shutdown()
            self.initialized = False
            logger.info("MT5 connection closed")
    
    def get_data(self, symbol: str, timeframe: str, start: datetime, 
                 end: datetime = None) -> pd.DataFrame:
        """
        Retrieve historical candle data from MT5.
        
        Args:
            symbol: Currency pair (e.g., "EURUSD")
            timeframe: Timeframe string ("1m", "5m", "15m", "1H", "4H", "1D")
            start: Start datetime
            end: End datetime (default: now)
            
        Returns:
            DataFrame with OHLCV data and clean timestamps
        """
        if not self.initialized:
            raise RuntimeError("MT5 not initialized. Call initialize_mt5() first.")
        
        # Map timeframe strings to MT5 constants
        tf_map = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1H": mt5.TIMEFRAME_H1,
            "4H": mt5.TIMEFRAME_H4,
            "1D": mt5.TIMEFRAME_D1
        }
        
        if timeframe not in tf_map:
            raise ValueError(f"Invalid timeframe. Use: {list(tf_map.keys())}")
        
        mt5_timeframe = tf_map[timeframe]
        
        # Default end to now if not specified
        if end is None:
            end = datetime.now()
        
        # Fetch data from MT5
        rates = mt5.copy_rates_range(symbol, mt5_timeframe, start, end)
        
        if rates is None or len(rates) == 0:
            logger.warning(f"No data retrieved for {symbol} {timeframe}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        
        # Convert time to datetime and set as index
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Rename columns to standard format
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume',
            'real_volume': 'RealVolume'
        }, inplace=True)
        
        # Clean data
        df = self._clean_data(df)
        
        logger.info(f"Retrieved {len(df)} bars for {symbol} {timeframe}")
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate OHLCV data.
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by time
        df.sort_index(inplace=True)
        
        # Remove rows with zero or negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        df = df[(df[price_cols] > 0).all(axis=1)]
        
        # Validate OHLC relationships
        df = df[
            (df['High'] >= df['Low']) &
            (df['High'] >= df['Open']) &
            (df['High'] >= df['Close']) &
            (df['Low'] <= df['Open']) &
            (df['Low'] <= df['Close'])
        ]
        
        # Forward fill small gaps (max 3 periods)
        df = df.fillna(method='ffill', limit=3)
        
        # Drop remaining NaN
        df.dropna(inplace=True)
        
        return df
    
    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """
        Save DataFrame to Parquet file.
        
        Args:
            df: DataFrame to save
            symbol: Currency pair
            timeframe: Timeframe string
        """
        filename = f"{symbol}_{timeframe}.parquet"
        filepath = self.data_dir / filename
        
        df.to_parquet(filepath, compression='gzip')
        logger.info(f"Saved {len(df)} bars to {filepath}")
    
    def load_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Load DataFrame from Parquet file.
        
        Args:
            symbol: Currency pair
            timeframe: Timeframe string
            
        Returns:
            DataFrame with historical data
        """
        filename = f"{symbol}_{timeframe}.parquet"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()
        
        df = pd.read_parquet(filepath)
        logger.info(f"Loaded {len(df)} bars from {filepath}")
        
        return df
    
    def update_data(self, symbol: str, timeframe: str, lookback_days: int = 365):
        """
        Download and save historical data.
        
        Args:
            symbol: Currency pair
            timeframe: Timeframe string
            lookback_days: Days of history to download
        """
        end = datetime.now()
        start = end - timedelta(days=lookback_days)
        
        df = self.get_data(symbol, timeframe, start, end)
        
        if not df.empty:
            self.save_data(df, symbol, timeframe)
    
    def get_live_bar(self, symbol: str, timeframe: str, count: int = 1) -> pd.DataFrame:
        """
        Get most recent bars (for live trading).
        
        Args:
            symbol: Currency pair
            timeframe: Timeframe string
            count: Number of recent bars
            
        Returns:
            DataFrame with recent bars
        """
        tf_map = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1H": mt5.TIMEFRAME_H1,
            "4H": mt5.TIMEFRAME_H4,
            "1D": mt5.TIMEFRAME_D1
        }
        
        mt5_timeframe = tf_map[timeframe]
        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
        
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume'
        }, inplace=True)
        
        return df


# Example usage
if __name__ == "__main__":
    # Initialize handler
    handler = DataHandler(data_dir="forex_data")
    
    # Connect to MT5
    mt5_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    if handler.initialize_mt5(path=mt5_path):
        
        # Download historical data
        symbol = "EURUSD"
        timeframe = "15m"
        
        # Get 1 year of data
        end = datetime.now()
        start = end - timedelta(days=365)
        
        df = handler.get_data(symbol, timeframe, start, end)
        print(f"\nData shape: {df.shape}")
        print(f"\nFirst 5 rows:\n{df.head()}")
        print(f"\nLast 5 rows:\n{df.tail()}")
        
        # Save to disk
        handler.save_data(df, symbol, timeframe)
        
        # Load from disk
        df_loaded = handler.load_data(symbol, timeframe)
        print(f"\nLoaded data shape: {df_loaded.shape}")
        
        # Cleanup
        handler.shutdown_mt5()
    else:
        print("Failed to initialize MT5")