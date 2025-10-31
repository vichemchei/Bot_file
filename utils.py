"""
Utility Module for MT5 Forex Bot
Helper functions and configurations.
"""

import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the trading bot."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize Config.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """Load configuration from file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix == '.yaml':
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            'mt5': {
                'path': r'C:\Program Files\MetaTrader 5\terminal64.exe',
                'symbols': ['EURUSD', 'GBPUSD', 'USDJPY'],
                'timeframe': '15m'
            },
            'risk': {
                'initial_balance': 10000,
                'risk_per_trade': 0.01,
                'daily_loss_limit': 0.03,
                'weekly_loss_limit': 0.10,
                'max_drawdown_limit': 0.20,
                'max_positions_per_symbol': 2,
                'max_total_positions': 5,
                'cooldown_after_losses': 3,
                'cooldown_hours': 2.0
            },
            'trading': {
                'sl_atr_mult': 1.0,
                'tp_atr_mult': 2.0,
                'prob_threshold': 0.6,
                'check_interval': 900,
                'max_bars_in_trade': 48
            },
            'model': {
                'model_path': 'models/xgb_model.pkl',
                'scaler_path': 'models/xgb_model_scaler.pkl',
                'features_path': 'models/xgb_model_features.pkl'
            },
            'monitoring': {
                'dashboard_update_interval': 300,
                'drift_check_interval': 3600,
                'email_enabled': False,
                'telegram_enabled': False
            }
        }
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                if self.config_path.suffix == '.yaml':
                    yaml.dump(self.config, f, default_flow_style=False)
                else:
                    json.dump(self.config, f, indent=4)
            
            logger.info(f"Configuration saved to {self.config_path}")
        
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get(self, key: str, default=None):
        """
        Get configuration value by key path.
        
        Args:
            key: Dot-separated key path (e.g., 'risk.initial_balance')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value):
        """
        Set configuration value by key path.
        
        Args:
            key: Dot-separated key path
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value


class TradeLogger:
    """Custom logger for trading activities."""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize TradeLogger.
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup file handler
        log_file = self.log_dir / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(file_handler)
        
        logger.info(f"Trade logger initialized. Log file: {log_file}")
    
    def log_trade(self, trade_info: dict):
        """
        Log trade information to CSV.
        
        Args:
            trade_info: Dictionary with trade details
        """
        csv_file = self.log_dir / "trades.csv"
        
        # Convert to DataFrame
        df = pd.DataFrame([trade_info])
        
        # Append to CSV
        if csv_file.exists():
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file, mode='w', header=True, index=False)
        
        logger.info(f"Trade logged to {csv_file}")


def format_currency(value: float) -> str:
    """
    Format value as currency.
    
    Args:
        value: Numeric value
        
    Returns:
        Formatted string
    """
    return f"${value:,.2f}"


def calculate_pip_value(symbol: str, account_currency: str = 'USD', 
                       lot_size: float = 1.0) -> float:
    """
    Calculate pip value for a currency pair.
    
    Args:
        symbol: Currency pair
        account_currency: Account currency
        lot_size: Position size in lots
        
    Returns:
        Pip value in account currency
    """
    # Standard pip values for major pairs (per 1 lot)
    pip_values = {
        'EURUSD': 10.0,
        'GBPUSD': 10.0,
        'AUDUSD': 10.0,
        'NZDUSD': 10.0,
        'USDCAD': 10.0,
        'USDCHF': 10.0,
        'USDJPY': 10.0
    }
    
    base_pip_value = pip_values.get(symbol, 10.0)
    
    return base_pip_value * lot_size


def validate_symbol(symbol: str) -> bool:
    """
    Validate currency pair symbol format.
    
    Args:
        symbol: Currency pair symbol
        
    Returns:
        True if valid
    """
    if len(symbol) != 6:
        return False
    
    if not symbol.isalpha():
        return False
    
    if not symbol.isupper():
        return False
    
    return True


def calculate_required_margin(symbol: str, lot_size: float, 
                              leverage: int = 100) -> float:
    """
    Calculate required margin for a position.
    
    Args:
        symbol: Currency pair
        lot_size: Position size in lots
        leverage: Account leverage
        
    Returns:
        Required margin in account currency
    """
    # Contract size (standard lot)
    contract_size = 100000
    
    # Margin = (Contract Size * Lot Size) / Leverage
    margin = (contract_size * lot_size) / leverage
    
    return margin


def get_trading_session(hour: int) -> str:
    """
    Determine trading session based on UTC hour.
    
    Args:
        hour: Hour of day (0-23) in UTC
        
    Returns:
        Session name
    """
    if 0 <= hour < 8:
        return 'Asian'
    elif 8 <= hour < 16:
        return 'London'
    elif 16 <= hour < 24:
        return 'New York'
    else:
        return 'Unknown'


def is_high_impact_news_time(current_time: datetime) -> bool:
    """
    Check if current time is during typical high-impact news releases.
    
    Args:
        current_time: Current datetime
        
    Returns:
        True if during news time
    """
    # Typical news times (UTC):
    # - 8:30 AM (London open, ECB, UK data)
    # - 12:30 PM (US pre-market data)
    # - 2:00 PM (FOMC, major US data)
    
    hour = current_time.hour
    minute = current_time.minute
    
    news_times = [
        (8, 15, 8, 45),   # 8:15 - 8:45
        (12, 15, 12, 45), # 12:15 - 12:45
        (14, 0, 14, 30)   # 14:00 - 14:30
    ]
    
    for start_h, start_m, end_h, end_m in news_times:
        if (hour > start_h or (hour == start_h and minute >= start_m)) and \
           (hour < end_h or (hour == end_h and minute <= end_m)):
            return True
    
    return False


def create_default_config_file():
    """Create default configuration file."""
    config = Config()
    config.save_config()
    logger.info("Default configuration file created")


# Example usage
if __name__ == "__main__":
    # Test configuration
    print("Testing Configuration Manager")
    config = Config()
    
    print(f"\nMT5 Path: {config.get('mt5.path')}")
    print(f"Risk per trade: {config.get('risk.risk_per_trade')}")
    print(f"Symbols: {config.get('mt5.symbols')}")
    
    # Test trade logger
    print("\nTesting Trade Logger")
    trade_logger = TradeLogger()
    
    test_trade = {
        'timestamp': datetime.now(),
        'symbol': 'EURUSD',
        'direction': 'BUY',
        'entry_price': 1.1000,
        'position_size': 0.1,
        'stop_loss': 1.0950,
        'take_profit': 1.1100
    }
    
    trade_logger.log_trade(test_trade)
    print("Trade logged successfully")
    
    # Test utility functions
    print("\nTesting Utility Functions")
    print(f"Formatted currency: {format_currency(12345.67)}")
    print(f"Pip value for EURUSD: {calculate_pip_value('EURUSD')}")
    print(f"Symbol validation: {validate_symbol('EURUSD')}")
    print(f"Current session: {get_trading_session(datetime.now().hour)}")
    print(f"High impact news time: {is_high_impact_news_time(datetime.now())}")