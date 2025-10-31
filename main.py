"""
Main Entry Point for MT5 Forex Trading Bot
Complete end-to-end trading system with ML and rule-based logic.
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
import logging

from data_handler import DataHandler
from feature_engineering import FeatureEngineer
from label_generator import LabelGenerator
from model_trainer import ModelTrainer
from backtester import Backtester
from executor import TradeExecutor
from risk_manager import RiskManager
from monitor import BotMonitor
from utils import Config, TradeLogger

import pandas as pd
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ForexBot:
    """Main Forex Trading Bot orchestrator."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize Forex Bot.
        
        Args:
            config_path: Path to configuration file
        """
        logger.info("="*60)
        logger.info("FOREX TRADING BOT - INITIALIZATION")
        logger.info("="*60)
        
        # Load configuration
        self.config = Config(config_path)
        logger.info("Configuration loaded")
        
        # Initialize components
        self.data_handler = DataHandler(data_dir="forex_data")
        self.feature_engineer = FeatureEngineer()
        self.label_generator = LabelGenerator(
            sl_atr_mult=self.config.get('trading.sl_atr_mult', 1.0),
            tp_atr_mult=self.config.get('trading.tp_atr_mult', 2.0),
            max_bars=self.config.get('trading.max_bars_in_trade', 48)
        )
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            initial_balance=self.config.get('risk.initial_balance', 10000),
            risk_per_trade=self.config.get('risk.risk_per_trade', 0.01),
            daily_loss_limit=self.config.get('risk.daily_loss_limit', 0.03),
            weekly_loss_limit=self.config.get('risk.weekly_loss_limit', 0.10),
            max_drawdown_limit=self.config.get('risk.max_drawdown_limit', 0.20),
            max_positions_per_symbol=self.config.get('risk.max_positions_per_symbol', 2),
            max_total_positions=self.config.get('risk.max_total_positions', 5),
            cooldown_after_losses=self.config.get('risk.cooldown_after_losses', 3),
            cooldown_hours=self.config.get('risk.cooldown_hours', 2.0)
        )
        
        # Initialize monitor
        self.monitor = BotMonitor()
        
        # Initialize trade logger
        self.trade_logger = TradeLogger()
        
        logger.info("All components initialized successfully")
    
    def download_data(self, symbol: str, lookback_days: int = 365):
        """
        Download historical data for a symbol.
        
        Args:
            symbol: Currency pair
            lookback_days: Days of history to download
        """
        logger.info(f"Downloading {lookback_days} days of data for {symbol}")
        
        # Connect to MT5
        mt5_path = self.config.get('mt5.path')
        if not self.data_handler.initialize_mt5(path=mt5_path):
            logger.error("Failed to initialize MT5")
            return
        
        # Download data
        timeframe = self.config.get('mt5.timeframe', '15m')
        self.data_handler.update_data(symbol, timeframe, lookback_days)
        
        # Cleanup
        self.data_handler.shutdown_mt5()
        
        logger.info("Data download complete")
    
    def train_model(self, symbol: str, optimize: bool = False, n_trials: int = 50):
        """
        Train ML model for a symbol.
        
        Args:
            symbol: Currency pair
            optimize: Whether to run hyperparameter optimization
            n_trials: Number of optimization trials
        """
        logger.info("="*60)
        logger.info(f"TRAINING MODEL FOR {symbol}")
        logger.info("="*60)
        
        # Load data
        timeframe = self.config.get('mt5.timeframe', '15m')
        df = self.data_handler.load_data(symbol, timeframe)
        
        if df.empty:
            logger.error(f"No data found for {symbol}. Run download_data first.")
            return
        
        logger.info(f"Loaded {len(df)} bars")
        
        # Compute features
        logger.info("Computing features...")
        df = self.feature_engineer.compute_features(df)
        
        # Generate labels
        logger.info("Generating labels...")
        df = self.label_generator.generate_labels(df, label_type='classification')
        
        # Get label statistics
        stats = self.label_generator.get_label_statistics(df)
        logger.info("Label Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Prepare training data
        feature_cols = self.feature_engineer.get_feature_list()
        trainer = ModelTrainer(model_dir="models")
        
        X, y, features = trainer.prepare_data(df, feature_cols, target_col='label')
        
        # Split data chronologically
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Test samples: {len(X_test)}")
        
        # Normalize features
        X_train_scaled, X_test_scaled = trainer.normalize_features(X_train, X_test)
        
        # Optimize or train with defaults
        if optimize:
            logger.info(f"Optimizing hyperparameters ({n_trials} trials)...")
            best_params = trainer.optimize_hyperparameters(X_train_scaled, y_train, n_trials)
            
            logger.info("Training with best parameters...")
            trainer.train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test, best_params)
        else:
            logger.info("Training with default parameters...")
            trainer.train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Evaluate
        logger.info("Evaluating model...")
        metrics = trainer.evaluate_model(X_test_scaled, y_test)
        
        # Plot feature importance
        trainer.plot_feature_importance(top_n=15)
        
        # Save model
        model_name = f"{symbol}_{timeframe}_model"
        trainer.save_model(model_name)
        
        logger.info("="*60)
        logger.info("MODEL TRAINING COMPLETE")
        logger.info("="*60)
    
    def backtest(self, symbol: str, use_model: bool = True):
        """
        Run backtest for a symbol.
        
        Args:
            symbol: Currency pair
            use_model: Whether to use ML model predictions
        """
        logger.info("="*60)
        logger.info(f"BACKTESTING {symbol}")
        logger.info("="*60)
        
        # Load data
        timeframe = self.config.get('mt5.timeframe', '15m')
        df = self.data_handler.load_data(symbol, timeframe)
        
        if df.empty:
            logger.error(f"No data found for {symbol}")
            return
        
        # Compute features
        logger.info("Computing features...")
        df = self.feature_engineer.compute_features(df)
        
        # Generate signals from rules
        logger.info("Generating rule-based signals...")
        signals = self._generate_signals_from_rules(df)
        
        # Load model if requested
        model_probs = None
        if use_model:
            try:
                from model_trainer import ModelTrainer
                import joblib
                
                model_name = f"{symbol}_{timeframe}_model"
                trainer = ModelTrainer(model_dir="models")
                trainer.load_model(model_name)
                
                # Generate probabilities
                features = df[self.feature_engineer.get_feature_list()]
                X = features.dropna().values
                
                if trainer.scaler:
                    X_scaled = trainer.scaler.transform(X)
                else:
                    X_scaled = X
                
                probs = trainer.model.predict_proba(X_scaled)[:, 1]
                model_probs = pd.Series(probs, index=features.dropna().index)
                
                logger.info("ML model loaded and probabilities generated")
            
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
                logger.info("Proceeding with rule-based signals only")
        
        # Run backtest
        logger.info("Running backtest...")
        backtester = Backtester(
            initial_balance=self.config.get('risk.initial_balance', 10000),
            risk_per_trade=self.config.get('risk.risk_per_trade', 0.01)
        )
        
        trades_df = backtester.backtest(
            df, signals, model_probs,
            prob_threshold=self.config.get('trading.prob_threshold', 0.6),
            sl_atr_mult=self.config.get('trading.sl_atr_mult', 1.0),
            tp_atr_mult=self.config.get('trading.tp_atr_mult', 2.0)
        )
        
        # Calculate metrics
        metrics = backtester.calculate_metrics(trades_df)
        
        logger.info("="*60)
        logger.info("BACKTEST RESULTS")
        logger.info("="*60)
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.2f}")
            else:
                logger.info(f"{key}: {value}")
        
        # Plot results
        backtester.plot_results(output_dir="backtest_results")
        
        # Save trades
        trades_df.to_csv(f"backtest_results/{symbol}_{timeframe}_trades.csv", index=False)
        
        logger.info("="*60)
        logger.info("BACKTEST COMPLETE")
        logger.info("="*60)
    
    def _generate_signals_from_rules(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals from rule-based conditions."""
        
        
        signals = pd.Series(0, index=df.index)
        
        for i in range(len(df)):
            if i < 200:  # Need history for indicators
                continue
            
            current = df.iloc[i]
            
            # Simple EMA + RSI + Stoch rules
            ema_bullish = current['ema50'] > current['ema200']
            ema_bearish = current['ema50'] < current['ema200']
            
            rsi_oversold = current['rsi14'] < 30
            rsi_overbought = current['rsi14'] > 70
            
            stoch_oversold = current['stoch_k'] < 20
            stoch_overbought = current['stoch_k'] > 80
            
            # Long signal
            if ema_bullish and (rsi_oversold or stoch_oversold):
                signals.iloc[i] = 1
            
            # Short signal
            elif ema_bearish and (rsi_overbought or stoch_overbought):
                signals.iloc[i] = -1
        
        return signals
    
    def run_live(self, symbol: str):
        """
        Run live trading for a symbol.
        
        Args:
            symbol: Currency pair to trade
        """
        logger.info("="*60)
        logger.info(f"STARTING LIVE TRADING: {symbol}")
        logger.info("="*60)
        
        # Connect to MT5
        mt5_path = self.config.get('mt5.path')
        if not self.data_handler.initialize_mt5(path=mt5_path):
            logger.error("Failed to initialize MT5")
            return
        
        # Load model
        timeframe = self.config.get('mt5.timeframe', '15m')
        model_name = f"{symbol}_{timeframe}_model"
        
        model_path = f"models/{model_name}.pkl"
        scaler_path = f"models/{model_name}_scaler.pkl"
        features_path = f"models/{model_name}_features.pkl"
        
        # Initialize executor
        executor = TradeExecutor(
            data_handler=self.data_handler,
            feature_engineer=self.feature_engineer,
            risk_manager=self.risk_manager,
            model_path=model_path,
            scaler_path=scaler_path,
            features_path=features_path
        )
        
        # Run trading loop
        try:
            executor.run_trading_loop(
                symbol=symbol,
                timeframe=timeframe,
                check_interval=self.config.get('trading.check_interval', 900),
                prob_threshold=self.config.get('trading.prob_threshold', 0.6)
            )
        except KeyboardInterrupt:
            logger.info("Live trading interrupted by user")
        finally:
            self.data_handler.shutdown_mt5()
            logger.info("MT5 connection closed")
        
        logger.info("="*60)
        logger.info("LIVE TRADING ENDED")
        logger.info("="*60)


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(description='Forex Trading Bot with ML')
    
    parser.add_argument('command', choices=['download', 'train', 'backtest', 'live'],
                       help='Command to execute')
    parser.add_argument('--symbol', type=str, default='EURUSD',
                       help='Currency pair (default: EURUSD)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--optimize', action='store_true',
                       help='Run hyperparameter optimization (for train command)')
    parser.add_argument('--trials', type=int, default=50,
                       help='Number of optimization trials')
    parser.add_argument('--days', type=int, default=365,
                       help='Days of historical data to download')
    parser.add_argument('--no-model', action='store_true',
                       help='Run backtest without ML model')
    
    args = parser.parse_args()
    
    # Initialize bot
    bot = ForexBot(config_path=args.config)
    
    # Execute command
    if args.command == 'download':
        bot.download_data(args.symbol, lookback_days=args.days)
    
    elif args.command == 'train':
        bot.train_model(args.symbol, optimize=args.optimize, n_trials=args.trials)
    
    elif args.command == 'backtest':
        bot.backtest(args.symbol, use_model=not args.no_model)
    
    elif args.command == 'live':
        bot.run_live(args.symbol)


if __name__ == "__main__":
    main()


"""
USAGE EXAMPLES:

1. Download historical data:
   python main.py download --symbol EURUSD --days 730

2. Train ML model (with optimization):
   python main.py train --symbol EURUSD --optimize --trials 100

3. Train ML model (fast, default params):
   python main.py train --symbol EURUSD

4. Run backtest with ML model:
   python main.py backtest --symbol EURUSD

5. Run backtest without ML (rule-based only):
   python main.py backtest --symbol EURUSD --no-model

6. Run live trading:
   python main.py live --symbol EURUSD

7. Use custom config:
   python main.py train --symbol GBPUSD --config my_config.yaml
"""