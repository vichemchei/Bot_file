"""
Executor Module for MT5 Forex Bot
Handles live trade execution with rule-based filters and ML predictions.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
import joblib
from pathlib import Path

from data_handler import DataHandler
from feature_engineering import FeatureEngineer
from risk_manager import RiskManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradeExecutor:
    """
    Executes trades on MetaTrader 5 with rule-based and ML-based signals.
    """
    
    def __init__(self, data_handler: DataHandler, feature_engineer: FeatureEngineer,
                 risk_manager: RiskManager, model_path: str = None,
                 scaler_path: str = None, features_path: str = None):
        """
        Initialize TradeExecutor.
        
        Args:
            data_handler: DataHandler instance
            feature_engineer: FeatureEngineer instance
            risk_manager: RiskManager instance
            model_path: Path to trained ML model
            scaler_path: Path to feature scaler
            features_path: Path to feature names list
        """
        self.data_handler = data_handler
        self.feature_engineer = feature_engineer
        self.risk_manager = risk_manager
        
        # Load ML model if provided
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        if model_path and Path(model_path).exists():
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            
            if scaler_path and Path(scaler_path).exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Scaler loaded from {scaler_path}")
            
            if features_path and Path(features_path).exists():
                self.feature_names = joblib.load(features_path)
                logger.info(f"Feature names loaded from {features_path}")
        
        # Trading state
        self.active_trades = {}  # ticket -> trade_info
        self.trade_log = []
    
    def check_rule_based_conditions(self, df: pd.DataFrame) -> dict:
        """
        Check rule-based trading conditions.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary with signals and conditions
        """
        if len(df) == 0:
            return {'signal': 0, 'conditions_met': False}
        
        current = df.iloc[-1]
        
        # Initialize signal
        signal = 0
        conditions = {}
        
        # EMA trend filter
        ema_bullish = current['ema50'] > current['ema200']
        ema_bearish = current['ema50'] < current['ema200']
        conditions['ema_trend'] = 'BULLISH' if ema_bullish else 'BEARISH' if ema_bearish else 'NEUTRAL'
        
        # Price position relative to EMAs
        price_above_ema50 = current['Close'] > current['ema50']
        price_below_ema50 = current['Close'] < current['ema50']
        
        # RSI conditions
        rsi_oversold = current['rsi14'] < 30
        rsi_overbought = current['rsi14'] > 70
        rsi_neutral = 30 <= current['rsi14'] <= 70
        conditions['rsi_level'] = 'OVERSOLD' if rsi_oversold else 'OVERBOUGHT' if rsi_overbought else 'NEUTRAL'
        
        # Stochastic conditions
        stoch_oversold = current['stoch_k'] < 20
        stoch_overbought = current['stoch_k'] > 80
        stoch_crossover_up = current['stoch_k'] > current['stoch_d']
        stoch_crossover_down = current['stoch_k'] < current['stoch_d']
        conditions['stoch_signal'] = 'BULLISH' if stoch_crossover_up else 'BEARISH' if stoch_crossover_down else 'NEUTRAL'
        
        # FVG (Fair Value Gap) signals
        fvg_bullish = current['fvg_bullish'] == 1
        fvg_bearish = current['fvg_bearish'] == 1
        conditions['fvg'] = 'BULLISH' if fvg_bullish else 'BEARISH' if fvg_bearish else 'NONE'
        
        # BOS (Break of Structure) signals
        bos_bullish = current['bos_bullish'] == 1
        bos_bearish = current['bos_bearish'] == 1
        conditions['bos'] = 'BULLISH' if bos_bullish else 'BEARISH' if bos_bearish else 'NONE'
        
        # LONG signal conditions
        long_conditions = [
            ema_bullish,  # Trend is up
            price_above_ema50 or (rsi_oversold and stoch_oversold),  # Price pullback or oversold
            stoch_crossover_up,  # Momentum turning up
            fvg_bullish or bos_bullish,  # Institutional confirmation
            rsi_neutral or rsi_oversold  # Not overbought
        ]
        
        # SHORT signal conditions
        short_conditions = [
            ema_bearish,  # Trend is down
            price_below_ema50 or (rsi_overbought and stoch_overbought),  # Price pullback or overbought
            stoch_crossover_down,  # Momentum turning down
            fvg_bearish or bos_bearish,  # Institutional confirmation
            rsi_neutral or rsi_overbought  # Not oversold
        ]
        
        # Count met conditions
        long_score = sum(long_conditions)
        short_score = sum(short_conditions)
        
        conditions['long_score'] = long_score
        conditions['short_score'] = short_score
        
        # Minimum 3 out of 5 conditions for signal
        if long_score >= 3:
            signal = 1
            conditions['signal_type'] = 'LONG'
        elif short_score >= 3:
            signal = -1
            conditions['signal_type'] = 'SHORT'
        else:
            conditions['signal_type'] = 'NONE'
        
        conditions['signal'] = signal
        conditions['conditions_met'] = signal != 0
        
        return conditions
    
    def get_ml_prediction(self, df: pd.DataFrame) -> float:
        """
        Get ML model prediction probability.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Probability of successful trade (0-1)
        """
        if self.model is None or self.feature_names is None:
            return 0.5  # Neutral if no model
        
        try:
            current = df.iloc[-1]
            
            # Extract features
            features = current[self.feature_names].values.reshape(1, -1)
            
            # Scale if scaler available
            if self.scaler is not None:
                features = self.scaler.transform(features)
            
            # Predict probability
            prob = self.model.predict_proba(features)[0, 1]
            
            return prob
        
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return 0.5
    
    def execute_trade(self, symbol: str, direction: str, entry_price: float,
                     stop_loss: float, take_profit: float, position_size: float,
                     probability: float = None) -> int:
        """
        Execute trade on MT5.
        
        Args:
            symbol: Trading pair
            direction: 'BUY' or 'SELL'
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            position_size: Position size in lots
            probability: ML probability (for logging)
            
        Returns:
            Ticket number (0 if failed)
        """
        if not self.data_handler.initialized:
            logger.error("MT5 not initialized")
            return 0
        
        # Prepare request
        order_type = mt5.ORDER_TYPE_BUY if direction == 'BUY' else mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position_size,
            "type": order_type,
            "price": entry_price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 10,
            "magic": 234000,
            "comment": f"ML_Bot_p{probability:.2f}" if probability else "ML_Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.comment}")
            return 0
        
        # Log trade
        ticket = result.order
        trade_info = {
            'ticket': ticket,
            'symbol': symbol,
            'direction': direction,
            'entry_time': datetime.now(),
            'entry_price': result.price,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'probability': probability
        }
        
        self.active_trades[ticket] = trade_info
        self.trade_log.append(trade_info)
        
        # Update risk manager
        self.risk_manager.open_position(symbol)
        
        logger.info(f"Trade executed: {direction} {symbol} @ {result.price} | SL: {stop_loss} | TP: {take_profit}")
        
        return ticket
    
    def close_trade(self, ticket: int) -> bool:
        """
        Close trade by ticket.
        
        Args:
            ticket: Trade ticket number
            
        Returns:
            True if successful
        """
        if ticket not in self.active_trades:
            logger.warning(f"Ticket {ticket} not found in active trades")
            return False
        
        trade_info = self.active_trades[ticket]
        
        # Get position
        position = mt5.positions_get(ticket=ticket)
        
        if not position:
            logger.warning(f"Position {ticket} not found")
            return False
        
        position = position[0]
        
        # Prepare close request
        close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": close_type,
            "position": ticket,
            "price": mt5.symbol_info_tick(position.symbol).bid if close_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(position.symbol).ask,
            "deviation": 10,
            "magic": 234000,
            "comment": "ML_Bot_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Close failed: {result.comment}")
            return False
        
        # Calculate profit
        profit = position.profit
        
        # Update trade info
        trade_info['exit_time'] = datetime.now()
        trade_info['exit_price'] = result.price
        trade_info['profit'] = profit
        trade_info['outcome'] = 'WIN' if profit > 0 else 'LOSS' if profit < 0 else 'BREAKEVEN'
        
        # Update risk manager
        self.risk_manager.record_trade(position.symbol, profit, trade_info['outcome'])
        self.risk_manager.close_position(position.symbol)
        
        # Remove from active trades
        del self.active_trades[ticket]
        
        logger.info(f"Trade closed: {position.symbol} | Profit: ${profit:.2f}")
        
        return True
    
    def monitor_active_trades(self):
        """Monitor and manage active trades."""
        for ticket in list(self.active_trades.keys()):
            position = mt5.positions_get(ticket=ticket)
            
            if not position:
                # Position closed by TP/SL
                trade_info = self.active_trades[ticket]
                
                # Try to get historical deal to check outcome
                deals = mt5.history_deals_get(position=ticket)
                
                if deals and len(deals) > 0:
                    last_deal = deals[-1]
                    profit = last_deal.profit
                    
                    trade_info['exit_time'] = datetime.fromtimestamp(last_deal.time)
                    trade_info['exit_price'] = last_deal.price
                    trade_info['profit'] = profit
                    trade_info['outcome'] = 'WIN' if profit > 0 else 'LOSS' if profit < 0 else 'BREAKEVEN'
                    
                    # Update risk manager
                    self.risk_manager.record_trade(trade_info['symbol'], profit, trade_info['outcome'])
                    self.risk_manager.close_position(trade_info['symbol'])
                    
                    logger.info(f"Trade auto-closed: {trade_info['symbol']} | Profit: ${profit:.2f}")
                
                del self.active_trades[ticket]
    
    def run_trading_loop(self, symbol: str, timeframe: str, 
                        check_interval: int = 900, prob_threshold: float = 0.6):
        """
        Main trading loop.
        
        Args:
            symbol: Trading pair
            timeframe: Timeframe for analysis
            check_interval: Seconds between checks (900 = 15 min)
            prob_threshold: Minimum ML probability to trade
        """
        logger.info(f"Starting trading loop for {symbol} on {timeframe}")
        logger.info(f"Check interval: {check_interval}s | Probability threshold: {prob_threshold}")
        
        while True:
            try:
                # Monitor existing trades
                self.monitor_active_trades()
                
                # Check if can trade
                can_trade, reason = self.risk_manager.can_trade(symbol)
                
                if not can_trade:
                    logger.info(f"Trading not allowed: {reason}")
                    time.sleep(check_interval)
                    continue
                
                # Get latest data
                df = self.data_handler.get_live_bar(symbol, timeframe, count=200)
                
                if df.empty:
                    logger.warning("No data retrieved")
                    time.sleep(check_interval)
                    continue
                
                # Compute features
                df = self.feature_engineer.compute_features(df)
                
                # Check rule-based conditions
                conditions = self.check_rule_based_conditions(df)
                
                if not conditions['conditions_met']:
                    logger.info(f"No signal: {conditions}")
                    time.sleep(check_interval)
                    continue
                
                # Get ML prediction
                probability = self.get_ml_prediction(df)
                
                logger.info(f"Signal detected: {conditions['signal_type']} | Probability: {probability:.3f}")
                
                # Check probability threshold
                if probability < prob_threshold:
                    logger.info(f"Probability too low ({probability:.3f} < {prob_threshold})")
                    time.sleep(check_interval)
                    continue
                
                # Prepare trade parameters
                current = df.iloc[-1]
                entry_price = current['Close']
                atr = current['atr14']
                
                direction = 'BUY' if conditions['signal'] == 1 else 'SELL'
                
                # Calculate SL and TP
                if direction == 'BUY':
                    stop_loss = entry_price - (atr * 1.0)
                    take_profit = entry_price + (atr * 2.0)
                else:
                    stop_loss = entry_price + (atr * 1.0)
                    take_profit = entry_price - (atr * 2.0)
                
                # Calculate position size
                position_size = self.risk_manager.calculate_position_size(
                    symbol, entry_price, stop_loss
                )
                
                # Check volatility filter
                if not self.risk_manager.check_volatility_filter(atr, df['atr14']):
                    logger.info("Volatility too low, skipping trade")
                    time.sleep(check_interval)
                    continue
                
                # Execute trade
                logger.info(f"Executing {direction} trade on {symbol}")
                ticket = self.execute_trade(
                    symbol, direction, entry_price,
                    stop_loss, take_profit, position_size, probability
                )
                
                if ticket > 0:
                    logger.info(f"Trade executed successfully. Ticket: {ticket}")
                else:
                    logger.error("Trade execution failed")
                
                # Wait before next check
                time.sleep(check_interval)
            
            except KeyboardInterrupt:
                logger.info("Trading loop interrupted by user")
                break
            
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(check_interval)
        
        logger.info("Trading loop ended")
    
    def get_trade_log_df(self) -> pd.DataFrame:
        """Get trade log as DataFrame."""
        if len(self.trade_log) == 0:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trade_log)


# Example usage
if __name__ == "__main__":
    # Initialize components
    data_handler = DataHandler()
    
    # Initialize MT5
    mt5_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    if not data_handler.initialize_mt5(path=mt5_path):
        print("Failed to initialize MT5")
        exit()
    
    feature_engineer = FeatureEngineer()
    risk_manager = RiskManager(initial_balance=10000, risk_per_trade=0.01)
    
    # Initialize executor (without ML model for demo)
    executor = TradeExecutor(
        data_handler=data_handler,
        feature_engineer=feature_engineer,
        risk_manager=risk_manager
    )
    
    # Run trading loop (dry run - comment out for live trading)
    print("Trading executor initialized")
    print("To start live trading, uncomment the run_trading_loop call")
    
    # executor.run_trading_loop("EURUSD", "15m", check_interval=900, prob_threshold=0.6)
    
    data_handler.shutdown_mt5()