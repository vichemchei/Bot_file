"""
Risk Manager Module for MT5 Forex Bot
Manages risk limits, position sizing, and safety controls.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskManager:
    """
    Manages all risk-related decisions and constraints for trading bot.
    """
    
    def __init__(self, initial_balance: float = 10000.0,
                 risk_per_trade: float = 0.01,
                 daily_loss_limit: float = 0.03,
                 weekly_loss_limit: float = 0.10,
                 max_drawdown_limit: float = 0.20,
                 max_positions_per_symbol: int = 2,
                 max_total_positions: int = 5,
                 cooldown_after_losses: int = 3,
                 cooldown_hours: float = 2.0,
                 min_atr_percentile: float = 0.10):
        """
        Initialize RiskManager.
        
        Args:
            initial_balance: Starting account balance
            risk_per_trade: Risk percentage per trade (0.01 = 1%)
            daily_loss_limit: Max daily loss as fraction of balance (0.03 = 3%)
            weekly_loss_limit: Max weekly loss as fraction of balance
            max_drawdown_limit: Max drawdown before stop trading
            max_positions_per_symbol: Max concurrent positions per pair
            max_total_positions: Max total concurrent positions
            cooldown_after_losses: Number of consecutive losses before cooldown
            cooldown_hours: Hours to wait after loss streak
            min_atr_percentile: Minimum ATR percentile to trade (volatility filter)
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.daily_loss_limit = daily_loss_limit
        self.weekly_loss_limit = weekly_loss_limit
        self.max_drawdown_limit = max_drawdown_limit
        self.max_positions_per_symbol = max_positions_per_symbol
        self.max_total_positions = max_total_positions
        self.cooldown_after_losses = cooldown_after_losses
        self.cooldown_hours = cooldown_hours
        self.min_atr_percentile = min_atr_percentile
        
        # State tracking
        self.peak_balance = initial_balance
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.current_day = None
        self.current_week = None
        self.recent_trades = deque(maxlen=cooldown_after_losses)
        self.cooldown_until = None
        self.open_positions = {}  # symbol -> count
        self.total_positions = 0
        self.trading_enabled = True
        self.last_reset_time = datetime.now()
        
        # Performance tracking
        self.trade_history = []
    
    def update_balance(self, new_balance: float):
        """
        Update current balance and peak balance.
        
        Args:
            new_balance: New account balance
        """
        self.current_balance = new_balance
        
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
    
    def can_trade(self, symbol: str = None) -> tuple:
        """
        Check if trading is allowed based on risk constraints.
        
        Args:
            symbol: Trading pair (optional)
            
        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        now = datetime.now()
        
        # Check if trading is manually disabled
        if not self.trading_enabled:
            return False, "Trading manually disabled"
        
        # Check cooldown period
        if self.cooldown_until and now < self.cooldown_until:
            remaining = (self.cooldown_until - now).total_seconds() / 3600
            return False, f"In cooldown period ({remaining:.1f} hours remaining)"
        
        # Reset daily P&L at start of new day
        if self.current_day != now.date():
            self.daily_pnl = 0.0
            self.current_day = now.date()
        
        # Reset weekly P&L at start of new week
        week_num = now.isocalendar()[1]
        if self.current_week != week_num:
            self.weekly_pnl = 0.0
            self.current_week = week_num
        
        # Check daily loss limit
        daily_loss_pct = abs(self.daily_pnl) / self.current_balance
        if self.daily_pnl < 0 and daily_loss_pct >= self.daily_loss_limit:
            return False, f"Daily loss limit reached ({daily_loss_pct*100:.2f}%)"
        
        # Check weekly loss limit
        weekly_loss_pct = abs(self.weekly_pnl) / self.current_balance
        if self.weekly_pnl < 0 and weekly_loss_pct >= self.weekly_loss_limit:
            return False, f"Weekly loss limit reached ({weekly_loss_pct*100:.2f}%)"
        
        # Check max drawdown
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        if current_drawdown >= self.max_drawdown_limit:
            return False, f"Max drawdown limit reached ({current_drawdown*100:.2f}%)"
        
        # Check total position limit
        if self.total_positions >= self.max_total_positions:
            return False, f"Max total positions reached ({self.total_positions})"
        
        # Check per-symbol position limit
        if symbol:
            symbol_positions = self.open_positions.get(symbol, 0)
            if symbol_positions >= self.max_positions_per_symbol:
                return False, f"Max positions for {symbol} reached ({symbol_positions})"
        
        return True, "OK"
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                               stop_loss: float, pip_value: float = 10.0) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol: Trading pair
            entry_price: Entry price
            stop_loss: Stop loss price
            pip_value: Value of 1 pip for 1 lot
            
        Returns:
            Position size in lots
        """
        # Risk amount in dollars
        risk_amount = self.current_balance * self.risk_per_trade
        
        # Stop distance in pips
        stop_distance = abs(entry_price - stop_loss)
        stop_pips = stop_distance * 10000  # Assuming 4 decimal places
        
        # Calculate position size
        if stop_pips > 0:
            position_size = risk_amount / (stop_pips * pip_value)
        else:
            position_size = 0.0
        
        # Ensure minimum and maximum lot sizes
        position_size = max(0.01, min(position_size, 10.0))  # Min 0.01, Max 10 lots
        
        return round(position_size, 2)
    
    def record_trade(self, symbol: str, profit: float, outcome: str):
        """
        Record trade result and update risk state.
        
        Args:
            symbol: Trading pair
            profit: Trade profit/loss
            outcome: 'WIN', 'LOSS', or 'BREAKEVEN'
        """
        # Update P&L
        self.daily_pnl += profit
        self.weekly_pnl += profit
        
        # Update balance
        self.update_balance(self.current_balance + profit)
        
        # Record trade
        trade_record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'profit': profit,
            'outcome': outcome,
            'balance': self.current_balance,
            'daily_pnl': self.daily_pnl,
            'weekly_pnl': self.weekly_pnl
        }
        self.trade_history.append(trade_record)
        
        # Track recent outcomes for loss streak detection
        self.recent_trades.append(outcome)
        
        # Check for consecutive losses
        if len(self.recent_trades) == self.cooldown_after_losses:
            if all(t == 'LOSS' for t in self.recent_trades):
                self.cooldown_until = datetime.now() + timedelta(hours=self.cooldown_hours)
                logger.warning(f"Cooldown activated until {self.cooldown_until} after {self.cooldown_after_losses} consecutive losses")
                self.recent_trades.clear()
    
    def open_position(self, symbol: str):
        """
        Register opened position.
        
        Args:
            symbol: Trading pair
        """
        self.open_positions[symbol] = self.open_positions.get(symbol, 0) + 1
        self.total_positions += 1
        logger.info(f"Position opened: {symbol} (Total: {self.total_positions})")
    
    def close_position(self, symbol: str):
        """
        Register closed position.
        
        Args:
            symbol: Trading pair
        """
        if symbol in self.open_positions and self.open_positions[symbol] > 0:
            self.open_positions[symbol] -= 1
            self.total_positions -= 1
            logger.info(f"Position closed: {symbol} (Total: {self.total_positions})")
    
    def check_volatility_filter(self, current_atr: float, atr_series: pd.Series) -> bool:
        """
        Check if current ATR meets minimum volatility requirement.
        
        Args:
            current_atr: Current ATR value
            atr_series: Historical ATR series
            
        Returns:
            True if volatility is sufficient
        """
        if len(atr_series) < 100:
            return True  # Not enough data, allow trade
        
        # Calculate percentile
        percentile_threshold = np.percentile(atr_series.dropna(), 
                                            self.min_atr_percentile * 100)
        
        return current_atr >= percentile_threshold
    
    def get_risk_summary(self) -> dict:
        """
        Get current risk status summary.
        
        Returns:
            Dictionary with risk metrics
        """
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        daily_loss_pct = abs(self.daily_pnl) / self.current_balance if self.daily_pnl < 0 else 0
        weekly_loss_pct = abs(self.weekly_pnl) / self.current_balance if self.weekly_pnl < 0 else 0
        
        can_trade, reason = self.can_trade()
        
        summary = {
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'current_drawdown_pct': current_drawdown * 100,
            'daily_pnl': self.daily_pnl,
            'daily_loss_pct': daily_loss_pct * 100,
            'weekly_pnl': self.weekly_pnl,
            'weekly_loss_pct': weekly_loss_pct * 100,
            'total_positions': self.total_positions,
            'can_trade': can_trade,
            'status_reason': reason,
            'in_cooldown': self.cooldown_until is not None and datetime.now() < self.cooldown_until,
            'trading_enabled': self.trading_enabled
        }
        
        return summary
    
    def enable_trading(self):
        """Enable trading."""
        self.trading_enabled = True
        logger.info("Trading enabled")
    
    def disable_trading(self):
        """Disable trading."""
        self.trading_enabled = False
        logger.info("Trading disabled")
    
    def reset_daily_limits(self):
        """Reset daily limits (for testing or manual reset)."""
        self.daily_pnl = 0.0
        self.current_day = datetime.now().date()
        logger.info("Daily limits reset")
    
    def reset_weekly_limits(self):
        """Reset weekly limits."""
        self.weekly_pnl = 0.0
        self.current_week = datetime.now().isocalendar()[1]
        logger.info("Weekly limits reset")
    
    def get_trade_history_df(self) -> pd.DataFrame:
        """
        Get trade history as DataFrame.
        
        Returns:
            DataFrame with trade history
        """
        if len(self.trade_history) == 0:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trade_history)


# Example usage
if __name__ == "__main__":
    # Initialize risk manager
    risk_mgr = RiskManager(
        initial_balance=10000,
        risk_per_trade=0.01,
        daily_loss_limit=0.03,
        cooldown_after_losses=3
    )
    
    print("Risk Manager initialized")
    print(f"Initial balance: ${risk_mgr.current_balance}")
    
    # Check if can trade
    can_trade, reason = risk_mgr.can_trade("EURUSD")
    print(f"\nCan trade EURUSD: {can_trade} ({reason})")
    
    # Calculate position size
    entry = 1.1000
    stop_loss = 1.0950
    position_size = risk_mgr.calculate_position_size("EURUSD", entry, stop_loss)
    print(f"\nPosition size for {entry} -> {stop_loss}: {position_size} lots")
    
    # Simulate some trades
    print("\nSimulating trades:")
    
    # Trade 1: Loss
    risk_mgr.open_position("EURUSD")
    risk_mgr.record_trade("EURUSD", -100, "LOSS")
    risk_mgr.close_position("EURUSD")
    print(f"Trade 1: Loss -$100, Balance: ${risk_mgr.current_balance:.2f}")
    
    # Trade 2: Win
    risk_mgr.open_position("GBPUSD")
    risk_mgr.record_trade("GBPUSD", 200, "WIN")
    risk_mgr.close_position("GBPUSD")
    print(f"Trade 2: Win +$200, Balance: ${risk_mgr.current_balance:.2f}")
    
    # Trade 3: Loss
    risk_mgr.open_position("EURUSD")
    risk_mgr.record_trade("EURUSD", -100, "LOSS")
    risk_mgr.close_position("EURUSD")
    print(f"Trade 3: Loss -$100, Balance: ${risk_mgr.current_balance:.2f}")
    
    # Get risk summary
    print("\nRisk Summary:")
    summary = risk_mgr.get_risk_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")