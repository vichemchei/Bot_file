"""
Backtester Module for MT5 Forex Bot
Performs walk-forward simulation with realistic transaction costs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtests trading strategies with realistic costs and risk management.
    """
    
    def __init__(self, initial_balance: float = 10000.0, risk_per_trade: float = 0.01,
                 spread_pips: float = 2.0, commission_per_lot: float = 7.0,
                 slippage_pips: float = 0.5, pip_value: float = 10.0):
        """
        Initialize Backtester.
        
        Args:
            initial_balance: Starting account balance
            risk_per_trade: Risk percentage per trade (0.01 = 1%)
            spread_pips: Bid-ask spread in pips
            commission_per_lot: Commission per standard lot
            slippage_pips: Average slippage in pips
            pip_value: Value of 1 pip for 1 standard lot
        """
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.spread_pips = spread_pips
        self.commission_per_lot = commission_per_lot
        self.slippage_pips = slippage_pips
        self.pip_value = pip_value
        
        self.trades = []
        self.equity_curve = []
        self.balance = initial_balance
    
    def reset(self):
        """Reset backtest state."""
        self.trades = []
        self.equity_curve = []
        self.balance = self.initial_balance
    
    def backtest(self, df: pd.DataFrame, signals: pd.Series, 
                model_probs: pd.Series = None, prob_threshold: float = 0.6,
                sl_atr_mult: float = 1.0, tp_atr_mult: float = 2.0,
                max_bars_in_trade: int = 48) -> pd.DataFrame:
        """
        Run backtest on historical data.
        
        Args:
            df: DataFrame with OHLCV and features
            signals: Series with trade signals (1=long, -1=short, 0=no trade)
            model_probs: Series with ML model probabilities (optional)
            prob_threshold: Minimum probability to take trade
            sl_atr_mult: Stop-loss ATR multiplier
            tp_atr_mult: Take-profit ATR multiplier
            max_bars_in_trade: Maximum bars to hold position
            
        Returns:
            DataFrame with trade log
        """
        logger.info(f"Starting backtest on {len(df)} bars")
        self.reset()
        
        in_trade = False
        trade_entry_idx = None
        trade_direction = None
        entry_price = None
        stop_loss = None
        take_profit = None
        position_size = None
        
        for i in range(len(df)):
            current_bar = df.iloc[i]
            current_time = df.index[i]
            
            # Record equity
            self.equity_curve.append({
                'time': current_time,
                'balance': self.balance,
                'in_trade': in_trade
            })
            
            # Check if we should exit current trade
            if in_trade:
                bars_in_trade = i - trade_entry_idx
                
                # Check TP/SL
                if trade_direction == 1:  # Long trade
                    if current_bar['High'] >= take_profit:
                        # Take profit hit
                        exit_price = take_profit
                        profit_pips = (exit_price - entry_price) * 10000
                        self._close_trade(current_time, exit_price, profit_pips, 
                                        position_size, 'TP')
                        in_trade = False
                    elif current_bar['Low'] <= stop_loss:
                        # Stop loss hit
                        exit_price = stop_loss
                        profit_pips = (exit_price - entry_price) * 10000
                        self._close_trade(current_time, exit_price, profit_pips, 
                                        position_size, 'SL')
                        in_trade = False
                    elif bars_in_trade >= max_bars_in_trade:
                        # Time-based exit
                        exit_price = current_bar['Close']
                        profit_pips = (exit_price - entry_price) * 10000
                        self._close_trade(current_time, exit_price, profit_pips, 
                                        position_size, 'TIME')
                        in_trade = False
                
                elif trade_direction == -1:  # Short trade
                    if current_bar['Low'] <= take_profit:
                        # Take profit hit
                        exit_price = take_profit
                        profit_pips = (entry_price - exit_price) * 10000
                        self._close_trade(current_time, exit_price, profit_pips, 
                                        position_size, 'TP')
                        in_trade = False
                    elif current_bar['High'] >= stop_loss:
                        # Stop loss hit
                        exit_price = stop_loss
                        profit_pips = (entry_price - exit_price) * 10000
                        self._close_trade(current_time, exit_price, profit_pips, 
                                        position_size, 'SL')
                        in_trade = False
                    elif bars_in_trade >= max_bars_in_trade:
                        # Time-based exit
                        exit_price = current_bar['Close']
                        profit_pips = (entry_price - exit_price) * 10000
                        self._close_trade(current_time, exit_price, profit_pips, 
                                        position_size, 'TIME')
                        in_trade = False
            
            # Check for new trade entry
            if not in_trade and i < len(signals):
                signal = signals.iloc[i] if i < len(signals) else 0
                
                # Check ML probability if provided
                if model_probs is not None and i < len(model_probs):
                    prob = model_probs.iloc[i]
                    if prob < prob_threshold:
                        continue
                
                if signal != 0 and not pd.isna(current_bar['atr14']):
                    # Enter trade
                    trade_direction = signal
                    entry_price = current_bar['Close']
                    atr = current_bar['atr14']
                    
                    # Calculate SL and TP
                    if trade_direction == 1:  # Long
                        stop_loss = entry_price - (atr * sl_atr_mult)
                        take_profit = entry_price + (atr * tp_atr_mult)
                    else:  # Short
                        stop_loss = entry_price + (atr * sl_atr_mult)
                        take_profit = entry_price - (atr * tp_atr_mult)
                    
                    # Calculate position size
                    risk_amount = self.balance * self.risk_per_trade
                    stop_pips = abs(entry_price - stop_loss) * 10000
                    position_size = risk_amount / (stop_pips * self.pip_value)
                    
                    # Open trade
                    trade_entry_idx = i
                    in_trade = True
                    
                    self.trades.append({
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'direction': 'LONG' if trade_direction == 1 else 'SHORT',
                        'position_size': position_size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'atr': atr,
                        'balance_at_entry': self.balance
                    })
        
        # Close any remaining open trade
        if in_trade:
            exit_price = df.iloc[-1]['Close']
            if trade_direction == 1:
                profit_pips = (exit_price - entry_price) * 10000
            else:
                profit_pips = (entry_price - exit_price) * 10000
            self._close_trade(df.index[-1], exit_price, profit_pips, 
                            position_size, 'EOD')
        
        logger.info(f"Backtest complete. {len(self.trades)} trades executed.")
        
        return pd.DataFrame(self.trades)
    
    def _close_trade(self, exit_time, exit_price, profit_pips, position_size, reason):
        """Close trade and update balance."""
        # Calculate P&L
        gross_profit = profit_pips * position_size * self.pip_value
        
        # Deduct costs
        spread_cost = self.spread_pips * position_size * self.pip_value
        commission = self.commission_per_lot * position_size
        slippage_cost = self.slippage_pips * position_size * self.pip_value
        
        total_costs = spread_cost + commission + slippage_cost
        net_profit = gross_profit - total_costs
        
        # Update balance
        self.balance += net_profit
        
        # Update last trade record
        self.trades[-1].update({
            'exit_time': exit_time,
            'exit_price': exit_price,
            'profit_pips': profit_pips,
            'gross_profit': gross_profit,
            'total_costs': total_costs,
            'net_profit': net_profit,
            'balance_after': self.balance,
            'exit_reason': reason,
            'return_pct': (net_profit / self.trades[-1]['balance_at_entry']) * 100
        })
    
    def calculate_metrics(self, trades_df: pd.DataFrame = None) -> dict:
        """
        Calculate performance metrics.
        
        Args:
            trades_df: DataFrame with trades (uses self.trades if None)
            
        Returns:
            Dictionary of metrics
        """
        if trades_df is None:
            trades_df = pd.DataFrame(self.trades)
        
        if len(trades_df) == 0:
            logger.warning("No trades to analyze")
            return {}
        
        # Filter complete trades
        trades_df = trades_df[trades_df['exit_time'].notna()].copy()
        
        if len(trades_df) == 0:
            return {}
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = (trades_df['net_profit'] > 0).sum()
        losing_trades = (trades_df['net_profit'] < 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit metrics
        total_profit = trades_df['net_profit'].sum()
        total_return_pct = ((self.balance - self.initial_balance) / 
                           self.initial_balance) * 100
        
        avg_win = trades_df[trades_df['net_profit'] > 0]['net_profit'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['net_profit'] < 0]['net_profit'].mean()) if losing_trades > 0 else 0
        
        profit_factor = (trades_df[trades_df['net_profit'] > 0]['net_profit'].sum() /
                        abs(trades_df[trades_df['net_profit'] < 0]['net_profit'].sum())
                        if losing_trades > 0 else np.inf)
        
        # Risk metrics
        returns = trades_df['return_pct'].values / 100
        equity_series = pd.Series([t['balance_after'] for t in self.trades])
        
        # Drawdown
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        # Sharpe and Sortino ratios (annualized, assuming ~250 trading days)
        if len(returns) > 1:
            sharpe_ratio = np.sqrt(250) * (returns.mean() / returns.std()) if returns.std() > 0 else 0
            downside_returns = returns[returns < 0]
            sortino_ratio = (np.sqrt(250) * returns.mean() / downside_returns.std() 
                           if len(downside_returns) > 0 and downside_returns.std() > 0 else 0)
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate * 100,
            'total_profit': total_profit,
            'total_return_pct': total_return_pct,
            'final_balance': self.balance,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'expectancy': expectancy
        }
        
        return metrics
    
    def plot_results(self, output_dir: str = "backtest_results"):
        """Plot backtest results."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if len(self.equity_curve) == 0:
            logger.warning("No equity curve to plot")
            return
        
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Plot equity curve
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Equity curve
        ax1.plot(equity_df['time'], equity_df['balance'], label='Balance', linewidth=2)
        ax1.axhline(y=self.initial_balance, color='r', linestyle='--', 
                   label='Initial Balance')
        ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Balance ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        cummax = equity_df['balance'].cummax()
        drawdown = (equity_df['balance'] - cummax) / cummax * 100
        ax2.fill_between(equity_df['time'], drawdown, 0, alpha=0.3, color='red')
        ax2.set_title('Drawdown (%)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/equity_drawdown.png', dpi=150)
        plt.close()
        
        logger.info(f"Results plot saved to {output_dir}/equity_drawdown.png")


# Example usage
if __name__ == "__main__":
    from data_handler import DataHandler
    from feature_engineering import FeatureEngineer
    from label_generator import LabelGenerator
    
    # Load data
    handler = DataHandler(data_dir="forex_data")
    df = handler.load_data("EURUSD", "15m")
    
    if df.empty:
        print("No data available.")
        exit()
    
    # Compute features
    engineer = FeatureEngineer()
    df = engineer.compute_features(df)
    
    # Generate simple signals (EMA crossover for demonstration)
    signals = pd.Series(0, index=df.index)
    signals[df['ema50'] > df['ema200']] = 1  # Long when 50 EMA > 200 EMA
    
    # Run backtest
    backtester = Backtester(initial_balance=10000, risk_per_trade=0.01)
    trades_df = backtester.backtest(df, signals, sl_atr_mult=1.0, tp_atr_mult=2.0)
    
    # Calculate metrics
    metrics = backtester.calculate_metrics(trades_df)
    
    print("\nBacktest Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Plot results
    backtester.plot_results()
    
    print(f"\nSample trades:\n{trades_df.head()}")