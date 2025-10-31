"""
Monitor Module for MT5 Forex Bot
Real-time dashboards, drift detection, and alerting system.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from scipy import stats
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BotMonitor:
    """
    Monitors bot performance, detects anomalies, and sends alerts.
    """
    
    def __init__(self, email_config: dict = None, telegram_config: dict = None):
        """
        Initialize BotMonitor.
        
        Args:
            email_config: Dict with 'smtp_server', 'port', 'sender', 'password', 'recipients'
            telegram_config: Dict with 'bot_token', 'chat_id'
        """
        self.email_config = email_config
        self.telegram_config = telegram_config
        
        # Performance tracking
        self.performance_history = []
        self.feature_distributions = {}
        self.baseline_distributions = {}
        
        # Alert tracking
        self.alert_cooldown = {}
        self.alert_cooldown_minutes = 60
    
    def update_performance(self, metrics: dict):
        """
        Update performance tracking.
        
        Args:
            metrics: Dictionary of performance metrics
        """
        metrics['timestamp'] = datetime.now()
        self.performance_history.append(metrics)
        
        # Keep last 1000 records
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_performance_df(self) -> pd.DataFrame:
        """Get performance history as DataFrame."""
        if len(self.performance_history) == 0:
            return pd.DataFrame()
        
        return pd.DataFrame(self.performance_history)
    
    def plot_realtime_dashboard(self, output_path: str = "dashboard.png"):
        """
        Create real-time performance dashboard.
        
        Args:
            output_path: Path to save dashboard image
        """
        df = self.get_performance_df()
        
        if df.empty:
            logger.warning("No performance data to plot")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Forex Bot Real-Time Dashboard', fontsize=16, fontweight='bold')
        
        # Balance over time
        if 'current_balance' in df.columns:
            axes[0, 0].plot(df['timestamp'], df['current_balance'], linewidth=2)
            axes[0, 0].set_title('Account Balance')
            axes[0, 0].set_ylabel('Balance ($)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Win rate over time (rolling window)
        if 'win_rate' in df.columns:
            axes[0, 1].plot(df['timestamp'], df['win_rate'], linewidth=2, color='green')
            axes[0, 1].axhline(y=50, color='r', linestyle='--', alpha=0.5)
            axes[0, 1].set_title('Win Rate (%)')
            axes[0, 1].set_ylabel('Win Rate')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Daily P&L
        if 'daily_pnl' in df.columns:
            colors = ['green' if x >= 0 else 'red' for x in df['daily_pnl']]
            axes[1, 0].bar(range(len(df)), df['daily_pnl'], color=colors, alpha=0.6)
            axes[1, 0].set_title('Daily P&L')
            axes[1, 0].set_ylabel('P&L ($)')
            axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Drawdown
        if 'current_drawdown_pct' in df.columns:
            axes[1, 1].fill_between(range(len(df)), df['current_drawdown_pct'], 
                                    0, alpha=0.3, color='red')
            axes[1, 1].set_title('Drawdown (%)')
            axes[1, 1].set_ylabel('Drawdown')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Trade count
        if 'total_trades' in df.columns:
            axes[2, 0].plot(df['timestamp'], df['total_trades'], linewidth=2, color='blue')
            axes[2, 0].set_title('Cumulative Trades')
            axes[2, 0].set_ylabel('Trade Count')
            axes[2, 0].grid(True, alpha=0.3)
        
        # Profit factor
        if 'profit_factor' in df.columns:
            # Cap profit factor at 5 for visualization
            pf = df['profit_factor'].clip(upper=5)
            axes[2, 1].plot(df['timestamp'], pf, linewidth=2, color='purple')
            axes[2, 1].axhline(y=1, color='r', linestyle='--', alpha=0.5)
            axes[2, 1].set_title('Profit Factor (capped at 5)')
            axes[2, 1].set_ylabel('Profit Factor')
            axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Dashboard saved to {output_path}")
    
    def detect_feature_drift(self, current_features: pd.DataFrame, 
                            threshold: float = 0.05) -> dict:
        """
        Detect distribution drift in features using KS test.
        
        Args:
            current_features: Recent feature data
            threshold: P-value threshold for drift detection
            
        Returns:
            Dictionary with drift detection results
        """
        if len(self.baseline_distributions) == 0:
            # Set baseline
            for col in current_features.columns:
                self.baseline_distributions[col] = current_features[col].dropna().values
            logger.info("Baseline distributions set")
            return {'drift_detected': False, 'drifted_features': []}
        
        drifted_features = []
        drift_stats = {}
        
        for col in current_features.columns:
            if col not in self.baseline_distributions:
                continue
            
            baseline = self.baseline_distributions[col]
            current = current_features[col].dropna().values
            
            if len(current) < 30 or len(baseline) < 30:
                continue
            
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(baseline, current)
            
            drift_stats[col] = {
                'statistic': statistic,
                'p_value': p_value,
                'drifted': p_value < threshold
            }
            
            if p_value < threshold:
                drifted_features.append(col)
                logger.warning(f"Feature drift detected in {col}: p-value={p_value:.4f}")
        
        result = {
            'drift_detected': len(drifted_features) > 0,
            'drifted_features': drifted_features,
            'drift_stats': drift_stats,
            'timestamp': datetime.now()
        }
        
        return result
    
    def send_email_alert(self, subject: str, message: str):
        """
        Send email alert.
        
        Args:
            subject: Email subject
            message: Email body
        """
        if not self.email_config:
            logger.warning("Email config not provided")
            return
        
        # Check cooldown
        if subject in self.alert_cooldown:
            last_sent = self.alert_cooldown[subject]
            if datetime.now() - last_sent < timedelta(minutes=self.alert_cooldown_minutes):
                logger.info(f"Alert '{subject}' in cooldown period")
                return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender']
            msg['To'] = ', '.join(self.email_config['recipients'])
            msg['Subject'] = f"[Forex Bot Alert] {subject}"
            
            body = f"""
Forex Trading Bot Alert

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{message}

---
This is an automated alert from your Forex trading bot.
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], 
                                 self.email_config['port'])
            server.starttls()
            server.login(self.email_config['sender'], 
                        self.email_config['password'])
            
            server.send_message(msg)
            server.quit()
            
            self.alert_cooldown[subject] = datetime.now()
            logger.info(f"Email alert sent: {subject}")
        
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
    
    def send_telegram_alert(self, message: str):
        """
        Send Telegram alert.
        
        Args:
            message: Alert message
        """
        if not self.telegram_config:
            logger.warning("Telegram config not provided")
            return
        
        try:
            import requests
            
            bot_token = self.telegram_config['bot_token']
            chat_id = self.telegram_config['chat_id']
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            
            text = f"🤖 *Forex Bot Alert*\n\n{message}\n\n_Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
            
            payload = {
                'chat_id': chat_id,
                'text': text,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                logger.info("Telegram alert sent")
            else:
                logger.error(f"Telegram alert failed: {response.text}")
        
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
    
    def check_and_alert(self, risk_summary: dict, drift_result: dict = None):
        """
        Check conditions and send alerts if needed.
        
        Args:
            risk_summary: Risk manager summary
            drift_result: Drift detection results
        """
        alerts = []
        
        # Check daily loss limit
        if risk_summary.get('daily_loss_pct', 0) > 2.5:
            alerts.append(f"⚠️ Daily loss approaching limit: {risk_summary['daily_loss_pct']:.2f}%")
        
        # Check drawdown
        if risk_summary.get('current_drawdown_pct', 0) > 15:
            alerts.append(f"⚠️ High drawdown: {risk_summary['current_drawdown_pct']:.2f}%")
        
        # Check if trading disabled
        if not risk_summary.get('can_trade', True):
            reason = risk_summary.get('status_reason', 'Unknown')
            alerts.append(f"🛑 Trading halted: {reason}")
        
        # Check feature drift
        if drift_result and drift_result.get('drift_detected', False):
            drifted = ', '.join(drift_result['drifted_features'][:5])
            alerts.append(f"📊 Feature drift detected in: {drifted}")
        
        # Send alerts
        if alerts:
            alert_message = '\n'.join(alerts)
            
            # Email
            self.send_email_alert("Trading Bot Alert", alert_message)
            
            # Telegram
            self.send_telegram_alert(alert_message)
    
    def generate_daily_report(self, trades_df: pd.DataFrame, 
                             risk_summary: dict) -> str:
        """
        Generate daily performance report.
        
        Args:
            trades_df: DataFrame with trades
            risk_summary: Risk summary dict
            
        Returns:
            Report as string
        """
        report = f"""
DAILY TRADING REPORT
{'='*50}
Date: {datetime.now().strftime('%Y-%m-%d')}

ACCOUNT SUMMARY
Balance: ${risk_summary.get('current_balance', 0):.2f}
Peak Balance: ${risk_summary.get('peak_balance', 0):.2f}
Drawdown: {risk_summary.get('current_drawdown_pct', 0):.2f}%

DAILY PERFORMANCE
Daily P&L: ${risk_summary.get('daily_pnl', 0):.2f}
Daily Loss %: {risk_summary.get('daily_loss_pct', 0):.2f}%

TRADING ACTIVITY
Total Positions: {risk_summary.get('total_positions', 0)}
Can Trade: {risk_summary.get('can_trade', False)}
Status: {risk_summary.get('status_reason', 'Unknown')}

"""
        
        if not trades_df.empty:
            today_trades = trades_df[
                trades_df['entry_time'].dt.date == datetime.now().date()
            ]
            
            if len(today_trades) > 0:
                wins = (today_trades['profit'] > 0).sum()
                losses = (today_trades['profit'] < 0).sum()
                win_rate = (wins / len(today_trades) * 100) if len(today_trades) > 0 else 0
                
                report += f"""
TODAY'S TRADES
Total Trades: {len(today_trades)}
Wins: {wins}
Losses: {losses}
Win Rate: {win_rate:.1f}%
Total Profit: ${today_trades['profit'].sum():.2f}
"""
        
        report += f"\n{'='*50}\n"
        
        return report


# Example usage
if __name__ == "__main__":
    # Initialize monitor
    monitor = BotMonitor()
    
    # Simulate performance data
    for i in range(100):
        metrics = {
            'current_balance': 10000 + np.random.randn() * 500,
            'win_rate': 50 + np.random.randn() * 10,
            'daily_pnl': np.random.randn() * 100,
            'current_drawdown_pct': abs(np.random.randn() * 5),
            'total_trades': i,
            'profit_factor': 1.5 + np.random.randn() * 0.5
        }
        monitor.update_performance(metrics)
    
    # Plot dashboard
    monitor.plot_realtime_dashboard("test_dashboard.png")
    print("Dashboard created: test_dashboard.png")
    
    # Generate report
    risk_summary = {
        'current_balance': 10000,
        'peak_balance': 10500,
        'current_drawdown_pct': 4.76,
        'daily_pnl': -50,
        'daily_loss_pct': 0.5,
        'total_positions': 2,
        'can_trade': True,
        'status_reason': 'OK'
    }
    
    report = monitor.generate_daily_report(pd.DataFrame(), risk_summary)
    print("\nDaily Report:")
    print(report)