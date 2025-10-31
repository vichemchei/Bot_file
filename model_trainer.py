"""
Model Training and Optimization Module for MT5 Forex Bot
Handles XGBoost training, hyperparameter optimization, and evaluation.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    brier_score_loss, precision_recall_curve, roc_curve
)
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains and optimizes XGBoost models for forex trading predictions.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize ModelTrainer.
        
        Args:
            model_dir: Directory to save models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.features = None
        self.training_history = {}
        
        logger.info(f"ModelTrainer initialized. Model directory: {self.model_dir}")
    
    def prepare_data(self, df: pd.DataFrame, feature_cols: list, 
                    target_col: str = 'label'):
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Target column name
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info("Preparing training data...")
        
        # Extract features and target
        features = df[feature_cols].copy()
        target = df[target_col].copy()
        
        # Remove rows with NaN values
        valid_idx = features.dropna().index.intersection(target.dropna().index)
        X = features.loc[valid_idx].values
        y = target.loc[valid_idx].values
        
        self.features = feature_cols
        
        logger.info(f"Data prepared: {len(X)} samples, {len(feature_cols)} features")
        logger.info(f"Class distribution: {np.bincount(y.astype(int))}")
        
        return X, y, feature_cols
    
    def normalize_features(self, X_train: np.ndarray, X_test: np.ndarray = None):
        """
        Normalize features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            
        Returns:
            Scaled features (X_train_scaled, X_test_scaled) or just X_train_scaled
        """
        logger.info("Normalizing features...")
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def train_xgboost(self, X_train, y_train, X_test=None, y_test=None, params=None):
        """
        Train XGBoost model with proper configuration.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Validation features (optional)
            y_test: Validation labels (optional)
            params: Model hyperparameters
        """
        logger.info("Training XGBoost model...")
        
        # Calculate class weights for imbalanced data
        class_counts = np.bincount(y_train.astype(int))
        scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0
        
        logger.info(f"Class imbalance ratio: {scale_pos_weight:.2f}")
        
        # Default parameters
        if params is None:
            params = {
                'max_depth': 4,
                'learning_rate': 0.05,
                'n_estimators': 300,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'min_child_weight': 3
            }
        
        # Add class weight and other settings
        params['scale_pos_weight'] = scale_pos_weight
        params['random_state'] = 42
        params['eval_metric'] = 'auc'
        params['use_label_encoder'] = False
        
        # Setup early stopping if validation data provided
        eval_set = None
        if X_test is not None and y_test is not None:
            eval_set = [(X_train, y_train), (X_test, y_test)]
            params['early_stopping_rounds'] = 50
            params['verbose'] = 100
        
        # Train model
        self.model = xgb.XGBClassifier(**params)
        
        if eval_set:
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            
            # Get best iteration
            best_iteration = self.model.best_iteration
            logger.info(f"Best iteration: {best_iteration}")
            self.training_history['best_iteration'] = best_iteration
        else:
            self.model.fit(X_train, y_train)
        
        logger.info("Model training complete")
        
        # Store training info
        self.training_history['params'] = params
        self.training_history['n_features'] = X_train.shape[1]
        self.training_history['n_samples'] = X_train.shape[0]
        self.training_history['timestamp'] = datetime.now().isoformat()
    
    def optimize_hyperparameters(self, X_train, y_train, n_trials=50, 
                                 cv_splits=5, timeout=3600):
        """
        Optimize hyperparameters using Optuna with proper time-series CV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_trials: Number of optimization trials
            cv_splits: Number of cross-validation splits
            timeout: Maximum optimization time in seconds
            
        Returns:
            Best hyperparameters
        """
        logger.info(f"Starting hyperparameter optimization ({n_trials} trials)...")
        
        # Calculate class weight
        class_counts = np.bincount(y_train.astype(int))
        scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0
        
        def objective(trial):
            """Optuna objective function with time-series cross-validation."""
            
            # Suggest hyperparameters (using updated Optuna API)
            params = {
                'max_depth': trial.suggest_int('max_depth', 2, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'eval_metric': 'auc',
                'use_label_encoder': False
            }
            
            # Time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_splits)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=30,
                    verbose=False
                )
                
                # Predict probabilities
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # Calculate AUC
                try:
                    auc = roc_auc_score(y_val, y_pred_proba)
                    cv_scores.append(auc)
                except:
                    cv_scores.append(0.5)
            
            # Return negative mean AUC (Optuna minimizes)
            mean_auc = np.mean(cv_scores)
            return -mean_auc
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
        )
        
        # Optimize
        study.optimize(
            objective, 
            n_trials=n_trials, 
            timeout=timeout,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        best_score = -study.best_value
        
        logger.info(f"Optimization complete. Best AUC: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Store optimization history
        self.training_history['optimization'] = {
            'best_score': best_score,
            'best_params': best_params,
            'n_trials': len(study.trials)
        }
        
        return best_params
    
    def evaluate_model(self, X_test, y_test, plot_dir=None):
        """
        Comprehensive model evaluation.
        
        Args:
            X_test: Test features
            y_test: Test labels
            plot_dir: Directory to save evaluation plots
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model...")
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_xgboost() first.")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'brier_score': brier_score_loss(y_test, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['confusion_matrix'] = cm
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Profit expectancy (assuming 2:1 R:R)
        win_rate = tp / (tp + fp) if (tp + fp) > 0 else 0
        loss_rate = 1 - win_rate
        avg_win = 2.0  # 2R
        avg_loss = 1.0  # 1R
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        
        metrics['win_rate'] = win_rate
        metrics['expectancy'] = expectancy
        
        # Log results
        logger.info("="*60)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
        logger.info(f"Brier:     {metrics['brier_score']:.4f}")
        logger.info(f"Win Rate:  {metrics['win_rate']:.2%}")
        logger.info(f"Expectancy: {metrics['expectancy']:.3f}R")
        logger.info("="*60)
        
        # Generate evaluation plots
        if plot_dir:
            self._plot_evaluation(y_test, y_pred, y_pred_proba, cm, plot_dir)
        
        return metrics
    
    def _plot_evaluation(self, y_test, y_pred, y_pred_proba, cm, plot_dir):
        """Generate evaluation plots."""
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[0, 1].set_title('ROC Curve', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        axes[1, 0].plot(recall, precision, linewidth=2)
        axes[1, 0].set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Prediction Distribution
        axes[1, 1].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.5, 
                       label='Class 0', color='red')
        axes[1, 1].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.5, 
                       label='Class 1', color='green')
        axes[1, 1].set_title('Prediction Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'model_evaluation.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Evaluation plots saved to {plot_dir}")
    
    def plot_feature_importance(self, top_n=20, output_path=None):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to display
            output_path: Path to save plot (optional)
        """
        if self.model is None:
            logger.warning("Model not trained. Cannot plot feature importance.")
            return
        
        if self.features is None:
            logger.warning("Feature names not available.")
            return
        
        logger.info("Plotting feature importance...")
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, max(6, top_n * 0.3)))
        plt.barh(range(len(importance)), importance['importance'])
        plt.yticks(range(len(importance)), importance['feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        # Save or display
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {output_path}")
        else:
            plt.savefig(self.model_dir / 'feature_importance.png', dpi=150)
            logger.info(f"Feature importance saved to {self.model_dir}")
        
        plt.close()
    
    def save_model(self, model_name: str):
        """
        Save model and associated objects with version tracking.
        
        Args:
            model_name: Base name for saved files
        """
        if self.model is None:
            logger.error("No model to save")
            return
        
        # Add timestamp to model name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        versioned_name = f"{model_name}_{timestamp}"
        
        # Save model
        model_path = self.model_dir / f"{versioned_name}.pkl"
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save scaler
        if self.scaler:
            scaler_path = self.model_dir / f"{versioned_name}_scaler.pkl"
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
        
        # Save features
        if self.features:
            features_path = self.model_dir / f"{versioned_name}_features.pkl"
            joblib.dump(self.features, features_path)
            logger.info(f"Features saved to {features_path}")
        
        # Save training history
        history_path = self.model_dir / f"{versioned_name}_history.pkl"
        joblib.dump(self.training_history, history_path)
        logger.info(f"Training history saved to {history_path}")
        
        # Also save with non-versioned name for easy loading
        joblib.dump(self.model, self.model_dir / f"{model_name}.pkl")
        if self.scaler:
            joblib.dump(self.scaler, self.model_dir / f"{model_name}_scaler.pkl")
        if self.features:
            joblib.dump(self.features, self.model_dir / f"{model_name}_features.pkl")
        
        logger.info(f"Model saved with both versioned and current names")
    
    def load_model(self, model_name: str):
        """
        Load model and associated objects.
        
        Args:
            model_name: Base name of saved model
        """
        try:
            model_path = self.model_dir / f"{model_name}.pkl"
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            
            scaler_path = self.model_dir / f"{model_name}_scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Scaler loaded from {scaler_path}")
            
            features_path = self.model_dir / f"{model_name}_features.pkl"
            if features_path.exists():
                self.features = joblib.load(features_path)
                logger.info(f"Features loaded from {features_path}")
            
            history_path = self.model_dir / f"{model_name}_history.pkl"
            if history_path.exists():
                self.training_history = joblib.load(history_path)
                logger.info(f"Training history loaded")
            
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


# Example usage
if __name__ == "__main__":
    from data_handler import DataHandler
    from feature_engineering import FeatureEngineer
    from label_generator import LabelGenerator
    
    logger.info("Testing ModelTrainer...")
    
    # Load data
    handler = DataHandler(data_dir="forex_data")
    df = handler.load_data("EURUSD", "15m")
    
    if df.empty:
        logger.error("No data available")
        exit()
    
    # Compute features
    engineer = FeatureEngineer()
    df = engineer.compute_features(df)
    
    # Generate labels
    label_gen = LabelGenerator()
    df = label_gen.generate_labels(df, label_type='classification')
    
    # Prepare data
    trainer = ModelTrainer()
    feature_cols = engineer.get_feature_list()
    X, y, features = trainer.prepare_data(df, feature_cols, target_col='label')
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Normalize
    X_train_scaled, X_test_scaled = trainer.normalize_features(X_train, X_test)
    
    # Train
    logger.info("Training model...")
    trainer.train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Evaluate
    metrics = trainer.evaluate_model(X_test_scaled, y_test, plot_dir="model_evaluation")
    
    # Feature importance
    trainer.plot_feature_importance(top_n=15)
    
    # Save
    trainer.save_model("test_model")
    
    logger.info("Testing complete!")