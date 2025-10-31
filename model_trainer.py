"""Model training and optimization module."""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
from pathlib import Path

class ModelTrainer:
    def __init__(self, model_dir: str = "models"):
        """Initialize ModelTrainer."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model = None
        self.scaler = None
        self.features = None

    def prepare_data(self, df: pd.DataFrame, feature_cols: list, target_col: str):
        """Prepare data for training."""
        features = df[feature_cols].copy()
        target = df[target_col].copy()
        
        # Remove rows with NaN values
        valid_idx = features.dropna().index
        X = features.loc[valid_idx].values
        y = target.loc[valid_idx].values
        
        self.features = feature_cols
        return X, y, feature_cols

    def normalize_features(self, X_train: np.ndarray, X_test: np.ndarray):
        """Normalize features using StandardScaler."""
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def train_xgboost(self, X_train, y_train, X_test, y_test, params=None):
        """Train XGBoost model."""
        if params is None:
            params = {
                'max_depth': 4,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        
        self.model = xgb.XGBClassifier(**params, random_state=42)
        self.model.fit(X_train, y_train)

    def optimize_hyperparameters(self, X_train, y_train, n_trials=50):
        """Optimize hyperparameters using Optuna."""
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 2, 8),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
            }
            
            model = xgb.XGBClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            
            return -model.score(X_train, y_train)
        
        study = optuna.create_study()
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_pred = self.model.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }

    def plot_feature_importance(self, top_n=15):
        """Plot feature importance."""
        importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance['feature'], importance['importance'])
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(self.model_dir / 'feature_importance.png')
        plt.close()

    def save_model(self, model_name: str):
        """Save model and associated objects."""
        joblib.dump(self.model, self.model_dir / f"{model_name}.pkl")
        joblib.dump(self.scaler, self.model_dir / f"{model_name}_scaler.pkl")
        joblib.dump(self.features, self.model_dir / f"{model_name}_features.pkl")

    def load_model(self, model_name: str):
        """Load model and associated objects."""
        self.model = joblib.load(self.model_dir / f"{model_name}.pkl")
        self.scaler = joblib.load(self.model_dir / f"{model_name}_scaler.pkl")
        self.features = joblib.load(self.model_dir / f"{model_name}_features.pkl")