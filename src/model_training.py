"""
=============================================================================
Customer Churn Prediction — Model Training & Evaluation
=============================================================================
Trains 4 classifiers, evaluates them, and selects the best model.
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              confusion_matrix)
import joblib
import os
import warnings

warnings.filterwarnings("ignore")


class ChurnModelTrainer:
    """Train, evaluate, and compare multiple ML models for churn prediction."""
    
    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=42, class_weight="balanced"
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=42,
                class_weight="balanced", n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, eval_metric="logloss",
                scale_pos_weight=3, use_label_encoder=False
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                random_state=42
            )
        }
        self.results = {}
        self.best_model_name = None
        self.best_model = None
    
    def train_all(self, X_train, y_train):
        """Train all models on the training data."""
        print("\n" + "=" * 60)
        print("  🤖 MODEL TRAINING")
        print("=" * 60)
        for name, model in self.models.items():
            print(f"\n   Training {name}...", end=" ")
            model.fit(X_train, y_train)
            print("✅ Done")
        print("\n" + "=" * 60)
    
    def evaluate_all(self, X_test, y_test):
        """Evaluate all models and return comparison DataFrame."""
        print("\n" + "=" * 60)
        print("  📊 MODEL EVALUATION")
        print("=" * 60)
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                "Accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
                "Precision": round(precision_score(y_test, y_pred) * 100, 2),
                "Recall": round(recall_score(y_test, y_pred) * 100, 2),
                "F1 Score": round(f1_score(y_test, y_pred) * 100, 2),
                "ROC-AUC": round(roc_auc_score(y_test, y_proba) * 100, 2),
                "y_pred": y_pred,
                "y_proba": y_proba
            }
            self.results[name] = metrics
            
            print(f"\n   📌 {name}:")
            print(f"      Accuracy:  {metrics['Accuracy']}%")
            print(f"      Precision: {metrics['Precision']}%")
            print(f"      Recall:    {metrics['Recall']}%")
            print(f"      F1 Score:  {metrics['F1 Score']}%")
            print(f"      ROC-AUC:   {metrics['ROC-AUC']}%")
        
        # Select best model by F1 Score
        self.best_model_name = max(
            self.results, key=lambda k: self.results[k]["F1 Score"]
        )
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n🏆 Best Model: {self.best_model_name} "
              f"(F1={self.results[self.best_model_name]['F1 Score']}%)")
        print("=" * 60)
        
        return self.get_comparison_df()
    
    def get_comparison_df(self):
        """Return a DataFrame comparing all model metrics."""
        rows = []
        for name, metrics in self.results.items():
            rows.append({
                "Model": name,
                "Accuracy (%)": metrics["Accuracy"],
                "Precision (%)": metrics["Precision"],
                "Recall (%)": metrics["Recall"],
                "F1 Score (%)": metrics["F1 Score"],
                "ROC-AUC (%)": metrics["ROC-AUC"]
            })
        return pd.DataFrame(rows).sort_values("F1 Score (%)", ascending=False)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from the best model."""
        model = self.best_model
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            return None
        
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(15)
        return fi_df
    
    def save_best_model(self, output_dir="models"):
        """Save the best model to disk."""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, "best_model.pkl")
        joblib.dump(self.best_model, filepath)
        print(f"💾 Best model saved: {filepath}")
        
        # Also save model info
        info = {
            "model_name": self.best_model_name,
            "metrics": {k: v for k, v in self.results[self.best_model_name].items() 
                       if k not in ["y_pred", "y_proba"]}
        }
        info_path = os.path.join(output_dir, "model_info.txt")
        with open(info_path, "w") as f:
            f.write(f"Best Model: {info['model_name']}\n")
            for k, v in info["metrics"].items():
                f.write(f"{k}: {v}%\n")
        
        return filepath
