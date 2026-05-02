"""
=============================================================================
Customer Churn Prediction — Predictor Module
=============================================================================
Load trained model and predict churn for individual customers.
=============================================================================
"""

import joblib
import numpy as np
import pandas as pd


class ChurnPredictor:
    """Predict churn for new/individual customers using the saved model."""
    
    def __init__(self, model_path="models/best_model.pkl"):
        self.model = joblib.load(model_path)
        print(f"✅ Model loaded from: {model_path}")
    
    def predict_single(self, features: dict) -> dict:
        """
        Predict churn for a single customer.
        
        Parameters
        ----------
        features : dict
            Dictionary of customer features (already preprocessed/encoded).
        
        Returns
        -------
        dict with prediction, probability, and risk level.
        """
        X = pd.DataFrame([features])
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        churn_prob = probability[1] * 100
        risk_level = (
            "🔴 HIGH RISK" if churn_prob > 70 else
            "🟡 MEDIUM RISK" if churn_prob > 40 else
            "🟢 LOW RISK"
        )
        
        return {
            "prediction": int(prediction),
            "churn_label": "WILL CHURN" if prediction == 1 else "WILL STAY",
            "churn_probability": round(churn_prob, 2),
            "stay_probability": round(probability[0] * 100, 2),
            "risk_level": risk_level
        }
    
    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict churn for a batch of customers."""
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        result = pd.DataFrame({
            "Churn_Prediction": predictions,
            "Churn_Probability": (probabilities * 100).round(2),
            "Risk_Level": pd.cut(
                probabilities, bins=[0, 0.3, 0.6, 1.0],
                labels=["Low", "Medium", "High"]
            )
        })
        return result
    
    @staticmethod
    def get_retention_strategy(churn_prob: float) -> list:
        """Suggest retention strategies based on churn probability."""
        strategies = []
        if churn_prob > 70:
            strategies = [
                "🎁 Offer exclusive loyalty discount (20-30%)",
                "📞 Priority customer support call within 24 hours",
                "📦 Free upgrade to premium plan for 3 months",
                "💳 Waive any pending late payment fees"
            ]
        elif churn_prob > 40:
            strategies = [
                "📧 Send personalized retention email",
                "🎯 Offer 10-15% discount on next billing",
                "📊 Schedule quarterly satisfaction survey",
                "🔔 Enable proactive support notifications"
            ]
        else:
            strategies = [
                "⭐ Include in loyalty rewards program",
                "📱 Send product updates and new feature alerts",
                "🎉 Celebrate customer milestones (anniversary, usage)"
            ]
        return strategies
