"""
=============================================================================
Customer Churn Prediction — Main Pipeline
=============================================================================
Orchestrates the full ML pipeline:
  1. Generate synthetic data
  2. Preprocess & engineer features
  3. Generate EDA visualizations
  4. Train & evaluate models
  5. Generate evaluation charts
  6. Save best model
=============================================================================
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import generate_customer_data, save_dataset
from src.preprocessing import ChurnPreprocessor
from src.model_training import ChurnModelTrainer
from src.visualization import generate_all_eda, generate_all_evaluation


def main():
    """Run the complete Customer Churn Prediction pipeline."""
    
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " 🚀 CUSTOMER CHURN PREDICTION PIPELINE ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
    # ── Phase 1: Generate Synthetic Data ─────────────────────────
    print("\n" + "━" * 60)
    print("  📦 PHASE 1: Data Generation")
    print("━" * 60)
    df = generate_customer_data(n_customers=10000)
    save_dataset(df)
    
    # ── Phase 2: EDA Visualizations (raw data) ───────────────────
    print("\n" + "━" * 60)
    print("  📊 PHASE 2: Exploratory Data Analysis")
    print("━" * 60)
    generate_all_eda(df)
    
    # ── Phase 3: Preprocessing & Feature Engineering ─────────────
    print("\n" + "━" * 60)
    print("  🔧 PHASE 3: Preprocessing & Feature Engineering")
    print("━" * 60)
    preprocessor = ChurnPreprocessor()
    X_train, X_test, y_train, y_test, processed_df = preprocessor.fit_transform(df)
    
    # ── Phase 4: Model Training ──────────────────────────────────
    print("\n" + "━" * 60)
    print("  🤖 PHASE 4: Model Training")
    print("━" * 60)
    trainer = ChurnModelTrainer()
    trainer.train_all(X_train, y_train)
    
    # ── Phase 5: Model Evaluation ────────────────────────────────
    print("\n" + "━" * 60)
    print("  📈 PHASE 5: Model Evaluation")
    print("━" * 60)
    comparison_df = trainer.evaluate_all(X_test, y_test)
    print("\n📊 Model Comparison:")
    print(comparison_df.to_string(index=False))
    
    # ── Phase 6: Feature Importance & Charts ─────────────────────
    print("\n" + "━" * 60)
    print("  📊 PHASE 6: Evaluation Visualizations")
    print("━" * 60)
    fi_df = trainer.get_feature_importance(preprocessor.feature_names)
    generate_all_evaluation(trainer, X_test, y_test, comparison_df, fi_df)
    
    # ── Phase 7: Save Best Model ─────────────────────────────────
    print("\n" + "━" * 60)
    print("  💾 PHASE 7: Save Best Model")
    print("━" * 60)
    trainer.save_best_model()
    
    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " ✅ PIPELINE COMPLETE ".center(58) + "║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  📦 Dataset:       10,000 customers generated".ljust(59) + "║")
    print(f"║  📊 Features:      {X_train.shape[1]} features engineered".ljust(59) + "║")
    print(f"║  🤖 Models:        4 classifiers trained".ljust(59) + "║")
    print(f"║  🏆 Best Model:    {trainer.best_model_name}".ljust(59) + "║")
    best_f1 = trainer.results[trainer.best_model_name]["F1 Score"]
    print(f"║  📈 Best F1:       {best_f1}%".ljust(59) + "║")
    print(f"║  💾 Model saved:   models/best_model.pkl".ljust(59) + "║")
    print(f"║  📊 Charts saved:  outputs/".ljust(59) + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n🚀 Run 'streamlit run app.py' to launch the dashboard!")


if __name__ == "__main__":
    main()
