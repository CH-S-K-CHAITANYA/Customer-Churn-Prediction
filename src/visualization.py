"""
=============================================================================
Customer Churn Prediction — Visualization Module
=============================================================================
Generate all EDA and model evaluation charts.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os
import warnings

warnings.filterwarnings("ignore")

# Style configuration
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = ["#6C5CE7", "#00CEC9", "#FD79A8", "#FDCB6E", "#0984E3", "#E17055"]
OUTPUT_DIR = "outputs"


def setup_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_churn_distribution(df, save=True):
    """Pie + bar chart of churn distribution."""
    setup_output_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    counts = df["Churn"].value_counts()
    labels = ["Stayed", "Churned"]
    colors = [COLORS[1], COLORS[2]]
    
    # Pie chart
    axes[0].pie(counts, labels=labels, autopct="%1.1f%%", colors=colors,
                startangle=90, explode=(0, 0.05), shadow=True,
                textprops={"fontsize": 12, "fontweight": "bold"})
    axes[0].set_title("Churn Distribution", fontsize=14, fontweight="bold")
    
    # Bar chart
    bars = axes[1].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=2)
    axes[1].set_title("Customer Count by Churn Status", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Count", fontsize=12)
    for bar, val in zip(bars, counts.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                     str(val), ha="center", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUT_DIR}/churn_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved: churn_distribution.png")


def plot_correlation_heatmap(df, save=True):
    """Correlation heatmap of numeric features."""
    setup_output_dir()
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                center=0, ax=ax, square=True, linewidths=0.5,
                annot_kws={"size": 7})
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=20)
    
    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved: correlation_heatmap.png")


def plot_feature_distributions(df, save=True):
    """Distribution plots for key numeric features by churn status."""
    setup_output_dir()
    features = ["Monthly_Charges", "Tenure_Months", "Support_Calls",
                 "Satisfaction_Score", "Data_Usage_GB", "Late_Payments"]
    features = [f for f in features if f in df.columns]
    
    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()
    
    for i, feat in enumerate(features):
        for churn_val, color, label in [(0, COLORS[1], "Stayed"), (1, COLORS[2], "Churned")]:
            subset = df[df["Churn"] == churn_val][feat].dropna()
            axes[i].hist(subset, bins=30, alpha=0.6, color=color, label=label, edgecolor="white")
        axes[i].set_title(feat, fontsize=13, fontweight="bold")
        axes[i].legend()
        axes[i].set_xlabel("")
    
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle("Feature Distributions by Churn Status", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUT_DIR}/feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved: feature_distributions.png")


def plot_churn_by_category(df, save=True):
    """Churn rate by categorical features."""
    setup_output_dir()
    cat_features = ["Contract_Type", "Payment_Method", "Internet_Service", "Gender", "Location"]
    cat_features = [f for f in cat_features if f in df.columns]
    
    n_cols = min(3, len(cat_features))
    n_rows = (len(cat_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    if len(cat_features) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, feat in enumerate(cat_features):
        churn_rate = df.groupby(feat)["Churn"].mean() * 100
        bars = axes[i].bar(churn_rate.index, churn_rate.values, 
                           color=[COLORS[j % len(COLORS)] for j in range(len(churn_rate))],
                           edgecolor="white", linewidth=2)
        axes[i].set_title(f"Churn Rate by {feat}", fontsize=13, fontweight="bold")
        axes[i].set_ylabel("Churn Rate (%)")
        axes[i].tick_params(axis="x", rotation=45)
        for bar, val in zip(bars, churn_rate.values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
    
    for j in range(len(cat_features), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUT_DIR}/churn_by_category.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved: churn_by_category.png")


def plot_confusion_matrices(trainer, X_test, y_test, save=True):
    """Confusion matrices for all models."""
    setup_output_dir()
    model_names = list(trainer.models.keys())
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5))
    if len(model_names) == 1:
        axes = [axes]
    
    for i, name in enumerate(model_names):
        y_pred = trainer.models[name].predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["Stayed", "Churned"],
                    yticklabels=["Stayed", "Churned"],
                    annot_kws={"size": 14})
        axes[i].set_title(name, fontsize=13, fontweight="bold")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
    
    plt.suptitle("Confusion Matrices", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUT_DIR}/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved: confusion_matrices.png")


def plot_roc_curves(trainer, X_test, y_test, save=True):
    """ROC curves for all models."""
    setup_output_dir()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, (name, model) in enumerate(trainer.models.items()):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[i], linewidth=2.5,
                label=f"{name} (AUC={roc_auc:.3f})")
    
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curves — All Models", fontsize=16, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUT_DIR}/roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved: roc_curves.png")


def plot_model_comparison(comparison_df, save=True):
    """Bar chart comparing all model metrics."""
    setup_output_dir()
    fig, ax = plt.subplots(figsize=(14, 7))
    
    metrics = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)", "ROC-AUC (%)"]
    x = np.arange(len(comparison_df))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        bars = ax.bar(x + i * width, comparison_df[metric], width,
                      label=metric, color=COLORS[i], edgecolor="white", linewidth=1)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}", ha="center", fontsize=8, fontweight="bold")
    
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(comparison_df["Model"], fontsize=11, fontweight="bold")
    ax.set_ylabel("Score (%)", fontsize=13)
    ax.set_title("Model Performance Comparison", fontsize=16, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved: model_comparison.png")


def plot_feature_importance(fi_df, save=True):
    """Horizontal bar chart of top feature importances."""
    setup_output_dir()
    fig, ax = plt.subplots(figsize=(12, 8))
    
    fi_sorted = fi_df.sort_values("Importance", ascending=True)
    colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(fi_sorted)))
    
    ax.barh(fi_sorted["Feature"], fi_sorted["Importance"],
            color=colors_gradient, edgecolor="white", linewidth=1)
    ax.set_xlabel("Importance", fontsize=13)
    ax.set_title("Top 15 Feature Importances (Best Model)", fontsize=16, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved: feature_importance.png")


def generate_all_eda(df):
    """Run all EDA visualizations."""
    print("\n" + "=" * 60)
    print("  📊 GENERATING EDA VISUALIZATIONS")
    print("=" * 60)
    plot_churn_distribution(df)
    plot_correlation_heatmap(df)
    plot_feature_distributions(df)
    plot_churn_by_category(df)
    print("=" * 60)


def generate_all_evaluation(trainer, X_test, y_test, comparison_df, fi_df):
    """Run all model evaluation visualizations."""
    print("\n" + "=" * 60)
    print("  📊 GENERATING EVALUATION CHARTS")
    print("=" * 60)
    plot_confusion_matrices(trainer, X_test, y_test)
    plot_roc_curves(trainer, X_test, y_test)
    plot_model_comparison(comparison_df)
    if fi_df is not None:
        plot_feature_importance(fi_df)
    print("=" * 60)
