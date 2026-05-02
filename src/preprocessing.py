"""
=============================================================================
Customer Churn Prediction — Data Preprocessing & Feature Engineering
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import warnings

warnings.filterwarnings("ignore")


class ChurnPreprocessor:
    """End-to-end preprocessing pipeline for customer churn data."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        print(f"✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        missing_before = df.isnull().sum().sum()
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        print(f"✅ Missing values: {missing_before} → {df.isnull().sum().sum()}")
        return df
    
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        columns = ["Monthly_Charges", "Total_Charges", "Monthly_Minutes", 
                    "Data_Usage_GB", "Support_Calls", "Late_Payments"]
        columns = [c for c in columns if c in df.columns]
        for col in columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            n = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
            if n > 0:
                df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
                print(f"   📌 {col}: capped {n} outliers")
        print("✅ Outlier treatment complete")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Avg_Charge_Per_Month"] = (df["Total_Charges"] / df["Tenure_Months"].clip(lower=1)).round(2)
        df["Charge_Deviation"] = (df["Monthly_Charges"] - df["Monthly_Charges"].mean()).round(2)
        df["Support_Call_Rate"] = (df["Support_Calls"] / df["Tenure_Months"].clip(lower=1)).round(4)
        df["CLV_Proxy"] = (df["Monthly_Charges"] * df["Tenure_Months"]).round(2)
        df["Late_Payment_Ratio"] = (df["Late_Payments"] / df["Tenure_Months"].clip(lower=1)).round(4)
        df["Engagement_Score"] = (
            (df["Monthly_Minutes"] / df["Monthly_Minutes"].max()) * 0.4 +
            (df["Data_Usage_GB"] / df["Data_Usage_GB"].max()) * 0.3 +
            (df["Satisfaction_Score"] / 5.0) * 0.3
        ).round(4)
        df["High_Risk_Flag"] = (
            (df["Monthly_Charges"] > df["Monthly_Charges"].quantile(0.75)) &
            (df["Satisfaction_Score"] <= 2) & (df["Complaints"] >= 2)
        ).astype(int)
        df["Tenure_Category"] = pd.cut(
            df["Tenure_Months"], bins=[0, 6, 12, 24, 48, 72],
            labels=["0-6m", "6-12m", "12-24m", "24-48m", "48-72m"]
        ).astype(str)
        print(f"✅ Feature engineering complete — 8 new features created")
        return df
    
    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "CustomerID" in df.columns:
            df = df.drop("CustomerID", axis=1)
        for col in ["Gender", "Phone_Service"]:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        onehot_cols = [c for c in ["Location", "Contract_Type", "Payment_Method",
                       "Internet_Service", "Tenure_Category"] if c in df.columns]
        df = pd.get_dummies(df, columns=onehot_cols, drop_first=True, dtype=int)
        print(f"✅ Encoding complete")
        return df
    
    def scale_features(self, X_train, X_test):
        self.feature_names = X_train.columns.tolist()
        X_train_s = pd.DataFrame(self.scaler.fit_transform(X_train), 
                                  columns=self.feature_names, index=X_train.index)
        X_test_s = pd.DataFrame(self.scaler.transform(X_test),
                                 columns=self.feature_names, index=X_test.index)
        print("✅ Feature scaling complete")
        return X_train_s, X_test_s
    
    def fit_transform(self, df, test_size=0.2, random_state=42):
        print("\n" + "=" * 60)
        print("  🔧 PREPROCESSING PIPELINE")
        print("=" * 60)
        df = self.handle_missing_values(df)
        df = self.remove_outliers(df)
        df = self.engineer_features(df)
        df = self.encode_features(df)
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv("data/processed/processed_data.csv", index=False)
        X = df.drop("Churn", axis=1)
        y = df["Churn"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)
        print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Features: {X_train.shape[1]}")
        X_train, X_test = self.scale_features(X_train, X_test)
        print("=" * 60)
        return X_train, X_test, y_train, y_test, df
