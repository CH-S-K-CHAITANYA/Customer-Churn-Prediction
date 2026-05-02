"""
=============================================================================
Customer Churn Prediction — Synthetic Data Generator
=============================================================================
Generates a realistic synthetic dataset of 10,000 telecom customers with 
churn patterns that mirror real-world behavior.

Churn Logic:
  - Higher charges + low tenure → more churn
  - Many support calls → more churn
  - Month-to-month contracts → more churn
  - Low satisfaction → more churn
  - Late payments → more churn
=============================================================================
"""

import numpy as np
import pandas as pd
import os


def generate_customer_data(n_customers: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic customer churn dataset.
    
    Parameters
    ----------
    n_customers : int
        Number of customer records to generate (default: 10,000).
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with customer features and churn label.
    """
    np.random.seed(seed)
    
    # ── Customer IDs ─────────────────────────────────────────────
    customer_ids = [f"CUST-{str(i).zfill(5)}" for i in range(1, n_customers + 1)]
    
    # ── Demographics ─────────────────────────────────────────────
    gender = np.random.choice(["Male", "Female"], size=n_customers)
    age = np.random.randint(18, 70, size=n_customers)
    location = np.random.choice(
        ["Urban", "Suburban", "Rural"], 
        size=n_customers, 
        p=[0.50, 0.35, 0.15]
    )
    
    # ── Account Details ──────────────────────────────────────────
    tenure = np.random.exponential(scale=24, size=n_customers).clip(1, 72).astype(int)
    contract_type = np.random.choice(
        ["Month-to-Month", "One Year", "Two Year"], 
        size=n_customers, 
        p=[0.50, 0.30, 0.20]
    )
    payment_method = np.random.choice(
        ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"],
        size=n_customers,
        p=[0.30, 0.25, 0.30, 0.15]
    )
    
    # ── Usage Metrics ────────────────────────────────────────────
    monthly_minutes = np.random.normal(loc=500, scale=200, size=n_customers).clip(50, 1500).astype(int)
    data_usage_gb = np.random.exponential(scale=5, size=n_customers).clip(0.1, 50).round(1)
    support_calls = np.random.poisson(lam=1.5, size=n_customers).clip(0, 10)
    
    # ── Billing ──────────────────────────────────────────────────
    monthly_charges = np.random.normal(loc=65, scale=25, size=n_customers).clip(20, 150).round(2)
    total_charges = (monthly_charges * tenure + np.random.normal(0, 50, n_customers)).clip(20, 10000).round(2)
    late_payments = np.random.poisson(lam=1, size=n_customers).clip(0, 12)
    
    # ── Satisfaction ─────────────────────────────────────────────
    satisfaction_score = np.random.randint(1, 6, size=n_customers)  # 1-5
    complaints = np.random.poisson(lam=0.8, size=n_customers).clip(0, 8)
    
    # ── Internet & Phone Service ─────────────────────────────────
    internet_service = np.random.choice(
        ["Fiber Optic", "DSL", "No"], 
        size=n_customers, 
        p=[0.45, 0.35, 0.20]
    )
    phone_service = np.random.choice(["Yes", "No"], size=n_customers, p=[0.85, 0.15])
    
    # ══════════════════════════════════════════════════════════════
    # CHURN SIMULATION — Realistic probability based on features
    # ══════════════════════════════════════════════════════════════
    churn_probability = np.zeros(n_customers)
    
    # Base churn rate ~20%
    churn_probability += 0.15
    
    # Contract type influence (biggest factor)
    churn_probability += np.where(
        np.array(contract_type) == "Month-to-Month", 0.20, 
        np.where(np.array(contract_type) == "One Year", 0.05, -0.10)
    )
    
    # Tenure influence (new customers churn more)
    churn_probability += np.where(tenure < 6, 0.15, 
                          np.where(tenure < 12, 0.05,
                          np.where(tenure > 48, -0.15, -0.05)))
    
    # Support calls (frustration → churn)
    churn_probability += (support_calls - 1.5) * 0.04
    
    # Monthly charges (high cost → churn)
    churn_probability += (monthly_charges - 65) / 65 * 0.10
    
    # Satisfaction score (low satisfaction → churn)
    churn_probability += (3 - satisfaction_score) * 0.06
    
    # Late payments (payment issues → churn)
    churn_probability += late_payments * 0.03
    
    # Complaints (dissatisfaction → churn)
    churn_probability += complaints * 0.04
    
    # Electronic check users churn more (industry pattern)
    churn_probability += np.where(
        np.array(payment_method) == "Electronic Check", 0.08, 0.0
    )
    
    # Age influence (younger users churn slightly more)
    churn_probability += np.where(age < 30, 0.05, np.where(age > 55, -0.03, 0.0))
    
    # Clip probabilities to valid range
    churn_probability = np.clip(churn_probability, 0.02, 0.95)
    
    # Generate churn labels
    churn = (np.random.random(n_customers) < churn_probability).astype(int)
    
    # ── Introduce some missing values (realistic) ────────────────
    # About 2% missing in total_charges, 1% in satisfaction
    missing_idx_charges = np.random.choice(n_customers, size=int(n_customers * 0.02), replace=False)
    missing_idx_satisfaction = np.random.choice(n_customers, size=int(n_customers * 0.01), replace=False)
    
    total_charges_series = pd.Series(total_charges)
    satisfaction_series = pd.Series(satisfaction_score, dtype=float)
    
    total_charges_series.iloc[missing_idx_charges] = np.nan
    satisfaction_series.iloc[missing_idx_satisfaction] = np.nan
    
    # ── Build DataFrame ──────────────────────────────────────────
    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Gender": gender,
        "Age": age,
        "Location": location,
        "Tenure_Months": tenure,
        "Contract_Type": contract_type,
        "Payment_Method": payment_method,
        "Monthly_Minutes": monthly_minutes,
        "Data_Usage_GB": data_usage_gb,
        "Support_Calls": support_calls,
        "Monthly_Charges": monthly_charges,
        "Total_Charges": total_charges_series,
        "Late_Payments": late_payments,
        "Satisfaction_Score": satisfaction_series,
        "Complaints": complaints,
        "Internet_Service": internet_service,
        "Phone_Service": phone_service,
        "Churn": churn
    })
    
    return df


def save_dataset(df: pd.DataFrame, output_dir: str = "data/raw") -> str:
    """
    Save generated dataset to CSV.
    
    Parameters
    ----------
    df : pd.DataFrame
        The generated customer DataFrame.
    output_dir : str
        Directory to save the CSV file.
    
    Returns
    -------
    str
        Path to the saved CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "customer_churn_data.csv")
    df.to_csv(filepath, index=False)
    print(f"✅ Dataset saved: {filepath}")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Churn Rate: {df['Churn'].mean()*100:.1f}%")
    return filepath


if __name__ == "__main__":
    print("=" * 60)
    print("  Generating Synthetic Customer Churn Dataset...")
    print("=" * 60)
    df = generate_customer_data()
    save_dataset(df)
    print("\n📊 Sample Records:")
    print(df.head())
    print(f"\n📈 Dataset Info:")
    print(f"   Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
