# app.py """"Customer Churn Prediction — Streamlit Dashboard"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Churn Prediction Dashboard", page_icon="🔮", layout="wide")

# ── Custom CSS ──
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
.main { background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%); }
.stApp { background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%); }
.metric-card { background: linear-gradient(135deg, rgba(108,92,231,0.2), rgba(0,206,201,0.2));
    border: 1px solid rgba(108,92,231,0.3); border-radius: 16px; padding: 20px;
    text-align: center; backdrop-filter: blur(10px); }
.metric-card h3 { color: #a29bfe; font-size: 14px; margin: 0; }
.metric-card h1 { color: #ffffff; font-size: 32px; margin: 5px 0 0 0; }
.section-header { background: linear-gradient(90deg, #6C5CE7, #00CEC9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 28px; font-weight: 700; margin: 30px 0 15px 0; }
.risk-high { color: #ff6b6b; font-weight: 700; font-size: 24px; }
.risk-medium { color: #feca57; font-weight: 700; font-size: 24px; }
.risk-low { color: #00CEC9; font-weight: 700; font-size: 24px; }
div[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a3e 0%, #0f0c29 100%); }
</style>""", unsafe_allow_html=True)

# ── Data Loading ──
@st.cache_data
def load_data():
    raw_path = "data/raw/customer_churn_data.csv"
    proc_path = "data/processed/processed_data.csv"
    raw = pd.read_csv(raw_path) if os.path.exists(raw_path) else None
    proc = pd.read_csv(proc_path) if os.path.exists(proc_path) else None
    return raw, proc

@st.cache_resource
def load_model():
    path = "models/best_model.pkl"
    return joblib.load(path) if os.path.exists(path) else None

raw_df, proc_df = load_data()
model = load_model()

if raw_df is None:
    st.error("⚠️ Run `python main.py` first to generate data and train models!")
    st.stop()

# ── Sidebar ──
st.sidebar.markdown("## 🔮 Navigation")
page = st.sidebar.radio("Go to", ["📊 Overview", "🔍 EDA", "🤖 Model Results",
                                     "🎯 Predict Churn", "💡 Business Insights"])
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Quick Stats")
st.sidebar.metric("Total Customers", f"{len(raw_df):,}")
st.sidebar.metric("Churn Rate", f"{raw_df['Churn'].mean()*100:.1f}%")
st.sidebar.metric("Avg Tenure", f"{raw_df['Tenure_Months'].mean():.0f} months")

# ═══════════════ PAGE 1: OVERVIEW ═══════════════
if page == "📊 Overview":
    st.markdown("# 🔮 Customer Churn Prediction Dashboard")
    st.markdown("*AI-powered customer retention analytics for telecom industry*")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    churned = raw_df["Churn"].sum()
    stayed = len(raw_df) - churned
    with c1:
        st.markdown(f'<div class="metric-card"><h3>TOTAL CUSTOMERS</h3><h1>{len(raw_df):,}</h1></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h3>CHURNED</h3><h1 style="color:#ff6b6b">{churned:,}</h1></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h3>RETAINED</h3><h1 style="color:#00CEC9">{stayed:,}</h1></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h3>CHURN RATE</h3><h1>{raw_df["Churn"].mean()*100:.1f}%</h1></div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(raw_df, names=raw_df["Churn"].map({0:"Stayed",1:"Churned"}),
                     title="Churn Distribution", color_discrete_sequence=["#00CEC9","#ff6b6b"],
                     hole=0.4)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white")
        # Plotly gets use_container_width
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        churn_by_contract = raw_df.groupby("Contract_Type")["Churn"].mean().reset_index()
        churn_by_contract["Churn"] *= 100
        fig = px.bar(churn_by_contract, x="Contract_Type", y="Churn",
                     title="Churn Rate by Contract Type", color="Contract_Type",
                     color_discrete_sequence=["#6C5CE7","#00CEC9","#FD79A8"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white", showlegend=False, yaxis_title="Churn Rate (%)")
        # Plotly gets use_container_width
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Sample Data")
    # DataFrame gets width='stretch'
    st.dataframe(raw_df.head(10), width='stretch')

# ═══════════════ PAGE 2: EDA ═══════════════
elif page == "🔍 EDA":
    st.markdown('<p class="section-header">🔍 Exploratory Data Analysis</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "🔗 Correlations", "📦 Categories", "📈 Trends"])

    with tab1:
        feat = st.selectbox("Select Feature", ["Monthly_Charges","Tenure_Months","Support_Calls",
                            "Data_Usage_GB","Satisfaction_Score","Age","Late_Payments"])
        fig = px.histogram(raw_df, x=feat, color=raw_df["Churn"].map({0:"Stayed",1:"Churned"}),
                           barmode="overlay", color_discrete_sequence=["#00CEC9","#ff6b6b"],
                           opacity=0.7, title=f"Distribution of {feat} by Churn Status")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
        corr = raw_df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlBu_r",
                        title="Feature Correlation Heatmap", aspect="auto")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=700)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        cat = st.selectbox("Category", ["Contract_Type","Payment_Method","Internet_Service","Location","Gender"])
        churn_rate = raw_df.groupby(cat)["Churn"].mean().reset_index()
        churn_rate["Churn"] *= 100
        fig = px.bar(churn_rate, x=cat, y="Churn", color=cat, title=f"Churn Rate by {cat}",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white", yaxis_title="Churn Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = px.scatter(raw_df, x="Tenure_Months", y="Monthly_Charges",
                         color=raw_df["Churn"].map({0:"Stayed",1:"Churned"}),
                         color_discrete_sequence=["#00CEC9","#ff6b6b"],
                         opacity=0.5, title="Tenure vs Monthly Charges")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════ PAGE 3: MODEL RESULTS ═══════════════
elif page == "🤖 Model Results":
    st.markdown('<p class="section-header">🤖 Model Performance</p>', unsafe_allow_html=True)

    info_path = "models/model_info.txt"
    if os.path.exists(info_path):
        with open(info_path) as f:
            lines = f.readlines()
        best_name = lines[0].split(":")[1].strip() if lines else "N/A"
        st.success(f"🏆 **Best Model: {best_name}**")
        metrics = {}
        for line in lines[1:]:
            if ":" in line:
                k, v = line.strip().split(":")
                metrics[k.strip()] = v.strip()
        if metrics:
            cols = st.columns(len(metrics))
            for i, (k, v) in enumerate(metrics.items()):
                cols[i].metric(k, v)

    st.markdown("---")
    # Show saved charts
    chart_files = ["model_comparison.png","confusion_matrices.png","roc_curves.png","feature_importance.png"]
    for cf in chart_files:
        path = f"outputs/{cf}"
        if os.path.exists(path):
            # Image gets width='stretch'
            st.image(path, caption=cf.replace("_"," ").replace(".png","").title(), width='stretch')
            st.markdown("---")

# ═══════════════ PAGE 4: PREDICT ═══════════════
elif page == "🎯 Predict Churn":
    st.markdown('<p class="section-header">🎯 Customer Churn Predictor</p>', unsafe_allow_html=True)
    st.markdown("*Enter customer details below to predict churn risk*")

    if model is None or proc_df is None:
        st.error("⚠️ Run `python main.py` first!")
        st.stop()

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Male","Female"])
            age = st.slider("Age", 18, 70, 35)
            location = st.selectbox("Location", ["Urban","Suburban","Rural"])
            tenure = st.slider("Tenure (months)", 1, 72, 12)
            contract = st.selectbox("Contract Type", ["Month-to-Month","One Year","Two Year"])
        with c2:
            payment = st.selectbox("Payment Method", ["Credit Card","Bank Transfer","Electronic Check","Mailed Check"])
            minutes = st.slider("Monthly Minutes", 50, 1500, 500)
            data_gb = st.slider("Data Usage (GB)", 0.1, 50.0, 5.0)
            support = st.slider("Support Calls", 0, 10, 1)
            internet = st.selectbox("Internet Service", ["Fiber Optic","DSL","No"])
        with c3:
            phone = st.selectbox("Phone Service", ["Yes","No"])
            charges = st.slider("Monthly Charges ($)", 20.0, 150.0, 65.0)
            total_ch = st.slider("Total Charges ($)", 20.0, 10000.0, 780.0)
            late = st.slider("Late Payments", 0, 12, 1)
            satisfaction = st.slider("Satisfaction (1-5)", 1, 5, 3)
            complaints = st.slider("Complaints", 0, 8, 1)

        # Button gets width='stretch'
        submitted = st.form_submit_button("🔮 Predict Churn", width='stretch')

    if submitted:
        # Build feature dict matching processed columns
        feat = {
            "Gender": 1 if gender == "Male" else 0,
            "Age": age, "Tenure_Months": tenure,
            "Monthly_Minutes": minutes, "Data_Usage_GB": data_gb,
            "Support_Calls": support, "Monthly_Charges": charges,
            "Total_Charges": total_ch, "Late_Payments": late,
            "Satisfaction_Score": float(satisfaction), "Complaints": complaints,
            "Phone_Service": 1 if phone == "Yes" else 0,
            "Avg_Charge_Per_Month": round(total_ch / max(tenure, 1), 2),
            "Charge_Deviation": round(charges - 65, 2),
            "Support_Call_Rate": round(support / max(tenure, 1), 4),
            "CLV_Proxy": round(charges * tenure, 2),
            "Late_Payment_Ratio": round(late / max(tenure, 1), 4),
            "Engagement_Score": round((minutes/1500)*0.4 + (data_gb/50)*0.3 + (satisfaction/5)*0.3, 4),
            "High_Risk_Flag": 1 if (charges > 90 and satisfaction <= 2 and complaints >= 2) else 0,
        }
        # One-hot features
        for col_val in [("Location_Suburban", location=="Suburban"), ("Location_Urban", location=="Urban"),
                        ("Contract_Type_One Year", contract=="One Year"), ("Contract_Type_Two Year", contract=="Two Year"),
                        ("Payment_Method_Credit Card", payment=="Credit Card"),
                        ("Payment_Method_Electronic Check", payment=="Electronic Check"),
                        ("Payment_Method_Mailed Check", payment=="Mailed Check"),
                        ("Internet_Service_Fiber Optic", internet=="Fiber Optic"),
                        ("Internet_Service_No", internet=="No"),
                        ("Tenure_Category_12-24m", 12<=tenure<24), ("Tenure_Category_24-48m", 24<=tenure<48),
                        ("Tenure_Category_48-72m", 48<=tenure<=72), ("Tenure_Category_6-12m", 6<=tenure<12)]:
            feat[col_val[0]] = int(col_val[1])

        # Align with model features
        try:
            expected = proc_df.drop("Churn", axis=1).columns.tolist()
            for c in expected:
                if c not in feat:
                    feat[c] = 0
            X_input = pd.DataFrame([feat])[expected]

            from sklearn.preprocessing import StandardScaler
            ref = proc_df.drop("Churn", axis=1)
            scaler = StandardScaler().fit(ref)
            X_scaled = pd.DataFrame(scaler.transform(X_input), columns=expected)

            pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            churn_pct = proba[1] * 100

            st.markdown("---")
            r1, r2, r3 = st.columns(3)
            with r1:
                if pred == 1:
                    st.markdown(f'<div class="metric-card"><h3>PREDICTION</h3><h1 style="color:#ff6b6b">⚠️ WILL CHURN</h1></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-card"><h3>PREDICTION</h3><h1 style="color:#00CEC9">✅ WILL STAY</h1></div>', unsafe_allow_html=True)
            with r2:
                st.markdown(f'<div class="metric-card"><h3>CHURN PROBABILITY</h3><h1>{churn_pct:.1f}%</h1></div>', unsafe_allow_html=True)
            with r3:
                risk = "HIGH" if churn_pct > 70 else "MEDIUM" if churn_pct > 40 else "LOW"
                rclass = "risk-high" if risk=="HIGH" else "risk-medium" if risk=="MEDIUM" else "risk-low"
                emoji = "🔴" if risk=="HIGH" else "🟡" if risk=="MEDIUM" else "🟢"
                st.markdown(f'<div class="metric-card"><h3>RISK LEVEL</h3><h1 class="{rclass}">{emoji} {risk}</h1></div>', unsafe_allow_html=True)

            # Gauge chart
            fig = go.Figure(go.Indicator(mode="gauge+number", value=churn_pct,
                title={"text": "Churn Risk Meter"}, number={"suffix": "%"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#6C5CE7"},
                       "steps": [{"range":[0,30],"color":"#00CEC9"},{"range":[30,60],"color":"#feca57"},
                                 {"range":[60,100],"color":"#ff6b6b"}]}))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=300)
            st.plotly_chart(fig, use_container_width=True)

            # Retention strategies
            st.markdown("### 💡 Recommended Retention Strategies")
            if churn_pct > 70:
                for s in ["🎁 Offer exclusive loyalty discount (20-30%)", "📞 Priority support call within 24hrs",
                          "📦 Free upgrade to premium for 3 months", "💳 Waive pending late fees"]:
                    st.markdown(f"- {s}")
            elif churn_pct > 40:
                for s in ["📧 Send personalized retention email", "🎯 Offer 10-15% discount",
                          "📊 Schedule satisfaction survey"]:
                    st.markdown(f"- {s}")
            else:
                for s in ["⭐ Include in loyalty rewards program", "📱 Send product updates",
                          "🎉 Celebrate customer milestones"]:
                    st.markdown(f"- {s}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ═══════════════ PAGE 5: BUSINESS INSIGHTS ═══════════════
elif page == "💡 Business Insights":
    st.markdown('<p class="section-header">💡 Business Insights & Recommendations</p>', unsafe_allow_html=True)

    # Key findings
    st.markdown("### 🔑 Key Findings")
    m2m_churn = raw_df[raw_df["Contract_Type"]=="Month-to-Month"]["Churn"].mean()*100
    echeck_churn = raw_df[raw_df["Payment_Method"]=="Electronic Check"]["Churn"].mean()*100
    low_sat = raw_df[raw_df["Satisfaction_Score"]<=2]["Churn"].mean()*100 if "Satisfaction_Score" in raw_df.columns else 0
    high_support = raw_df[raw_df["Support_Calls"]>=4]["Churn"].mean()*100

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Month-to-Month Churn", f"{m2m_churn:.1f}%", "Highest risk segment")
    c2.metric("E-Check Churn", f"{echeck_churn:.1f}%", "Payment friction")
    c3.metric("Low Satisfaction Churn", f"{low_sat:.1f}%", "Score ≤ 2")
    c4.metric("High Support Calls", f"{high_support:.1f}%", "≥ 4 calls")

    st.markdown("---")
    st.markdown("### 📊 Revenue Impact Analysis")
    avg_monthly = raw_df["Monthly_Charges"].mean()
    churned_count = raw_df["Churn"].sum()
    annual_loss = churned_count * avg_monthly * 12
    st.metric("Estimated Annual Revenue at Risk", f"${annual_loss:,.0f}")

    col1, col2 = st.columns(2)
    with col1:
        tenure_churn = raw_df.groupby(pd.cut(raw_df["Tenure_Months"], bins=[0,6,12,24,48,72]), observed=False)["Churn"].mean()*100
        fig = px.bar(x=[str(x) for x in tenure_churn.index], y=tenure_churn.values,
                     title="Churn Rate by Tenure Group", labels={"x":"Tenure","y":"Churn Rate (%)"},
                     color_discrete_sequence=["#6C5CE7"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        sat_churn = raw_df.groupby("Satisfaction_Score")["Churn"].mean()*100
        fig = px.line(x=sat_churn.index, y=sat_churn.values, markers=True,
                      title="Churn Rate by Satisfaction Score",
                      labels={"x":"Satisfaction Score","y":"Churn Rate (%)"},
                      color_discrete_sequence=["#FD79A8"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🎯 Actionable Recommendations")
    recs = [
        ("🔒 Convert Month-to-Month contracts", "Offer discounts for annual commitment — could reduce churn by 15-20%"),
        ("💳 Migrate Electronic Check users", "Incentivize credit card/bank transfer — reduces payment friction"),
        ("📞 Proactive support for new customers", "First 6 months are critical — assign dedicated support reps"),
        ("⭐ Satisfaction improvement program", "Target customers with score ≤ 2 with personalized outreach"),
        ("🎁 Loyalty rewards for long-tenure users", "Reward customers who stay 24+ months to prevent late-stage churn"),
    ]
    for title, desc in recs:
        st.markdown(f"**{title}**  \n{desc}")

st.sidebar.markdown("---")
st.sidebar.markdown("*Built with ❤️ using Streamlit & Scikit-learn*")