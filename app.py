import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


# 1. Page Configuration

st.set_page_config(page_title="RiskGuard | Credit Predictor", page_icon="🏦", layout="wide")

@st.cache_resource
def load_assets():
    with open("final_rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open("top_15.pkl", "rb") as f:
        top_15 = pickle.load(f)
    with open("top_15_indices.pkl", "rb") as f:
        top_15_indices = pickle.load(f)
    return model, preprocessor, top_15, top_15_indices

model, preprocessor, top_15, top_15_indices = load_assets()

# 2. Sidebar - Detailed Inputs

st.sidebar.header("📋 Customer Credit Profile")

# Basic Info
LIMIT_BAL = st.sidebar.number_input("Credit Limit ($)", 1000, 1000000, 50000)
AGE = st.sidebar.slider("Age", 18, 80, 35)
SEX = st.sidebar.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
EDUCATION = st.sidebar.selectbox("Education", [1, 2, 3, 4], format_func=lambda x: {1:"Grad School", 2:"University", 3:"High School", 4:"Others"}[x])
MARRIAGE = st.sidebar.selectbox("Marital Status", [1, 2, 3], format_func=lambda x: {1:"Married", 2:"Single", 3:"Others"}[x])

# Mapping for Repayment Status
pay_map = {
    -2: "No consumption",
    -1: "Paid duly",
    0: "Paid on time (revolving)",
    1: "1 month delay", 
    2: "2 month delay",
    3: "3 month delay",
    4: "4 month delay"
}

st.sidebar.divider()

# 6 Months Repayment Status
with st.sidebar.expander("🕒 Repayment History (Past 6 Months)"):
    p0 = st.selectbox("Sept (Current)", options=list(pay_map.keys()), format_func=lambda x: pay_map[x], index=2)
    p2 = st.selectbox("Aug", options=list(pay_map.keys()), format_func=lambda x: pay_map[x], index=2)
    p3 = st.selectbox("July", options=list(pay_map.keys()), format_func=lambda x: pay_map[x], index=2)
    p4 = st.selectbox("June", options=list(pay_map.keys()), format_func=lambda x: pay_map[x], index=2)
    p5 = st.selectbox("May", options=list(pay_map.keys()), format_func=lambda x: pay_map[x], index=2)
    p6 = st.selectbox("April", options=list(pay_map.keys()), format_func=lambda x: pay_map[x], index=2)

# 6 Months Bill Amounts
with st.sidebar.expander("💸 Bill Amounts (Past 6 Months)"):
    b1 = st.number_input("Sept Bill ($)", 0, 500000, 20000)
    b2 = st.number_input("Aug Bill ($)", 0, 500000, 18000)
    b3 = st.number_input("July Bill ($)", 0, 500000, 15000)
    b4 = st.number_input("June Bill ($)", 0, 500000, 10000)
    b5 = st.number_input("May Bill ($)", 0, 500000, 5000)
    b6 = st.number_input("April Bill ($)", 0, 500000, 2000)

# 6 Months Payment Amounts
with st.sidebar.expander("💰 Payments Made (Past 6 Months)"):
    pa1 = st.number_input("Sept Pay ($)", 0, 500000, 5000)
    pa2 = st.number_input("Aug Pay ($)", 0, 500000, 5000)
    pa3 = st.number_input("July Pay ($)", 0, 500000, 5000)
    pa4 = st.number_input("June Pay ($)", 0, 500000, 5000)
    pa5 = st.number_input("May Pay ($)", 0, 500000, 5000)
    pa6 = st.number_input("April Pay ($)", 0, 500000, 5000)

# Threshold Slider 
threshold = st.sidebar.slider("Risk Threshold", 0.1, 0.9, 0.5)



# 3. Logic & Feature Engineering

# Create raw dataframe
input_data = {
    "LIMIT_BAL": LIMIT_BAL, "AGE": AGE, "SEX": SEX, "EDUCATION": EDUCATION, "MARRIAGE": MARRIAGE,
    "PAY_0": p0, "PAY_2": p2, "PAY_3": p3, "PAY_4": p4, "PAY_5": p5, "PAY_6": p6,
    "BILL_AMT1": b1, "BILL_AMT2": b2, "BILL_AMT3": b3, "BILL_AMT4": b4, "BILL_AMT5": b5, "BILL_AMT6": b6,
    "PAY_AMT1": pa1, "PAY_AMT2": pa2, "PAY_AMT3": pa3, "PAY_AMT4": pa4, "PAY_AMT5": pa5, "PAY_AMT6": pa6
}

# Calculated Features (Summary)
bills = [b1, b2, b3, b4, b5, b6]
pays = [pa1, pa2, pa3, pa4, pa5, pa6]

input_data["avg_bill_amt"] = np.mean(bills)
input_data["total_bill_amt"] = np.sum(bills)
input_data["avg_pay_amt"] = np.mean(pays)
input_data["total_pay_amt"] = np.sum(pays)
input_data["utilisation_ratio"] = (
    input_data["total_bill_amt"] / (LIMIT_BAL + 1)
)


input_df = pd.DataFrame([input_data])


# 4. Main View
st.title("🏦 Credit Default Analysis Portal")

st.markdown(
    "This application estimates the **probability of credit default** using customer "
    "demographics and recent financial behaviour."
)

predict_btn = st.button("🔍 Analyze Risk", type="primary")

if predict_btn:
    # Preprocessing & Prediction
    # Ensure all features required by preprocessor are present
    for col in preprocessor.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[preprocessor.feature_names_in_]
    X_processed = preprocessor.transform(input_df)

    X_selected = X_processed[:, top_15_indices]
    prob = model.predict_proba(X_selected)[0][1]

    # Result Card
    st.markdown(f"""
        <div style="background-color: white; padding: 40px; border-radius: 15px; border-left: 10px solid {'#ef4444' if prob >= threshold else '#10b981'}; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);">
            <h1 style="margin:0; font-size: 50px;">{prob:.1%}</h1>
            <p style="color: gray;">Predicted Probability of Default</p>
        </div>
    """, unsafe_allow_html=True)

    # Risk Meter & Label
    st.write("")
    if prob >= threshold:
        st.error(f"🚨 **High Risk**: Probability exceeds your threshold of {threshold:.0%}")
    elif prob >= (threshold - 0.2):
        st.warning(f"🟡 **Medium Risk**: Probability is approaching the limit.")
    else:
        st.success(f"✅ **Low Risk**: Account is within safe limits.")

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Risk Factors")
        feat_imp = pd.DataFrame({"Feature": top_15, "Score": model.feature_importances_}).sort_values("Score")
        fig, ax = plt.subplots()
        ax.barh(feat_imp["Feature"], feat_imp["Score"], color="#3b82f6")
        st.pyplot(fig)
        
    with col2:
        st.subheader("Financial Summary")
        st.write(f"**Total Billed (6m):** ${input_data['total_bill_amt']:,.2f}")
        st.write(f"**Total Paid (6m):** ${input_data['total_pay_amt']:,.2f}")
        st.write(f"**Credit Utilisation:** {input_data['utilisation_ratio']:.1%}")

else:
    st.info("Fill in the customer data on the left and click 'Analyze Risk' to begin.")