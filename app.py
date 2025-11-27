import os
import streamlit as st
import pandas as pd
import json
from langchain_groq import ChatGroq

# =========================
# Page Setup
# =========================
st.set_page_config(page_title="CredLense Malaysia", layout="centered")
st.title("CredLense Malaysia – Hybrid AI + Rule Loan Approval")
st.markdown("### Malaysia's Smartest Loan Approval AI | Credit + Application Analysis")

# =========================
# Initialize Groq LLaMA
# =========================
api_key = os.environ.get("GROK_API_KEY")  # Hugging Face Secret

if not api_key:
    raise ValueError("GROQ_API_KEY not found! Make sure you added it to Secrets.")

os.environ["GROQ_API_KEY"] = api_key
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# =========================
# Session State Initialization
# =========================
if "run_prediction" not in st.session_state:
    st.session_state.run_prediction = False
if "new_prediction" not in st.session_state:
    st.session_state.new_prediction = False
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# Rule-based functions
# =========================
def credit_score_component(credit):
    if credit >= 800: return 95
    if credit >= 740: return 85
    if credit >= 700: return 75
    if credit >= 650: return 60
    if credit >= 600: return 45
    if credit >= 550: return 30
    return 15

def dti_component(dti):
    if dti <= 20: return 95
    if dti <= 30: return 80
    if dti <= 40: return 60
    if dti <= 50: return 40
    if dti <= 60: return 25
    return 10

def lti_component(loan, income):
    if income <= 0: return 0
    ratio = loan / income
    if ratio <= 0.2: return 95
    if ratio <= 0.4: return 80
    if ratio <= 0.6: return 60
    if ratio <= 0.8: return 40
    return 15

def income_floor_component(income):
    if income >= 80000: return 90
    if income >= 50000: return 75
    if income >= 30000: return 60
    if income >= 20000: return 45
    return 25

def employment_component(status):
    if status == "Employed": return 95
    if status == "Self-Employed": return 75
    if status == "Student": return 50
    if status == "Retired": return 40
    if status == "Unemployed": return 20
    return 30

def rule_based_probability(income, credit, loan, dti, employment):
    w_credit = 0.3
    w_dti = 0.25
    w_lti = 0.2
    w_income = 0.15
    w_employment = 0.1

    c1 = credit_score_component(credit)
    c2 = dti_component(dti)
    c3 = lti_component(loan, income)
    c4 = income_floor_component(income)
    c5 = employment_component(employment)

    probability = (
        w_credit * c1 + 
        w_dti * c2 + 
        w_lti * c3 + 
        w_income * c4 +
        w_employment * c5
    )

    return max(0, min(100, probability)), {
        "credit_component": c1,
        "dti_component": c2,
        "lti_component": c3,
        "income_component": c4,
        "employment_component": c5
    }

def risk_tier(prob):
    if prob >= 80: return "Low"
    if prob >= 60: return "Medium"
    if prob >= 40: return "High"
    return "Very High"

# =========================
# Sidebar Inputs
# =========================
with st.sidebar:
    st.header("Applicant Info")
    income = st.number_input("Annual Income (RM)", 20000, 500000, 80000)
    credit = st.slider("Credit Score", 300, 850, 680)
    loan = st.number_input("Loan Amount (RM)", 1000, 200000, 25000)
    dti = st.slider("DTI (%)", 0, 100, 42)
    tenure = st.slider("Tenure (years)", 1, 30, 5)
    employment = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed", "Student", "Retired"])
    reason = st.text_area("Application Reason", height=120, value="I need money urgently for hospital bills, my daughter is very sick")
    st.markdown("---")
    st.caption("Powered by Groq LLaMA (llama-3.1-8b-instant)")

# =========================
# Buttons
# =========================
col_buttons = st.columns(2)
with col_buttons[0]:
    if st.button("Start New Prediction"):
        st.session_state.run_prediction = False
        st.session_state.new_prediction = True
with col_buttons[1]:
    if st.button("Run Hybrid AI Approval"):
        st.session_state.run_prediction = True
        st.session_state.new_prediction = False

# =========================
# Show rule-based metrics
# =========================
rule_prob, rule_details = rule_based_probability(income, credit, loan, dti, employment)
tier = risk_tier(rule_prob)
col1, col2, col3 = st.columns(3)
col1.metric("Rule-Based Probability", f"{rule_prob:.0f}%")
col2.metric("Risk Tier", tier)
col3.metric("Loan / Income Ratio", f"{loan/income:.2f}x")

# =========================
# Run prediction only when user clicks button
# =========================
if st.session_state.run_prediction:
    with st.spinner("Evaluating with Groq LLaMA..."):
        # Build prompt
        prompt = f"""
You are a Malaysian bank credit officer.
Annual income: RM{income}
Credit score: {credit}
Loan amount: RM{loan}
DTI: {dti}%
Tenure: {tenure} years
Employment Status: {employment}
Applicant reason: "{reason}"
Return JSON only in English:
{{
    "ai_probability": integer 0-100,
    "ai_reason": "one short sentence",
    "ai_key_factors": ["factor1", "factor2"]
}}
        """
        # Call Groq LLaMA
        try:
            resp = llm.invoke(prompt)
            raw = resp.content.strip()
            start = raw.find("{")
            end = raw.rfind("}") + 1
            parsed = json.loads(raw[start:end])
            ai_prob = parsed.get("ai_probability", 50)
            ai_reason = parsed.get("ai_reason", "")
            ai_factors = parsed.get("ai_key_factors", [])
        except Exception:
            ai_prob = 50
            ai_reason = "AI parsing failed"
            ai_factors = []

        # Hybrid probability
        combined_prob = 0.6 * rule_prob + 0.4 * ai_prob
        combined_prob = max(0, min(100, combined_prob))
        final_tier = risk_tier(combined_prob)
        decision = "Approved" if combined_prob >= 60 else "Declined"

        # Weakest factor
        comps = {
            "Credit score": rule_details["credit_component"],
            "DTI": rule_details["dti_component"],
            "Loan-to-Income": rule_details["lti_component"],
            "Income": rule_details["income_component"],
            "Employment Status": rule_details["employment_component"]
        }
        worst_factor = min(comps.items(), key=lambda x: x[1])[0]

        summary_reason = (
            f"Declined due to weak {worst_factor.lower()}, AI noted: {ai_reason}"
            if decision == "Declined"
            else f"Approved — strong enough rule score and AI agrees: {ai_reason}"
        )

        # Display results
        st.subheader("Final Approval Result")
        st.markdown(f"**Decision:** {decision}")
        st.markdown(f"**Final Probability:** {combined_prob:.0f}%")
        st.markdown(f"**Risk Tier:** {final_tier}")
        st.markdown(f"**Reason:** {summary_reason}")
        st.markdown("### AI Key Factors")
        st.write(ai_factors)

        # Repayment calculator
        st.subheader("Repayment Estimator")
        rate = st.number_input("Interest Rate (%)", 0.1, 20.0, 6.0)
        monthly_rate = rate / 100 / 12
        n = tenure * 12
        if monthly_rate == 0:
            monthly_payment = loan / n
        else:
            monthly_payment = loan * (monthly_rate * (1 + monthly_rate)**n) / ((1 + monthly_rate)**n - 1)
        st.write(f"**Estimated Monthly Payment:** RM {monthly_payment:,.2f}")

        # Save session history
        st.session_state.history.append({
            "Income": income,
            "Credit": credit,
            "Loan": loan,
            "DTI": dti,
            "Tenure": tenure,
            "Employment": employment,
            "Rule_Prob": rule_prob,
            "AI_Prob": ai_prob,
            "Combined": combined_prob,
            "Decision": decision
        })

# =========================
# Sidebar session history
# =========================
st.sidebar.markdown("### Session History")
if st.session_state.history:
    st.sidebar.dataframe(pd.DataFrame(st.session_state.history))