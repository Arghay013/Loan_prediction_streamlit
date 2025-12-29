import streamlit as st
import pandas as pd
import pickle

st.title("🏦 Loan Approval Prediction App")

# Load trained model
model = pickle.load(open("model/loan_model.pkl", "rb"))

st.subheader("Enter Applicant Details")

def user_input_form():
    person_age = st.number_input("Age", 18, 100, 25)
    person_gender = st.selectbox("Gender", ["male","female"])
    person_education = st.selectbox("Education",
        ["High School","Bachelor","Master","PhD"])
    person_income = st.number_input("Annual Income", 0, 500000, 30000)
    person_emp_exp = st.number_input("Employment Experience (years)", 0, 40, 2)
    person_home_ownership = st.selectbox("Home Ownership",
        ["RENT","OWN","MORTGAGE","OTHER"])
    loan_amnt = st.number_input("Loan Amount", 500, 50000, 5000)
    loan_intent = st.selectbox("Loan Intent",
        ["PERSONAL","EDUCATION","MEDICAL","VENTURE","DEBTCONSOLIDATION"])
    loan_int_rate = st.number_input("Interest Rate", 1.0, 40.0, 12.0)
    loan_percent_income = st.number_input("Percent of Income", 0.0, 1.0, 0.2)
    cb_person_cred_hist_length = st.number_input("Credit History Length", 0, 50, 3)
    credit_score = st.number_input("Credit Score", 300, 900, 600)
    previous_loan_defaults_on_file = st.selectbox("Previous Defaults", ["Yes","No"])

    data = pd.DataFrame({
        "person_age":[person_age],
        "person_gender":[person_gender],
        "person_education":[person_education],
        "person_income":[person_income],
        "person_emp_exp":[person_emp_exp],
        "person_home_ownership":[person_home_ownership],
        "loan_amnt":[loan_amnt],
        "loan_intent":[loan_intent],
        "loan_int_rate":[loan_int_rate],
        "loan_percent_income":[loan_percent_income],
        "cb_person_cred_hist_length":[cb_person_cred_hist_length],
        "credit_score":[credit_score],
        "previous_loan_defaults_on_file":[previous_loan_defaults_on_file],
    })

    return data

input_df = user_input_form()

if st.button("Predict Loan Status"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
