import streamlit as st
import pandas as pd
from nlp_utils import parse_user_sentence
from model_utils import (
    extract_inputs_for_prediction,
    predict_revpou,
    predict_revpou_with_new_capex,
    model_rlm
)

st.set_page_config(page_title="Capex ROI Predictor", layout="centered")

st.title("💰 Enter an Investment Plan to Simulate Performance")
input_mode = st.radio("Choose input mode", ["Type a sentence", "Manual entry"])

parsed = {}

# Input Mode 1: NLP sentence
if input_mode == "Type a sentence":
    sentence = st.text_input("Describe your investment idea", value="I want to invest $10000 in 2 units on property ATL000 in Jan 2023")
    if sentence:
        parsed = parse_user_sentence(sentence)

# Input Mode 2: Manual entry
else:
    parsed["property_id"] = st.text_input("Property ID", "ATL000")
    parsed["investment_amount"] = st.number_input("Investment Amount", value=10000.0)
    parsed["num_units"] = st.number_input("Number of Units", value=2)
    parsed["capex_date_str"] = st.date_input("Capex Date").strftime("%Y-%m-%d")

# Let user input prediction target date
target_date_input = st.date_input("Prediction Target Date", value=pd.Timestamp("2025-03-31"))
target_date = pd.Timestamp(target_date_input)

# Show summary table of parsed input
if parsed and all(k in parsed for k in ["property_id", "investment_amount", "num_units", "capex_date_str"]):
    st.success("📑 Ready to use this investment plan:")
    summary_df = pd.DataFrame({
        "Field": ["Property ID", "Investment Amount", "Number of Units", "Capex Date", "Prediction Date"],
        "Value": [parsed["property_id"], parsed["investment_amount"], parsed["num_units"], parsed["capex_date_str"], target_date.strftime("%Y-%m-%d")]
    })
    st.table(summary_df)

    if st.button("🚀 Run Prediction"):
        perf_df = st.session_state.get("perf_df")
        demo_df = st.session_state.get("demo_df")

        if perf_df is not None and demo_df is not None:
            try:
                # Prediction logic
                current_features = extract_inputs_for_prediction(perf_df, parsed["property_id"], target_date)
                revpou_current = predict_revpou(model_rlm, current_features)
                revpou_new = predict_revpou_with_new_capex(
                    model_rlm, perf_df, parsed["property_id"],
                    parsed["investment_amount"], parsed["num_units"],
                    parsed["capex_date_str"], target_date
                )

                st.header("📈 Prediction Result")
                st.metric("Current REVPOU_6M", f"${revpou_current:.2f}")
                st.metric("Predicted REVPOU_6M (with CAPEX)", f"${revpou_new:.2f}")
                st.metric("Change in REVPOU", f"${revpou_new - revpou_current:.2f}")

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
        else:
            st.warning("📂 Please upload both performance and demographic data.")

# Sidebar: Upload section
st.sidebar.header("📁 Upload Data")
perf_upload = st.sidebar.file_uploader("Upload Performance Data (.xlsx or .csv)", type=["xlsx", "csv"])
demo_upload = st.sidebar.file_uploader("Upload Demographic Data (.xlsx or .csv)", type=["xlsx", "csv"])

# Handle uploads
if perf_upload and demo_upload:
    perf_df = pd.read_excel(perf_upload) if perf_upload.name.endswith("xlsx") else pd.read_csv(perf_upload)
    demo_df = pd.read_excel(demo_upload) if demo_upload.name.endswith("xlsx") else pd.read_csv(demo_upload)
    st.session_state["perf_df"] = perf_df
    st.session_state["demo_df"] = demo_df
    st.sidebar.success("✅ Files uploaded successfully!")
