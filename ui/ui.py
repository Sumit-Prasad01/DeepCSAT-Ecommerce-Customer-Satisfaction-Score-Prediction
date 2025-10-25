import streamlit as st
import requests
import json

FASTAPI_URL = "http://localhost:8000/predict" 

POSSIBLE_VALUES = {
    "channel_name": ["Outcall", "Inbound", "Chat", "Email"],
    "category": ["Product Queries", "Order Related", "Returns", "Technical Support"],
    "sub_category": ["Life Insurance", "Product Specific Information", "Installation/demo", "Reverse Pickup Enquiry"],
    "agent_name": ["Richard Buchanan", "Vicki Collins", "Duane Norman", "Patrick Flores"],
    "supervisor": ["Mason Gupta", "Dylan Kim", "Jackson Park", "Olivia Wang"],
    "manager": ["Jennifer Nguyen", "Michael Lee", "William Kim", "John Smith"],
    "tenure_bucket": ["On Job Training", "0-30", "31-60", "61-90", ">90"],
    "agent_shift": ["Morning", "Evening", "Night"],
    "customer_remarks": ["No remarks", "Great service", "Issue resolved", "Need follow up"],
    "customer_city": ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata"],
    "product_category": ["Electronics", "Insurance", "Home Appliances", "Fashion"],
}

st.set_page_config(
    page_title="CSAT Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("Customer Satisfaction (CSAT) Prediction App")
    st.markdown("Enter the transaction details to predict the CSAT score (1-5).")

    with st.sidebar:
        st.header("Transaction Details")

        input_data = {}

        st.subheader("Categorical Features")
        input_data["channel_name"] = st.selectbox("Channel Name", POSSIBLE_VALUES["channel_name"])
        input_data["category"] = st.selectbox("Category", POSSIBLE_VALUES["category"])
        input_data["sub_category"] = st.selectbox("Sub-category", POSSIBLE_VALUES["sub_category"])
        input_data["agent_name"] = st.selectbox("Agent Name", POSSIBLE_VALUES["agent_name"])
        input_data["supervisor"] = st.selectbox("Supervisor", POSSIBLE_VALUES["supervisor"])
        input_data["manager"] = st.selectbox("Manager", POSSIBLE_VALUES["manager"])
        input_data["agent_shift"] = st.selectbox("Agent Shift", POSSIBLE_VALUES["agent_shift"])
        input_data["customer_remarks"] = st.selectbox("Customer Remarks", POSSIBLE_VALUES["customer_remarks"])
        input_data["customer_city"] = st.selectbox("Customer City", POSSIBLE_VALUES["customer_city"])
        input_data["product_category"] = st.selectbox("Product Category", POSSIBLE_VALUES["product_category"])
        
        st.subheader("Tenure")
        input_data["tenure_bucket"] = st.selectbox("Tenure Bucket", POSSIBLE_VALUES["tenure_bucket"])

        st.subheader("Numerical Features")
        input_data["item_price"] = st.number_input("Item Price", min_value=0.0, value=15000.0, step=100.0, format="%.2f")
        input_data["connected_handling_time"] = st.number_input("Handling Time (Minutes)", min_value=0.0, value=120.0, step=1.0, format="%.2f")

    st.header("Prediction Result")
    
    if st.button("Predict CSAT Score", type="primary"):
        with st.spinner('Sending request to FastAPI...'):
            try:
                headers = {"Content-Type": "application/json"}
                response = requests.post(FASTAPI_URL, headers=headers, data=json.dumps(input_data))

                if response.status_code == 200:
                    result = response.json()
                    predicted_score = result.get("predicted_csat_score")
                    confidence = result.get("confidence") * 100

                    st.success("Prediction successful!")
                    st.metric(label="Predicted CSAT Score (1-5)", value=predicted_score)
                    st.info(f"Confidence: **{confidence:.2f}%**")
                    
                    if predicted_score >= 4:
                        st.balloons()
                        st.markdown(f"### 🎉 High Predicted Satisfaction!")
                    elif predicted_score <= 2:
                        st.warning(f"### ⚠️ Low Predicted Satisfaction. Review this transaction.")

                else:
                    error_detail = response.json().get("detail", "Unknown error occurred.")
                    st.error(f"Error connecting to FastAPI: Status {response.status_code}")
                    st.json(error_detail)

            except requests.exceptions.ConnectionError:
                st.error(f"Failed to connect to FastAPI at {FASTAPI_URL}. Please ensure your FastAPI backend is running.")
            except Exception as e:
                st.exception(e)
                st.error("An unexpected error occurred during prediction.")

if __name__ == '__main__':
    main()