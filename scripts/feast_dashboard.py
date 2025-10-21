"""
Simple Streamlit dashboard to visualize Feast features.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from feast import FeatureStore


# Initialize Feast
@st.cache_resource
def get_feast_store():
    feast_repo = Path(__file__).parent.parent / "src" / "features" / "feature_repo"
    return FeatureStore(repo_path=str(feast_repo))

store = get_feast_store()

st.title("Feast Feature Store Viewer")

# Sidebar
st.sidebar.header("Options")
view_mode = st.sidebar.radio("View Mode", ["Offline Store", "Online Store", "Registry Info"])

if view_mode == "Registry Info":
    st.header("Registered Features")
    
    # Show entities
    st.subheader("Entities")
    entities = store.list_entities()
    for entity in entities:
        st.write(f"- **{entity.name}**: {entity.description}")
    
    # Show feature views
    st.subheader("Feature Views")
    feature_views = store.list_feature_views()
    for fv in feature_views:
        with st.expander(f"🔹 {fv.name}"):
            st.write(f"**TTL:** {fv.ttl}")
            st.write(f"**Online:** {fv.online}")
            st.write("**Features:**")
            for field in fv.schema:
                st.write(f"  - {field.name} ({field.dtype})")

elif view_mode == "Offline Store":
    st.header(" Offline Store (Parquet)")
    
    # Load parquet file
    parquet_path = Path(__file__).parent.parent / "data" / "processed" / "features" / "customer_features.parquet"
    df = pd.read_parquet(parquet_path)
    
    st.write(f"**Total Records:** {len(df):,}")
    st.write(f"**Columns:** {len(df.columns)}")
    
    # Show sample
    st.subheader("Sample Data")
    st.dataframe(df.head(20))
    
    # Show statistics
    st.subheader("Statistics")
    st.dataframe(df.describe())

elif view_mode == "Online Store":
    st.header(" Online Store (Redis)")
    
    # Input customer ID
    customer_id = st.text_input("Customer ID", "7590-VHVEG")
    
    if st.button("Fetch Features"):
        try:
            # Get online features
            features = store.get_online_features(
                entity_rows=[{"customer_id": customer_id}],
                features=[
                    "customer_demographics:gender",
                    "customer_demographics:SeniorCitizen",
                    "customer_demographics:Partner",
                    "customer_demographics:Dependents",
                    "customer_account:tenure",
                    "customer_account:Contract",
                    "customer_account:PaperlessBilling",
                    "customer_account:PaymentMethod",
                    "customer_account:MonthlyCharges",
                    "customer_account:TotalCharges",
                    "customer_services:PhoneService",
                    "customer_services:MultipleLines",
                    "customer_services:InternetService",
                    "customer_services:OnlineSecurity",
                    "customer_services:OnlineBackup",
                    "customer_services:DeviceProtection",
                    "customer_services:TechSupport",
                    "customer_services:StreamingTV",
                    "customer_services:StreamingMovies",
                ]
            ).to_df()
            
            st.success(f"Found features for {customer_id}")
            st.dataframe(features.T)  # Transpose for better view
            
        except Exception as e:
            st.error(f"Error: {str(e)}")