"""
Feast feature definitions for churn prediction.

Defines entities, feature views, and feature services for the churn prediction model.
"""

from datetime import timedelta
from pathlib import Path
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int32, String

# Get absolute path to the parquet file
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
PARQUET_PATH = str(PROJECT_ROOT / "data" / "processed" / "features" / "customer_features.parquet")

# Define the customer entity
customer = Entity(
    name="customer",
    join_keys=["customer_id"],
    description="Customer identifier"
)

# Define the data source - Parquet file with historical customer data
customer_source = FileSource(
    path=PARQUET_PATH,  # Use absolute path
    timestamp_field="event_timestamp",
)

# Define feature view for customer demographics
customer_demographics = FeatureView(
    name="customer_demographics",
    entities=[customer],
    ttl=timedelta(days=365),  # Features valid for 1 year
    schema=[
        Field(name="gender", dtype=String),
        Field(name="SeniorCitizen", dtype=Int32),
        Field(name="Partner", dtype=String),
        Field(name="Dependents", dtype=String),
    ],
    online=True,
    source=customer_source,
    tags={"category": "demographics"},
)

# Define feature view for account information
customer_account = FeatureView(
    name="customer_account",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="tenure", dtype=Int32),
        Field(name="Contract", dtype=String),
        Field(name="PaperlessBilling", dtype=String),
        Field(name="PaymentMethod", dtype=String),
        Field(name="MonthlyCharges", dtype=Float32),
        Field(name="TotalCharges", dtype=Float32),
    ],
    online=True,
    source=customer_source,
    tags={"category": "account"},
)

# Define feature view for services
customer_services = FeatureView(
    name="customer_services",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="PhoneService", dtype=String),
        Field(name="MultipleLines", dtype=String),
        Field(name="InternetService", dtype=String),
        Field(name="OnlineSecurity", dtype=String),
        Field(name="OnlineBackup", dtype=String),
        Field(name="DeviceProtection", dtype=String),
        Field(name="TechSupport", dtype=String),
        Field(name="StreamingTV", dtype=String),
        Field(name="StreamingMovies", dtype=String),
    ],
    online=True,
    source=customer_source,
    tags={"category": "services"},
)