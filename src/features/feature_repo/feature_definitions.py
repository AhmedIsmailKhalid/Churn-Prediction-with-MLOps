"""
Feast feature definitions for churn prediction.

Defines entities, feature views, and feature services for the churn prediction model.
"""

from datetime import timedelta
import os

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int32, String

# Use container path for Docker, fallback to relative path for local
if os.path.exists("/app/data/processed/features/customer_features.parquet"):
    PARQUET_PATH = "/app/data/processed/features/customer_features.parquet"
else:
    PARQUET_PATH = "data/processed/features/customer_features.parquet"

# Define the customer entity
customer = Entity(
    name="customer_id",  # Changed from "customer" to match data column
    description="Customer identifier"
)


# Define the data source - Parquet file with historical customer data
customer_source = FileSource(
    path=PARQUET_PATH,
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