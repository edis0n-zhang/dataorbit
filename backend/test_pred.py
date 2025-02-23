import pandas as pd
import numpy as np
import pickle
import os

# Get all condition columns from the CSV header
cols_path = os.path.join(os.path.dirname(__file__), "cols.csv")
with open(cols_path, "r") as f:
    header = f.readline().strip().split(",")

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "xgboost_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load the input data
input_path = os.path.join(os.path.dirname(__file__), "input.csv")
df = pd.read_csv(input_path)

# Reorder columns to match cols.csv order
df = df[header]

# Define numerical columns that need to be converted to Float64
numerical_cols = [
    "Body Weight",
    "Body Height",
    "Body Mass Index",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Heart rate",
    "Respiratory rate",
    "AGE",
]

# Convert numerical columns to Float64 and handle None values
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].astype("Float64")

# Convert categorical columns and handle None values
categorical_cols = [col for col in df.columns if col not in numerical_cols]
for col in categorical_cols:
    df[col] = df[col].fillna(0)
    df[col] = df[col].astype("category")

# Make prediction
try:
    prediction = model.predict(df)
    print("fullpred: ", prediction)
    print(f"Input data:")
    print(df[numerical_cols].to_string())
    print("\nActive conditions:", [col for col in header if df[col].iloc[0] == 1])
    print(f"\nPredicted cost: ${prediction[0]:,.2f}")
except Exception as e:
    print(f"Prediction failed: {str(e)}")
    # Print additional debugging information
    print("\nDataFrame info:")
    print(df.info())
    print("\nDataFrame columns:")
    print(df.columns.tolist())
