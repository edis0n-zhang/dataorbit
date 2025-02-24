from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from groq import Groq
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Get all condition columns from the CSV header (excluding the ones we already have)
cols_path = os.path.join(os.path.dirname(__file__), "cols.csv")
with open(cols_path, "r") as f:
    header = f.readline().strip().split(",")

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "xgboost_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load the column dtypes
dtype_path = os.path.join(os.path.dirname(__file__), "column_dtypes.pkl")
with open(dtype_path, "rb") as f:
    column_dtypes = pickle.load(f)

scaler_path = os.path.join(os.path.dirname(__file__), "robust_scaler.pkl")
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

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


def match_conditions(user_conditions, available_conditions):
    """
    Match user input conditions with available conditions using Groq API
    Returns a dictionary mapping user conditions to matched conditions
    """
    print("match0")
    user_conditions_str = str(user_conditions)
    print("user conditions: ", user_conditions_str)
    available_conditions_str = str(available_conditions)
    prompt = (
        f"Given a list of user-provided medical conditions and a list of standardized medical conditions, "
        f"match each user condition to the most similar standardized condition.\n\n"
        f"User conditions: {user_conditions_str}\n"
        f"Standardized conditions: {available_conditions_str}\n\n"
        f"The JSON schema should include:\n"
        f'{{"matched_conditions": []}}\n\n'
        f"Example output format:\n"
        f'{{"matched_conditions": ["condition1", "condition2"]}}\n\n'
        f"Only return the JSON object, no other text."
    )
    print("prompt: ", prompt)
    print("match1")
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )
    print("match2")

    try:
        print("trying")
        output = completion.choices[0].message.content
        output_dict = json.loads(output)

        print("reach here")
        # output is in json, convert it to a dictionary
        output = output_dict["matched_conditions"]
        print(output)
        return output
    except:
        print("failed match?")
        return []


@app.route("/api/submit-health-data", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        print(data)

        # Create a dictionary for the input data with None handling
        input_data = {
            "RACE": data.get("race"),
            "ETHNICITY": data.get("ethnicity"),
            "GENDER": data.get("gender"),
            "Body Height": data.get("height"),
            "Body Weight": data.get("weight"),
            "Body Mass Index": data.get("bmi"),
            "Diastolic Blood Pressure": data.get("diastolicBP"),
            "Systolic Blood Pressure": data.get("systolicBP"),
            "Heart rate": data.get("heartRate"),
            "Respiratory rate": data.get("respiratoryRate"),
            "AGE": data.get("age"),
        }

        # Add all medical conditions as binary columns (0 or 1)
        conditions = data.get("conditions", [])
        condition_cols = [
            col
            for col in header
            if col
            not in [
                "RACE",
                "ETHNICITY",
                "GENDER",
                "TOTAL_FUTURE_COST",
                "TOTAL_FUTURE_COVERAGE",
                "FUTURE_OUT_OF_POCKET",
                "Body Height",
                "Body Weight",
                "Body Mass Index",
                "Diastolic Blood Pressure",
                "Systolic Blood Pressure",
                "Heart rate",
                "Respiratory rate",
                "AGE",
            ]
        ]

        matched_conditions = []

        if len(conditions) > 0:
            # Match conditions using Groq API
            print("begin match")
            matched_conditions = match_conditions(conditions, condition_cols)
            print("Condition matches:", matched_conditions)  # Debug print

        # Initialize all conditions to 0
        for col in condition_cols:
            input_data[col] = 0

        for matched_condition in matched_conditions:
            if matched_condition in condition_cols:
                print(matched_condition)
                input_data[matched_condition] = 1

        print("init data")

        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        for column in df.columns:
            if column in column_dtypes:
                df[column] = df[column].astype(column_dtypes[column])
        print("conversion successful")

        print("loaded scaler")

        scaled_cols = [
            "Body Weight",
            "Body Height",
            "Body Mass Index",
            "Systolic Blood Pressure",
            "Diastolic Blood Pressure",
            "Heart rate",
            "Respiratory rate",
            "AGE",
        ]

        # Use RobustScaler instead of StandardScaler for better handling of outliers
        df[scaled_cols] = scaler.transform(df[scaled_cols])

        print("scaled")

        cols_path = os.path.join(os.path.dirname(__file__), "cols.csv")
        with open(cols_path, "r") as f:
            COLUMN_ORDER = f.readline().strip().split(",")

        df = df[COLUMN_ORDER]

        df.to_csv("input.csv", index=False)
        print("saved input")

        # Make prediction
        try:
            prediction = model.predict(df)
            print("pred successful")
            output = jsonify(
                {
                    "prediction": prediction.tolist()[0],
                    "warning": "Some input fields were empty"
                    if df.isna().any().any()
                    else None,
                }
            )
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

        print(output)
        return output

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
