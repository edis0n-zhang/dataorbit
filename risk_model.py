import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from datetime import datetime
import os


class HealthRiskModel:
    def __init__(self):
        self.condition_encoder = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.risk_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective="binary:logistic",
        )
        self.cost_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1
        )

    def load_and_process_data(self, data_dir):
        # Load data
        conditions_df = pd.read_csv(os.path.join(data_dir, "conditions.csv"))
        patients_df = pd.read_csv(os.path.join(data_dir, "patients.csv"))
        encounters_df = pd.read_csv(os.path.join(data_dir, "encounters.csv"))
        medications_df = pd.read_csv(os.path.join(data_dir, "medications.csv"))
        procedures_df = pd.read_csv(os.path.join(data_dir, "procedures.csv"))

        # Process conditions - create one-hot encoding
        condition_pivot = pd.pivot_table(
            conditions_df,
            values="START",
            index="PATIENT",
            columns="DESCRIPTION",
            aggfunc="count",
            fill_value=0,
        )

        # Calculate age from birthdate
        patients_df["BIRTHDATE"] = pd.to_datetime(patients_df["BIRTHDATE"])
        current_year = datetime.now().year
        patients_df["AGE"] = current_year - patients_df["BIRTHDATE"].dt.year

        # Encode categorical variables
        if "GENDER" not in self.label_encoders:
            self.label_encoders["GENDER"] = LabelEncoder()
            patients_df["GENDER"] = self.label_encoders["GENDER"].fit_transform(
                patients_df["GENDER"]
            )
        else:
            patients_df["GENDER"] = self.label_encoders["GENDER"].transform(
                patients_df["GENDER"]
            )

        # Calculate encounter frequencies and costs
        encounter_stats = (
            encounters_df.groupby("PATIENT")
            .agg(
                {
                    "BASE_ENCOUNTER_COST": ["sum", "mean", "count"],
                    "TOTAL_CLAIM_COST": ["sum", "mean"],
                }
            )
            .fillna(0)
        )
        encounter_stats.columns = [
            "total_encounter_cost",
            "avg_encounter_cost",
            "encounter_count",
            "total_claim_cost",
            "avg_claim_cost",
        ]

        # Calculate medication statistics
        med_stats = (
            medications_df.groupby("PATIENT")
            .agg({"BASE_COST": ["sum", "mean", "count"]})
            .fillna(0)
        )
        med_stats.columns = ["total_med_cost", "avg_med_cost", "med_count"]

        # Calculate procedure statistics
        proc_stats = (
            procedures_df.groupby("PATIENT")
            .agg({"BASE_COST": ["sum", "mean", "count"]})
            .fillna(0)
        )
        proc_stats.columns = ["total_proc_cost", "avg_proc_cost", "proc_count"]

        # Combine all features
        features = pd.concat(
            [
                condition_pivot,
                patients_df.set_index("Id")[["AGE", "GENDER"]],
                encounter_stats,
                med_stats,
                proc_stats,
            ],
            axis=1,
        ).fillna(0)

        # Get target variables
        targets = patients_df.set_index("Id")[
            ["HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE"]
        ]

        # Create binary risk target (high risk if expenses > 75th percentile)
        expense_threshold = targets["HEALTHCARE_EXPENSES"].quantile(0.75)
        targets["HIGH_RISK"] = (
            targets["HEALTHCARE_EXPENSES"] > expense_threshold
        ).astype(int)

        return features, targets

    def train(self, features, targets):
        # Split features for risk and cost prediction
        X_train, X_val, y_train, y_val = train_test_split(
            features,
            targets[["HIGH_RISK", "HEALTHCARE_EXPENSES"]],
            test_size=0.2,
            random_state=42,
        )

        # Scale numerical features
        numerical_cols = ["AGE"] + [
            col for col in features.columns if "cost" in col or "count" in col
        ]
        X_train[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        X_val[numerical_cols] = self.scaler.transform(X_val[numerical_cols])

        # Train risk model
        print("Training risk classification model...")
        self.risk_model.fit(
            X_train,
            y_train["HIGH_RISK"],
            eval_set=[(X_val, y_val["HIGH_RISK"])],
            verbose=False,
        )

        # Train cost prediction model
        print("Training cost prediction model...")
        self.cost_model.fit(
            X_train,
            y_train["HEALTHCARE_EXPENSES"],
            eval_set=[(X_val, y_val["HEALTHCARE_EXPENSES"])],
            verbose=False,
        )

    def predict(self, features):
        # Get the features that were present during training
        model_features = self.risk_model.get_booster().feature_names
        
        # Create a DataFrame with all training features, initialized to 0
        aligned_features = pd.DataFrame(0, index=features.index, columns=model_features)
        
        # Copy over values for features that exist in both
        common_features = features.columns.intersection(model_features)
        aligned_features[common_features] = features[common_features]
        
        # Scale numerical features that exist in both datasets
        numerical_cols = ["AGE"] + [col for col in aligned_features.columns if "cost" in col or "count" in col]
        existing_num_cols = [col for col in numerical_cols if col in aligned_features.columns]
        if existing_num_cols:
            aligned_features[existing_num_cols] = self.scaler.transform(aligned_features[existing_num_cols])

        # Make predictions
        risk_prob = self.risk_model.predict_proba(aligned_features)[:, 1]  # Get probability of positive class
        estimated_cost = self.cost_model.predict(aligned_features)

        return pd.DataFrame(
            {"risk_probability": risk_prob, "estimated_cost": estimated_cost},
            index=features.index,
        )


def main():
    # Initialize model
    model = HealthRiskModel()

    # Load and process training data
    print("Loading data from pop1...")
    features, targets = model.load_and_process_data("dataverse_files/pop1")
    
    # Split into train and validation sets
    print("Splitting into train and validation sets...")
    train_features, val_features, train_targets, val_targets = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )

    # Train model
    print("Training model...")
    model.train(train_features, train_targets)

    # Make predictions on validation set
    print("Making predictions on validation set...")
    predictions = model.predict(val_features)

    # Calculate metrics
    from sklearn.metrics import roc_auc_score, mean_squared_error
    import numpy as np

    # Risk prediction metrics
    risk_auc = roc_auc_score(val_targets["HIGH_RISK"], predictions["risk_probability"])
    print(f"Risk prediction AUC-ROC: {risk_auc:.3f}")

    # Cost prediction metrics
    cost_rmse = np.sqrt(mean_squared_error(val_targets["HEALTHCARE_EXPENSES"], predictions["estimated_cost"]))
    print(f"Cost prediction RMSE: ${cost_rmse:,.2f}")


if __name__ == "__main__":
    main()
