import os
import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
import xgboost as xgb

def handle_outliers(df, columns, n_sigmas=3):
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = df[col].clip(mean - n_sigmas * std, mean + n_sigmas * std)
    return df

def main():
    # Load and preprocess data
    X = pd.read_csv('final_df.csv')
    
    # Drop correlated values to FUTURE_OUT_OF_POCKET
    X.drop(['TOTAL_FUTURE_COVERAGE', 'TOTAL_FUTURE_COST'], axis=1, inplace=True)
    X.dropna(axis=0, subset=['FUTURE_OUT_OF_POCKET'], inplace=True)
    
    # Handle outliers in target variable
    # y = X.FUTURE_OUT_OF_POCKET
    # q1, q3 = y.quantile(0.25), y.quantile(0.75)
    # iqr = q3 - q1
    # y = y.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    y = X.FUTURE_OUT_OF_POCKET
    p1, p99 = y.quantile(0.02), y.quantile(0.98)
    print(f"Original number of samples: {len(X)}")
    mask = (y >= p1) & (y <= p99)
    X = X[mask]
    y = y[mask]
    print(f"Number of samples after removing outliers: {len(X)}")
    print(f"Kept values in range: ${p1:.2f} - ${p99:.2f}")
    
    X.drop(['FUTURE_OUT_OF_POCKET'], axis=1, inplace=True)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=42
    )
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    
    # Handle outliers in numerical columns
    numerical_cols = ['Body Weight', 'Body Height', 'Body Mass Index', 'Systolic Blood Pressure', 'AGE']
    X_train = handle_outliers(X_train, numerical_cols)
    X_test = handle_outliers(X_test, numerical_cols)
    
    # Use RobustScaler instead of StandardScaler for better handling of outliers
    scaler = RobustScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # Initialize model with carefully tuned parameters
    model = xgb.XGBRegressor(
        max_depth=4,               # Reduced to prevent overfitting
        learning_rate=0.03,        # Smaller learning rate for better generalization
        n_estimators=300,          # Increased due to smaller learning rate
        min_child_weight=5,        # Increased to reduce overfitting
        subsample=0.9,             # Increased for better stability
        colsample_bytree=0.9,      # Increased for better stability
        gamma=1,                   # Added to control tree growth
        reg_alpha=0.1,            # L1 regularization
        reg_lambda=1,             # L2 regularization
        objective='reg:squarederror',
        eval_metric='mae',
        early_stopping_rounds=20,  # Increased patience
        random_state=42
    )
    
    # Fit the model with more evaluation sets
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    # Make predictions
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"Mean Absolute Error: {mae}")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

if __name__ == "__main__":
    main()


