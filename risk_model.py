import os
import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

def main():
    X = pd.read_csv('final_df.csv')

    # Separate features and target variable
    # Drop correlated values to FUTURE_OUT_OF_POCKET
    X.drop(['TOTAL_FUTURE_COVERAGE', 'TOTAL_FUTURE_COST'], axis = 1, inplace=True)
    X.dropna(axis=0, subset=['FUTURE_OUT_OF_POCKET'], inplace=True)
    y = X.FUTURE_OUT_OF_POCKET
    X.drop(['FUTURE_OUT_OF_POCKET'], axis = 1, inplace=True)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size = 0.8, test_size=0.2, random_state=0
    )
    
    # Initialize and train the XGBoost classifier
    model = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=5, missing=np.nan)
    model.fit(X_train, y_train,  
             eval_set=[(X_test, y_test)],
             verbose=False)
    
    # Predict and evaluate the model on the test set
    prediction = model.predict(X_test)
    mae = mean_absolute_error(y_test, prediction)
    print("Mean Absolute Error:", mae)

if __name__ == "__main__":
    main()


