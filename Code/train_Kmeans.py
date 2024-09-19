# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Akshat Rastogi, Shubh Gupta and Rupal Mishra
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (04 July 2024)
            # Developers: Akshat Rastogi, Shubh Gupta and Rupal Mishra
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This Streamlit app allows users to input features and make predictions using Unsupervised Learning.
        # SQLite: Yes
        # MQs: No
        # Cloud: No
        # Data versioning: No
        # Data masking: No

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # Python 3.11.5
        # Streamlit 1.36.0


import pandas as pd # For data manipulation and analysis
import joblib # For loading the trained model
from sklearn.cluster import KMeans # For training the model
from ingest_transform import preprocess_data, scale_data, feature_selection

# Importing helper functions from the local .py files
from load import load_train
from evaluate import evaluate_model

def train_model(df):
    df = preprocess_data(df)
    X = feature_selection(df)
    X_pca = scale_data(X)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X_pca)


    # Save the trained model
    # joblib.dump(model, model_path)

    # # Print a success message
    # print("Model training completed successfully.")
    
    # # Return the train model metrics
    # return evaluate_model(model, train_data)