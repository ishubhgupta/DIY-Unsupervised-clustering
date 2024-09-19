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
from ingest_transform import preprocess_data, scale_data

# Importing helper functions from the local .py files
# from load import load_train
from evaluate import evaluate_model

def train_model(df):
    X = preprocess_data(df)
    X_pca = scale_data(X)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X_pca)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    evals = evaluate_model(X_pca, labels, "K-Means", centroids)
    print(evals)

    return evals