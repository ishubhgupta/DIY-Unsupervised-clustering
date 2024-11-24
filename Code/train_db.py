# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Rupal Mishra
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (21 September 2024)
            # Developers: Rupal Mishra
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This script implements DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm for customer segmentation. It provides functionality to train and evaluate DBSCAN clustering models.
        # SQLite: Yes
        # MQs: No

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # Python 3.10.11
        # numpy 1.24.3
        # pandas 1.5.3
        # scikit-learn 1.2.2
        # joblib 1.3.1


import pandas as pd  # For data manipulation and analysis
import joblib  # For saving and loading trained models
from sklearn.cluster import DBSCAN  # Importing the DBSCAN clustering algorithm
from ingest_transform import preprocess_data, scale_data  # Custom functions for preprocessing and scaling data

# Importing helper functions from local files
# from load import load_train  # (Commented out as it is not used in this snippet)
from evaluate import evaluate_model  # Function to evaluate the clustering model's performance

def train_model(df, eps, minsm, minmax):
    """Train DBSCAN model using DataFrame directly"""
    try:
        X = df.values
        X_pca = scale_data(X, minmax)
        dbscan = DBSCAN(eps=eps, min_samples=minsm).fit(X_pca)
        labels = dbscan.labels_
        evals = evaluate_model(X_pca, labels, 'DBSCAN')
        joblib.dump(dbscan, 'Code/saved model/dbscan.pkl')
        return evals
    except Exception as e:
        print(f"Error in DBSCAN training: {e}")
        return None
