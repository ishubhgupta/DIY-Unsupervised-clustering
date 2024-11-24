# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Akshat Rastogi, Shubh Gupta and Rupal Mishra
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (21 September 2024)
            # Developers: Akshat Rastogi, Shubh Gupta and Rupal Mishra
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This script implements Gaussian Mixture Model (GMM) clustering algorithm for customer segmentation. It provides functionality to train and evaluate GMM clustering models.
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
from sklearn.mixture import GaussianMixture  # Importing the Gaussian Mixture Model for clustering
from ingest_transform import preprocess_data, scale_data  # Custom functions for preprocessing and scaling data

# Importing helper functions from local files
# from load import load_train  # (Commented out as it is not used in this snippet)
from evaluate import evaluate_model  # Function to evaluate the clustering model's performance

def train_model(df, n):
    """Train GMM model using DataFrame directly"""
    try:
        X = df.values
        X_pca = scale_data(X)
        gmm = GaussianMixture(n_components=n, random_state=42)
        labels = gmm.fit_predict(X_pca)
        evals = evaluate_model(X_pca, labels, 'Gaussian Mixture Model')
        joblib.dump(gmm, 'Code/saved model/gmm.pkl')
        return evals
    except Exception as e:
        print(f"Error in GMM training: {e}")
        return None
