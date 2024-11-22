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
    """
    Train the DBSCAN clustering model using the provided DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing data for clustering.
        eps (float): The maximum distance between two samples for them to be considered as neighbors.
        minsm (int): The minimum number of samples in a neighborhood for a point to be considered as a core point.
        minmax (bool): Indicates whether to use MinMaxScaler for scaling (True) or StandardScaler (False).

    Returns:
        evals: Evaluation results from the evaluate_model function.
    """
    # Preprocess the input DataFrame to prepare it for clustering
    X = preprocess_data(df)
    
    # Scale the data and apply PCA transformation for dimensionality reduction
    X_pca = scale_data(X, minmax)
    
    # Initialize the DBSCAN model with specified parameters and fit it to the PCA-transformed data
    dbscan = DBSCAN(eps=eps, min_samples=minsm).fit(X_pca)
    
    # Retrieve the cluster labels assigned by the DBSCAN algorithm
    labels = dbscan.labels_
    
    # Evaluate the model using the custom evaluation function
    evals = evaluate_model(X_pca, labels, 'DBSCAN')
    
    # Save the trained DBSCAN model to a file for future use
    joblib.dump(dbscan, 'Code\\saved model\\dbscan.pkl')

    # Return the evaluation results for further analysis
    return evals
