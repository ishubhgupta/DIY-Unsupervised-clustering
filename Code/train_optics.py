# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Shubh Gupta and Rupal Mishra
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (21 September 2024)
            # Developers: Shubh Gupta and Rupal Mishra
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This script implements OPTICS (Ordering Points To Identify Clustering Structure) algorithm for customer segmentation. It provides functionality to train and evaluate OPTICS clustering models.
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
from sklearn.cluster import DBSCAN, OPTICS  # Importing clustering algorithms
from ingest_transform import preprocess_data, scale_data  # Custom functions for preprocessing and scaling data

# Importing helper functions from local files
# from load import load_train  # (Commented out as it is not used in this snippet)
from evaluate import evaluate_model  # Function to evaluate the clustering model's performance

def train_model(df, min_sample, xi, cluster, minmax):
    """
    Train the OPTICS clustering algorithm using the provided DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing data for clustering.
        min_sample (int): The minimum number of samples in a neighborhood for a point to be considered a core point.
        xi (float): The steepness parameter for determining the cluster stability.
        cluster (float): The minimum size of clusters (as a proportion of the dataset).
        minmax (bool): Flag to indicate whether to use MinMaxScaler for scaling.

    Returns:
        evals: Evaluation results from the evaluate_model function.
    """
    # Preprocess the input DataFrame to prepare it for clustering
    X = preprocess_data(df)
    
    # Scale the data and apply PCA transformation for dimensionality reduction
    X_pca = scale_data(X, minmax)
    
    # Initialize the OPTICS model with the specified parameters and fit it to the PCA-transformed data
    optics = OPTICS(min_samples=min_sample, xi=xi, min_cluster_size=cluster).fit(X_pca)
    
    # Retrieve the predicted labels for each sample in the dataset
    labels = optics.labels_
    
    # Evaluate the model's performance using the custom evaluation function
    evals = evaluate_model(X_pca, labels, 'OPTICS')
    
    # Save the trained OPTICS model to a file for future use
    joblib.dump(optics, 'Code\\saved model\\optics.pkl')

    # Return the evaluation results for further analysis
    return evals
