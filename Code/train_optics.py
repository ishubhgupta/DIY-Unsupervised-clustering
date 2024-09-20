# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Akshat Rastogi, Shubh Gupta and Rupal Mishra
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (19 September 2024)
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
