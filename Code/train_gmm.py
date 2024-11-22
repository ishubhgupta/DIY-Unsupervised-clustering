# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

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
    """
    Train the Gaussian Mixture Model (GMM) clustering algorithm using the provided DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing data for clustering.
        n (int): The number of mixture components (clusters) for the GMM.

    Returns:
        evals: Evaluation results from the evaluate_model function.
    """
    # Preprocess the input DataFrame to prepare it for clustering
    X = preprocess_data(df)
    
    # Scale the data and apply PCA transformation for dimensionality reduction
    X_pca = scale_data(X)
    
    # Initialize the Gaussian Mixture Model with the specified number of components and random state
    gmm = GaussianMixture(n_components=n, random_state=42)
    
    # Fit the GMM to the PCA-transformed data and retrieve the predicted labels for each sample
    labels = gmm.fit_predict(X_pca)
    
    # Evaluate the model using the custom evaluation function
    evals = evaluate_model(X_pca, labels, 'Gaussian Mixture Model')
    
    # Save the trained Gaussian Mixture Model to a file for future use
    joblib.dump(gmm, 'Code\\saved model\\gmm.pkl')

    # Return the evaluation results for further analysis
    return evals
