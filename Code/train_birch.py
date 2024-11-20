# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Shubh Gupta
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (21 September 2024)
            # Developers: Shubh Gupta
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This Streamlit app allows users to input features and make predictions using Unsupervised Learning.
        # SQLite: Yes
        # MQs: No

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # Python 3.10.11
        # Streamlit 1.40.0


import pandas as pd  # For data manipulation and analysis
import joblib  # For saving and loading trained models
from sklearn.cluster import Birch  # BIRCH clustering algorithm
from ingest_transform import preprocess_data, scale_data  # Custom preprocessing and scaling functions

# Importing helper functions from the local .py files
# from load import load_train # (Commented out as it is not used in this snippet)
from evaluate import evaluate_model  # Custom function to evaluate the model's performance

def train_model(df, n_cluster, threas, minmax):
    """
    Train the BIRCH clustering model using the input DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing data to be used for training.
        n_cluster (int): Number of clusters to form. If set to None, BIRCH will automatically determine clusters.
        threas (float): Threshold for the BIRCH model, which controls the radius of subclusters.
        minmax (bool): Indicates whether to use MinMaxScaler for scaling (True) or StandardScaler (False).

    Returns:
        evals: The evaluation results from the evaluate_model function, which assesses the model's performance.
    """
    # Preprocess the data using custom preprocessing function
    X = preprocess_data(df)
    
    # Scale the data and apply PCA transformation
    X_pca = scale_data(X, minm=minmax)
    
    # Initialize the BIRCH clustering model with specified parameters
    birch = Birch(n_clusters=n_cluster, threshold=threas)
    
    # Fit the model to the data and predict cluster labels
    labels = birch.fit_predict(X_pca)
    
    # Evaluate the model using custom evaluation function
    evals = evaluate_model(X_pca, labels, 'BIRCH')
    
    # Save the trained BIRCH model to a file using joblib
    joblib.dump(birch, r'Code\saved model\birch.pkl')

    # Return the evaluation results
    return evals
