# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Akshat Rastogi and Rupal Mishra
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (21 September 2024)
            # Developers: Akshat Rastogi and Rupal Mishra
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
from sklearn.cluster import KMeans  # Importing the K-Means clustering algorithm
from ingest_transform import preprocess_data, scale_data  # Custom functions for data preprocessing and scaling

# Importing helper functions from local files
# from load import load_train  # (Commented out as it is not used in this snippet)
from evaluate import evaluate_model  # Function to evaluate the clustering model's performance

def train_model(df, n):
    """
    Train the K-Means clustering algorithm using the provided DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing data for clustering.
        n (int): The number of clusters for the K-Means algorithm.

    Returns:
        evals: Evaluation results from the evaluate_model function.
    """
    # Preprocess the input DataFrame to prepare it for clustering
    X = preprocess_data(df)
    
    # Scale the data and apply PCA transformation for dimensionality reduction
    X_pca = scale_data(X)
    
    # Initialize the K-Means model with the specified number of clusters and a fixed random state
    kmeans = KMeans(n_clusters=n, random_state=42).fit(X_pca)
    
    # Retrieve the coordinates of the cluster centroids
    centroids = kmeans.cluster_centers_
    
    # Get the predicted labels for each sample in the dataset
    labels = kmeans.labels_
    
    # Evaluate the model's performance using the custom evaluation function
    evals = evaluate_model(X_pca, labels, "K-Means", centroids)
    
    # Save the trained K-Means model to a file for future use
    joblib.dump(kmeans, 'Code\\saved model\\kmeans.pkl')

    # Return the evaluation results for further analysis
    return evals
