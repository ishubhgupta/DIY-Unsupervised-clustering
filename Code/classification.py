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
     
    # Description: This Streamlit app allows users to input features and make predictions using Unsupervised Learning.
        # SQLite: Yes
        # MQs: No

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # Python 3.10.11
        # Streamlit 1.40.0
        # pandas 1.5.3
        # scikit-learn 1.2.2


# Import necessary libraries
import joblib  # For loading saved models
import pandas as pd  # Not used directly here but useful for handling data in other parts of the application
from sklearn.cluster import KMeans  # Not used directly, kept if needed for debugging or new implementations
from ingest_transform import scale_back  # Custom function to scale back or preprocess input data

def classify(algorithm, items):
    """
    Classifies the input items into clusters based on the selected algorithm.

    Parameters:
    algorithm (str): The name of the clustering algorithm selected by the user.
    items (numpy array): The input data consisting of customer details.

    Returns:
    numpy array: Cluster labels assigned to the input data.
    """

    # Check if the selected algorithm is K-Means
    if algorithm == 'K-Means':
        # Scale the input data using the custom 'scale_back' function
        scaled_data = scale_back(items)
        
        # Load the pre-trained K-Means model
        model = joblib.load(r'Code\saved model\kmeans.pkl')
        
        # Predict clusters using the K-Means model
        clusters = model.predict(scaled_data)
        
        # Return the predicted cluster labels
        return clusters

    # Check if the selected algorithm is Gaussian Mixture Model
    elif algorithm == 'Gaussian Mixture Model':
        # Scale the input data using the custom 'scale_back' function
        scaled_data = scale_back(items)
        
        # Load the pre-trained Gaussian Mixture Model
        model = joblib.load(r'Code\saved model\gmm.pkl')
        
        # Predict clusters using the GMM model
        clusters = model.predict(scaled_data)
        
        # Return the predicted cluster labels
        return clusters

    # Uncomment the following sections if implementing DBSCAN or OPTICS
    
    # # Check if the selected algorithm is DBSCAN
    # elif algorithm == 'DBSCAN':
    #     # Scale the input data with additional scaling requirements (e.g., minimum value)
    #     scaled_data = scale_back(items, minm=True)
        
    #     # Load the pre-trained DBSCAN model
    #     model = joblib.load(r'Code\saved model\dbscan.pkl')
        
    #     # Predict clusters using the DBSCAN model
    #     clusters = model.predict(scaled_data)
        
    #     # Return the predicted cluster labels
    #     return clusters

    # # Check if the selected algorithm is OPTICS
    # elif algorithm == 'OPTICS':
    #     # Scale the input data with additional scaling requirements (e.g., minimum value)
    #     scaled_data = scale_back(items, minm=True)
        
    #     # Load the pre-trained OPTICS model
    #     model = joblib.load(r'Code\saved model\optics.pkl')
        
    #     # Fit and predict clusters using the OPTICS model
    #     clusters = model.fit_predict(scaled_data)
        
    #     # Return the predicted cluster labels
    #     return clusters

    # Check if the selected algorithm is BIRCH
    elif algorithm == 'BIRCH':
        # Scale the input data using the custom 'scale_back' function
        scaled_data = scale_back(items)
        
        # Load the pre-trained BIRCH model
        model = joblib.load(r'Code\saved model\birch.pkl')
        
        # Predict clusters using the BIRCH model
        clusters = model.predict(scaled_data)
        
        # Return the predicted cluster labels
        return clusters
