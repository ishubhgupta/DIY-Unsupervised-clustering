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
     
    # Description: This script handles the classification of new data points using trained clustering models. It loads saved models and provides functionality to assign cluster labels to new customer data.
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

# Import necessary libraries
import joblib  # For loading saved models
import pandas as pd  # Not used directly here but useful for handling data in other parts of the application
from sklearn.cluster import KMeans  # Not used directly, kept if needed for debugging or new implementations
from ingest_transform import scale_back  # Custom function to scale back or preprocess input data

def classify(algorithm, items):
    """Modified classify function to work with database"""
    try:
        scaled_data = scale_back(items)
        
        if algorithm == 'K-Means':
            model = joblib.load('Code/saved model/kmeans.pkl')
        elif algorithm == 'Gaussian Mixture Model':
            model = joblib.load('Code/saved model/gmm.pkl')
        elif algorithm == 'BIRCH':
            model = joblib.load('Code/saved model/birch.pkl')
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
        clusters = model.predict(scaled_data)
        return clusters
    except Exception as e:
        print(f"Error in classification: {e}")
        return None
