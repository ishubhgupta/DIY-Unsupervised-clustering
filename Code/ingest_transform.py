# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Akshat Rastogi, Shubh Gupta and Rupal Mishra
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (04 July 2024)
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



import pandas as pd # For data manipulation and analysis
# import sqlite3 # For connecting to and interacting with SQLite databases
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

le = LabelEncoder()

def preprocess_data(df):
    global le
    
    df['gender'] = le.fit_transform(df['gender'])
    df['region'] = le.fit_transform(df['region'])
    df['customer_type'] = le.fit_transform(df['customer_type'])
    
    col = ['age', 'income', 'purchase_history', 'customer_spending_score', 'freq_of_visit', 'gender', 'region', 'customer_type']
    
    imputer = SimpleImputer(strategy='mean')
    df[col] = imputer.fit_transform(df[col])

    X = df[col].values

    return X

def scale_data(X, minmax = False):
    if(minmax == False):
        scaler = StandardScaler()  
    else:
        scaler = MinMaxScaler()  # Changed to StandardScaler

    X_scaled = scaler.fit_transform(X)

    # Dimensionality Reduction using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca

def preprocess_test(data):
    global le
    data = le.transform(data)
    