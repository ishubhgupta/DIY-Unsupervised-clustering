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


import pandas as pd  # For data manipulation and analysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Corrected import
from sklearn.impute import SimpleImputer
import sqlite3

# Example mappings for gender, region, and customer type
gender_mapping = {'Male': 0, 'Female': 1, 'Agender': 2, 'Genderqueer': 3, 'Polygender': 4, 'Genderfluid': 5, 'Non-binary': 6, 'Bigender': 7}
region_mapping = {'North': 0, 'South': 1, 'East': 2, 'West': 3}
customer_type_mapping = {'budget': 0, 'regular': 1, 'premium': 2}  # Corrected mapping

# Initialize scalers and PCA
standard = StandardScaler()
minmax = MinMaxScaler()
pca = PCA(n_components=2)

# Preprocessing function for the data
def preprocess_data(df):
    global gender_mapping, region_mapping, customer_type_mapping

    # Map categorical columns to numeric values
    df['gender'] = df['gender'].map(gender_mapping)
    df['region'] = df['region'].map(region_mapping)
    df['customer_type'] = df['customer_type'].map(customer_type_mapping)

    col = ['age', 'income', 'purchase_history', 'customer_spending_score', 'freq_of_visit', 'gender', 'region', 'customer_type']
    
    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    df[col] = imputer.fit_transform(df[col])

    X = df[col].values

    return X

# Function to scale data and apply PCA
def scale_data(X, minm=False):
    global standard, minmax, pca
    scaler = minmax if minm else standard  # Use MinMaxScaler if minm is True, otherwise StandardScaler
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    X_pca = pca.fit_transform(X_scaled)

    return X_pca

# Function to preprocess single test data points
def preprocess_test(data):
    global gender_mapping, region_mapping, customer_type_mapping
    
    # Apply the correct mapping for each column
    if data in gender_mapping.keys():
        data = gender_mapping[data]
    elif data in region_mapping.keys():
        data = region_mapping[data]
    elif data in customer_type_mapping.keys():
        data = customer_type_mapping[data]

    return data

# Function to reverse scaling and PCA transformation
def scale_back(items, minm=False):
    global standard, minmax, pca
    scaler = minmax if minm else standard
    X_scaled = scaler.transform(items)
    scaled_pca = pca.transform(X_scaled)

    return scaled_pca

# Function to store the master data path in SQLite
def store_path_to_sqlite(path, db_path):
    try:
        # Create or connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create a table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                master_data_path TEXT
            );
        ''')

        # Insert or update the master data path
        cursor.execute('''
            INSERT INTO config (master_data_path)
            VALUES (?)
        ''', (path,))
        
        conn.commit()
        conn.close()
        print("Path stored successfully.")
    except Exception as e:
        print(f"Error storing path to SQLite: {e}")


# Function to retrieve the master data path from SQLite
def retrieve_path_from_sqlite(db_path):
    """Retrieve the master data path from SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT master_data_path FROM config ORDER BY id DESC LIMIT 1')
        row = cursor.fetchone()
        conn.close()

        if row:
            return row[0]
        else:
            print("No path found in the database.")
            return None
    except Exception as e:
        print(f"Error retrieving path from SQLite: {e}")
        return None
