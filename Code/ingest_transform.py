# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Akshat Rastogi, Shubh Gupta
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (21 September 2024)
            # Developers: Akshat Rastogi, Shubh Gupta and Rupal Mishra
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This script handles data ingestion, preprocessing, and transformation operations. It includes functions for data scaling, PCA transformation, and SQLite database operations for storing and retrieving data paths.
        # SQLite: Yes
        # MQs: No

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # Python 3.10.11
        # numpy 1.24.3
        # pandas 1.5.3
        # scikit-learn 1.2.2
        # sqlite3 (built-in)

# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
from sklearn.decomposition import PCA  # For dimensionality reduction
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # For data scaling
from sklearn.impute import SimpleImputer  # For handling missing values
import sqlite3  # For interacting with SQLite database

# Example mappings for gender, region, and customer type to convert categorical data to numerical values
gender_mapping = {'Male': 0, 'Female': 1, 'Agender': 2, 'Genderqueer': 3, 'Polygender': 4, 
                  'Genderfluid': 5, 'Non-binary': 6, 'Bigender': 7}
region_mapping = {'North': 0, 'South': 1, 'East': 2, 'West': 3}
customer_type_mapping = {'budget': 0, 'regular': 1, 'premium': 2}  # Corrected mapping

# Initialize scalers and PCA
standard = StandardScaler()  # Standard scaler for standardizing features
minmax = MinMaxScaler()  # MinMax scaler for scaling features to a range
pca = PCA(n_components=2)  # PCA to reduce data to 2 dimensions

# Preprocessing function for the data
def preprocess_data(df):
    """
    Preprocesses the data by mapping categorical values to numeric, imputing missing values,
    and selecting relevant columns for modeling.

    Parameters:
    df (DataFrame): The input DataFrame containing customer data.

    Returns:
    numpy array: Processed numerical data ready for scaling and modeling.
    """
    global gender_mapping, region_mapping, customer_type_mapping

    # Map categorical columns (gender, region, customer_type) to numeric values using predefined mappings
    df['gender'] = df['gender'].map(gender_mapping)
    df['region'] = df['region'].map(region_mapping)
    df['customer_type'] = df['customer_type'].map(customer_type_mapping)

    # Define the columns to be used in the analysis
    col = ['age', 'income', 'purchase_history', 'customer_spending_score', 
           'freq_of_visit', 'gender', 'region', 'customer_type']
    
    # Impute missing values in the selected columns using the mean strategy
    imputer = SimpleImputer(strategy='mean')
    df[col] = imputer.fit_transform(df[col])

    # Convert the processed DataFrame columns to a numpy array
    X = df[col].values

    return X

# Function to scale data and apply PCA
def scale_data(X, minm=False):
    """
    Scales the data using StandardScaler or MinMaxScaler and applies PCA.

    Parameters:
    X (numpy array): The input data to be scaled and transformed.
    minm (bool): If True, use MinMaxScaler; otherwise, use StandardScaler.

    Returns:
    numpy array: The data transformed by scaling and PCA.
    """
    global standard, minmax, pca
    scaler = minmax if minm else standard  # Choose the scaler based on the minm flag
    X_scaled = scaler.fit_transform(X)  # Scale the data

    # Apply PCA to reduce the data to 2 dimensions
    X_pca = pca.fit_transform(X_scaled)

    return X_pca

# Function to preprocess single test data points
def preprocess_test(data):
    """
    Preprocesses a single data point by mapping it to its numeric equivalent.

    Parameters:
    data (str): A categorical value that needs to be mapped.

    Returns:
    int: Mapped numeric value of the input data.
    """
    global gender_mapping, region_mapping, customer_type_mapping
    
    # Map the input data to its corresponding numeric value based on predefined mappings
    if data in gender_mapping.keys():
        data = gender_mapping[data]
    elif data in region_mapping.keys():
        data = region_mapping[data]
    elif data in customer_type_mapping.keys():
        data = customer_type_mapping[data]

    return data

# Function to reverse scaling and PCA transformation
def scale_back(items, minm=False):
    """
    Reverses the scaling and PCA transformation for items.

    Parameters:
    items (numpy array): The input data items to reverse scale.
    minm (bool): If True, use MinMaxScaler; otherwise, use StandardScaler.

    Returns:
    numpy array: Data transformed back by inverse scaling and PCA transformation.
    """
    global standard, minmax, pca
    scaler = minmax if minm else standard
    X_scaled = scaler.transform(items)  # Scale the data using the selected scaler
    scaled_pca = pca.transform(X_scaled)  # Apply PCA transformation

    return scaled_pca

# Function to store the master data path in SQLite
def store_path_to_sqlite(path, db_path):
    """
    Stores the master data path in an SQLite database.

    Parameters:
    path (str): The file path to be stored.
    db_path (str): The path to the SQLite database file.
    """
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

        # Insert the master data path into the table
        cursor.execute('''
            INSERT INTO config (master_data_path)
            VALUES (?)
        ''', (path,))
        
        conn.commit()  # Commit the changes to the database
        conn.close()  # Close the database connection
        print("Path stored successfully.")
    except Exception as e:
        print(f"Error storing path to SQLite: {e}")

# Function to retrieve the master data path from SQLite
def retrieve_path_from_sqlite(db_path):
    """
    Retrieves the master data path from the SQLite database and reads the corresponding CSV file.

    Parameters:
    db_path (str): The path to the SQLite database file.

    Returns:
    DataFrame or None: The DataFrame read from the CSV file if successful; otherwise, None.
    """
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Retrieve the latest master data path from the config table
        cursor.execute('SELECT master_data_path FROM config ORDER BY id DESC LIMIT 1')
        row = cursor.fetchone()  # Fetch the latest record
        conn.close()  # Close the database connection

        if row:
            csv_path = row[0]  # Get the path from the fetched row
            try:
                # Read the CSV file using the retrieved path
                df = pd.read_csv(csv_path)
                return df
            except FileNotFoundError:
                print(f"File not found at path: {csv_path}")
                return None
            except pd.errors.EmptyDataError:
                print(f"No data found in the file at path: {csv_path}")
                return None
            except Exception as e:
                print(f"Error reading CSV file: {e}")
                return None
        else:
            print("No path found in the database.")
            return None
    except Exception as e:
        print(f"Error retrieving path from SQLite: {e}")
        return None
