import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score

# Title for the app
st.title("Clustering with OPTICS and PCA")

# Step 1: Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Step 2: Display the dataset
    st.write("Here is a preview of your data:")
    st.write(df.head())

    # Step 3: Preprocessing
    # Checking for required columns
    required_columns = ['age', 'income', 'purchase_history', 'customer_spending_score', 'freq_of_visit', 'gender', 'region', 'customer_type']
    
    if all(col in df.columns for col in required_columns):
        # Encode categorical features
        le = LabelEncoder()
        df['gender'] = le.fit_transform(df['gender'])
        df['region'] = le.fit_transform(df['region'])
        df['customer_type'] = le.fit_transform(df['customer_type'])

        # Feature Selection
        X = df[required_columns].values

        # Scaling the dataset
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Dimensionality Reduction using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # Step 4: Apply OPTICS Clustering
        optics = OPTICS(min_samples=10, xi=0.02, min_cluster_size=0.25)
        optics.fit(X_pca)
        optics_labels = optics.labels_

        # Step 5: Plotting clusters
        st.write("Cluster Visualization:")
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=optics_labels, cmap='viridis')
        ax.set_title('OPTICS Clustering')
        ax.set_xlabel('Feature 1 (PCA)')
        ax.set_ylabel('Feature 2 (PCA)')
        fig.colorbar(scatter, label='Cluster Label')
        st.pyplot(fig)

        # Step 6: Display Silhouette Score
        score = silhouette_score(X_pca, optics_labels)
        st.write("Silhouette Score:", score)
    else:
        st.error("The uploaded dataset must contain the following columns: " + ", ".join(required_columns))
