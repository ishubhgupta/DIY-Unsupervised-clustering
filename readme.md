# DIY-UnsupervisedClustering

This repository provides a comprehensive solution for unsupervised clustering using several algorithms. The project includes data preprocessing, model training, evaluation, and visualization for clustering tasks with a focus on BIRCH, DBSCAN, GMM, KMeans, and OPTICS algorithms.

## Unsupervised Clustering Algorithms

Unsupervised clustering methods are used to group data points into clusters based on their similarities. This repository includes implementations of the following clustering algorithms:

- *BIRCH*: Balanced Iterative Reducing and Clustering using Hierarchies.
- *DBSCAN*: Density-Based Spatial Clustering of Applications with Noise.
- *GMM*: Gaussian Mixture Model.
- *KMeans*: K-Means Clustering.
- *OPTICS*: Ordering Points To Identify the Clustering Structure.

## Data Definition

The dataset used for clustering tasks is designed to facilitate clustering analysis and includes features suitable for unsupervised learning. Proper preprocessing and scaling of the data are essential before applying the clustering algorithms.

## Directory Structure
```plaintext
DIY-Unsupervised-clustering/
├── Code/
│   ├── app.py               # Main Streamlit application
│   ├── ingest_transform.py  # Data preprocessing
│   ├── train_brich.py          # Model training scripts
│   ├── train_db.py          # Model training scripts
│   ├── train_gmm.py          # Model training scripts
│   ├── train_kmeans.py          # Model training scripts
│   ├── train_optics.py          # Model training scripts
│   ├── evaluate.py         # Model evaluation
│   └── classification.py   # Prediction functionality
├── Data/
│   ├── Master/            # Raw data storage
│   └── Processed/         # Processed data and SQLite DB
└── saved images/          # Generated visualizations
```

## Program Flow

1. *Data Ingestion*: Extract data and preprocess it. [ingest_transform.py]
2. *Data Transformation*: Encode categorical features, handle missing values, scale data, and apply PCA for dimensionality reduction. [ingest_transform.py]
3. *Model Training*: Train various clustering models (BIRCH, DBSCAN, GMM, KMeans, OPTICS) using the preprocessed data. [train_birch.py, train_dbscan.py, train_gmm.py, train_kmeans.py, train_optics.py]
4. *Model Evaluation*: Evaluate the performance of the clustering models using silhouette scores and generate visualizations. [evaluate.py]
5. *Visualization*: Save and display clustering results. [evaluate.py]

## Steps to Run

1. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the Application**:
    ```bash
    streamlit run Code/app.py
    ```

3. **Input Data Path**:
    - Open the application in your web browser (Streamlit will provide a local URL).
    - In the "Model Config" tab, enter the path to your master data CSV file in the text input box. For example:
      ```
      Data/Master/MOCK_DATA.csv
      ```
    - Click outside the text box to load the data. A preview of the data will be displayed.

4. **Train the Model**:
    - Go to the "Model Training & Evaluation" tab.
    - Select the clustering algorithm you want to train (e.g., K-Means, Gaussian Mixture Model, DBSCAN, OPTICS, BIRCH).
    - Configure the model parameters (e.g., number of clusters for K-Means, EPS for DBSCAN).
    - Click the "Train [Model Name] Model" button to start training.
    - Once training is complete, the silhouette score and a cluster graph image will be displayed.

5. **Make Predictions on New Data**:
    - Go to the "Classification" tab.
    - Select the clustering algorithm you trained earlier from the dropdown menu.
    - Enter the customer details for clustering (e.g., age, income, purchase history, etc.).
    - Click the "Cluster" button to classify the new data.
    - The predicted cluster label and the corresponding cluster graph image will be displayed.

6. **View Evaluation Results**:
    - Check the `saved images/` directory for saved plots and evaluation results.
    - The cluster graph images for each trained model will be saved in this directory.


## Contact

- *Developers*:
  - Akshat Rastogi
  - Shubh Gupta
  - Rupal Mishra
- *Code Ownership Rights*: PreProd Corp
