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

- *Code/*: Contains all scripts for data ingestion, transformation, model training, evaluation, and visualization.
- *saved images/*: Directory where generated plots are saved.


## Program Flow

1. *Data Ingestion*: Extract data and preprocess it. [ingest_transform.py]
2. *Data Transformation*: Encode categorical features, handle missing values, scale data, and apply PCA for dimensionality reduction. [ingest_transform.py]
3. *Model Training*: Train various clustering models (BIRCH, DBSCAN, GMM, KMeans, OPTICS) using the preprocessed data. [train_birch.py, train_dbscan.py, train_gmm.py, train_kmeans.py, train_optics.py]
4. *Model Evaluation*: Evaluate the performance o
