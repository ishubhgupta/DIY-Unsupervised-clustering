# DIY-UnsupervisedClustering

This repository provides a comprehensive solution for unsupervised clustering using several algorithms. The project includes data preprocessing, model training, evaluation, and visualization for clustering tasks with a focus on BIRCH, DBSCAN, GMM, KMeans, and OPTICS algorithms.

## Unsupervised Clustering Algorithms

Unsupervised clustering methods are used to group data points into clusters based on their similarities. This repository includes implementations of the following clustering algorithms:

- **BIRCH**: Balanced Iterative Reducing and Clustering using Hierarchies.
- **DBSCAN**: Density-Based Spatial Clustering of Applications with Noise.
- **GMM**: Gaussian Mixture Model.
- **KMeans**: K-Means Clustering.
- **OPTICS**: Ordering Points To Identify the Clustering Structure.

## Data Definition

The dataset used for clustering tasks is designed to facilitate clustering analysis and includes features suitable for unsupervised learning. Proper preprocessing and scaling of the data are essential before applying the clustering algorithms.

## Directory Structure

- **Code/**: Contains all scripts for data ingestion, transformation, model training, evaluation, and visualization.
- **Data/**: Contains the raw and processed data.
- **saved images/**: Directory where generated plots are saved.

## Data Splitting

The data is split into training, testing, and validation sets as follows:

- **Training Samples**: [Specify number if applicable]
- **Testing Samples**: [Specify number if applicable]
- **Validation Samples**: [Specify number if applicable]
- **Supervalidation Samples**: [Specify number if applicable]

## Program Flow

1. **Data Ingestion**: Extract data and preprocess it. [`ingest_transform.py`]
2. **Data Transformation**: Encode categorical features, handle missing values, scale data, and apply PCA for dimensionality reduction. [`ingest_transform.py`]
3. **Model Training**: Train various clustering models (BIRCH, DBSCAN, GMM, KMeans, OPTICS) using the preprocessed data. [`train_birch.py`, `train_dbscan.py`, `train_gmm.py`, `train_kmeans.py`, `train_optics.py`]
4. **Model Evaluation**: Evaluate the performance of the clustering models using silhouette scores and generate visualizations. [`evaluate.py`]
5. **Visualization**: Save and display clustering results. [`evaluate.py`]

## Steps to Run

1. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Run Data Ingestion and Transformation**:
    ```bash
    python Code/ingest_transform.py
    ```

3. **Train Clustering Models**:
    ```bash
    python Code/train_birch.py
    python Code/train_dbscan.py
    python Code/train_gmm.py
    python Code/train_kmeans.py
    python Code/train_optics.py
    ```

4. **View Evaluation Results**: Check the `saved images/` directory for saved plots and evaluation results.

## Contact

- **Developers**:
  - Akshat Rastogi
  - Shubh Gupta
  - Rupal Mishra
- **Code Ownership Rights**: PreProd Corp

