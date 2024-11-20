# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Akshat Rastogi
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

# Import necessary libraries
import numpy as np  # For numerical operations, especially working with arrays
from sklearn.metrics import silhouette_score  # To evaluate the quality of clustering
import matplotlib.pyplot as plt  # For plotting graphs

def evaluate_model(X_pca, labels, algorithm_name, centroids=None):
    """
    Evaluates the clustering model by plotting the data and calculating the silhouette score.

    Parameters:
    X_pca (numpy array): The transformed data in two dimensions (e.g., after PCA).
    labels (numpy array): Cluster labels assigned to each data point.
    algorithm_name (str): The name of the clustering algorithm used, for display purposes.
    centroids (numpy array, optional): Centroid positions, if applicable. Defaults to None.

    Returns:
    float: The average silhouette score, which measures how well data points fit within their clusters.
    """

    # Calculate the silhouette score to evaluate the clustering performance
    silhouette_avg = silhouette_score(X_pca, labels)

    # Create a scatter plot of the data points, colored by their cluster labels
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o', s=100, label="Data points")

    # If centroids are provided, plot them on the scatter plot
    if centroids is not None:
        # Plot the centroids in red with an 'x' marker
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='x', label="Centroids")
        
        # Label each centroid with its cluster number
        for i, centroid in enumerate(centroids):
            plt.text(centroid[0], centroid[1], f'Cluster {i+1}', fontsize=12, fontweight='bold',
                     ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))
    else:
        # If centroids are not provided, calculate and label the mean position of each cluster
        for cluster in np.unique(labels):
            # Get all points belonging to the current cluster
            cluster_points = X_pca[labels == cluster]
            
            # Calculate the mean position of these points
            cluster_mean = np.mean(cluster_points, axis=0)
            
            # Label the mean position with the cluster number
            plt.text(cluster_mean[0], cluster_mean[1], f'Cluster {cluster + 1}', fontsize=12, fontweight='bold',
                     ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))

    # Add title and axis labels to the plot
    plt.title(f'{algorithm_name} Clustering on Dataset')
    plt.xlabel('Feature 1 (Standardized)')
    plt.ylabel('Feature 2 (Standardized)')
    
    # Add legend and grid for better visualization
    plt.legend()
    plt.grid(True)
    
    # Add a color bar to indicate cluster labels
    plt.colorbar(label='Cluster Label')
    
    # Save the plot as an image in the specified directory
    plt.savefig(f'Code/saved images/{algorithm_name}.jpg', format="jpg", dpi=300)
    
    # Display the plot
    plt.show()

    # Return the silhouette score as the evaluation metric
    return silhouette_avg
