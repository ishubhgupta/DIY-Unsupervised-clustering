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
     
    # Description: This is the main Streamlit application script that provides a web interface for customer segmentation. It integrates all clustering algorithms and allows users to train models, evaluate performance, and classify new customers through an interactive interface.
        # SQLite: Yes
        # MQs: No

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # Python 3.10.11
        # numpy 1.24.3
        # pandas 1.5.3
        # streamlit 1.40.0
        # joblib 1.3.1
        # scikit-learn 1.2.2

# importing all dependencies
import numpy as np
import pandas as pd
import streamlit as st # For building the web app
from train_Kmeans import train_model as train
from train_gmm import train_model as train_gmm
from train_db import train_model as train_db
from train_optics import train_model as train_optics
from train_birch import train_model as train_birch
from classification import classify
from ingest_transform import preprocess_test, store_path_to_sqlite, retrieve_data_from_sqlite



# Set up the Streamlit page
st.set_page_config(page_title="Customer Segmentation", page_icon=":cash:", layout="centered")
st.markdown("<h1 style='text-align: center; color: white;'>Customer Segmentation</h1>", unsafe_allow_html=True)
st.divider()

# Initialize session state variables if not already set
if "sqlite_db_path" not in st.session_state:
    st.session_state.sqlite_db_path = "Data/Processed/sqlite.db"

if "master_data_path" not in st.session_state:
    st.session_state.master_data_path = r"Data\Master\MOCK_DATA.csv"

tab1, tab2, tab3 = st.tabs(["Model Config", "Model Training & Evaluation", "Classification"])


# Inside tab1:
with tab1:
    uploaded_file = st.text_input("Enter the path to the Master data", value=st.session_state.master_data_path)

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Here is a preview of your data:")
            st.write(df.head())
            store_path_to_sqlite(uploaded_file, st.session_state.sqlite_db_path)
            st.session_state.master_data_path = uploaded_file
            st.success("Data successfully loaded and stored in database!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        df = retrieve_data_from_sqlite(st.session_state.sqlite_db_path, processed=False)
        if df is not None:
            st.write("Data preview from database:")
            st.write(df.head())
        else:
            st.warning("No data found. Please load a CSV file.")


# Inside tab2: This code block executes when the second tab is active in Streamlit.
with tab2:
    # Display a subheader for the model training section.
    st.subheader("Model Training")
    
    # Display a brief description of this section.
    st.write("This is where you can train the model.")
    
    # Add a horizontal divider line for better separation of sections.
    st.divider()

    # ---- K-Means Clustering Model Training ----
    
    # Set the model name to be used in headers and captions.
    model_name = 'K-Means'
    
    # Display the model name as a header, centered and styled with custom CSS.
    st.markdown(f"<h3 style='text-align: center; color: white;'>{model_name}</h3>", unsafe_allow_html=True)

    # Input for the number of clusters for the K-Means model.
    # Users can select between 2 and 10 clusters, with a default value of 3.
    num_clusters = st.number_input('Number of clusters:', min_value=2, max_value=10, value=3, step=1)

    # Button to trigger the training of the K-Means model.
    if st.button(f"Train {model_name} Model", use_container_width=True):
        # Display a status message indicating that training is in progress.
        with st.status("Training K-Means Model..."):
            df = retrieve_data_from_sqlite(st.session_state.sqlite_db_path, processed=True)
            if df is not None:
                score = train(df, num_clusters)
                st.write(f"Training complete! The score is: {score}")
                # Display the training completion message with the score.
                st.write(f"Training complete! The score is: {score}")

        # Show a success message upon successful training of the model.
        st.success(f"{model_name} Trained Successfully")

        # Display a header for model evaluation results, centered and styled.
        st.markdown("<h4 style='text-align: center; color: white;'>Model Evaluation </h4>", unsafe_allow_html=True)

        # Display the silhouette score as part of the evaluation metric.
        st.write(f"Silhouette: {score}")

        # Show an image of the cluster graph saved under the specified path.
        st.image(f"Code/saved images/{model_name}.jpg", caption=f"Cluster graph of {model_name}", width=600)

    # Add a divider to separate this section from others.
    st.divider()
    
    # ---- Gaussian Mixture Model Training ----
    
    model_name = 'Gaussian Mixture Model'
    
    # Display the model name header with styling.
    st.markdown(f"<h3 style='text-align: center; color: white;'>{model_name}</h3>", unsafe_allow_html=True)

    # Slider to select the number of components for the Gaussian Mixture Model.
    n_component = st.slider('n_component:', min_value=2, max_value=10, step=1)

    # Button to trigger training of the Gaussian Mixture Model.
    if st.button(f"Train {model_name} Model", use_container_width=True):
        # Display status message while the model is training.
        with st.status(f"Training {model_name}..."):
            df = retrieve_data_from_sqlite(st.session_state.sqlite_db_path, processed=True)
            if df is not None:
                score = train_gmm(df, n_component)
                st.write(f"Training complete! The score is: {score}")

        # Display success message upon completion.
        st.success(f"{model_name} Trained Successfully")

        # Display the evaluation header and the silhouette score.
        st.markdown("<h4 style='text-align: center; color: white;'>Model Evaluation </h4>", unsafe_allow_html=True)
        st.write(f"Silhouette: {score}")

        # Display the cluster graph image for the trained model.
        st.image(f"Code/saved images/{model_name}.jpg", caption=f"Cluster graph of {model_name}", width=600)

    # Add a divider to separate the sections.
    st.divider()
    
    # ---- DBSCAN Model Training ----
    
    model_name = 'DBSCAN'
    
    # Display the model name header with styling.
    st.markdown(f"<h3 style='text-align: center; color: white;'>{model_name}</h3>", unsafe_allow_html=True)

    # Slider to select the EPS value, which is the distance threshold for clustering in DBSCAN.
    eps = st.slider('EPS (distance threshold):', min_value=0.30, max_value=2.0, value=0.5, step=0.1)

    # Slider to select the minimum samples required to form a cluster.
    min_sm = st.slider('Min Samples', min_value=20, max_value=50, step=2)

    # Button to trigger training of the DBSCAN model.
    if st.button(f"Train {model_name} Model", use_container_width=True):
        # Display status message while the model is training.
        with st.status(f"Training {model_name}..."):
            df = retrieve_data_from_sqlite(st.session_state.sqlite_db_path, processed=True)
            if df is not None:
                score = train_db(df, eps, min_sm, minmax=True)
                st.write(f"Training complete! The score is: {score}")

        # Display success message upon completion.
        st.success(f"{model_name} Trained Successfully")

        # Display the evaluation header and the silhouette score.
        st.markdown("<h4 style='text-align: center; color: white;'>Model Evaluation </h4>", unsafe_allow_html=True)
        st.write(f"Silhouette: {score}")

        # Display the cluster graph image for the trained model.
        st.image(f"Code/saved images/{model_name}.jpg", caption=f"Cluster graph of {model_name}", width=600)

    # Add a divider to separate the sections.
    st.divider()
    
    # ---- OPTICS Model Training ----
    
    model_name = 'OPTICS'
    
    # Display the model name header with styling.
    st.markdown(f"<h3 style='text-align: center; color: white;'>{model_name}</h3>", unsafe_allow_html=True)

    # Slider to select the minimum samples required to form a cluster.
    min_sample = st.slider('min_samples:', min_value=2, max_value=20, step=1)

    # Slider to select the xi value, which is a steepness threshold used in OPTICS clustering.
    xi = st.slider('xi:', min_value=0.1, max_value=0.75, step=0.01)

    # Slider to select the minimum cluster size.
    cluster = st.slider('min_cluster_size:', min_value=0.1, max_value=0.25, step=0.05)

    # Button to trigger training of the OPTICS model.
    if st.button(f"Train {model_name} Model", use_container_width=True):
        # Display status message while the model is training.
        with st.status(f"Training {model_name}..."):
            df = retrieve_data_from_sqlite(st.session_state.sqlite_db_path, processed=True)
            if df is not None:
                score = train_optics(df, min_sample, xi, cluster, minmax=True)
                st.write(f"Training complete! The score is: {score}")

        # Display success message upon completion.
        st.success(f"{model_name} Trained Successfully")

        # Display the evaluation header and the silhouette score.
        st.markdown("<h4 style='text-align: center; color: white;'>Model Evaluation </h4>", unsafe_allow_html=True)
        st.write(f"Silhouette: {score}")

        # Display the cluster graph image for the trained model.
        st.image(f"Code/saved images/{model_name}.jpg", caption=f"Cluster graph of {model_name}", width=600)

    # Add a divider to separate the sections.
    st.divider()
    
    # ---- BIRCH Model Training ----
    
    model_name = 'BIRCH'
    
    # Display the model name header with styling.
    st.markdown(f"<h3 style='text-align: center; color: white;'>{model_name}</h3>", unsafe_allow_html=True)

    # Input for the number of clusters, allowing users to select between 2 and 10 clusters.
    n_clus1 = st.number_input('Number of clusters:', min_value=2, max_value=10, value=3, step=1, key='clusters_option_1')

    # Slider to select the threshold value for forming clusters in BIRCH.
    threas = st.slider('min_cluster_size:', min_value=0.1, max_value=0.55, step=0.05, key='thres_option_2')

    # Button to trigger training of the BIRCH model.
    if st.button(f"Train {model_name} Model", use_container_width=True):
        # Display status message while the model is training.
        with st.status(f"Training {model_name}..."):
            df = retrieve_data_from_sqlite(st.session_state.sqlite_db_path, processed=True)
            if df is not None:
                score = train_birch(df, n_clus1, threas, minmax=True)
                st.write(f"Training complete! The score is: {score}")

        # Display success message upon completion.
        st.success(f"{model_name} Trained Successfully")

        # Display the evaluation header and the silhouette score.
        st.markdown("<h4 style='text-align: center; color: white;'>Model Evaluation </h4>", unsafe_allow_html=True)
        st.write(f"Silhouette: {score}")

        # Display the cluster graph image for the trained model.
        st.image(f"Code/saved images/{model_name}.jpg", caption=f"Cluster graph of {model_name}", width=600)

    # Add a divider to mark the end of the training section.
    st.divider()

# Tab 3 for Customer Clustering
with tab3:
    # Dropdown to select the clustering algorithm
    algorithm = st.selectbox("Select algorithm:", ("K-Means", "Gaussian Mixture Model", "BIRCH"))

    # Form to enter customer details for clustering
    with st.form(key="clustering_form"):
        st.subheader("Clustering")
        st.write("Enter customer details for clustering.")

        # Input for age with validation between 18 and 80
        age = st.number_input('Enter age (18-80):', min_value=18, max_value=80, value=30)

        # Input for annual income with validation and formatting
        income = st.number_input(
            'Enter annual income (30000.00-100000.00):', 
            min_value=3000.00, max_value=300000.00, 
            value=50000.00, step=0.01, format="%.2f"
        )

        # Input for purchase history with min and max validation
        purchase_history = st.number_input('Enter Purchase history :', min_value=100, max_value=50000, value=500)

        # Input for customer spending score with a range of 0 to 100
        customer_spending_score = st.number_input('Enter Customer Spending Score :', min_value=0, max_value=100, value=50)

        # Input for frequency of visits with a range of 0 to 100
        freq_of_visit = st.number_input('Enter frequency of visit:', min_value=0, max_value=100, value=50)

        # List of gender options for the customer
        gender_opt = [
            'Male', 'Female', 'Agender', 'Genderqueer', 'Polygender', 
            'Genderfluid', 'Non-binary', 'Bigender'
        ]

        # Radio button for gender selection
        gender = preprocess_test(st.radio('Choose an option:', gender_opt, horizontal=True))

        # Radio button to select the region
        region = preprocess_test(st.radio('Choose an option:', ['East', 'West', 'North', 'South'], horizontal=True))

        # Radio button to select customer type
        customer_type = preprocess_test(st.radio('Choose an option:', ['budget', 'regular', 'premium'], horizontal=True))

        # Combine all inputs into a numpy array
        items = np.array([
            [age, income, purchase_history, customer_spending_score, 
             freq_of_visit, gender, region, customer_type]
        ])

        # Submit button to perform clustering
        if st.form_submit_button("Cluster", use_container_width=True):
            # Call classify function to determine the cluster for the input data
            new_cluster = classify(algorithm, items)
            
            # Display the cluster result
            st.write(f"The Data belong to {new_cluster}")
            
            # Display the corresponding cluster graph image for the selected algorithm
            st.image(f"Code/saved images/{algorithm}.jpg", caption=f"Cluster graph of {algorithm}", width=600)
