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


import pandas as pd
import streamlit as st # For building the web app
from train_Kmeans import train_model as train
from train_gmm import train_model as train_gmm
from train_db import train_model as train_db

st.set_page_config(page_title="Unsupervised Clustering", page_icon=":cash:", layout="centered")
st.markdown("<h1 style='text-align: center; color: white;'>Unsupervised Clustering </h1>", unsafe_allow_html=True)
st.divider()


tab1, tab2=  st.tabs(["Model Config", "Model Training & Evaluation"])

with tab1:  
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

    else:
        df = pd.read_csv("MOCK_DATA.csv")

    st.write("Here is a preview of your data:")
    st.write(df.head())

with tab2:
    st.subheader("Model Training")
    st.write("This is where you can train the model.")
    st.divider()

    # K-Means

    model_name = 'K-Means'
    st.markdown(f"<h3 style='text-align: center; color: white;'>{model_name}</h3>", unsafe_allow_html=True)

    if st.button(f"Train {model_name} Model", use_container_width=True):
        
        with st.status("Training K-Means Model..."):
            score= train(df)

        st.success(f"{model_name} Trained Sucessully")
        st.markdown("<h4 style='text-align: center; color: white;'>Model Evaluation </h4>", unsafe_allow_html=True)

        st.write(f"Silhouette: {score}")
        st.image(f"Code/saved images/{model_name}.jpg", caption=f"Cluster graph of {model_name}", width=600)

        st.divider()
        
    # Gaussian Mixture Model
    
    model_name = 'Gaussian Mixture Model'
    st.markdown(f"<h3 style='text-align: center; color: white;'>{model_name}</h3>", unsafe_allow_html=True)

    if st.button(f"Train {model_name} Model", use_container_width=True):
        
        with st.status(f"Training {model_name}..."):
            score= train_gmm(df)

        st.success(f"{model_name} Trained Sucessully")
        st.markdown("<h4 style='text-align: center; color: white;'>Model Evaluation </h4>", unsafe_allow_html=True)

        st.write(f"Silhouette: {score}")
        st.image(f"Code/saved images/{model_name}.jpg", caption=f"Cluster graph of {model_name}", width=600)

        st.divider()
        
        
    # DBSCAN
    
    model_name = 'DBSCAN'
    st.markdown(f"<h3 style='text-align: center; color: white;'>{model_name}</h3>", unsafe_allow_html=True)

    if st.button(f"Train {model_name} Model", use_container_width=True):
        
        with st.status(f"Training {model_name}..."):
            score= train_db(df)

        st.success(f"{model_name} Trained Sucessully")
        st.markdown("<h4 style='text-align: center; color: white;'>Model Evaluation </h4>", unsafe_allow_html=True)

        st.write(f"Silhouette: {score}")
        st.image(f"Code/saved images/{model_name}.jpg", caption=f"Cluster graph of {model_name}", width=600)

        st.divider()

