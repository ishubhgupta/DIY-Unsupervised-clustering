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

st.set_page_config(page_title="Unsupervised Clustering", page_icon=":cash:", layout="centered")
st.markdown("<h1 style='text-align: center; color: white;'>Unsupervised Clustering </h1>", unsafe_allow_html=True)
st.divider()


tab1, tab2, tab3, tab4 =  st.tabs(["Model Config", "Model Training", "Model Evaluation", "Model Prediction"])

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
    
    st.markdown("<h3 style='text-align: center; color: white;'>K-Means</h3>", unsafe_allow_html=True)