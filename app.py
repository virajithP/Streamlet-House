import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Set page config
st.set_page_config(page_title="Boston Housing Prices", layout="wide")

# Load data function
@st.cache_data
def load_data():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :3]])
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = raw_df.values[1::2, 2]
    return df

# Load model function
@st.cache_resource
def load_data():
    # Use the local CSV file
    df = pd.read_csv('boston_housing.csv') 
    
    # Ensure the required columns are present (optional check)
    if 'MEDV' not in df.columns:
        st.error("Error: 'MEDV' column not found in boston_housing.csv")
    
    return df
# Main App
st.title("Boston Housing Price Prediction 🏠")

tab1, tab2, tab3 = st.tabs(["Home", "EDA", "Prediction"])

df = load_data()

with tab1:
    st.header("Project Overview")
    st.write("""
    ### Welcome to the Boston Housing Price Prediction App!
    
    This application allows you to explore the famous Boston Housing dataset and predict the median value of owner-occupied homes (MEDV) based on various features.
    
    **Key Features:**
    - **Exploratory Data Analysis (EDA):** Visualize data distributions and correlations.
    - **Real-time Prediction:** Use machine learning to estimate house prices.
    
    **Dataset:**
    The dataset contains 506 samples and 13 feature variables, originally collected by the U.S. Census Service.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Boston_Back_Bay.jpg/800px-Boston_Back_Bay.jpg", caption="Boston Back Bay (Representative Image)")

with tab2:
    st.header("Exploratory Data Analysis")
    
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)
        
    with col2:
        st.subheader("Distributions")
        feature_to_plot = st.selectbox("Select Feature to Visualize", df.columns)
        fig_dist, ax_dist = plt.subplots()
        sns.histplot(df[feature_to_plot], kde=True, ax=ax_dist)
        st.pyplot(fig_dist)

with tab3:
    st.header("Model Prediction")
    
    model = load_model()
    
    if model:
        st.write("Adjust the sliders below to estimate the house price.")
        
        # Input features
        # We need inputs for all features used in training. 
        # For simplicity in this demo, we'll expose key ones and set mean for others, 
        # OR better yet, expose the ones requested: RM, LSTAT, PTRATIO, CRIM.
        
        col_pred1, col_pred2 = st.columns(2)
        
        with col_pred1:
            rm = st.slider("RM (Average number of rooms per dwelling)", 
                           min_value=float(df['RM'].min()), 
                           max_value=float(df['RM'].max()), 
                           value=float(df['RM'].mean()))
            
            lstat = st.slider("LSTAT (% lower status of the population)", 
                              min_value=float(df['LSTAT'].min()), 
                              max_value=float(df['LSTAT'].max()), 
                              value=float(df['LSTAT'].mean()))

        with col_pred2:
            ptratio = st.slider("PTRATIO (Pupil-teacher ratio by town)", 
                                min_value=float(df['PTRATIO'].min()), 
                                max_value=float(df['PTRATIO'].max()), 
                                value=float(df['PTRATIO'].mean()))
            
            crim = st.number_input("CRIM (Per capita crime rate by town)", 
                                   min_value=float(df['CRIM'].min()), 
                                   max_value=float(df['CRIM'].max()), 
                                   value=float(df['CRIM'].mean()))
        
        # Prepare input vector
        # We need to construction the full vector. We'll use means for non-input features.
        input_data = pd.DataFrame([df.drop('MEDV', axis=1).mean()], columns=df.drop('MEDV', axis=1).columns)
        
        # Update user inputs
        input_data['RM'] = rm
        input_data['LSTAT'] = lstat
        input_data['PTRATIO'] = ptratio
        input_data['CRIM'] = crim
        
        if st.button("Predict Price"):
            prediction = model.predict(input_data)[0]
            st.success(f"Estimated Median Value: ${prediction * 1000:,.2f}")
            
            st.info("Note: Prediction is based on the selected inputs while keeping other features at their average values.")
