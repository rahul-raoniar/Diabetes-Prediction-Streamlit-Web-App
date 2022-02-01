import streamlit as st

# Load EDA packages
import pandas as pd

#Load visualisation libraries
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import plotly.express as px

# Load dataset
@st.cache
def load_data(data):
    df = pd.read_csv(data)
    return df

def run_eda_app():
    st.success("Exploratory Data Analysis")
    #df = pd.read_csv("data/diabetes_data_upload.csv")
    df = load_data("diabetes_data_upload.csv")
    df_encoded = load_data("diabetes_data_upload_clean.csv")
    freq_df = load_data("freqdist_of_age_data.csv")
    
    
    submenu = st.sidebar.selectbox("Submenu", ["Descriptive", "Plots"])
    if submenu == "Descriptive":
        st.subheader("Descriptive Stats.")
        st.dataframe(df)
        
        with st.beta_expander("Data Types"):
            st.write(df.dtypes)
            
        with st.beta_expander("Descriptive Summary"):
            st.write(df_encoded.describe())
            
        with st.beta_expander("Class Distribution"):
            st.write(df["class"].value_counts())
            
        with st.beta_expander("Gender Distribution"):
            st.write(df["Gender"].value_counts())
    
    elif submenu == "Plots":
        st.subheader("Plots")
        
        #Layout
        col1, col2 = st.beta_columns([2, 1])
        
        with col1:
            # Gender Frequency Distribution
            with st.beta_expander("Frequency Plot of Gender"):
                # Using seaborn
                # fig, ax = plt.subplots(figsize = (2,2))
                # sns.countplot(df["Gender"], ax = ax)
                # st.pyplot(fig)
                
                gen_df = df["Gender"].value_counts().to_frame()
                gen_df = gen_df.reset_index()
                gen_df.columns = ["Gender Type", "Counts"]
                
                # Using plotly
                p1 = px.pie(gen_df, names = "Gender Type", values="Counts")
                st.plotly_chart(p1, use_container_width=True)
                
            # For class distribution
            with st.beta_expander("Frequency Plot of Class"):
                #Using seaborn
                # fig, ax = plt.subplots(figsize = (2,2))
                # sns.countplot(df["class"], ax = ax)
                # st.pyplot(fig)
                
                class_df = df["class"].value_counts().to_frame()
                class_df = class_df.reset_index()
                class_df.columns = ["Class Type", "Counts"]
                
                # Using plotly
                p2 = px.pie(class_df, names = "Class Type", values="Counts")
                st.plotly_chart(p2, use_container_width=True)   
                
            # Age Frequency Distribution
            with st.beta_expander("Distribution of Age"):
                 p3 = px.bar(freq_df, x = "Age", y = "count")
                 st.plotly_chart(p3, use_container_width=True)       
        
        
        with col2:
            with st.beta_expander("Gender Frequency Count"):
                st.dataframe(gen_df)
                
            with st.beta_expander("Class Frequency Count"):
                st.dataframe(class_df)
                
            with st.beta_expander("Distribution of Age"):
                st.dataframe(freq_df)
                
        
        # Outlier detection using boxplot
        with st.beta_expander("Box Plot for Outlier Detection"):
            p4 = px.box(df, x = df["Age"], color = "Gender")
            st.plotly_chart(p4)
            
        # Correlation plot
        with st.beta_expander("Correlation Plot"):
            corr_matrix = df_encoded.corr()
            p5 = px.imshow(corr_matrix, labels=dict(color="Correlation"))
            st.plotly_chart(p5, use_container_width=True)
                
    
        

        
    
    
    
    
    