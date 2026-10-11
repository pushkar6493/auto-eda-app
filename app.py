import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Auto EDA App", layout="wide")
st.title("📊 Auto EDA App - Day 1 Project")
st.write("Upload any CSV and get instant analysis")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("1. Data Preview")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("2. Data Types & Missing Values")
    st.write(pd.DataFrame({"dtype": df.dtypes, "missing": df.isnull().sum(), "missing_%": (df.isnull().sum()/len(df)*100).round(2)}))

    st.subheader("3. Numerical Summary")
    st.dataframe(df.describe())

    st.subheader("4. Correlation Heatmap")
    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.write("No numeric columns for correlation")

    st.subheader("5. Distribution Plot")
    col_to_plot = st.selectbox("Select column to plot", df.columns)
    if col_to_plot:
        fig2, ax2 = plt.subplots()
        if pd.api.types.is_numeric_dtype(df[col_to_plot]):
            df[col_to_plot].hist(ax=ax2, bins=30)
        else:
            df[col_to_plot].value_counts().head(10).plot(kind='bar', ax=ax2)
        st.pyplot(fig2)

else:
    st.info("Try with sample: Bangalore House Prices dataset from Kaggle")
