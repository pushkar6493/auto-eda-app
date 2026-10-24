import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Auto EDA App", layout="wide")
st.title("📊 Auto EDA App - Pro")
st.markdown("Upload any CSV and get full EDA + Seaborn visuals + Report")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.success(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📉 Missing & Stats", "📊 Seaborn Plots", "📥 Download Report"])
    
    with tab1:
        st.subheader("Preview")
        st.dataframe(df.head())
        st.write("Columns:", list(df.columns))
        st.write("Dtypes:")
        st.write(df.dtypes)

    with tab2:
        st.subheader("Missing Values")
        missing = df.isnull().sum()
        st.dataframe(missing[missing > 0])
        if missing.sum() == 0:
            st.info("No missing values! ✅")
        
        st.subheader("Summary Stats")
        st.dataframe(df.describe())

    with tab3:
        st.subheader("Seaborn Visuals")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # 1. Correlation Heatmap
        if len(numeric_cols) > 1:
            st.write("**Correlation Heatmap**")
            fig, ax = plt.subplots()
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        
        # 2. Distribution Plot
        if numeric_cols:
            col = st.selectbox("Select numeric column for Distribution", numeric_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)
        
        # 3. Count Plot
        if cat_cols:
            col2 = st.selectbox("Select categorical column for Countplot", cat_cols)
            fig, ax = plt.subplots()
            sns.countplot(y=df[col2], ax=ax, order=df[col2].value_counts().index[:10])
            st.pyplot(fig)
            
        # 4. Scatter Plot
        if len(numeric_cols) >= 2:
            st.write("**Scatter Plot**")
            x_col = st.selectbox("X-axis", numeric_cols, key="x")
            y_col = st.selectbox("Y-axis", numeric_cols, key="y")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
            st.pyplot(fig)

    with tab4:
        st.subheader("Download EDA Report")
        report = f"""
        Auto EDA Report
        Rows: {df.shape[0]}
        Columns: {df.shape[1]}
        Columns List: {list(df.columns)}
        
        Missing Values:
        {df.isnull().sum().to_string()}
        
        Summary:
        {df.describe().to_string()}
        """
        st.download_button("Download Report as TXT", report, file_name="eda_report.txt")
        st.download_button("Download Cleaned Data (Original)", df.to_csv(index=False), file_name="data.csv")

else:
    st.info("👆 Upload a CSV to start - Try Bangalore House Price CSV")
