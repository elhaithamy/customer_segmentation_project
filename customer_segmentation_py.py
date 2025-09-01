import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import datetime as dt

# Streamlit App Title
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
st.title("📊 Customer Segmentation Dashboard")
st.write("Analyze customer behavior and segment them using clustering techniques.")

# File Upload (CSV + Excel Support)
uploaded_file = st.file_uploader("📂 Upload your file", type=["csv", "xlsx"])

if uploaded_file:
    # Read CSV or Excel dynamically
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"⚠️ Error reading the file: {e}")
        st.stop()

    # Show raw data preview
    st.subheader("📌 Raw Data Preview")
    st.dataframe(df.head())

    # Convert date column to datetime if exists
    if 'last_order' in df.columns:
        df['last_order'] = pd.to_datetime(df['last_order'], errors='coerce')

    # Fill missing values if any
    df.fillna(0, inplace=True)

    # Calculate RFM Metrics
    st.subheader("📌 Calculating RFM Metrics")
    today = dt.datetime.today()
    df['Recency'] = (today - df['last_order']).dt.days
    df.rename(columns={'n_order': 'Frequency', 'AVG_Value': 'Monetary'}, inplace=True)

    rfm_df = df[['customer_phone', 'Recency', 'Frequency', 'Monetary']].copy()
    st.dataframe(rfm_df.head())

    # Scale the RFM data
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

    # KMeans Clustering
    st.subheader("📌 KMeans Clustering")
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Merge Clusters with Original Data
    final_df = df.merge(rfm_df[['customer_phone', 'Cluster']], on='customer_phone', how='left')

    # Display Cluster Summary
    st.subheader("📌 Customer Segments Overview")
    cluster_summary = final_df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'customer_phone': 'count'
    }).rename(columns={'customer_phone': 'Customer Count'})

    st.dataframe(cluster_summary)

    # Visualize Cluster Sizes
    st.subheader("📊 Cluster Size Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='Cluster', data=rfm_df, palette="viridis", ax=ax)
    ax.set_title("Customer Distribution per Cluster")
    st.pyplot(fig)

    # Visualize Monetary Value per Cluster
    st.subheader("📊 Average Monetary Value by Cluster")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=cluster_summary.index, y=cluster_summary['Monetary'], palette="coolwarm", ax=ax)
    ax.set_title("Average Monetary Value by Cluster")
    st.pyplot(fig)

    # Download Segmented Data
    st.subheader("📥 Download Segmented Customer Data")
    csv_download = final_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Segmented Data as CSV",
        data=csv_download,
        file_name="segmented_customers.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Please upload a CSV or Excel file to get started.")

st.markdown("---")
st.caption("Developed by Haitham Hassan | Growth & Performance Analytics")
