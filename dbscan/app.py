import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

st.set_page_config(page_title="DBSCAN Clustering", layout="wide")

st.title("DBSCAN Clustering - Mall Customers Dataset")

# Load Dataset
df = pd.read_csv("Mall_Customers.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Feature Selection
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sidebar Controls
st.sidebar.header("DBSCAN Parameters")

eps = st.sidebar.slider(
    "Epsilon (eps)",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.1
)

min_samples = st.sidebar.slider(
    "Min Samples",
    min_value=2,
    max_value=20,
    value=5
)

# DBSCAN Model
dbscan = DBSCAN(
    eps=eps,
    min_samples=min_samples
)

clusters = dbscan.fit_predict(X_scaled)

df["Cluster"] = clusters

# Number of Clusters
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

# Noise Points
noise_points = list(clusters).count(-1)

col1, col2 = st.columns(2)

with col1:
    st.metric("Clusters Found", n_clusters)

with col2:
    st.metric("Noise Points", noise_points)

# Cluster Visualization
st.subheader("DBSCAN Cluster Visualization")

fig, ax = plt.subplots(figsize=(8,6))

scatter = ax.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster']
)

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title("DBSCAN Clustering")

st.pyplot(fig)

# Cluster Distribution
st.subheader("Cluster Distribution")

cluster_counts = df["Cluster"].value_counts().sort_index()

st.bar_chart(cluster_counts)

# Clustered Dataset
st.subheader("Clustered Dataset")

st.dataframe(df)

# Download Results
csv = df.to_csv(index=False)

st.download_button(
    label="Download Clustered Dataset",
    data=csv,
    file_name="dbscan_output.csv",
    mime="text/csv"
)
