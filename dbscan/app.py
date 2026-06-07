import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from pathlib import Path

st.set_page_config(page_title="DBSCAN Clustering", layout="wide")

st.title("DBSCAN Clustering - Mall Customers")

# Find CSV automatically
base_dir = Path(__file__).parent

csv_files = list(base_dir.glob("*.csv"))

if not csv_files:
    st.error("No CSV file found in project folder.")
    st.write("Files available:")
    for file in base_dir.iterdir():
        st.write(file.name)
    st.stop()

df = pd.read_csv(csv_files[0])

st.success(f"Loaded: {csv_files[0].name}")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Parameters
eps = st.sidebar.slider("EPS", 0.1, 2.0, 0.5, 0.1)
min_samples = st.sidebar.slider("Min Samples", 2, 20, 5)

# DBSCAN
model = DBSCAN(
    eps=eps,
    min_samples=min_samples
)

clusters = model.fit_predict(X_scaled)

df["Cluster"] = clusters

# Metrics
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
noise = list(clusters).count(-1)

col1, col2 = st.columns(2)

with col1:
    st.metric("Clusters", n_clusters)

with col2:
    st.metric("Noise Points", noise)

# Visualization
st.subheader("Cluster Visualization")

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster']
)

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title("DBSCAN Clustering")

st.pyplot(fig)

# Cluster Counts
st.subheader("Cluster Distribution")
st.bar_chart(df["Cluster"].value_counts())

# Data
st.subheader("Clustered Dataset")
st.dataframe(df)

# Download
csv = df.to_csv(index=False)

st.download_button(
    "Download Results",
    csv,
    "dbscan_results.csv",
    "text/csv"
)
