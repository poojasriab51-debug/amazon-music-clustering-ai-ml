import streamlit as st
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.title("🎵 Music Clustering Dashboard")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    # Load Data
    # Some CSVs may have inconsistent delimiters/quoting; try a forgiving read first.
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=None, engine="python")


    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Features for clustering
    features = [
        'danceability',
        'energy',
        'loudness',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'liveness',
        'valence',
        'tempo',
        'duration_ms'
    ]

    X = df[features]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    df['Cluster'] = clusters

    # PCA for visualization
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        'PCA1': pca_features[:, 0],
        'PCA2': pca_features[:, 1],
        'Cluster': clusters
    })

    # PCA Plot
    st.subheader("PCA Cluster Visualization")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        x='PCA1',
        y='PCA2',
        hue='Cluster',
        data=pca_df,
        palette='Set1',
        ax=ax
    )

    ax.set_title("Music Clusters using PCA")
    st.pyplot(fig)

    # Cluster Counts
    st.subheader("Songs in Each Cluster")

    cluster_counts = df['Cluster'].value_counts().sort_index()
    st.dataframe(cluster_counts)

    # Cluster Meaning
    st.subheader("Cluster Interpretation")

    st.markdown("""
    🔴 **Cluster 0** → Acoustic / Calm Songs

    🔵 **Cluster 1** → Energetic / Dance Songs

    🟢 **Cluster 2** → Balanced / Moderate Songs
    """)

    # Cluster Statistics
    st.subheader("Cluster Summary Statistics")

    cluster_summary = df.groupby('Cluster')[[
        'danceability',
        'energy',
        'acousticness',
        'valence',
        'tempo'
    ]].mean()

    st.dataframe(cluster_summary.round(3))

    # Final Dataset
    st.subheader("Clustered Dataset Preview")
    st.dataframe(df.head())