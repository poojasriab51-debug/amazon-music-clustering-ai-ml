import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.title("Music Clustering Dashboard")

uploaded_file = st.file_uploader("Upload CSV File")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write(df.head())

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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)

    clusters = kmeans.fit_predict(X_scaled)

    df['Cluster'] = clusters

    pca = PCA(n_components=2)

    pca_features = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        'PCA1': pca_features[:,0],
        'PCA2': pca_features[:,1],
        'Cluster': clusters
    })

    fig, ax = plt.subplots(figsize=(10,6))

    sns.scatterplot(
        x='PCA1',
        y='PCA2',
        hue='Cluster',
        data=pca_df,
        palette='Set1'
    )

    st.pyplot(fig)

    st.write(df.head())