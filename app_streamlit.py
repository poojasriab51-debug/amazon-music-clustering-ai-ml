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

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # -------------------------
    # Load Dataset
    # -------------------------

    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(
            uploaded_file,
            sep=None,
            engine="python"
        )

    # -------------------------
    # Dataset Preview
    # -------------------------

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------
    # EDA - Correlation Heatmap
    # -------------------------

    eda_features = [
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

    st.subheader("Feature Correlation Heatmap")

    fig_heat, ax_heat = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        df[eda_features].corr(),
        cmap='coolwarm',
        annot=False,
        ax=ax_heat
    )

    ax_heat.set_title("Correlation Between Audio Features")

    st.pyplot(fig_heat)

    # -------------------------
    # Feature Selection
    # -------------------------

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

    # -------------------------
    # Standardization
    # -------------------------

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------------
    # KMeans Clustering
    # -------------------------

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_scaled)

    df['Cluster'] = clusters

    # -------------------------
    # PCA
    # -------------------------

    pca = PCA(n_components=2)

    pca_features = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        'PCA1': pca_features[:, 0],
        'PCA2': pca_features[:, 1],
        'Cluster': clusters
    })

    
    # PCA Visualization
    

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

    cluster_counts = (
        df['Cluster']
        .value_counts()
        .sort_index()
    )

    st.dataframe(cluster_counts)

    
    # Cluster Percentages
    

    st.subheader("Cluster Distribution (%)")

    cluster_percent = (
        df['Cluster']
        .value_counts(normalize=True)
        .sort_index() * 100
    )

    st.dataframe(
        cluster_percent.round(2)
        .rename("Percentage")
    )


    # Cluster Interpretation


    st.subheader("Cluster Interpretation")

    st.markdown("""
    🔴 **Cluster 0** → Acoustic / Calm Songs

    🔵 **Cluster 1** → Energetic / Dance Songs

    🟢 **Cluster 2** → Balanced / Moderate Songs
    """)

    
    # Cluster Summary Statistics


    st.subheader("Cluster Summary Statistics")

    cluster_summary = df.groupby('Cluster')[[
        'danceability',
        'energy',
        'acousticness',
        'valence',
        'tempo'
    ]].mean()

    st.dataframe(
        cluster_summary.round(3)
    )

    
    # Sample Songs
    

    if 'name_song' in df.columns:

        st.subheader("Sample Songs From Each Cluster")

        for cluster in sorted(df['Cluster'].unique()):

            st.write(f"### Cluster {cluster}")

            sample_songs = (
                df[df['Cluster'] == cluster]
                [['name_song']]
                .head(10)
            )

            st.dataframe(sample_songs)

    
    # Song Recommendation
    

    if 'name_song' in df.columns:

        st.subheader("🎧 Song Recommendation System")

        song_list = sorted(
            df['name_song']
            .dropna()
            .astype(str)
            .unique()
        )

        selected_song = st.selectbox(
            "Choose a Song",
            song_list
        )

        selected_cluster = df[
            df['name_song'] == selected_song
        ]['Cluster'].iloc[0]

        recommendations = (
            df[
                (df['Cluster'] == selected_cluster)
                &
                (df['name_song'] != selected_song)
            ][['name_song']]
            .drop_duplicates()
            .head(10)
        )

        st.write(
            f"Recommended songs from Cluster {selected_cluster}"
        )

        st.dataframe(recommendations)

    
    # Final Dataset Preview
    

    st.subheader("Clustered Dataset Preview")

    st.dataframe(df.head())
