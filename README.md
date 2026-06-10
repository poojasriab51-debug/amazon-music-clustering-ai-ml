# Amazon Music Clustering AI/ML (Streamlit)

This repo contains a Streamlit app to explore clustering results for the Amazon music dataset.

# Music Clustering and Recommendation using K-Means

## Project Overview

With thousands of songs available on music platforms, it becomes difficult to organize and discover similar music efficiently. This project uses Unsupervised Machine Learning to group songs based on their audio characteristics without using predefined labels.

The goal is to identify hidden patterns in music data and create meaningful clusters of songs with similar properties such as energy, danceability, acousticness, and tempo.

## Problem Statement

Music datasets contain a large number of songs with different audio features. Since these songs are not categorized into predefined groups, it is challenging to understand relationships between them.

This project applies K-Means Clustering to automatically discover groups of similar songs and visualize them using PCA.

## Dataset

The dataset contains song-level information including:

* Danceability
* Energy
* Loudness
* Speechiness
* Acousticness
* Instrumentalness
* Liveness
* Valence
* Tempo
* Duration

The dataset consists of approximately 95,000 songs.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit

## Methodology

### Data Preprocessing

Relevant audio features were selected from the dataset. The features were standardized using StandardScaler to ensure that all variables contributed equally during clustering.

### K-Means Clustering

K-Means clustering was applied to group songs with similar audio characteristics.

### Finding Optimal Clusters

The Elbow Method and Silhouette Score were used to determine the optimal number of clusters.

The best clustering result was obtained with:

* Number of Clusters (k): 3
* Silhouette Score: ~0.24

### PCA Visualization

Principal Component Analysis (PCA) was used to reduce the feature space into two dimensions for visualization purposes.

## Results

The model identified three major groups of songs:

### Cluster 0 – Acoustic / Calm Songs

* High acousticness
* Lower energy
* Lower danceability

### Cluster 1 – Energetic / Dance Songs

* High energy
* High danceability
* Lower acousticness

### Cluster 2 – Balanced Songs

* Moderate energy
* Moderate acousticness
* Balanced musical characteristics

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard that allows users to:

* Upload a music dataset
* View dataset previews
* Visualize clusters using PCA
* Explore cluster distributions
* Analyze cluster characteristics

## Key Learnings

Through this project, I learned:

* Data preprocessing techniques
* Feature scaling
* Unsupervised machine learning
* K-Means clustering
* Cluster evaluation using Silhouette Score
* Dimensionality reduction using PCA
* Building interactive dashboards with Streamlit

## Conclusion

This project successfully grouped songs into meaningful clusters using K-Means Clustering. The results demonstrate how unsupervised learning can uncover hidden patterns in music data and support music discovery, recommendation, and organization systems.
