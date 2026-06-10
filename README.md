# Amazon Music Clustering (Streamlit)

This repo contains a Streamlit app to explore clustering results for the Amazon music dataset.

## Run the app

```bash
pip install streamlit pandas matplotlib seaborn scikit-learn
streamlit run app_streamlit.py
```

## Data

Upload a CSV from your clustering pipeline (e.g., `clustered_songs.csv`).

The app expects numeric feature columns such as:
- `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`,
  `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`

and it will add a `Cluster` column for visualization.

