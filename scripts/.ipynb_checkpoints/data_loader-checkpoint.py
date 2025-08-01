import pandas as pd
import numpy as np
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
import hdbscan

def retrieve_data(index):
    if index == "SP500":
        summary_df = pd.read_pickle("/data/processed/sp500_summary.pkl")
        df_clusters = pd.read_pickle("/data/processed/df_clusters_sp500.pkl")
    if index == "FTSE100":        
        summary_df = pd.read_pickle("/data/processed/ftse100_summary.pkl")
        df_clusters = pd.read_pickle("/data/processed/df_clusters_ftse100.pkl")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(summary_df)
    return summary_df, df_clusters, scaled_features    

def compute_anomalies(df, threshold=1.5):
    df = df.copy()
    df["Sharpe_z"] = df.groupby("Cluster")["Sharpe Ratio"].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    anomalies = df[np.abs(df["Sharpe_z"]) > threshold]
    return df, anomalies

def umap_embedding(scaled_features, summary_df):
    umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
    umap_embedding = umap_model.fit_transform(scaled_features)
    tickers = summary_df.index.tolist()
    x = umap_embedding[:, 0]
    y = umap_embedding[:, 1]
    return x, y, tickers, umap_embedding

def clustering(umap_embedding):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
    cluster_labels = clusterer.fit_predict(umap_embedding)    
    return cluster_labels