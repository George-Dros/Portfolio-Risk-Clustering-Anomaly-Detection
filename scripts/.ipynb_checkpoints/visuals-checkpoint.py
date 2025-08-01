import matplotlib.pyplot as plt
import data_loader as dl
import seaborn as sns

def plot_cluster_pie(df):
    cluster_counts = df["Cluster"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title("Cluster Distribution")
    fig.tight_layout()

    return fig

def plot_umap_scatter(x, y, tickers, title="UMAP 2D Embedding"):
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(x, y, alpha=0.7)

    # Ticker labels
    for i, ticker in enumerate(tickers):
        ax.text(x[i] + 0.05, y[i] + 0.05, ticker, fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True)

    fig.tight_layout()

    return fig  

def plot_cluster_scatter(x, y, tickers,cluster_labels, title="HDBSCAN Clustering on UMAP Embedding"):
    fig, ax = plt.subplots(figsize=(8, 6))

    palette = sns.color_palette('hls', len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0))
    colors = [palette[label] if label != -1 else (0.5, 0.5, 0.5) for label in cluster_labels]

    scatter = ax.scatter(x, y, c=colors, s=100, alpha=0.7)    
    
    for i, ticker in enumerate(tickers):
        ax.text(x[i] + 0.05, y[i] + 0.05, ticker, fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True)
    fig.tight_layout()

    return fig   

def plot_feature_bar(selected_data):
    
    features = selected_data.drop(labels=["Cluster", "Sharpe_z"], errors="ignore")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(features.index, features.values)
    
    ax.set_title(f"Feature Breakdown for {getattr(selected_data, 'name', 'Unknown')}")
    ax.set_ylabel("Value")
    ax.set_xticklabels(features.index, rotation=45)
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_sharpe_zscore_distribution(df, selected_ticker):
    selected_data = df.loc[selected_ticker]
    cluster_id = selected_data["Cluster"]
    cluster_members = df[df["Cluster"] == cluster_id]

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(cluster_members["Sharpe_z"], bins=20, kde=True, ax=ax, color="skyblue")
    ax.axvline(selected_data["Sharpe_z"], color="red", linestyle="--", label=f"{selected_ticker}")
    ax.set_title(f"Z-Score Distribution of Sharpe Ratio (Cluster {cluster_id})")
    ax.set_xlabel("Sharpe Ratio Z-Score")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_cluster_heatmap(cluster_summary):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cluster_summary, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, ax=ax)

    ax.set_title("Cluster Feature Averages")
    ax.set_ylabel("Cluster")
    ax.set_xlabel("Feature")
    fig.tight_layout()
    return fig   


def plot_cluster_feature_distribution(df, feature, index="S&P 500"):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="Cluster", y=feature, data=df, ax=ax, palette="Set2")
    
    ax.set_title(f"{feature} Distribution Across Clusters ({index})")
    ax.set_xlabel("Cluster")
    ax.set_ylabel(feature)
    ax.grid(True)
    fig.tight_layout()
    return fig    

def plot_correlation_heatmap(corr_matrix, title="Correlation Heatmap"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig