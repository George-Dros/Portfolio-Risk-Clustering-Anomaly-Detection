import matplotlib.pyplot as plt
import data_loader as dl

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