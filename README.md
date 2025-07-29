# Portfolio Risk Clustering & Anomaly Detection

Uncovering hidden asset relationships and surfacing abnormal risk behavior using unsupervised machine learning.

---

## Project Overview

This project explores patterns of financial risk by analyzing stocks from the **S&P 500** and **FTSE 100** indices.  
Using dimensionality reduction and unsupervised clustering techniques, we group stocks based on volatility, correlation, and other statistical properties — and detect outliers with unusual behavior.

The goal: provide insights for portfolio construction, risk management, and anomaly detection.

---

## Tools & Libraries

| Layer        | Tools Used                                                  |
|--------------|-------------------------------------------------------------|
| **Data**     | `yfinance`, `pandas`, `numpy`                               |
| **ML**       | `scikit-learn`, `umap-learn`, `hdbscan`, `statsmodels`      |
| **Visuals**  | `matplotlib`, `seaborn`                                     |
| **Notebook** | `JupyterLab`, `joblib`, `pickle`                            |
| **Repo**     | `GitHub`, `Markdown`                                        |

---

## Core Methods

- **Feature Engineering**: returns, volatility, drawdowns, skewness, kurtosis, Sharpe ratio, correlation
- **Dimensionality Reduction**: PCA, UMAP
- **Clustering**: HDBSCAN
- **Anomaly Detection**: HDBSCAN outliers flagged for further analysis
- **Visualizations**: cluster maps, anomaly overlays, Sharpe ratio gradients

---

## 📁 Project Structure

```
portfolio-risk-clustering/
│
├── data/
│   ├── raw/                   # Raw OHLCV data (from yfinance)
│   ├── processed/             # Cleaned returns and metrics per asset
│   └── outputs/               # Cluster results, PCA/UMAP coordinates
│
├── notebooks/
│   ├── 1_data_collection.ipynb         # Ticker selection and OHLCV retrieval
│   ├── 2_feature_engineering.ipynb     # Returns, volatility, drawdowns, Sharpe
│   ├── 3_dimensionality_clustering.ipynb  # PCA, UMAP, HDBSCAN analysis
│   └── 4_insights_and_anomalies.ipynb  # Cluster analysis and anomaly detection
│
├── .gitignore
├── README.md
└── requirements.txt
```


---

## 📈 Pipeline Summary

### Phase 1 – Data Collection & Cleaning
- Retrieved 5-year OHLCV data for S&P 500 and FTSE 100 via `yfinance`
- Forward-filled missing values
- Aligned all time series data

### Phase 2 – Feature Engineering
- Calculated log returns, 30-day rolling volatility, drawdowns, correlation, skewness, kurtosis, and Sharpe ratios
- Constructed summary DataFrames for each asset

### Phase 3 – Dimensionality Reduction & Clustering
- PCA applied for linear inspection
- UMAP applied for non-linear projection (n_neighbors=5, min_dist=0.3)
- HDBSCAN used to reveal clusters and outliers
- Visualized clusters with Sharpe ratio overlays and labeled tickers

### Phase 4 – Insights & Outlier Detection
- Identified clusters of high-volatility or negative Sharpe assets
- Isolated HDBSCAN outliers for anomaly labeling
- Compared S&P 500 vs FTSE 100 risk behavior
- Derived conclusions on risk concentration and market dynamics

---

## Key Insights

### S&P 500
- Formed ~4-5 clusters of similar risk profiles
- Several outliers displayed abnormal volatility or highly negative Sharpe ratios (e.g., specific tech or biotech stocks)

### FTSE 100
- Showed more conservative clustering with tighter Sharpe ratios
- Outliers were mostly isolated financial or commodity-linked companies
- Stronger presence of high-kurtosis assets

---

## (Optional Enhancements)

- Add **Isolation Forest** or **Elliptic Envelope** for statistical anomaly detection
- Deploy **interactive Streamlit dashboard**
- Add **macroeconomic overlays** (e.g., VIX, CPI) for anomaly context

---

## How to Use

1. Clone the repo  
2. Create environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
Run notebooks in order:
```
    1. Data Cleaning.ipynb

    2. Feature Engineering.ipynb

    3. ML & Clustering.ipynb

    4. Insights.ipynb
```

## Author

**Georgios Drosogiannis**  
 MSc in Applied Mathematics (NTUA)    
[LinkedIn](https://www.linkedin.com/in/georgios-drosogiannis/)  
[GitHub](https://github.com/George-Dros)

---