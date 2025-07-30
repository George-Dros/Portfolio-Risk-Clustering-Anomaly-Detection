import streamlit as st
import data_loader as dl
import visuals as vs

# Streamlit App
st. set_page_config(layout="wide",page_title="Clustering Analysis 📊",page_icon="" )
st.title("Anomaly Detection")
st.markdown("")

#sidebar
index = st.sidebar.selectbox(
    "Select select: SP500 / FTSE100:",
    options=["SP500", "FTSE100"],
    index=1,
)

summary, clusters, scaled_features = dl.retrieve_data(index)

outliers = st.sidebar.toggle("Enable Outliers", value=False)

filter_cluster = st.sidebar.selectbox(
    "Filter by Cluster:",
    options=[-1, 0, 1],
    index=1,
)

feature = st.sidebar.selectbox(
    "Select Feature for Coloring:",
    options=["Cluster", "Sharpe", "Volatility", "Drawdown", "Skewness", "Kurtosis"],
    index=0,
)

anomaly_threshold = st.sidebar.slider(
    "Set Sharpe Anomaly Threshold:",
    min_value = 0.5,
    max_value = 3.0,
    value = 0.5,
    step = 0.1
)


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview & Summary",
    "Interactive UMAP Cluster Explorer",
    "Anomaly Detection Explorer",
    "Cluster Profiles & Comparison",
    "Correlation Heatmap",
    "Full Feature Table"])

with tab1:
    st.write("Overview & Summary")
    st.markdown("""
    This dashboard provides an overview of the portfolio risk clustering and anomaly detection project.
    Explore stock cluster distributions, outlier behavior, and statistical breakdowns in a clean visual format.
    """)

    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Stocks", len(summary))
    with col2:
        st.metric("Clusters Found", len(clusters["Cluster"].unique()))
    with col3:
        clusters_with_z, anomalies = dl.compute_anomalies(clusters)
        st.metric("Anomalies Detected", len(anomalies))
    st.divider()

    st.subheader("Cluster Distribution")
    st.markdown("Visual breakdown of how stocks are grouped by risk cluster.")
    # Placeholder for pie chart
    fig = vs.plot_cluster_pie(clusters)
    st.pyplot(fig)

    st.divider()


with tab2:
    st.write("Interactive UMAP Cluster Explorer")

    x_umap, y_umap, tickers = dl.umap_embedding(scaled_features, summary)
    fig = vs.plot_umap_scatter(x_umap, y_umap, tickers)
    st.pyplot(fig)
with tab3:
    st.write("Anomaly Detection Explorer")
with tab4:
    st.write("Cluster Profiles & Comparison")
with tab5:
    st.write("Correlation Heatmap")
with tab6:
    st.write("Full Feature Table")
    