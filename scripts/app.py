import streamlit as st
import data_loader as dl
import visuals as vs


st. set_page_config(layout="wide",page_title="Clustering Analysis 📊",page_icon="" )
st.title("Anomaly Detection")
st.markdown("")

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

    x_umap, y_umap, tickers, umap_embedding = dl.umap_embedding(scaled_features, summary)
    cluster_labels = dl.clustering(umap_embedding)
    fig = vs.plot_cluster_scatter(x_umap, y_umap, tickers, cluster_labels)
    st.pyplot(fig)
with tab3:
    st.write("Anomaly Detection Explorer")
    threshold = st.slider("Select Z-Score Threshold", 0.5, 3.0, 1.5, 0.1)
    df, anomalies = dl.compute_anomalies(clusters, threshold)

    if len(anomalies) > 0:
        st.dataframe(anomalies)
        selected_ticker = st.selectbox("Select an anomalous stock", anomalies.index.tolist())
        selected_data = df.loc[selected_ticker]
        fig = vs.plot_feature_bar(selected_data)
        st.pyplot(fig)

        fig2 = vs.plot_sharpe_zscore_distribution(anomalies, selected_ticker)
        st.pyplot(fig2)
with tab4:
    st.write("Cluster Profile Explorer")

    st.subheader(f"{index} Cluster Profile Explorer")

    merged = summary.join(clusters["Cluster"])
    cluster_summary = merged.groupby("Cluster").mean()

    st.subheader("Cluster Feature Averages")
    st.markdown("Mean values of financial metrics per cluster.")

    fig = vs.plot_cluster_heatmap(cluster_summary)
    st.pyplot(fig)

    st.divider()

    feature = st.selectbox("Compare Feature Distribution Across Clusters", cluster_summary.columns)
    fig2 = vs.plot_cluster_feature_distribution(merged, feature, index)
    st.pyplot(fig2)
    
with tab5:
    st.write("Correlation Heatmap")

    st.markdown(f"Visualizing the correlation structure of features across all assets in **{index}**.")

    corr_matrix = summary.corr()

    fig = vs.plot_correlation_heatmap(corr_matrix, title=f"{index} – Feature Correlation Heatmap")
    st.pyplot(fig)

with tab6:
    st.write("Full Feature Table")

    st.markdown(f"Browse the full feature set for all assets in **{index}**. Use the interactive options below to explore and filter data.")

    selected_feature = st.selectbox("Sort by Feature", summary.columns.tolist(), index=0)
    ascending = st.radio("Sort Order", ["Descending", "Ascending"], horizontal=True)

    sorted_df = summary.sort_values(by=selected_feature, ascending=(ascending == "Ascending"))

    st.dataframe(sorted_df, use_container_width=True)

    