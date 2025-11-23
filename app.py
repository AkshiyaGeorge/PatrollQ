
# Step 8: Streamlit Crime Dashboard

import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import gdown
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min


st.set_page_config(page_title="Crime Dashboard", layout="wide")
st.title("🚨 Crime Analytics Dashboard")


# Google Drive direct download link (replace with your file ID if needed)
url = "https://drive.google.com/uc?id=1Z97ha7T2xSc28pIUhEpjQnMs7Nr3wBmp"
output = "Crimes_500K_Cleaned.csv"

# Download only if the file doesn't already exist locally
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# Load your data
df = pd.read_csv(output)

# Example: show first 5 rows in Streamlit
import streamlit as st
st.dataframe(df.head(), use_container_width=True)


# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🌍 Geographic Heatmap",
    "📅 Temporal Analysis",
    "🧬 Dimensionality Reduction",
    "📈 Model Performance"
])

# ------------------ Tab 1: Geographic Crime Heatmap ------------------
with tab1:
    st.subheader("🌍 Geographic Crime Heatmap")

    # Ensure datetime parsing
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['weekday'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month

    # Crime type filter
    crime_types = df['primary_type'].dropna().unique()
    selected_types = st.multiselect("Select Crime Types", crime_types, default=crime_types[:3])

    # Weekday filter
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    selected_days = st.multiselect("Select Days of Week", weekdays, default=weekdays)

    # Month filter
    selected_months = st.multiselect("Select Months", list(range(1, 13)), default=list(range(1, 13)))

    # Apply filters
    filtered_df = df[
        (df['primary_type'].isin(selected_types)) &
        (df['weekday'].isin(selected_days)) &
        (df['month'].isin(selected_months))
    ].copy()

    # Cluster count selector
    selected_k = st.selectbox("Select Number of Clusters (k)", [2, 5, 7, 10, 15])

    # Run K-Means clustering
    X_filtered = filtered_df[['latitude', 'longitude']].values
    kmeans = KMeans(n_clusters=selected_k, random_state=42)
    filtered_df['cluster'] = kmeans.fit_predict(X_filtered)

    # Define reference neighborhoods with coordinates
    neighborhoods = pd.DataFrame({
        'area_name': [
            'The Loop', 'Lincoln Park', 'Hyde Park', 'Englewood', 'Austin',
            'Lake View', 'South Shore', 'Rogers Park', 'West Town', 'Logan Square'
        ],
        'latitude': [41.8818, 41.9250, 41.8000, 41.7800, 41.9100,
                     41.9400, 41.7600, 42.0100, 41.9000, 41.9300],
        'longitude': [-87.6232, -87.6500, -87.5900, -87.6500, -87.7600,
                      -87.6500, -87.5600, -87.6700, -87.6800, -87.7100]
    })

    # Map cluster centroids to nearest neighborhood
    centroids = kmeans.cluster_centers_
    centroid_df = pd.DataFrame(centroids, columns=['latitude', 'longitude'])
    closest_idx, _ = pairwise_distances_argmin_min(
        centroid_df[['latitude', 'longitude']],
        neighborhoods[['latitude', 'longitude']]
    )
    centroid_df['area_name'] = neighborhoods.loc[closest_idx, 'area_name'].values

    # Map cluster labels to area names
    cluster_to_area = dict(enumerate(centroid_df['area_name']))
    filtered_df['area_name'] = filtered_df['cluster'].map(cluster_to_area)

    # Plot clusters using neighborhood names
    fig_cluster, ax_cluster = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        x='longitude', y='latitude',
        hue='area_name', data=filtered_df,
        palette='tab10', s=10, alpha=0.6, ax=ax_cluster
    )
    ax_cluster.set_title(f"Crime Hotspots by Neighborhood (K-Means k={selected_k})")
    ax_cluster.set_xlabel("Longitude")
    ax_cluster.set_ylabel("Latitude")
    ax_cluster.legend(title="Neighborhood", bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig_cluster)

    # ------------------ Patrol Recommendation ------------------
    st.markdown("### 🚔 Patrol Recommendation for Tonight")

    # Count crimes per neighborhood cluster
    area_counts = filtered_df['area_name'].value_counts().reset_index()
    area_counts.columns = ['Neighborhood', 'Crime Count']

    # Sort by highest crime count
    top_areas = area_counts.head(3)

    # Display recommendations
    st.write("Based on current clustering and filters, the top neighborhoods needing patrol are:")
    for idx, row in top_areas.iterrows():
        st.write(f"- **{row['Neighborhood']}** with {row['Crime Count']} incidents")

    # Optional: show as a table
    st.table(top_areas)

        # ------------------ Risk Heatmap (Sampled 50K) ------------------
    st.markdown("### 🔥 Crime Risk Heatmap")

    def plot_risk_heatmap_sampled(df, sample_size=50000):
        df_sample = df.sample(n=sample_size, random_state=42)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.kdeplot(
            data=df_sample, x='longitude', y='latitude',
            fill=True, cmap='RdYlGn_r', bw_adjust=0.5, levels=100, thresh=0.05,
            ax=ax
        )
        ax.set_title("Crime Risk Heatmap (Sampled 50K)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        return fig, df_sample

    fig_heatmap, df_sample = plot_risk_heatmap_sampled(df)
    st.pyplot(fig_heatmap)

    st.markdown("### 📍 DBSCAN Crime Hotspots by Neighborhood")

    # Sample and scale
    df_sample = df.iloc[::10, :].copy()
    X_sample = df_sample[['latitude', 'longitude']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)

    # DBSCAN clustering
    eps = 0.001
    min_samples = 20
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
    labels = db.labels_
    df_sample['cluster'] = labels

    # Metrics
    mask = labels != -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    sil = silhouette_score(X_scaled[mask], labels[mask]) if n_clusters > 1 else None
    dbi = davies_bouldin_score(X_scaled[mask], labels[mask]) if n_clusters > 1 else None

    # Compute cluster centroids
    centroids_df = df_sample[df_sample['cluster'] != -1].groupby('cluster')[['latitude', 'longitude']].mean().reset_index()

    # Reference neighborhoods
    neighborhoods = pd.DataFrame({
        'area_name': [
            'The Loop', 'Lincoln Park', 'Hyde Park', 'Englewood', 'Austin',
            'Lake View', 'South Shore', 'Rogers Park', 'West Town', 'Logan Square'
        ],
        'latitude': [41.8818, 41.9250, 41.8000, 41.7800, 41.9100,
                    41.9400, 41.7600, 42.0100, 41.9000, 41.9300],
        'longitude': [-87.6232, -87.6500, -87.5900, -87.6500, -87.7600,
                    -87.6500, -87.5600, -87.6700, -87.6800, -87.7100]
    })

    # Map centroids to nearest neighborhood
    closest_idx, _ = pairwise_distances_argmin_min(
        centroids_df[['latitude', 'longitude']],
        neighborhoods[['latitude', 'longitude']]
    )
    centroids_df['area_name'] = neighborhoods.loc[closest_idx, 'area_name'].values
    cluster_to_area = dict(zip(centroids_df['cluster'], centroids_df['area_name']))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    palette = sns.color_palette("tab20", n_clusters + 1)

    # Plot all clusters
    sns.scatterplot(
        x='longitude', y='latitude',
        hue='cluster', data=df_sample,
        palette=palette, s=10, alpha=0.5, ax=ax, legend=False
    )

    # Highlight top clusters with neighborhood names
    cluster_counts = df_sample[df_sample['cluster'] != -1]['cluster'].value_counts()
    top_clusters = cluster_counts.head(3).index.tolist()

    for cluster_id in top_clusters:
        subset = df_sample[df_sample['cluster'] == cluster_id]
        area_name = cluster_to_area.get(cluster_id, f"Cluster {cluster_id}")
        sns.scatterplot(
            x='longitude', y='latitude',
            data=subset, color='red', s=30, alpha=0.8, ax=ax,
            label=area_name
        )

    # Annotate metrics
    title = f"📍 DBSCAN Crime Hotspots (eps={eps}, min_samples={min_samples})\nClusters={n_clusters}, Silhouette={sil:.3f}, DBI={dbi:.3f}"
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(title="Highlighted Neighborhoods", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Display in Streamlit
    st.pyplot(fig)

        # ------------------ High-Risk Area Details ------------------
    st.markdown("### 📍 High-Risk Area Insights")

    # Approximate: group by latitude/longitude bins to find hotspots
    df_sample['lat_bin'] = pd.cut(df_sample['latitude'], bins=10)
    df_sample['lon_bin'] = pd.cut(df_sample['longitude'], bins=10)
    hotspot_counts = df_sample.groupby(['lat_bin','lon_bin']).size().reset_index(name='Crime Count')
    top_hotspots = hotspot_counts.sort_values('Crime Count', ascending=False).head(3)

    st.write("Top geographic bins with highest crime density:")
    st.table(top_hotspots)


# ------------------ Tab 2: Temporal Analysis ------------------

with tab2:
    st.subheader("📅 Temporal Crime Patterns")

    # Ensure datetime parsing
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['hour'] = df['date'].dt.hour
    df['weekday'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofweek'] = df['date'].dt.dayofweek  # 0=Monday
    df['weekday_num'] = df['date'].dt.weekday  # for mapping

    # ------------------ Filters ------------------
    st.markdown("### 🔍 Filter Crime Data")

    crime_types = df['primary_type'].dropna().unique()
    selected_types = st.multiselect(
        "Select Crime Types",
        crime_types,
        default=crime_types[:3],
        key="crime_types_tab2"
    )

    years = sorted(df['year'].dropna().unique())
    selected_year = st.selectbox(
        "Select Year",
        years,
        key="year_tab2"
    )

    months = list(range(1, 13))
    selected_months = st.multiselect(
        "Select Months",
        months,
        default=months,
        key="months_tab2"
    )

    # Apply filters
    filtered_df = df[
        (df['primary_type'].isin(selected_types)) &
        (df['year'] == selected_year) &
        (df['month'].isin(selected_months))
    ]

    # ------------------ Altair Interactive Charts ------------------
    st.markdown("### ⏰ Crimes by Hour")
    hourly = filtered_df.groupby('hour').size().reset_index(name='count')
    chart_hour = alt.Chart(hourly).mark_bar(color='steelblue').encode(
        x=alt.X('hour:O', title='Hour of Day'),
        y=alt.Y('count:Q', title='Crime Count'),
        tooltip=['hour', 'count']
    ).properties(height=300)
    st.altair_chart(chart_hour, use_container_width=True)

    st.markdown("### 📆 Crimes by Weekday")
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday = filtered_df.groupby('weekday').size().reindex(weekday_order).reset_index(name='count')
    chart_weekday = alt.Chart(weekday).mark_bar(color='darkorange').encode(
        x=alt.X('weekday:O', title='Day of Week'),
        y=alt.Y('count:Q', title='Crime Count'),
        tooltip=['weekday', 'count']
    ).properties(height=300)
    st.altair_chart(chart_weekday, use_container_width=True)

    st.markdown("### 📈 Monthly Crime Trend")
    monthly = filtered_df.groupby('month').size().reset_index(name='count')
    chart_month = alt.Chart(monthly).mark_line(point=True, color='seagreen').encode(
        x=alt.X('month:O', title='Month'),
        y=alt.Y('count:Q', title='Crime Count'),
        tooltip=['month', 'count']
    ).properties(height=300)
    st.altair_chart(chart_month, use_container_width=True)

    # ------------------ Seaborn/Matplotlib Visuals ------------------
    st.markdown("### 🔥 Hourly Crime Heatmap")
    heatmap_data = filtered_df.groupby(['dayofweek', 'hour']).size().unstack(fill_value=0)
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax_heatmap)
    ax_heatmap.set_title("Hourly Crime Heatmap (0=Monday, 6=Sunday)")
    ax_heatmap.set_xlabel("Hour of Day")
    ax_heatmap.set_ylabel("Day of Week")
    st.pyplot(fig_heatmap)

    st.markdown("### 📊 Crime Frequency by Hour")
    fig_hourly, ax_hourly = plt.subplots(figsize=(12, 6))
    filtered_df['hour'].value_counts().sort_index().plot(kind='bar', color='salmon', ax=ax_hourly)
    ax_hourly.set_title('Crime Frequency by Hour of Day')
    ax_hourly.set_xlabel('Hour')
    ax_hourly.set_ylabel('Number of Crimes')
    st.pyplot(fig_hourly)

    st.markdown("### 📅 Crime Distribution by Weekday")
    weekday_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    filtered_df['weekday_name'] = filtered_df['weekday_num'].map(weekday_map)
    fig_weekday, ax_weekday = plt.subplots(figsize=(10, 4))
    sns.countplot(
        x='weekday_name',
        data=filtered_df,
        order=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        palette='coolwarm',
        ax=ax_weekday
    )
    ax_weekday.set_title('Crime Distribution by Weekday')
    ax_weekday.set_xlabel('Day')
    ax_weekday.set_ylabel('Crime Count')
    st.pyplot(fig_weekday)

    st.markdown("### 📈 Monthly Crime Trends")
    fig_monthly, ax_monthly = plt.subplots(figsize=(10, 6))
    filtered_df['month'].value_counts().sort_index().plot(kind='line', marker='o', color='teal', ax=ax_monthly)
    ax_monthly.set_title('Monthly Crime Trends')
    ax_monthly.set_xlabel('Month')
    ax_monthly.set_ylabel('Crime Count')
    ax_monthly.set_xticks(range(1, 13))
    ax_monthly.grid(True)
    st.pyplot(fig_monthly)

    # ------------------ Insight Box ------------------
    st.markdown("### 🧠 Observations")
    if not hourly.empty and not weekday.empty and not monthly.empty:
        peak_hour = hourly.loc[hourly['count'].idxmax(), 'hour']
        peak_day = weekday.loc[weekday['count'].idxmax(), 'weekday']
        peak_month = monthly.loc[monthly['count'].idxmax(), 'month']
        st.info(f"🔺 Most crimes occur around **{peak_hour}:00 hrs** on **{peak_day}s**, especially in **month {peak_month}**.")
    else:
        st.warning("No data available for the selected filters.")

# ------------------ Tab 3: PCA Cluster Visualization ------------------

from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

with tab3:
    st.subheader("🧬 PCA Cluster Visualization")

    # --- Prepare numeric features (exclude non-numeric columns) ---
    exclude_cols = [
        'id','case_number','date','block','iucr','primary_type','description',
        'location_description','location','updated_on','cluster_label','weekday_name'
    ]
    features = df.drop(columns=exclude_cols, errors='ignore').select_dtypes(include=['int64','float64'])

    if features.empty:
        st.warning("No numeric features available for PCA after exclusions.")
    else:
        # --- Standardize features ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        # --- Apply PCA with 7 components ---
        pca = PCA(n_components=7)
        X_pca = pca.fit_transform(X_scaled)

        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()
        st.write(f"Explained variance by 7 components: **{cumulative_variance[-1]:.2%}**")

        # --- Loadings DataFrame ---
        component_labels = [f'PC{i+1}' for i in range(pca.n_components_)]
        loadings = pd.DataFrame(pca.components_.T, columns=component_labels, index=features.columns)

        # --- Top 5 contributing features ---
        top_features = loadings.abs().sum(axis=1).sort_values(ascending=False).head(5)
        st.markdown("### 🔎 Top 5 Features Driving PCA Components")
        st.write(top_features)

        # --- Bar chart of top features ---
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.barplot(x=top_features.values, y=top_features.index, palette="viridis", ax=ax1)
        ax1.set_title("Top 5 Features Driving Crime Patterns")
        ax1.set_xlabel("Contribution (sum of absolute loadings)")
        ax1.set_ylabel("Feature")
        plt.tight_layout()
        st.pyplot(fig1)

        # --- Heatmap of top feature contributions ---
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.heatmap(loadings.loc[top_features.index], annot=True, cmap='coolwarm', center=0, ax=ax2)
        ax2.set_title("Top Feature Contributions to Principal Components")
        plt.tight_layout()
        st.pyplot(fig2)

        st.markdown("---")
        st.subheader("📊 2D PCA Crime Type Clusters")

        # --- Build PCA DataFrame with crime type included ---
        df_pca = pd.DataFrame(X_pca[:, :3], columns=['PC1','PC2','PC3'])
        df_pca['primary_type'] = df['primary_type']

        # --- 2D PCA scatterplot colored by crime type ---
        fig3, ax3 = plt.subplots(figsize=(8,6))
        sns.scatterplot(
            data=df_pca, x='PC1', y='PC2',
            hue='primary_type', palette='tab10', alpha=0.6, s=20, ax=ax3
        )
        ax3.set_title("2D PCA: Crime Type Clusters")
        ax3.set_xlabel("PC1")
        ax3.set_ylabel("PC2")
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        st.pyplot(fig3)

        st.markdown("---")
        st.subheader("🌐 3D PCA Crime Type Clusters")

        # --- 3D PCA scatterplot ---
        fig4 = plt.figure(figsize=(8,7))
        ax4 = fig4.add_subplot(111, projection='3d')

        crime_types = df_pca['primary_type'].astype(str)
        unique_types = crime_types.unique()
        colors = plt.cm.get_cmap('tab20', len(unique_types))

        for i, crime in enumerate(unique_types):
            subset = df_pca[crime_types == crime]
            ax4.scatter(
                subset['PC1'], subset['PC2'], subset['PC3'],
                label=crime, alpha=0.6, s=20, color=colors(i)
            )

        ax4.set_title("3D PCA: Crime Type Clusters")
        ax4.set_xlabel("PC1")
        ax4.set_ylabel("PC2")
        ax4.set_zlabel("PC3")
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        st.pyplot(fig4)

# ------------------ Tab 4: Model Performance ------------------

with tab4:
    st.subheader("📈 Model Performance (MLflow)")

    # ------------------ Best Metrics ------------------
    st.markdown("### 🏆 Best Performing Model")
    st.metric("Best Silhouette Score", "0.991")   # replace with your best_score dynamically if needed
    st.metric("Best DBI", "0.011")                # replace with your best_dbi dynamically if needed
    st.write("📦 Best Model URI: `runs:/f374015fa454489497fcd6f745c83f8d/dbscan_model`")

    st.markdown("---")
    st.subheader("📊 All Runs and Links")

    # ------------------ Build Results DataFrame ------------------
    results_data = {
        "Algorithm": ["KMeans", "DBSCAN", "AgglomerativeClustering"],
        "Silhouette Score": [0.449589, 0.991977, 0.430136],
        "DBI": [0.824397, 0.011325, 0.893677],
        "Model URI": [
            "runs:/f61e9ffc41164f9281121316a0f1c2ab/kmeans_model",
            "runs:/f374015fa454489497fcd6f745c83f8d/dbscan_model",
            "runs:/e9df9e2fdc0b4d269261269fd92a26e0/hierarchical_model"
        ],
        "Run Link": [
            "http://localhost:5000/#/experiments/960592133619495839/runs/f61e9ffc41164f9281121316a0f1c2ab",
            "http://localhost:5000/#/experiments/960592133619495839/runs/f374015fa454489497fcd6f745c83f8d",
            "http://localhost:5000/#/experiments/960592133619495839/runs/e9df9e2fdc0b4d269261269fd92a26e0"
        ]
    }

    results_df = pd.DataFrame(results_data)

    # ------------------ Display Results ------------------
    st.dataframe(results_df, use_container_width=True)

    # ------------------ Clickable Links ------------------
    st.markdown("### 🔗 Direct Run Links")
    for algo, link in zip(results_df['Algorithm'], results_df['Run Link']):
        st.markdown(f"- **{algo}** → [View Run]({link})")