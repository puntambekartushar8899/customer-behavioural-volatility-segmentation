from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# FILE PATHS
# Update these paths if your files have different names
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CUSTOMER_DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "rfm_refined_behavioural_clustered.csv"
)

EVALUATION_DATA_PATH = (
    BASE_DIR
    / "outputs"
    / "tables"
    / "final_clustering_evaluation_metrics.csv"
)

BOOTSTRAP_DATA_PATH = (
    BASE_DIR
    / "outputs"
    / "tables"
    / "bootstrap_stability_summary.csv"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "Customer_Lifetime_Days",
    "Purchase_Velocity",
    "Purchase_Interval_Std",
    "Purchase_Regularity_Index",
    "Invoice_Value_CV",
]


# ============================================================
# CLUSTER NAMES
#
# IMPORTANT:
# K-Means cluster numbers are arbitrary.
# Check your final cluster profile and change these names if required.
# ============================================================

PERSONA_MAP = {
    0: "New / One-Time Customers",
    1: "High-Value Loyal Customers",
    2: "Growing / Potential Loyal Customers",
    3: "At-Risk / Dormant Customers",
}


# ============================================================
# CLUSTER DESCRIPTIONS
# ============================================================

CLUSTER_NOTES = {
    0: {
        "description": (
            "This cluster generally represents customers with a limited "
            "transaction history and relatively low purchase frequency. "
            "They may be new customers or customers who purchased only once."
        ),
        "business_action": (
            "Encourage a second purchase through welcome campaigns, "
            "repeat-purchase incentives and relevant product recommendations."
        ),
    },

    1: {
        "description": (
            "This cluster generally represents customers with high monetary "
            "value, frequent purchases and a longer relationship with the business."
        ),
        "business_action": (
            "Prioritise retention using loyalty rewards, personalised offers, "
            "premium services and early access to new products."
        ),
    },

    2: {
        "description": (
            "This cluster generally represents customers who show growing "
            "engagement and repeat-purchase potential. They may become "
            "high-value loyal customers."
        ),
        "business_action": (
            "Use loyalty programmes, cross-selling, bundle offers and targeted "
            "campaigns to increase frequency and customer value."
        ),
    },

    3: {
        "description": (
            "This cluster generally represents customers with high recency "
            "or declining engagement. They may be at risk of becoming inactive."
        ),
        "business_action": (
            "Run win-back campaigns, personalised re-engagement communication "
            "and time-limited incentives."
        ),
    },
}


# ============================================================
# FEATURE EXPLANATIONS
# ============================================================

FEATURE_EXPLANATIONS = {
    "Recency": (
        "The number of days since the customer's most recent purchase. "
        "A lower value means that the customer purchased more recently."
    ),

    "Frequency": (
        "The number of purchases or invoices associated with the customer. "
        "A higher value indicates stronger engagement."
    ),

    "Monetary": (
        "The total amount spent by the customer. This identifies customers "
        "who contribute the most financial value."
    ),

    "Customer_Lifetime_Days": (
        "The number of days between the customer's first and last purchase. "
        "It distinguishes long-term customers from newly acquired customers."
    ),

    "Purchase_Velocity": (
        "The rate at which the customer purchases during their active lifetime. "
        "A higher value indicates faster purchasing behaviour."
    ),

    "Purchase_Interval_Std": (
        "The variation in time between purchases. A higher value indicates "
        "less predictable purchasing behaviour."
    ),

    "Purchase_Regularity_Index": (
        "The consistency of the customer's purchasing pattern. A higher value "
        "usually indicates more stable engagement."
    ),

    "Invoice_Value_CV": (
        "The variation in invoice value relative to average invoice value. "
        "It shows whether customer spending is consistent or volatile."
    ),
}


# ============================================================
# FALLBACK MODEL RESULTS
# Used when the evaluation file is unavailable
# ============================================================

DEFAULT_MODEL_RESULTS = pd.DataFrame(
    {
        "Feature Set": [
            "Traditional RFM",
            "Traditional RFM",
            "Traditional RFM",
            "Refined Behavioural",
            "Refined Behavioural",
            "Refined Behavioural",
        ],
        "Algorithm": [
            "K-Means",
            "Hierarchical",
            "GMM",
            "K-Means",
            "Hierarchical",
            "GMM",
        ],
        "Silhouette Score": [
            0.3850,
            0.3230,
            0.1842,
            0.4367,
            0.4049,
            0.3018,
        ],
    }
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.35rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #8b949e;
            font-size: 1rem;
            margin-bottom: 1.4rem;
        }

        .information-box {
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid rgba(128,128,128,0.25);
            margin-bottom: 0.7rem;
        }

        .feature-box {
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #4f8bf9;
            background-color: rgba(79,139,249,0.08);
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


def detect_column(dataframe, possible_names):
    cleaned_columns = {
        column.lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", ""): column
        for column in dataframe.columns
    }

    for possible_name in possible_names:
        cleaned_name = (
            possible_name.lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )

        if cleaned_name in cleaned_columns:
            return cleaned_columns[cleaned_name]

    return None


def normalise_profile(profile):
    minimum = profile.min()
    maximum = profile.max()

    denominator = (maximum - minimum).replace(0, 1)

    return (profile - minimum) / denominator


def format_cluster_name(cluster):
    return PERSONA_MAP.get(
        int(cluster),
        f"Cluster {cluster}",
    )


def create_download_button(
    dataframe,
    filename,
    button_label,
):
    csv_data = dataframe.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label=button_label,
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# LOAD CUSTOMER DATA
# ============================================================

if not CUSTOMER_DATA_PATH.exists():
    st.error(
        "The final clustered customer file could not be found."
    )

    st.code(str(CUSTOMER_DATA_PATH))

    st.info(
        "Change CUSTOMER_DATA_PATH near the top of app.py so that it "
        "matches the actual location and filename of your clustered CSV."
    )

    st.stop()


df = load_csv(CUSTOMER_DATA_PATH)


# ============================================================
# VALIDATE FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    st.error(
        "The customer data is missing the following required features:"
    )

    for feature in missing_features:
        st.write(f"- {feature}")

    st.write("Available columns:", list(df.columns))

    st.stop()


cluster_column = detect_column(
    df,
    [
        "KMeans_Cluster",
        "KMeans Cluster",
        "Cluster",
        "Cluster_Label",
    ],
)

if cluster_column is None:
    st.error(
        "No K-Means cluster column was found. The file should contain "
        "`KMeans_Cluster`, `Cluster` or a similar cluster-label column."
    )

    st.stop()


customer_id_column = detect_column(
    df,
    [
        "CustomerID",
        "Customer ID",
        "customer_id",
    ],
)


# ============================================================
# CLEAN NUMERIC COLUMNS
# ============================================================

for feature in FEATURES:
    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce",
    )


df[cluster_column] = pd.to_numeric(
    df[cluster_column],
    errors="coerce",
)


df = df.dropna(
    subset=FEATURES + [cluster_column]
).copy()


df[cluster_column] = df[
    cluster_column
].astype(int)


df["Customer Segment"] = df[
    cluster_column
].apply(format_cluster_name)


# ============================================================
# LOAD EVALUATION DATA
# ============================================================

if EVALUATION_DATA_PATH.exists():
    evaluation_df = load_csv(EVALUATION_DATA_PATH)
else:
    evaluation_df = DEFAULT_MODEL_RESULTS.copy()


# ============================================================
# LOAD BOOTSTRAP DATA
# ============================================================

if BOOTSTRAP_DATA_PATH.exists():
    bootstrap_df = load_csv(BOOTSTRAP_DATA_PATH)
else:
    bootstrap_df = pd.DataFrame(
        {
            "Feature Set": [
                "Traditional RFM",
                "Refined Behavioural",
            ],
            "Mean Silhouette": [
                0.3861,
                0.4363,
            ],
            "Standard Deviation": [
                0.0049,
                0.0049,
            ],
        }
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        Behavioural Customer Segmentation Dashboard
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Customer cluster comparison, behavioural insights and model evaluation
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR CLUSTER FILTER
# ============================================================

st.sidebar.title("Dashboard Filters")

available_clusters = sorted(
    df[cluster_column]
    .unique()
    .tolist()
)


selected_clusters = st.sidebar.multiselect(
    "Select clusters",
    options=available_clusters,
    default=available_clusters,
    format_func=lambda cluster: (
        f"Cluster {cluster} — {format_cluster_name(cluster)}"
    ),
)


st.sidebar.caption(
    "Select one or more clusters to update the dashboard."
)


if not selected_clusters:
    st.warning(
        "Select at least one cluster from the sidebar."
    )

    st.stop()


filtered_df = df[
    df[cluster_column].isin(
        selected_clusters
    )
].copy()


# ============================================================
# MAIN KPI SECTION
# ============================================================

total_customers = (
    filtered_df[customer_id_column].nunique()
    if customer_id_column
    else len(filtered_df)
)

total_monetary = filtered_df[
    "Monetary"
].sum()

average_monetary = filtered_df[
    "Monetary"
].mean()

average_frequency = filtered_df[
    "Frequency"
].mean()

average_recency = filtered_df[
    "Recency"
].mean()


kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)


kpi1.metric(
    "Selected Customers",
    f"{total_customers:,}",
)


kpi2.metric(
    "Selected Clusters",
    len(selected_clusters),
)


kpi3.metric(
    "Total Monetary Value",
    f"£{total_monetary:,.0f}",
)


kpi4.metric(
    "Average Customer Value",
    f"£{average_monetary:,.2f}",
)


kpi5.metric(
    "Average Frequency",
    f"{average_frequency:.2f}",
)


st.caption(
    f"Average recency across the selected clusters: "
    f"{average_recency:.1f} days."
)


st.divider()


# ============================================================
# CUSTOMER DISTRIBUTION AND MONETARY CONTRIBUTION
# ============================================================

left_column, right_column = st.columns(2)


with left_column:

    cluster_distribution = (
        filtered_df.groupby(
            [
                cluster_column,
                "Customer Segment",
            ]
        )
        .size()
        .reset_index(
            name="Customers"
        )
        .sort_values(
            "Customers",
            ascending=False,
        )
    )


    customer_distribution_chart = px.bar(
        cluster_distribution,
        x="Customer Segment",
        y="Customers",
        color="Customer Segment",
        text="Customers",
        title="Customer Distribution by Cluster",
    )


    customer_distribution_chart.update_layout(
        showlegend=False,
        xaxis_title="",
        yaxis_title="Number of Customers",
    )


    customer_distribution_chart.update_traces(
        textposition="outside"
    )


    st.plotly_chart(
        customer_distribution_chart,
        use_container_width=True,
    )


with right_column:

    monetary_distribution = (
        filtered_df.groupby(
            "Customer Segment"
        )["Monetary"]
        .sum()
        .reset_index()
        .sort_values(
            "Monetary",
            ascending=False,
        )
    )


    monetary_distribution_chart = px.pie(
        monetary_distribution,
        names="Customer Segment",
        values="Monetary",
        hole=0.55,
        title="Monetary Contribution by Cluster",
    )


    st.plotly_chart(
        monetary_distribution_chart,
        use_container_width=True,
    )


# ============================================================
# CLUSTER COMPARISON TABLE
# ============================================================

st.header("Cluster Comparison")


cluster_comparison = (
    filtered_df.groupby(
        [
            cluster_column,
            "Customer Segment",
        ]
    )
    .agg(
        Customers=(
            cluster_column,
            "size",
        ),
        Average_Recency=(
            "Recency",
            "mean",
        ),
        Average_Frequency=(
            "Frequency",
            "mean",
        ),
        Average_Monetary=(
            "Monetary",
            "mean",
        ),
        Total_Monetary=(
            "Monetary",
            "sum",
        ),
        Average_Lifetime_Days=(
            "Customer_Lifetime_Days",
            "mean",
        ),
        Average_Purchase_Velocity=(
            "Purchase_Velocity",
            "mean",
        ),
        Average_Interval_Variability=(
            "Purchase_Interval_Std",
            "mean",
        ),
        Average_Regularity=(
            "Purchase_Regularity_Index",
            "mean",
        ),
        Average_Invoice_CV=(
            "Invoice_Value_CV",
            "mean",
        ),
    )
    .round(2)
    .reset_index()
)


st.dataframe(
    cluster_comparison,
    use_container_width=True,
    hide_index=True,
)


create_download_button(
    cluster_comparison,
    "cluster_comparison.csv",
    "Download Cluster Comparison",
)


# ============================================================
# NORMALISED FEATURE HEATMAP
# ============================================================

st.subheader("Behavioural Feature Comparison")


cluster_profile = (
    filtered_df.groupby(
        "Customer Segment"
    )[FEATURES]
    .mean()
)


if len(cluster_profile) > 1:

    normalised_profile = normalise_profile(
        cluster_profile
    )


    heatmap_chart = px.imshow(
        normalised_profile,
        text_auto=".2f",
        aspect="auto",
        title="Relative Feature Strength Across Selected Clusters",
    )


    heatmap_chart.update_layout(
        xaxis_title="Behavioural Features",
        yaxis_title="Customer Segment",
    )


    st.plotly_chart(
        heatmap_chart,
        use_container_width=True,
    )


    st.caption(
        "The heatmap values are normalised between 0 and 1. "
        "A higher value means that the cluster has a relatively higher "
        "average for that feature compared with the other selected clusters."
    )

else:
    st.info(
        "Select at least two clusters to display a comparative heatmap."
    )


# ============================================================
# INDIVIDUAL FEATURE COMPARISON
# ============================================================

st.subheader("Feature-Level Comparison")


selected_feature = st.selectbox(
    "Select a feature",
    FEATURES,
)


st.markdown(
    f"""
    <div class="feature-box">
        <strong>{selected_feature}</strong><br>
        {FEATURE_EXPLANATIONS[selected_feature]}
    </div>
    """,
    unsafe_allow_html=True,
)


feature_comparison = (
    filtered_df.groupby(
        "Customer Segment"
    )[selected_feature]
    .mean()
    .reset_index()
)


feature_chart = px.bar(
    feature_comparison,
    x="Customer Segment",
    y=selected_feature,
    color="Customer Segment",
    text=selected_feature,
    title=f"Average {selected_feature} by Cluster",
)


feature_chart.update_layout(
    showlegend=False,
    xaxis_title="",
)


feature_chart.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside",
)


st.plotly_chart(
    feature_chart,
    use_container_width=True,
)


# ============================================================
# CLUSTER NOTES
# ============================================================

st.header("Cluster Notes and Business Actions")


for cluster in selected_clusters:

    cluster_name = format_cluster_name(
        cluster
    )


    cluster_data = filtered_df[
        filtered_df[cluster_column]
        == cluster
    ]


    cluster_note = CLUSTER_NOTES.get(
        cluster,
        {
            "description": (
                "Review the cluster feature averages to identify "
                "its main behavioural characteristics."
            ),
            "business_action": (
                "Create a targeted strategy based on recency, frequency, "
                "monetary value and purchasing consistency."
            ),
        },
    )


    with st.expander(
        f"Cluster {cluster}: {cluster_name}",
        expanded=True,
    ):

        metric1, metric2, metric3, metric4 = st.columns(4)


        metric1.metric(
            "Customers",
            f"{len(cluster_data):,}",
        )


        metric2.metric(
            "Average Recency",
            f"{cluster_data['Recency'].mean():.1f}",
        )


        metric3.metric(
            "Average Frequency",
            f"{cluster_data['Frequency'].mean():.2f}",
        )


        metric4.metric(
            "Average Monetary",
            f"£{cluster_data['Monetary'].mean():,.2f}",
        )


        st.markdown(
            f"**Cluster interpretation:** "
            f"{cluster_note['description']}"
        )


        st.markdown(
            f"**Recommended business action:** "
            f"{cluster_note['business_action']}"
        )


# ============================================================
# AUTOMATIC BUSINESS FACTS
# ============================================================

st.header("Business Facts from Selected Clusters")


largest_cluster = cluster_comparison.loc[
    cluster_comparison[
        "Customers"
    ].idxmax()
]


highest_value_cluster = cluster_comparison.loc[
    cluster_comparison[
        "Average_Monetary"
    ].idxmax()
]


highest_revenue_cluster = cluster_comparison.loc[
    cluster_comparison[
        "Total_Monetary"
    ].idxmax()
]


most_frequent_cluster = cluster_comparison.loc[
    cluster_comparison[
        "Average_Frequency"
    ].idxmax()
]


most_recent_cluster = cluster_comparison.loc[
    cluster_comparison[
        "Average_Recency"
    ].idxmin()
]


least_recent_cluster = cluster_comparison.loc[
    cluster_comparison[
        "Average_Recency"
    ].idxmax()
]


fact_column1, fact_column2 = st.columns(2)


with fact_column1:

    st.success(
        f"""
        **Highest average customer value**

        {highest_value_cluster['Customer Segment']}

        Average monetary value:
        **£{highest_value_cluster['Average_Monetary']:,.2f}**
        """
    )


    st.success(
        f"""
        **Largest monetary contributor**

        {highest_revenue_cluster['Customer Segment']}

        Total monetary contribution:
        **£{highest_revenue_cluster['Total_Monetary']:,.2f}**
        """
    )


    st.info(
        f"""
        **Most frequent customer group**

        {most_frequent_cluster['Customer Segment']}

        Average frequency:
        **{most_frequent_cluster['Average_Frequency']:.2f}**
        """
    )


with fact_column2:

    st.info(
        f"""
        **Largest customer cluster**

        {largest_cluster['Customer Segment']}

        Number of customers:
        **{int(largest_cluster['Customers']):,}**
        """
    )


    st.info(
        f"""
        **Most recently active group**

        {most_recent_cluster['Customer Segment']}

        Average recency:
        **{most_recent_cluster['Average_Recency']:.1f} days**
        """
    )


    st.warning(
        f"""
        **Least recently active group**

        {least_recent_cluster['Customer Segment']}

        Average recency:
        **{least_recent_cluster['Average_Recency']:.1f} days**

        This cluster may require a re-engagement or win-back campaign.
        """
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.header("Clustering Model Comparison")


feature_set_column = detect_column(
    evaluation_df,
    [
        "Feature Set",
        "Dataset",
        "Feature_Set",
    ],
)


algorithm_column = detect_column(
    evaluation_df,
    [
        "Algorithm",
        "Model",
    ],
)


silhouette_column = detect_column(
    evaluation_df,
    [
        "Silhouette Score",
        "Silhouette",
    ],
)


if (
    feature_set_column is not None
    and algorithm_column is not None
    and silhouette_column is not None
):

    model_results = evaluation_df[
        [
            feature_set_column,
            algorithm_column,
            silhouette_column,
        ]
    ].copy()


    model_results.columns = [
        "Feature Set",
        "Algorithm",
        "Silhouette Score",
    ]


    model_results[
        "Silhouette Score"
    ] = pd.to_numeric(
        model_results[
            "Silhouette Score"
        ],
        errors="coerce",
    )


    model_results = model_results.dropna(
        subset=["Silhouette Score"]
    )


else:
    model_results = DEFAULT_MODEL_RESULTS.copy()


model_comparison_chart = px.bar(
    model_results,
    x="Algorithm",
    y="Silhouette Score",
    color="Feature Set",
    barmode="group",
    text="Silhouette Score",
    title="Traditional RFM vs Refined Behavioural Clustering",
)


model_comparison_chart.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside",
)


st.plotly_chart(
    model_comparison_chart,
    use_container_width=True,
)


st.dataframe(
    model_results.sort_values(
        "Silhouette Score",
        ascending=False,
    ),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# WHY K-MEANS WAS SELECTED
# ============================================================

st.subheader("Why Refined Behavioural K-Means Was Selected")


traditional_kmeans_score = 0.3850
refined_kmeans_score = 0.4367
hierarchical_score = 0.4049
gmm_score = 0.3018


percentage_improvement = (
    (
        refined_kmeans_score
        - traditional_kmeans_score
    )
    / traditional_kmeans_score
) * 100


reason1, reason2, reason3 = st.columns(3)


reason1.metric(
    "Best Silhouette Score",
    f"{refined_kmeans_score:.4f}",
)


reason2.metric(
    "Improvement over RFM",
    f"{percentage_improvement:.1f}%",
)


reason3.metric(
    "Bootstrap Result",
    "Better in 100% of samples",
)


st.success(
    """
    Refined Behavioural K-Means achieved the highest Silhouette Score of
    0.4367. This means that the customer groups were more compact internally
    and more clearly separated from one another.
    """
)


st.markdown(
    f"""
    ### K-Means compared with Hierarchical Clustering

    Refined Behavioural Hierarchical Clustering achieved a Silhouette Score
    of **{hierarchical_score:.4f}**, while Refined Behavioural K-Means achieved
    **{refined_kmeans_score:.4f}**.

    K-Means therefore created more clearly separated customer groups.
    Its centroid-based structure is also straightforward to interpret and can
    be used to assign additional customers to existing segments.

    ### K-Means compared with Gaussian Mixture Model

    Refined Behavioural GMM achieved a Silhouette Score of
    **{gmm_score:.4f}**, considerably lower than the K-Means result.

    Although GMM can represent probabilistic membership, its customer groups
    were less clearly separated in this dataset.

    ### Refined Behavioural K-Means compared with Traditional RFM K-Means

    Traditional RFM K-Means achieved **{traditional_kmeans_score:.4f}**,
    while Refined Behavioural K-Means achieved
    **{refined_kmeans_score:.4f}**.

    This is an improvement of approximately **{percentage_improvement:.1f}%**.
    The refined feature set captures customer lifetime, purchase speed,
    purchase regularity, interval variability and spending consistency,
    which are not measured by traditional RFM alone.
    """
)


st.info(
    """
    The final model was not selected using only one metric. The choice was
    based on cluster separation, bootstrap stability, business interpretability
    and the usefulness of the resulting customer segments.
    """
)


# ============================================================
# BOOTSTRAP VALIDATION
# ============================================================

st.header("Bootstrap Stability Validation")


bootstrap_feature_column = detect_column(
    bootstrap_df,
    [
        "Feature Set",
        "Dataset",
        "Model",
    ],
)


bootstrap_mean_column = detect_column(
    bootstrap_df,
    [
        "Mean Silhouette",
        "Mean_Silhouette",
        "Mean",
    ],
)


bootstrap_std_column = detect_column(
    bootstrap_df,
    [
        "Standard Deviation",
        "Std Dev",
        "Std",
    ],
)


if (
    bootstrap_feature_column is not None
    and bootstrap_mean_column is not None
):

    bootstrap_chart = px.bar(
        bootstrap_df,
        x=bootstrap_feature_column,
        y=bootstrap_mean_column,
        error_y=bootstrap_std_column,
        text=bootstrap_mean_column,
        title="Bootstrap Mean Silhouette Score",
    )


    bootstrap_chart.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside",
    )


    st.plotly_chart(
        bootstrap_chart,
        use_container_width=True,
    )


st.dataframe(
    bootstrap_df,
    use_container_width=True,
    hide_index=True,
)


st.success(
    """
    The refined behavioural model outperformed the traditional RFM baseline
    in 100% of bootstrap samples. This confirms that the improvement was
    stable and not dependent on one particular sample.
    """
)


# ============================================================
# SELECTED CUSTOMER TABLE
# ============================================================

st.header("Selected Customer Data")


display_columns = [
    "Customer Segment",
    cluster_column,
] + FEATURES


if customer_id_column:
    display_columns.insert(
        0,
        customer_id_column,
    )


st.dataframe(
    filtered_df[
        display_columns
    ],
    use_container_width=True,
    hide_index=True,
)


create_download_button(
    filtered_df,
    "selected_cluster_customers.csv",
    "Download Selected Customer Data",
)