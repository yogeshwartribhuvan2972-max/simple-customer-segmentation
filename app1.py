# -------------------------------------------------
# IMPORT LIBRARIES
# -------------------------------------------------

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛒",
    layout="wide"
)


# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("🛒 Customer Segmentation System")

st.write(
    "Segment customers based on Annual Spending "
    "and Order Count."
)


# -------------------------------------------------
# LOAD TRAINING DATA
# -------------------------------------------------

df = pd.read_excel(
    r"C:\Users\admin\OneDrive\Desktop\UPGRAD\ml\Projects\Flipkart_KMeans_Project\Customer_dataset1.xlsx"
)


# -------------------------------------------------
# SELECT FEATURES
# -------------------------------------------------

features = [
    "annual spending",
    "orderscount"
]

X = df[features]


# -------------------------------------------------
# SCALING
# -------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# -------------------------------------------------
# TRAIN K-MEANS MODEL
# -------------------------------------------------

model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = model.fit_predict(X_scaled)


# -------------------------------------------------
# CREATE CUSTOMER TYPE BASED ON CLUSTER
# -------------------------------------------------

# Find Average Spending of Every Cluster

cluster_spending = df.groupby("Cluster")[
    "annual spending"
].mean()


# Sort Clusters From Lowest Spending to Highest

sorted_clusters = cluster_spending.sort_values().index


# Assign Customer Type to Cluster

cluster_labels = {

    sorted_clusters[0]: "Poor Customer",

    sorted_clusters[1]: "Common Customer",

    sorted_clusters[2]: "Regular Customer",

    sorted_clusters[3]: "VIP Customer"

}


# Add Customer Type Column

df["Customer Type"] = df["Cluster"].map(
    cluster_labels
)


# -------------------------------------------------
# GET CENTROIDS
# -------------------------------------------------

centroids = scaler.inverse_transform(
    model.cluster_centers_
)


# =================================================
# GRAPH 1 - FULL DATASET
# =================================================

st.subheader(
    "Annual Spending vs Order Count - Full Dataset"
)

fig1, ax1 = plt.subplots()

ax1.scatter(
    df["annual spending"],
    df["orderscount"]
)

ax1.set_xlabel("Annual Spending")

ax1.set_ylabel("Order Count")

ax1.set_title(
    "Customers Before Clustering"
)

st.pyplot(fig1)


# =================================================
# GRAPH 2 - SAMPLE DATASET
# =================================================

st.subheader(
    "Annual Spending vs Order Count - Sample Data"
)


sample_df = df.sample(

    min(2000, len(df)),

    random_state=42

)


fig2, ax2 = plt.subplots()


ax2.scatter(

    sample_df["annual spending"],

    sample_df["orderscount"],

    s=10,

    alpha=0.5

)


ax2.set_xlabel(
    "Annual Spending"
)


ax2.set_ylabel(
    "Order Count"
)


ax2.set_title(
    "Annual Spending vs Order Count"
)


st.pyplot(fig2)


# =================================================
# GRAPH 3 - ELBOW METHOD
# =================================================

st.subheader("Elbow Method")


wcss = []


for i in range(1, 11):


    elbow_model = KMeans(

        n_clusters=i,

        random_state=42,

        n_init=10

    )


    elbow_model.fit(X_scaled)


    wcss.append(

        elbow_model.inertia_

    )


fig3, ax3 = plt.subplots()


ax3.plot(

    range(1, 11),

    wcss,

    marker="o"

)


ax3.set_xlabel(
    "Number of Clusters"
)


ax3.set_ylabel(
    "WCSS"
)


ax3.set_title(
    "Elbow Method"
)


st.pyplot(fig3)


# =================================================
# GRAPH 4 - FULL DATA CLUSTERS
# =================================================

st.subheader(
    "K-Means Customer Segmentation - Full Data"
)


fig4, ax4 = plt.subplots()


ax4.scatter(

    df["annual spending"],

    df["orderscount"],

    c=df["Cluster"]

)


# Display Centroids

ax4.scatter(

    centroids[:, 0],

    centroids[:, 1],

    marker="X",

    s=200,

    label="Centroids"

)


ax4.set_xlabel(
    "Annual Spending"
)


ax4.set_ylabel(
    "Order Count"
)


ax4.set_title(
    "K-Means Customer Segmentation"
)


ax4.legend()


st.pyplot(fig4)


# =================================================
# GRAPH 5 - SAMPLE CLUSTERS
# =================================================

st.subheader(
    "Customer Clusters - Sample Data"
)


cluster_sample_df = df.sample(

    min(5000, len(df)),

    random_state=42

)


fig5, ax5 = plt.subplots()


ax5.scatter(

    cluster_sample_df[
        "annual spending"
    ],

    cluster_sample_df[
        "orderscount"
    ],

    c=cluster_sample_df[
        "Cluster"
    ],

    s=10,

    alpha=0.6,

    cmap="viridis"

)


# Display Centroids

ax5.scatter(

    centroids[:, 0],

    centroids[:, 1],

    marker="X",

    s=250,

    label="Centroids"

)


ax5.set_xlabel(
    "Annual Spending"
)


ax5.set_ylabel(
    "Order Count"
)


ax5.set_title(
    "Customer Clusters"
)


ax5.legend()


st.pyplot(fig5)


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("Navigation")


option = st.sidebar.radio(

    "Choose Option",

    [

        "Single Customer",

        "Upload File"

    ]

)


# =================================================
# SINGLE CUSTOMER
# =================================================

if option == "Single Customer":


    st.header(
        "Customer Segmentation"
    )


    st.write(

        "Enter customer details to identify "

        "the customer segment."

    )


    col1, col2 = st.columns(2)


    # Annual Spending Input

    with col1:


        annual_spending = st.number_input(

            "Annual Spending",

            min_value=0.0,

            value=10000.0

        )


    # Order Count Input

    with col2:


        order_count = st.number_input(

            "Order Count",

            min_value=0,

            value=5

        )


    # Prediction Button

    if st.button(

        "Find Customer Segment",

        use_container_width=True

    ):


        # ------------------------------------------
        # CREATE CUSTOMER DATA
        # ------------------------------------------

        customer = pd.DataFrame({

            "annual spending": [

                annual_spending

            ],

            "orderscount": [

                order_count

            ]

        })


        # ------------------------------------------
        # SCALE CUSTOMER DATA
        # ------------------------------------------

        customer_scaled = scaler.transform(

            customer

        )


        # ------------------------------------------
        # PREDICT CLUSTER
        # ------------------------------------------

        cluster = model.predict(

            customer_scaled

        )[0]


        # ------------------------------------------
        # FIND CUSTOMER TYPE
        # ------------------------------------------

        customer_type = cluster_labels[cluster]


        # ------------------------------------------
        # DISPLAY RESULT
        # ------------------------------------------

        st.success(

            f"Customer belongs to Cluster {cluster} - {customer_type}"

        )


        # ------------------------------------------
        # CUSTOMER INFORMATION
        # ------------------------------------------

        st.subheader(
            "Customer Information"
        )


        result1, result2, result3, result4 = (

            st.columns(4)

        )


        result1.metric(

            "Annual Spending",

            f"₹{annual_spending:,.0f}"

        )


        result2.metric(

            "Order Count",

            order_count

        )


        result3.metric(

            "Customer Cluster",

            cluster

        )


        result4.metric(

            "Customer Type",

            customer_type

        )


# =================================================
# FILE UPLOAD
# =================================================

elif option == "Upload File":


    st.header(
        "Upload Customer Dataset"
    )


    st.write(

        "Upload CSV or Excel file containing "

        "Annual Spending and Order Count."

    )


    uploaded_file = st.file_uploader(

        "Choose File",

        type=[

            "csv",

            "xlsx"

        ]

    )


    if uploaded_file is not None:


        # ------------------------------------------
        # READ FILE
        # ------------------------------------------

        if uploaded_file.name.endswith(".csv"):


            uploaded_df = pd.read_csv(

                uploaded_file

            )


        else:


            uploaded_df = pd.read_excel(

                uploaded_file

            )


        # ------------------------------------------
        # DISPLAY UPLOADED DATA
        # ------------------------------------------

        st.subheader(
            "Uploaded Dataset"
        )


        st.dataframe(

            uploaded_df,

            use_container_width=True

        )


        # ------------------------------------------
        # REQUIRED COLUMNS
        # ------------------------------------------

        required_columns = [

            "annual spending",

            "orderscount"

        ]


        # ------------------------------------------
        # CHECK COLUMNS
        # ------------------------------------------

        if all(

            column in uploaded_df.columns

            for column in required_columns

        ):


            # --------------------------------------
            # SELECT FEATURES
            # --------------------------------------

            upload_X = uploaded_df[

                required_columns

            ]


            # --------------------------------------
            # SCALE DATA
            # --------------------------------------

            upload_scaled = scaler.transform(

                upload_X

            )


            # --------------------------------------
            # PREDICT CLUSTERS
            # --------------------------------------

            uploaded_df["Cluster"] = (

                model.predict(

                    upload_scaled

                )

            )


            # --------------------------------------
            # ADD CUSTOMER TYPE
            # --------------------------------------

            uploaded_df["Customer Type"] = (

                uploaded_df["Cluster"].map(

                    cluster_labels

                )

            )


            # --------------------------------------
            # SUCCESS MESSAGE
            # --------------------------------------

            st.success(

                "Customer segmentation completed successfully!"

            )


            # --------------------------------------
            # DISPLAY RESULTS
            # --------------------------------------

            st.subheader(

                "Customer Segmentation Results"

            )


            st.dataframe(

                uploaded_df,

                use_container_width=True

            )


            # --------------------------------------
            # VISUALIZATION
            # --------------------------------------

            st.subheader(

                "Customer Segments"

            )


            fig, ax = plt.subplots()


            ax.scatter(

                uploaded_df[

                    "annual spending"

                ],

                uploaded_df[

                    "orderscount"

                ],

                c=uploaded_df[

                    "Cluster"

                ],

                cmap="viridis"

            )


            # Display Centroids

            ax.scatter(

                centroids[:, 0],

                centroids[:, 1],

                marker="X",

                s=250,

                label="Centroids"

            )


            ax.set_xlabel(

                "Annual Spending"

            )


            ax.set_ylabel(

                "Order Count"

            )


            ax.set_title(

                "Customer Segmentation"

            )


            ax.legend()


            st.pyplot(fig)


            # --------------------------------------
            # DOWNLOAD RESULT
            # --------------------------------------

            csv = uploaded_df.to_csv(

                index=False

            ).encode("utf-8")


            st.download_button(

                label="Download Results",

                data=csv,

                file_name=(

                    "customer_segments.csv"

                ),

                mime="text/csv",

                use_container_width=True

            )


        else:


            st.error(

                "File must contain "

                "'annual spending' and "

                "'orderscount' columns."

            )