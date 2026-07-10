# Import Libraries

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Load Dataset

df = pd.read_excel(
    r"C:\Users\admin\OneDrive\Desktop\UPGRAD\ml\Projects\Flipkart_KMeans_Project\Customer_dataset1.xlsx"
)

print("Original Dataset:")
print(df)


# Check Null Values

print("\nNull Values:")
print(df.isnull().sum())


# Select Features

X = df[['annual spending', 'orderscount']]


# Scatter Plot Before Clustering

plt.scatter(
    X['annual spending'],
    X['orderscount']
)

plt.xlabel("Annual Spending")
plt.ylabel("Order Count")
plt.title("Customers Before Clustering")

plt.show()


# Sample Scatter Plot

sample_df = df.sample(
    min(2000, len(df)),
    random_state=42
)

plt.scatter(
    sample_df['annual spending'],
    sample_df['orderscount'],
    s=10,
    alpha=0.5
)

plt.xlabel('Annual Spending')
plt.ylabel('Order Count')
plt.title('Annual Spending vs Order Count')

plt.show()


# Scaling

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# Elbow Method

wcss = []

for i in range(1, 11):

    model = KMeans(
        n_clusters=i,
        random_state=42
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)


# Elbow Graph

plt.plot(
    range(1, 11),
    wcss,
    marker='o'
)

plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")

plt.show()


# Create Final K-Means Model

kmeans = KMeans(
    n_clusters=4,
    random_state=42
)


# Train Model and Create Clusters

df['Cluster'] = kmeans.fit_predict(X_scaled)


# -------------------------------------------------
# CREATE CUSTOMER TYPE BASED ON CLUSTER
# -------------------------------------------------

# Find Average Annual Spending of Each Cluster

cluster_spending = df.groupby('Cluster')[
    'annual spending'
].mean()


# Display Average Spending of Each Cluster

print("\nAverage Spending of Each Cluster:")

print(cluster_spending)


# Sort Clusters Based on Average Annual Spending

sorted_clusters = cluster_spending.sort_values().index


# Create Cluster Labels

cluster_labels = {

    sorted_clusters[0]: "Poor Customer",

    sorted_clusters[1]: "Common Customer",

    sorted_clusters[2]: "Regular Customer",

    sorted_clusters[3]: "VIP Customer"

}


# Add Customer Type Column

df['Customer Type'] = df['Cluster'].map(
    cluster_labels
)


# Display Cluster Labels

print("\nCluster Labels:")

print(cluster_labels)


# Display Final Dataset

print("\nDataset With Clusters and Customer Type:")

print(
    df[
        [
            'customer id',
            'annual spending',
            'orderscount',
            'Cluster',
            'Customer Type'
        ]
    ]
)


# Get Centroids

centroids = kmeans.cluster_centers_


# Convert Centroids to Original Scale

centroids = scaler.inverse_transform(
    centroids
)


# Final Scatter Plot

plt.scatter(
    df['annual spending'],
    df['orderscount'],
    c=df['Cluster']
)


# Display Centroids

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker='X',
    s=200
)


plt.xlabel("Annual Spending")
plt.ylabel("Order Count")
plt.title("K-Means Customer Segmentation")

plt.show()


# Sample Cluster Data

sample_df = df.sample(
    min(5000, len(df)),
    random_state=42
)


# Sample Cluster Scatter Plot

plt.scatter(
    sample_df['annual spending'],
    sample_df['orderscount'],
    c=sample_df['Cluster'],
    s=10,
    alpha=0.6,
    cmap='viridis'
)

plt.xlabel('Annual Spending')
plt.ylabel('Order Count')
plt.title('Customer Clusters')

plt.show()


# Calculate Silhouette Score

score = silhouette_score(
    X_scaled,
    df['Cluster'],
    sample_size=min(100000, len(df)),
    random_state=42
)


print("Silhouette Score:", score)