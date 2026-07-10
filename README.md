# 🛒 Customer Segmentation System Using K-Means Clustering

## 📌 Project Overview

The **Customer Segmentation System** is a Machine Learning project developed using the **K-Means Clustering Algorithm**.

The main objective of this project is to divide customers into different groups based on their purchasing behavior.

The model uses the following customer features:

- Annual Spending
- Order Count

Based on these features, customers are divided into **4 different clusters**.

Each cluster is also assigned a meaningful customer category:

- Poor Customer
- Common Customer
- Regular Customer
- VIP Customer

The project also includes an interactive **Streamlit Web Application** where users can predict the segment of a single customer or upload a complete customer dataset for segmentation.


## 🎯 Project Objectives

The main objectives of this project are:

- Analyze customer purchasing behavior
- Perform customer segmentation using Machine Learning
- Group similar customers using K-Means Clustering
- Identify valuable customer groups
- Visualize customer clusters
- Predict the cluster of a new customer
- Assign meaningful customer categories to clusters
- Allow users to upload customer datasets
- Download customer segmentation results


## 📊 Dataset Information

The dataset contains **100,000 customer records**.

The main columns used in the Machine Learning model are:

| Column | Description |
|---|---|
| customer id | Unique ID of the customer |
| annual spending | Total annual spending of the customer |
| orderscount | Total number of orders placed by the customer |

After applying the K-Means Clustering algorithm, two additional columns are created:

| Column | Description |
|---|---|
| Cluster | Cluster assigned by the K-Means model |
| Customer Type | Customer category based on the assigned cluster |


## 🤖 Machine Learning Algorithm

This project uses the **K-Means Clustering Algorithm**.

K-Means is an **Unsupervised Machine Learning Algorithm** used to divide data into groups of similar data points.

The model groups customers based on:

- Annual Spending
- Order Count


## ⚙️ Project Workflow

The complete Machine Learning workflow used in this project is:

1. Import Required Libraries
2. Load Customer Dataset
3. Check Missing Values
4. Select Features
5. Perform Data Visualization
6. Apply Feature Scaling
7. Use the Elbow Method
8. Select the Optimal Number of Clusters
9. Train the K-Means Model
10. Assign Customers to Clusters
11. Calculate Cluster Centroids
12. Visualize Customer Segments
13. Assign Customer Types
14. Evaluate Clustering Performance
15. Build Streamlit Web Application
16. Deploy the Application


## 📏 Feature Scaling

The project uses **StandardScaler** from Scikit-learn.

Feature scaling is important because Annual Spending and Order Count have different numerical ranges.

StandardScaler transforms the features so that both variables contribute properly to the clustering process.


## 📉 Elbow Method

The **Elbow Method** is used to determine the appropriate number of clusters.

The model calculates the **Within-Cluster Sum of Squares (WCSS)** for different values of K.

Based on the Elbow Method graph, the selected number of clusters is:

**K = 4**


## 👥 Customer Segments

The K-Means model divides customers into four clusters.

The clusters are categorized based on their average annual spending.

| Customer Type | Description |
|---|---|
| Poor Customer | Customers with the lowest average annual spending |
| Common Customer | Customers with relatively low annual spending |
| Regular Customer | Customers with moderate to high annual spending |
| VIP Customer | Customers with the highest average annual spending |


## 📈 Model Evaluation

Since K-Means is an **Unsupervised Machine Learning Algorithm**, traditional classification accuracy cannot be calculated.

Therefore, the model is evaluated using the **Silhouette Score**.

The Silhouette Score measures how well customers are grouped within their clusters.

A higher Silhouette Score indicates better cluster separation.


## 📊 Data Visualizations

The project includes multiple visualizations:

- Full Customer Dataset Scatter Plot
- Sample Customer Dataset Scatter Plot
- Elbow Method Graph
- Full Customer Cluster Visualization
- Sample Customer Cluster Visualization
- Cluster Centroids


## 💻 Streamlit Web Application

An interactive web application is developed using **Streamlit**.

The application provides two main features.


### 👤 Single Customer Prediction

Users can enter:

- Annual Spending
- Order Count

The Machine Learning model predicts:

- Customer Cluster
- Customer Type

Example Output:

```text
Customer belongs to Cluster 1 - VIP Customer
