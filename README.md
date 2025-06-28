# 🧠 Mall Customer Segmentation using K-Means Clustering

This project implements **K-Means Clustering** to segment customers based on their annual income and spending score using the **Mall Customer Segmentation Dataset**.

## 📁 Dataset

Dataset Source: [Kaggle - Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

Used Features:
- `Annual Income (k$)`
- `Spending Score (1-100)`

## 🚀 Objective

Group customers of a retail store into distinct clusters based on purchasing behavior, which helps businesses to:
- Identify high-value customers
- Tailor marketing strategies
- Optimize product targeting

## 📌 Steps Performed

1. Data loading and preprocessing using Pandas.
2. Feature selection (`Annual Income`, `Spending Score`).
3. Determining optimal number of clusters using the **Elbow Method**.
4. Applying **KMeans** algorithm.
5. Visualizing clusters with **Matplotlib** and **Seaborn**.

## 📊 Elbow Method Example

The Elbow Method helps determine the ideal number of clusters by plotting inertia (WCSS) against `k`.

![Elbow Curve](./screenshots/elbow_curve.png)

## 🎯 Final Cluster Plot

Visual representation of customer segments.

![Customer Clusters](./screenshots/customer_clusters.png)

## 🛠️ Technologies Used

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## 📂 How to Run

1. Clone this repository or download the code.
2. Install dependencies:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn
