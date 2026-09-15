# 📊 Instagram Engagement Prediction

A Machine Learning application that predicts Instagram post engagement based on post characteristics and audience-related features.

The project combines data preprocessing, feature engineering, exploratory data analysis, unsupervised learning, and supervised machine learning to build an engagement prediction system.

## 🚀 Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_APP_URL)

> Replace `YOUR_STREAMLIT_APP_URL` with your deployed Streamlit app link.

---

## 📌 Project Overview

Instagram engagement can be influenced by several factors such as followers, hashtags, caption length, posting time, content type, and whether a post is published during a weekend or holiday.

This project analyzes Instagram data and builds Machine Learning models to predict the expected engagement of a post.

The target variable is:

**Engagement = Likes + Comments + Shares + Saves**

The project also explores user behavior and groups posts into clusters based on their characteristics.

---

## 🧠 Machine Learning Approach

The project follows a complete Machine Learning pipeline:

### 1. Data Preprocessing
- Data cleaning
- Handling date and time features
- Removing unnecessary columns
- Checking duplicates and data types
- Combining multiple Instagram datasets

### 2. Feature Engineering
New features were created from the original data, including:

- Posting hour
- Weekend indicator
- Holiday indicator
- Reach-time bucket
- Hashtag bucket
- Caption-length bucket
- User post count
- Previous median user engagement
- Content-type features

### 3. Data Transformation
- Log transformation for skewed numerical features
- One-hot encoding for categorical features
- Standardization using `StandardScaler`
- Train/Test split

### 4. Exploratory Data Analysis
The project analyzes:
- Numerical feature distributions
- Categorical feature distributions
- Outliers
- Feature correlations
- Engagement patterns

---

## 🤖 Machine Learning Models

Several approaches were explored:

### HistGradientBoosting Regressor

A gradient boosting model used to predict Instagram engagement.

### Random Forest Regressor

An ensemble of Decision Trees used as another engagement prediction model.

### K-Means Clustering

K-Means was used to identify groups of similar Instagram posts.

### PCA

Principal Component Analysis was used for dimensionality reduction and visualization of the discovered clusters.

### Mixture of Experts (MoE)

A routed Mixture of Experts approach was implemented:

```text
Instagram Post
      │
      ▼
   K-Means
      │
      ▼
   Cluster
   /  |  \
  ▼   ▼   ▼
Expert Expert Expert
  0     1     2
   \    |    /
      ▼
  Engagement
  Prediction
```
## 📈 Model Evaluation

The models are evaluated using:

- **R² Score**
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **Silhouette Score** for clustering

Feature importance was also investigated using permutation importance.

---

## 🖥️ Streamlit Application

The Streamlit application provides an interactive interface for using the trained Machine Learning models.

Users can provide Instagram post characteristics and obtain an estimated engagement prediction.

The application is designed to demonstrate how the trained Machine Learning pipeline can be used in an interactive environment.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Pickle
- Google Colab

---

## 📂 Project Structure

```text
Instagram-Engagement-Prediction/
│
├── app.py
├── Grammy_prediction.sav
├── requirements.txt
├── README.md
│
├── data/
│   └── ...
│
└── notebooks/
    └── Instagram_Engagement_Prediction.ipynb
