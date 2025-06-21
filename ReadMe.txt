# 🎬 Movie Genre Classification using Machine Learning & NLP

This project focuses on classifying movie genres based on movie descriptions using advanced Natural Language Processing (NLP) and Machine Learning (ML) techniques. The aim is to build an accurate multi-class classifier capable of predicting genres from textual descriptions and some numeric metadata.

---

## 🚀 Project Overview

- **Dataset:** Movie metadata including `Description`, `Genre`, `Rating`, `Duration`, `Release Year`.
- **Text Processing:** TF-IDF vectorization of movie descriptions.
- **Data Balancing:** Techniques including SMOTE, Random OverSampler, Random UnderSampler.
- **Feature Scaling:** StandardScaler, MinMaxScaler, RobustScaler.
- **Dimensionality Reduction:** PCA, LDA, TruncatedSVD.
- **Models Used:** Logistic Regression, Decision Tree, Random Forest, SVM, KNN, XGBoost, Naive Bayes, Gradient Boosting, MLP, Extra Trees.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score.
- **Visualization:** Word clouds, PCA scatterplots, heatmaps, bar plots, and interactive plots using Plotly.

---

## 📁 Dataset

- **Source:** Kaggle  
- **File:** `movie_genre_classification_final.csv`  
- **Description:** Contains movie titles, textual descriptions, genres, and numeric features like rating, duration, and release year.

---

## 🧪 Methodology

### 1. Data Loading and Cleaning

- Load CSV data.
- Convert `Description` column to string type.
- Impute missing values using the most frequent value per column.

### 2. Exploratory Data Analysis (EDA)

- Plot genre distribution using logarithmic scale.
- Generate word clouds for each genre to visualize common words.
- Apply TF-IDF vectorization on movie descriptions.
- Use PCA (2 components) to visualize TF-IDF embeddings by genre.
- Calculate Mutual Information to find top informative words per genre.
- Visualize numerical feature distributions (`rating`, `duration`, `release_year`) across genres using violin plots.

### 3. Feature Engineering and Preprocessing

- Apply Label Encoding to categorical columns.
- Balance classes using different sampling methods: no balancing, random oversampling, random undersampling, SMOTE.
- Scale features with StandardScaler, MinMaxScaler, and RobustScaler.
- Reduce feature dimensions with PCA, LDA, and TruncatedSVD.

### 4. Model Training and Evaluation

- Train 10 classifiers over different combinations of encoding, balancing, scaling, and dimensionality reduction.
- Split data into train/test (80/20).
- Evaluate models using accuracy, precision, recall, and F1-score (weighted average).
- Store all results for analysis.

### 5. Result Visualization

- Bar plots of top models by F1-score.
- Heatmaps showing F1-score relationships between scalers, reducers, and models.
- Interactive Plotly scatterplot of precision vs recall sized by F1-score.

---

## 📊 Best Model Results (Selected Top Runs)

| Encoding       | Balancer            | Scaler         | Reducer     | Model            | Accuracy | Precision | Recall | F1-Score |
|----------------|---------------------|----------------|-------------|------------------|----------|-----------|--------|----------|
| LabelEncoding  | RandomUnderSampler   | MinMaxScaler   | PCA         | MLP              | 1.0      | 1.0       | 1.0    | 1.0      |
| LabelEncoding  | RandomUnderSampler   | MinMaxScaler   | LDA         | DecisionTree     | 1.0      | 1.0       | 1.0    | 1.0      |
| LabelEncoding  | SMOTE               | MinMaxScaler   | PCA         | SVM              | 1.0      | 1.0       | 1.0    | 1.0      |
| LabelEncoding  | SMOTE               | MinMaxScaler   | PCA         | MLP              | 1.0      | 1.0       | 1.0    | 1.0      |
| LabelEncoding  | SMOTE               | MinMaxScaler   | PCA         | ExtraTree        | 1.0      | 1.0       | 1.0    | 1.0      |
| LabelEncoding  | SMOTE               | MinMaxScaler   | LDA         | DecisionTree     | 1.0      | 1.0       | 1.0    | 1.0      |
| LabelEncoding  | SMOTE               | MinMaxScaler   | LDA         | RandomForest     | 1.0      | 1.0       | 1.0    | 1.0      |
| LabelEncoding  | SMOTE               | MinMaxScaler   | LDA         | SVM              | 1.0      | 1.0       | 1.0    | 1.0      |
| LabelEncoding  | SMOTE               | MinMaxScaler   | LDA         | KNN              | 1.0      | 1.0       | 1.0    | 1.0      |
| LabelEncoding  | SMOTE               | MinMaxScaler   | LDA         | XGBoost          | 1.0      | 1.0       | 1.0    | 1.0      |

*(Note: This is a sample of top results; many other combinations achieved similar performance.)*

---

## 📦 Installation

To install all required packages:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost imbalanced-learn wordcloud
