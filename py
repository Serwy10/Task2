import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Load the dataset (Assuming the Titanic dataset is in CSV format)
df = pd.read_csv('titanic.csv')

# 1. Handle Missing Values
# ------------------------
# We will fill missing values in numerical columns with the median of the column.
# For categorical columns, we'll use the most frequent value (mode).

numeric_features = ['Age', 'Fare']  # Replace with your actual numeric columns
categorical_features = ['Embarked', 'Sex']  # Replace with your actual categorical columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))  # Imputing missing values with the median for numeric columns
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputing missing values with the most frequent value for categorical columns
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encoding to handle categorical variables
])

# 2. Scaling/Normalization of Features
# ------------------------------------
# For numerical features, scaling is essential for algorithms like SVMs, K-Means, and Neural Networks.
# We'll use StandardScaler to standardize features (mean=0, variance=1) for these models.

scaler = StandardScaler()

# 3. Dimensionality Reduction (Optional)
# --------------------------------------
# PCA (Principal Component Analysis) is used to reduce the dimensionality of the dataset.
# It's helpful when dealing with high-dimensional data to avoid the curse of dimensionality.
# Here, we reduce the dataset to 2 principal components for simplicity.

pca = PCA(n_components=2)

# 4. Pipeline Creation
# --------------------
# We'll now create a full pipeline that includes all the steps defined above.
# This allows for consistent and repeatable preprocessing.

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', scaler),
    ('pca', pca)
])

# 5. Splitting the Dataset into Training and Testing Sets
# -------------------------------------------------------
# It's essential to split the data before applying the pipeline to prevent data leakage.

X = df.drop('Survived', axis=1)  # Features
y = df['Survived']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Fit the Pipeline on Training Data
# ------------------------------------
# The pipeline is fitted to the training data, ensuring all transformations are learned from it.

pipeline.fit(X_train)

# 7. Transform the Data
# ---------------------
# Transform both the training and testing data using the fitted pipeline.

X_train_transformed = pipeline.transform(X_train)
X_test_transformed = pipeline.transform(X_test)

# The pipeline now processes the raw data and outputs a transformed dataset ready for model training.
