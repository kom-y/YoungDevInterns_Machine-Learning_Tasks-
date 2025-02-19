import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Sample dataset with missing values and categorical data
data = {
    "Ad Spend ($)": [500, 1200, np.nan, 800, 1500, 700, np.nan, 1100],
    "Social Media Engagement": [2000, 4000, 3500, np.nan, 5000, 4500, 3200, np.nan],
    "Discount (%)": [10, 20, 15, 5, 25, np.nan, 30, 10],
    "Region": ["North", "South", "East", "West", "North", "South", "East", "West"],  # Categorical
    "Sales Revenue ($)": [7000, 8500, 7800, 6200, 9100, 7300, 6800, 8000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Separate features (X) and target (y)
X = df.drop(columns=["Sales Revenue ($)"])
y = df["Sales Revenue ($)"]

# Print missing values before imputation
print("Before Imputation:\n", X.isna().sum())

# Define preprocessing steps
numerical_features = ["Ad Spend ($)", "Social Media Engagement", "Discount (%)"]
categorical_features = ["Region"]

# Step 1: Handle missing values
num_imputer = SimpleImputer(strategy="mean")  # Replace NaNs with mean
X[numerical_features] = num_imputer.fit_transform(X[numerical_features])

# Step 2: Scale numerical features
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Step 3: Encode categorical variables
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_categories = encoder.fit_transform(X[categorical_features])

# Step 4: Convert categorical encoding to DataFrame
encoded_columns = encoder.get_feature_names_out(categorical_features)
X_encoded = pd.DataFrame(encoded_categories, columns=encoded_columns)

# Step 5: Concatenate numerical and categorical features
X_processed = pd.concat([pd.DataFrame(X[numerical_features]), X_encoded], axis=1)

# Check if NaNs are still present
print("After Preprocessing:\n", X_processed.isna().sum())

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Print results
print("Processed Feature Matrix (X_train):\n", X_train)
print("\nShape of X_train:", X_train.shape)
print("\nShape of X_test:", X_test.shape)
print("\nFirst row of X_train (after preprocessing):\n", X_train.iloc[0])
