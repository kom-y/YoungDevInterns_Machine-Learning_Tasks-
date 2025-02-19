# YoungDevInterns_Machine-Learning_Tasks-
# TASK 01
# Linear Regression Model for Sales Prediction
## 📌 Project Overview
This project implements a **Linear Regression Model** to predict **Sales Revenue** based on three key factors:
- **Advertising Budget ($)**
- **Social Media Engagement (likes, shares)**
- **Discount Percentage Offered (%)**
The model helps understand how each factor influences sales and allows users to input their own values for predictions.

---
## 🎯 How It Works
- **Generates a synthetic dataset** simulating real-world sales data.
- **Trains a Linear Regression Model** using `scikit-learn`.
- **Evaluates the model** using Mean Squared Error (MSE).
- **Allows user input** to predict sales revenue based on advertising, social media engagement, and discounts.
- **Visualizes relationships** between features using `seaborn` and `matplotlib`.

---

## 📊 Model Evaluation
- **Mean Squared Error (MSE):** Measures how far the predictions deviate from actual values.
- **Regression Coefficients:** Show the impact of each feature on sales revenue.

#TASK 02
## 📌 Project Overview
### 🔄 Data Preprocessing
Before training the model, the dataset is cleaned and transformed using the following steps:
- **Handling Missing Values**: Missing numerical values are replaced with the mean.
- **Feature Scaling**: Numerical features are standardized using `StandardScaler` to improve model performance.
- **Categorical Encoding**: The `Region` column is one-hot encoded to convert categorical data into numerical format.
- **Splitting Data**: The dataset is split into **training (80%)** and **testing (20%)** sets to evaluate model accuracy.

### 📊 Processed Data Summary
After preprocessing, the dataset is transformed into a numerical format, making it suitable for machine learning:
- **Standardized numerical values**: Adjusts values to a common scale.
- **One-hot encoded categorical variables**: Converts categories into separate binary columns.
- **Final shape of the dataset**: Training set contains **6 samples and 7 features**, while the test set contains **2 samples and 7 features**.

This ensures the model receives well-prepared input for accurate predictions.

---

## 🚀 Next Steps
With the preprocessed data, the next step is to **train the Linear Regression model** using scikit-learn and evaluate its performance using **Mean Squared Error (MSE)**.



Predicted Sales Revenue: $8750.45
