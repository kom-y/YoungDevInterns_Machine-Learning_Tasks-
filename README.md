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

### **Example Model Output**
```
Slope (Ad Spend Impact on Sales): 5.10
Slope (Social Media Engagement Impact on Sales): 0.78
Slope (Discounts Impact on Sales): -49.23
Intercept (Base Sales Without Any Factors): 5030.21

Enter advertising budget ($): 800
Enter social media engagement (likes/shares): 3500
Enter discount percentage offered: 10

Predicted Sales Revenue: $8750.45
