import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate a synthetic dataset
np.random.seed(42)
num_samples = 30

ad_spend = np.random.randint(100, 1000, (num_samples, 1))
social_media = np.random.randint(500, 5000, (num_samples, 1))  
discounts = np.random.randint(5, 30, (num_samples, 1))  

# Sales Revenue is influenced by all three factors
sales_revenue = (
    5 * ad_spend +  
    0.8 * social_media +  
    -50 * discounts +  # High discounts reduce revenue per sale
    np.random.randn(num_samples, 1) * 500 + 5000  
)

# Combine all features into a single dataset
X = np.hstack((ad_spend, social_media, discounts))
y = sales_revenue

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Print model coefficients
print(f"Slope (Ad Spend Impact on Sales): {model.coef_[0][0]:.2f}")
print(f"Slope (Social Media Engagement Impact on Sales): {model.coef_[0][1]:.2f}")
print(f"Slope (Discounts Impact on Sales - Negative means more discount lowers revenue): {model.coef_[0][2]:.2f}")
print(f"Intercept (Base Sales Without Any Factors): {model.intercept_[0]:.2f}")

# Take user input for prediction
user_ad_spend = float(input("Enter advertising budget ($): "))
user_social_media = float(input("Enter social media engagement (likes/shares): "))
user_discounts = float(input("Enter discount percentage offered: "))

# Predict sales revenue based on user input
user_input = np.array([[user_ad_spend, user_social_media, user_discounts]])
predicted_sales = model.predict(user_input)

print(f"Predicted Sales Revenue: ${predicted_sales[0][0]:.2f}")

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Convert NumPy array to Pandas DataFrame
df = pd.DataFrame(X, columns=['Ad Spend', 'Social Media Engagement', 'Discounts'])
df['Sales Revenue'] = y 

# Now use Seaborn pairplot
sns.pairplot(df, diag_kind='kde', plot_kws={'alpha': 0.5})
plt.show()
