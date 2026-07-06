# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Sample dataset
data = {
    'Area_sqft': [1000, 1200, 1500, 1800, 2000, 2200, 2500, 3000],
    'Price': [200000, 240000, 300000, 360000, 400000, 440000, 500000, 600000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features (X) and Target (y)
X = df[['Area_sqft']]
y = df['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
print("Actual Prices:   ", list(y_test))
print("Predicted Prices:", predictions)

print("\nMean Absolute Error:",
      mean_absolute_error(y_test, predictions))

print("R² Score:",
      r2_score(y_test, predictions))

# Predict the price of a new house
new_house = [[1700]]  # Area in square feet
predicted_price = model.predict(new_house)

print(f"\nPredicted price for a 1700 sq.ft house: ${predicted_price[0]:,.2f}")
