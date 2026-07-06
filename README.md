# House-Price-Prediction
# Simple House Price Predictor using Scikit-Learn

A beginner-friendly Machine Learning project that uses Linear Regression to predict housing prices based on square footage and the number of bedrooms.

## 🚀 Overview
This project demonstrates the core workflow of building a Machine Learning model:
- Generating synthetic tabular data with Pandas and NumPy
- Splitting data into training and testing sets
- Training a `LinearRegression` model using Scikit-Learn
- Evaluating performance using Mean Absolute Error (MAE) and $R^2$ Score

## 🛠️ Prerequisites & Installation
Ensure you have Python installed, then install the required libraries:
```bash
pip install numpy pandas scikit-learn
```

## 📈 Model Performance
- **Mean Absolute Error (MAE):** ~$11,000 (average prediction error)
- **$R^2$ Score:** ~0.94 (explains 94% of data variance)

## 🔮 Sample Prediction
- Inputs: 2,400 SqFt, 4 Bedrooms
- Predicted Price: ~$490,000
