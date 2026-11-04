# PRODIGY_ML_01 – House Price Prediction

## Project Overview
This project implements a Linear Regression Machine Learning model to predict house prices based on selected house features.

This project is completed as Task 01 of the Machine Learning Internship at Prodigy InfoTech.

## Objective
To predict house prices using:
- Above Ground Living Area (GrLivArea)
- Number of Bedrooms (BedroomAbvGr)
- Total Bathrooms (FullBath and HalfBath)

## Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn
- Linear Regression

## Project Workflow
1. Load the house price dataset.
2. Perform feature engineering.
3. Select input features and target variable.
4. Check missing values.
5. Split data into training and testing sets.
6. Train the Linear Regression model.
7. Predict house prices.
8. Evaluate model performance.
9. Visualize actual vs predicted prices.

## Model Evaluation
The model is evaluated using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

## Dataset
Kaggle – House Prices: Advanced Regression Techniques

https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

Download `train.csv` from Kaggle before running the project.

## How to Run

1. Clone this repository.
2. Install the required libraries:

   `pip install -r requirements.txt`

3. Download the Kaggle training dataset.
4. Update the dataset path in the Python code to match your local file location.
5. Run:

   `python house_price_prediction.py`

## Internship
Machine Learning Intern – Prodigy InfoTech

Task 01 – House Price Prediction using Linear Regression.
