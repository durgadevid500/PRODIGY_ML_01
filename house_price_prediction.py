
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
data = pd.read_csv(
    r"C:\Users\hp\Downloads\house-prices-advanced-regression-techniques\train.csv"
)

# 2. Select required features
data["TotalBathrooms"] = data["FullBath"] + (0.5 * data["HalfBath"])

X = data[["GrLivArea", "BedroomAbvGr", "TotalBathrooms"]]
y = data["SalePrice"]

# 3. Check missing values
print("Missing Values:")
print(X.isnull().sum())

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Create Linear Regression model
model = LinearRegression()

# 6. Train model
model.fit(X_train, y_train)

print("\nModel Training Completed!")

# 7. Predict house prices
predictions = model.predict(X_test)

# 8. Evaluate model
print("\nModel Evaluation:")
print("MAE:", mean_absolute_error(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))
print("RMSE:", mean_squared_error(y_test, predictions) ** 0.5)
print("R2 Score:", r2_score(y_test, predictions))

# 9. Predict price for a new house
new_house = pd.DataFrame({
    "GrLivArea": [2000],
    "BedroomAbvGr": [3],
    "TotalBathrooms": [2]
})

predicted_price = model.predict(new_house)

print("\nNew House Details:")
print(new_house)
print("Predicted House Price:", predicted_price[0])

# 10. Visualize actual vs predicted prices
plt.scatter(y_test, predictions)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()
