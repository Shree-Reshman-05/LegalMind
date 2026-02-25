# -------------------------------------------------
# LEGALMIND CHATBOT - MULTIPLE LINEAR REGRESSION
# WITH REGRESSION COEFFICIENT DISPLAY
# -------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -----------------------------
# Step 1: Create Evaluation Dataset
# -----------------------------

np.random.seed(42)

data = pd.DataFrame({
    "semantic_score": np.random.uniform(0.1, 0.95, 250),
    "num_sources": np.random.randint(1, 8, 250),
    "query_length": np.random.randint(3, 25, 250),
    "response_time": np.random.uniform(0.5, 4.0, 250)
})

# Target variable (User Rating 1–5 scale)
data["user_rating"] = (
    2.5
    + 1.8 * data["semantic_score"]
    + 0.25 * data["num_sources"]
    - 0.4 * data["response_time"]
    + 0.05 * data["query_length"]
    + np.random.normal(0, 0.3, 250)
)

# -----------------------------
# Step 2: Define X and Y
# -----------------------------

X = data[["semantic_score", "num_sources", "query_length", "response_time"]]
y = data["user_rating"]

# -----------------------------
# Step 3: Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 4: Train Model
# -----------------------------

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -----------------------------
# Step 5: Display Regression Coefficients
# -----------------------------

print("\n=========== MULTIPLE LINEAR REGRESSION RESULTS ===========\n")

print("Intercept (β0):", round(model.intercept_, 4))

print("\nRegression Coefficients (β values):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"β for {feature}: {round(coef, 4)}")

# Display Full Regression Equation
print("\nRegression Equation:")

equation = f"User_Rating = {round(model.intercept_,4)}"
for feature, coef in zip(X.columns, model.coef_):
    equation += f" + ({round(coef,4)} × {feature})"

print(equation)

# -----------------------------
# Step 6: Performance Metrics
# -----------------------------

print("\nModel Performance:")
print("R² Score:", round(r2_score(y_test, y_pred), 4))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 4))
print("MSE:", round(mean_squared_error(y_test, y_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))

# -----------------------------
# Step 7: Plot
# -----------------------------

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual User Rating")
plt.ylabel("Predicted User Rating")
plt.title("Actual vs Predicted Ratings (LegalMind Evaluation)")
plt.show()
