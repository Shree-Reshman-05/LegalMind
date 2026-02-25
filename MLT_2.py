import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
np.random.seed(42)
data = pd.DataFrame({
"Undernourishment": np.random.uniform(5, 40, 200),
"ChildWasting": np.random.uniform(2, 25, 200), 
"ChildStunting": np.random.uniform(10, 50, 200), 
"ChildMortality": np.random.uniform(1, 15, 200) 
})
data["GHI"] = (
0.25 * data["Undernourishment"]
+ 0.25 * data["ChildWasting"]
+ 0.25 * data["ChildStunting"]
+ 0.25 * data["ChildMortality"]
+ np.random.normal(0, 1.5, 200)
)
print("Synthetic Global Hunger Index Dataset:")
print(data.head())

X_simple = data[['Undernourishment']]
y = data['GHI']
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
X_simple, y, test_size=0.2, random_state=42
)
simple_lr = LinearRegression()
simple_lr.fit(X_train_s, y_train_s)
y_pred_s = simple_lr.predict(X_test_s)
print("\n--- Simple Linear Regression (Undernourishment → GHI) ---")

print("MAE:", mean_absolute_error(y_test_s, y_pred_s))
print("MSE:", mean_squared_error(y_test_s, y_pred_s))
print("R2 Score:", r2_score(y_test_s, y_pred_s))
# Multiple Linear Regression
X_multi = data[['Undernourishment', 'ChildWasting', 'ChildStunting', 'ChildMortality']]
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
X_multi, y, test_size=0.2, random_state=42
)
multi_lr = LinearRegression()
multi_lr.fit(X_train_m, y_train_m)
y_pred_m = multi_lr.predict(X_test_m)
print("\n--- Multiple Linear Regression (All Indicators → GHI) ---")
print("MAE:", mean_absolute_error(y_test_m, y_pred_m))
print("MSE:", mean_squared_error(y_test_m, y_pred_m))
print("R2 Score:", r2_score(y_test_m, y_pred_m))
plt.figure()
plt.scatter(y_test_m, y_pred_m)
plt.xlabel("Actual Global Hunger Index")
plt.ylabel("Predicted Global Hunger Index")
plt.title("Actual vs Predicted Global Hunger Index")
plt.show()

# Print coefficients & intercept
print("Regression Coefficients:")
for feature, coef in zip(X_multi.columns, multi_lr.coef_):
    print(f"{feature}: {coef}")

print("Intercept:", multi_lr.intercept_)

print("MAE:", mean_absolute_error(y_test_m, y_pred_m))
print("MSE:", mean_squared_error(y_test_m, y_pred_m))
print("R2 Score:", r2_score(y_test_m, y_pred_m))

