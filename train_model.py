import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. Data Ingestion
# The Boston dataset is deprecated in sklearn, so we load from the original source
data_url = "http://lib.stat.cmu.edu/datasets/boston"
print(f"Loading data from {data_url}...")

raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)

# The dataset is split into two lines per entry usually, but let's check the structure.
# Just following standard handling for this specific raw source which often needs careful parsing.
# Actually, the standard way to load the raw boston data now:
# Variables in order:
# CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV
# The file format has 506 lines, but some rows are split.
# Let's try a safer approach using the numpy trick often cited for this dataset source
# or simply use `pd.read_csv` with specific engine if needed.
# However, `read_csv` with `sep="\s+"` handles it if formatted correctly, but the raw file has 
# a jagged structure (some rows continue on next line).

# Let's use the parsing logic recommended for this specific URL structure:
# The first 11 columns are in one line, the next 3 (B, LSTAT, MEDV) are on the next line.
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :3]])
target = raw_df.values[1::2, 2]

feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.DataFrame(data, columns=feature_names)
# df['MEDV'] = target  # MEDV is already in data and feature_names

print("Data loaded successfully.")
print(df.head())

# 2. EDA
print("\nMissing values:")
print(df.isnull().sum())

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
# plt.show() # Commented out for script execution

# 3. Model Training
X = df.drop('MEDV', axis=1)
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"\nLinear Regression - MAE: {mae_lr:.2f}, R2: {r2_lr:.2f}")

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest - MAE: {mae_rf:.2f}, R2: {r2_rf:.2f}")

# 4. Compare and Save
best_model = rf if r2_rf > r2_lr else lr
print(f"\nBest model: {'Random Forest' if r2_rf > r2_lr else 'Linear Regression'}")

joblib.dump(best_model, 'model.pkl')
print("Model saved as model.pkl")
