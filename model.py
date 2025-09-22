# %% [markdown]
# Import the neccessary libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score,train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tabulate import tabulate

# %% [markdown]
# Data Loading

# %%
# Load the CSV file into a pandas DataFrame
try:
    df = pd.read_csv("coin_ChainLink.csv")
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: The file was not found. Please make sure it's in the same directory.")
    exit()
df

# %% [markdown]
# Data Preprocessing

# %%
# Check for missing values
df_missing = df.isnull().sum()
print("Missing Values")
print(df_missing)

# Check for duplicated rows
df_duplicated = df.duplicated().sum()
print("Dupicated Rows")
print(df_duplicated)

# Drop rows with any missing values to ensure the data is clean
df.dropna(inplace=True)

# Rename the columns for clarity and consistency
df.rename(columns={
    "SNo":"serial_number",
    "Name":"name",
    "Symbol":"symbol",
    "Date":"date",
    "High":"high",
    "Low":"low",
    "Open":"open",
    "Close":"close",
    "Volume":"volume",
    "Marketcap":"marketcap"
},inplace=True)



df

# %% [markdown]
# Features Engineering

# %%
# We'll use the "open", "high", and "low"
features = ["open","high","low"]
target = "close"

# Create the features matrix (X) and the target vector (y)
X = df[features]
y = df[target]

print("Features (X):",X.head())
print("Target:",y.head())
print("-"*50)

# %% [markdown]
# Visualization before Training

# %%
plt.figure(figsize=(10,6))
plt.scatter(df["open"],df["close"],color="red")
plt.title("Open vs Close Price")
plt.xlabel("Open Price")
plt.ylabel("Close Price")
plt.grid(True)
plt.show()

# %%
plt.figure(figsize=(10,6))
plt.scatter(df["high"],df["close"],color="blue")
plt.title("High vs Close Price")
plt.xlabel("High Price")
plt.ylabel("Close Price")
plt.grid(True)
plt.show()

# %%
plt.figure(figsize=(10,6))
plt.scatter(df["low"],df["close"],color="black")
plt.title("Low vs Close Price")
plt.xlabel("Low Price")
plt.ylabel("Close Price")
plt.grid(True)
plt.show()

# %% [markdown]
# Model Comparison and Evaluation

# %%
# We'll compare the three popular regression models to find the best one
# Define a dictionary of model to compare

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=0.1),
    "Lasso Regression": Lasso(alpha=0.1)
}


# Use the K-Fold Cross-validation for a more robust evaluation
# It splits the  data into 5 parts and train/tests the model 5 times
# providing a more reliable performance score than a single train/test split

kf = KFold(n_splits=5,shuffle=True,random_state=42)

results = []
best_model = None
best_model = -np.inf

# Loop through each model to train and evaluate it
for name,model in models.items():
    print(f"Evaluating {name}....")

    # Calculate the R-squared score using cross-validation
    r2_scores = cross_val_score(model,X,y,cv=kf,scoring="r2")

    # Calculate the Mean Squared Error (MSE) using cross-validation
    # We use  a negative sign because scikit-learn's scoring is based on maximizing the score
    mse_scores = -cross_val_score(model,X,y,cv=kf,scoring="neg_mean_squared_error")

    # Calculate the Mean Absolute Error (MAE) using a cross-validation
    mae_scores = -cross_val_score(model,X,y,cv=kf,scoring="neg_mean_absolute_error")

    # Store the average scores
    results.append([
        name,
        np.mean(r2_scores),
        np.mean(mse_scores),
        np.mean(mae_scores)
    ])

    '''# Check if this is the best performing model so far based on R-squared
    if np.mean(r2_scores) > best_score:
        best_model = np.mean(r2_scores)
        best_model = model'''

# Print the comaparison results in a formatted table
headers = ["Model","Average R-squared","Average MSE","Average MAE"]
print("Model Comparion Results")
print(tabulate(results,headers=headers,floatfmt=".6f",tablefmt="grid"))
print('-'*50)
print("Maximum R-squared")
print(max(r2_scores))

# %% [markdown]
# Data Splitting

# %%
# We'll use the 80% of the data for training and 20% for testing to evaluate the model performance.
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# %% [markdown]
# Model Training

# %%
ridge_reg = Ridge()
ridge_reg.fit(X_train,y_train)

y_pred_ridge = ridge_reg.predict(X_test)


# %% [markdown]
# Model Evaluation

# %%
# Ridge Regression Metrics
print("-----Ridge Regression-----")
print(f"R-squared: {r2_score(y_test,y_pred_ridge):.4f}")
print(f"MAE: {r2_score(y_test,y_pred_ridge):.4f}")
print(f"MSE: {mean_absolute_error(y_test,y_pred_ridge):.4f}")


