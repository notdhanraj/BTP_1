# -----------------------------------------------------
# IMPORTS
# -----------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import time

# -----------------------------------------------------
# 1. DATA LOADING (from Excel file)
# -----------------------------------------------------
# Load the dataset from the provided Excel file.
print("Loading dataset from DataSet.xlsx...")
file_path = "/content/Data Set.xlsx" # Assuming the file is uploaded to /content/
try:
    df = pd.read_excel(file_path)
    print(f"Dataset loaded with {df.shape[0]} samples and {df.shape[1]} columns.\n")
    print("Columns in dataset:", df.columns.tolist())
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure it's uploaded.")
    # Exit or handle the error appropriately if the file is essential
    exit()

# -----------------------------------------------------
# 2. SELECT FEATURES AND TARGET
# -----------------------------------------------------
target_column = "Factor Of Safety" # Updated to match the actual column name from DataSet.xlsx
X = df.drop(columns=[target_column])
y = df[target_column]

print(f"Features (X): {X.columns.tolist()}")
print(f"Target (y): {target_column}\n")

# -----------------------------------------------------
# 3. PREPROCESSING
# -----------------------------------------------------
# Identify numeric columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Create a preprocessing step (Scaling is good practice for pipelines,
# though RF is generally robust to unscaled data)
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols)
    ]
)

# -----------------------------------------------------
# 4. BUILD PIPELINE & TRAIN/TEST SPLIT
# -----------------------------------------------------
# Define the full pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(random_state=42))])

# Split data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}\n")

# -----------------------------------------------------
# 5. HYPERPARAMETER TUNING (GridSearchCV)
# -----------------------------------------------------
print("Starting GridSearchCV (Hyperparameter Tuning)...")
start_time = time.time()

# Define parameter grid to search
param_grid = {
    'regressor__n_estimators': [100, 200],      # Number of trees in the forest
    'regressor__max_depth': [10, 20, None],     # Max depth of trees
    'regressor__min_samples_split': [2, 3]      # Min samples to split a node
}

# Initialize GridSearch
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,                # 3-Fold Cross Validation
    n_jobs=-1,           # Use all CPU cores
    scoring='neg_mean_squared_error',
    verbose=1
)

# Train the model with GridSearch
grid_search.fit(X_train, y_train)

end_time = time.time()
print(f"\nGridSearch completed in {end_time - start_time:.2f} seconds.")
print(f"Best Parameters: {grid_search.best_params_}\n")

# -----------------------------------------------------
# 6. EVALUATE ACCURACY
# -----------------------------------------------------
# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Predict on Test Set
y_pred = best_model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("-" * 30)
print("MODEL PERFORMANCE REPORT")
print("-" * 30)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score (Accuracy):          {r2:.4f}")
print("-" * 30)

# -----------------------------------------------------
# 7. SAVE THE TRAINED MODEL
# -----------------------------------------------------
model_filename = "rf_fos_model.pkl"
joblib.dump(best_model, model_filename)
print(f"\n✅ Model saved successfully as '{model_filename}'")

# Empty file
