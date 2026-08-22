import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

XTrain = pd.read_csv('data/training/X_train.csv')
yTrain = pd.read_csv('data/training/y_train.csv', header=None).values.ravel()

XVal = pd.read_csv('data/training/X_val.csv')
yVal = pd.read_csv('data/training/y_val.csv', header=None).values.ravel()

print(f"\nTraining set: {XTrain.shape[0]} samples, {XTrain.shape[1]} features")
print(f"Validation set: {XVal.shape[0]} samples")

# Create Gradient Boosting model
model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.08,
    subsample=0.85,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    verbose=10
)

# Train model
model.fit(XTrain, yTrain)

# Predict on validation set
yPredVal = model.predict(XVal)

# Floor negative predictions at $50
yPredVal = np.maximum(yPredVal, 50.0)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(yVal, yPredVal))
mae = mean_absolute_error(yVal, yPredVal)
r2 = r2_score(yVal, yPredVal)

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)
print(f"RMSE: ${rmse:.2f}")
print(f"MAE:  ${mae:.2f}")
print(f"R²:   {r2:.4f}")

joblib.dump(model, 'models/model.pkl')
print("\nCompleted!")