import pandas as pd
from sklearn.model_selection import train_test_split

dataFrame = pd.read_csv('data/train-test-encoded.csv')

targetCol = 'posted_rate'
y = dataFrame[targetCol]

# Features for prediction
featureCols = ['distance', 'weight', 'market_index', 'quote_signal', 
               'month', 'dayOfYear', 'quarter', 'isWeekend',
               'weightPerDistance', 'marketXQuote', 
               'marketIndexLog', 'quoteSignalLog',
               'pickupEncoded', 'deliveryEncoded', 'equipmentEncoded']

X = dataFrame[featureCols]

print(f"\nAll columns in dataset: {len(dataFrame.columns)}")
print(f"Columns used for model: {len(featureCols)}")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

dataFrame.to_csv('data/training/train-test-final.csv', index=False)

# Stratified 80/20 split on distance
distanceQuartiles = pd.qcut(X['distance'], q=4, labels=False, duplicates='drop')

XTrain, XVal, yTrain, yVal = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=distanceQuartiles
)

print(f"Training set: {XTrain.shape[0]} samples")
print(f"Validation set: {XVal.shape[0]} samples")

XTrain.to_csv('data/training/X_train.csv', index=False)
XVal.to_csv('data/training/X_val.csv', index=False)
yTrain.to_csv('data/training/y_train.csv', index=False, header=False)
yVal.to_csv('data/training/y_val.csv', index=False, header=False)

print("\nComplted!")