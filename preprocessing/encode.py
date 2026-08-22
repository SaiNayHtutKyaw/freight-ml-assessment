import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

dataFrame = pd.read_csv('data/train-test-add-features.csv')

encoders = {}

# Encoding
for col in ['pickup', 'delivery', 'equipment']:
    encoders[col] = LabelEncoder()
    dataFrame[f'{col}Encoded'] = encoders[col].fit_transform(dataFrame[col])
    print(f"  {col}: {len(encoders[col].classes_)} unique values encoded")

joblib.dump(encoders, 'models/encoders.pkl')
print("\nEncoders saved to models/encoders.pkl")

dataFrame.to_csv('data/train-test-encoded.csv', index=False)

print("\nCompleted!")
print(f"Total columns now: {len(dataFrame.columns)}")