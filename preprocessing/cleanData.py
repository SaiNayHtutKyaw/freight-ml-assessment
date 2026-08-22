import pandas as pd

dataFrame = pd.read_csv('data/train-test.csv')

# Negative weights to positive (assuming they are due to data entry error)
negativeCount = len(dataFrame[dataFrame['weight'] < 0])
dataFrame['weight'] = dataFrame['weight'].abs()
print(f"{negativeCount} negative weights converted to positive.")

# Fix missing weight by filling with median weight of the same equipment
dataFrame['weight'] = dataFrame.groupby('equipment')['weight'].transform(
    lambda x: x.fillna(x.median())
)
dataFrame['weight'] = dataFrame['weight'].fillna(dataFrame['weight'].median())
print(f"Completed!! Remaining missing weights: {dataFrame['weight'].isnull().sum()}")

# Fix missing market_index by filling with median market_index
dataFrame['market_index'] = dataFrame['market_index'].fillna(dataFrame['market_index'].median())
print(f"Missing market_index after fix: {dataFrame['market_index'].isnull().sum()}")

# Convert date to datetime format
dataFrame['date'] = pd.to_datetime(dataFrame['date'])

dataFrame.to_csv('data/train-test-cleaned.csv', index=False)

print("\nCompleted!")
print(f"Total rows: {len(dataFrame)}")