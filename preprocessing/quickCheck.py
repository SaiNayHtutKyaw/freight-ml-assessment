import pandas as pd

dataFrame = pd.read_csv('data/train-test-cleaned.csv')

print("Missing values:")
print(dataFrame.isnull().sum())
print(f"\nDistance")
print(f"Min: {dataFrame['distance'].min():.2f} miles")
print(f"Max: {dataFrame['distance'].max():.2f} miles")
print(f"\nWeight")
print(f"Min: {dataFrame['weight'].min():.2f} lbs")
print(f"Max: {dataFrame['weight'].max():.2f} lbs")
print(f"\nPosted rate")
print(f"Min: ${dataFrame['posted_rate'].min():.2f}")
print(f"Max: ${dataFrame['posted_rate'].max():.2f}")
