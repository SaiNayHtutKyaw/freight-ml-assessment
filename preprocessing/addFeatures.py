import pandas as pd
import numpy as np

dataFrame = pd.read_csv('data/train-test-cleaned.csv')

# Datetime
dataFrame['date'] = pd.to_datetime(dataFrame['date'])
dataFrame['month'] = dataFrame['date'].dt.month
dataFrame['dayOfYear'] = dataFrame['date'].dt.dayofyear
dataFrame['quarter'] = dataFrame['date'].dt.quarter
dataFrame['isWeekend'] = dataFrame['date'].dt.dayofweek.isin([5, 6]).astype(int)

# Interaction
dataFrame['weightPerDistance'] = dataFrame['weight'] / (dataFrame['distance'] + 1)
dataFrame['marketXQuote'] = dataFrame['market_index'] * dataFrame['quote_signal']

# Log
dataFrame['marketIndexLog'] = np.log1p(dataFrame['market_index'])
dataFrame['quoteSignalLog'] = np.log1p(dataFrame['quote_signal'])

dataFrame.to_csv('data/train-test-add-features.csv', index=False)

print("\nCompleted!")
print(f"New columns added: 9")
print(f"Total columns now: {len(dataFrame.columns)}")