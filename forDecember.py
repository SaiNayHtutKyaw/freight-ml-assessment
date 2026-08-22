import pandas as pd
import numpy as np
import joblib

model = joblib.load('models/model.pkl')
encoders = joblib.load('models/encoders.pkl')

decemberDF = pd.read_csv('data/december-chart-inputs.csv')

decemberDF['date'] = pd.to_datetime(decemberDF['date'])
decemberDF['market_index'] = 1.0
decemberDF['quote_signal'] = 2.0

decemberDF['month'] = decemberDF['date'].dt.month
decemberDF['dayOfYear'] = decemberDF['date'].dt.dayofyear
decemberDF['quarter'] = decemberDF['date'].dt.quarter
decemberDF['isWeekend'] = decemberDF['date'].dt.dayofweek.isin([5, 6]).astype(int)

decemberDF['weightPerDistance'] = decemberDF['weight'] / (decemberDF['distance'] + 1)
decemberDF['marketXQuote'] = decemberDF['market_index'] * decemberDF['quote_signal']
decemberDF['marketIndexLog'] = np.log1p(decemberDF['market_index'])
decemberDF['quoteSignalLog'] = np.log1p(decemberDF['quote_signal'])

for col in ['pickup', 'delivery', 'equipment']:
    decemberDF[f'{col}Encoded'] = encoders[col].transform(decemberDF[col])

featureCols = ['distance', 'weight', 'market_index', 'quote_signal', 
               'month', 'dayOfYear', 'quarter', 'isWeekend',
               'weightPerDistance', 'marketXQuote', 
               'marketIndexLog', 'quoteSignalLog',
               'pickupEncoded', 'deliveryEncoded', 'equipmentEncoded']

X = decemberDF[featureCols]
predictions = model.predict(X)
predictions = np.maximum(predictions, 50.0)

decemberDF['predicted_rate'] = predictions
decemberDF = decemberDF[['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date', 'predicted_rate']]
decemberDF['date'] = decemberDF['date'].dt.strftime('%Y-%m-%d')

decemberDF.to_csv('data/december-chart-inputs.csv', index=False)
print("✓ December predictions complete")