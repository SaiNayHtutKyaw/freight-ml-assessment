import pandas as pd
import numpy as np
import joblib

model = joblib.load('models/model.pkl')
encoders = joblib.load('models/encoders.pkl')

validationDF = pd.read_csv('data/validation.csv')

validationDF['date'] = pd.to_datetime(validationDF['date'], format='%d/%m/%Y')
validationDF['weight'].fillna(validationDF['weight'].median(), inplace=True)
validationDF['market_index'].fillna(validationDF['market_index'].median(), inplace=True)

validationDF['month'] = validationDF['date'].dt.month
validationDF['dayOfYear'] = validationDF['date'].dt.dayofyear
validationDF['quarter'] = validationDF['date'].dt.quarter
validationDF['isWeekend'] = validationDF['date'].dt.dayofweek.isin([5, 6]).astype(int)

validationDF['weightPerDistance'] = validationDF['weight'] / (validationDF['distance'] + 1)
validationDF['marketXQuote'] = validationDF['market_index'] * validationDF['quote_signal']
validationDF['marketIndexLog'] = np.log1p(validationDF['market_index'])
validationDF['quoteSignalLog'] = np.log1p(validationDF['quote_signal'])

# Handle new cities
for col in ['pickup', 'delivery', 'equipment']:
    validationDF[col] = validationDF[col].apply(
        lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0]
    )
    validationDF[f'{col}Encoded'] = encoders[col].transform(validationDF[col])

featureCols = ['distance', 'weight', 'market_index', 'quote_signal', 
               'month', 'dayOfYear', 'quarter', 'isWeekend',
               'weightPerDistance', 'marketXQuote', 
               'marketIndexLog', 'quoteSignalLog',
               'pickupEncoded', 'deliveryEncoded', 'equipmentEncoded']

X = validationDF[featureCols]
predictions = model.predict(X)
predictions = np.maximum(predictions, 50.0)

results = pd.DataFrame({
    'load_id': validationDF['load_id'],
    'predicted_rate': predictions
})

results.to_csv('validation_predictions.csv', index=False)
print(f"✓ Predicted {len(results)} loads")