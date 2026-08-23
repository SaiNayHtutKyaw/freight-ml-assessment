# Freight Rate Prediction ML Assessment

## Overview
Machine learning model to predict freight load rates using historical data.

## Dataset
- Training: 48,000 loads (Jan-Oct 2025)
- Validation: 12,000 loads
- Target: Posted rate ($57 - $25,533)

## Model Performance
- **R²**: 0.8404 (explains 84% of variance)
- **RMSE**: $582.60
- **MAE**: $144.93

## Data Quality Issues Handled
- 292 negative weights → converted to positive
- 300 missing weights → filled with equipment-type median
- 374 missing market_index → filled with global median
- Unseen cities in validation → mapped to most common city
- Date format inconsistency → handled DD/MM/YYYY format

## Features Used
- Temporal: month, dayOfYear, quarter, isWeekend
- Interactions: weightPerDistance, marketXQuote
- Log-transformed: marketIndexLog, quoteSignalLog
- Encoded: pickupEncoded, deliveryEncoded, equipmentEncoded

## Model: Gradient Boosting
- n_estimators: 200
- max_depth: 8
- learning_rate: 0.08
- Stratified 80/20 split on distance quartiles

## Run Instructions

```bash
# 1. Clean data
python preprocessing/cleanData.py

# 2. Add features
python preprocessing/addFeatures.py

# 3. Encode the variables
python preprocessing/encode.py

# 4. Split into train/validation
python preprocessing/splitDataset.py

# 5. Train model
python preprocessing/trainModel.py

# 6. Prediction for validation dataset
python prediction.py

# 7. Prediction for December 
python forDecember.py

# 8. Validate with score.py
python score.py --predictions data/validation_predictions.csv --december-predictions data/december-chart-inputs.csv --output-dir result
```

## Outputs
- `data/validation_predictions.csv`: 12,000 predicted rates
- `result/candidate_december.png`: December trend chart
- `models/model.pkl`: Trained model
- `models/encoders.pkl`: Label encoders

## Key Findings
- The dataset had 48,000 loads
- Prices ranged from $57 to $25,533 depending on distance
- Found several data quality problems that needed fixing
- Built a model that predicts prices with 84% accuracy
- Model is stable and doesn't make crazy predictions