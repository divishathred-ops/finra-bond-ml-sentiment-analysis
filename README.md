# FINRA Bond Market Sentiment Analysis

## Project Overview
Enterprise ML system for OTC bond market sentiment analysis using real FINRA Fixed Income API data.
Real-time BUY/SELL/HOLD trading signals for traders, asset managers, and banking institutions.

## Data
- **Source**: FINRA Fixed Income Market Data API
- **Datasets**: Corporate Breadth, Corporate Sentiment, Agency Data, Treasury Aggregates
- **Records**: 2,706
- **Date Range**: 2022-12-14 00:00:00 to 2025-12-12 00:00:00

## Models
### LSTM (Recurrent Neural Network)
- Architecture: 2-layer LSTM with attention mechanism
- Test Accuracy: 53.89%
- Specializes in temporal market dynamics

### XGBoost (Gradient Boosting)
- Depth: 6, Learning Rate: 0.1
- Test Accuracy: 98.89%
- Excellent feature importance analysis

## Target Variable
- **NORMAL (0)**: 116 samples - Healthy market conditions
- **WARNING (1)**: 300 samples - Declining breadth/sentiment
- **CRISIS (2)**: 126 samples - Severe market stress

## Files
- `Models/` - Trained model files (LSTM, XGBoost, Scaler)
- `Data/` - Processed feature data
- `Visualizations/` - Training history and confusion matrices

## Usage
```python
import tensorflow as tf
import joblib

# Load models
lstm_model = tf.keras.models.load_model('finra_lstm_sentiment_model.h5')
scaler = joblib.load('feature_scaler.pkl')

# Make predictions
predictions = lstm_model.predict(your_data)
```

## Results Summary
- Total Features: 22
- Training Samples: 1,608
- Test Samples: 527
- Best Model: XGBoost

---
Generated: 2025-12-15 15:45:31
