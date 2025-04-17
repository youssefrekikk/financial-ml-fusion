#unfinished/ still expperimenting to try and run move to src
import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv('data/ticker.csv', parse_dates=['Date'])
data = data.rename(columns={'Unnamed: 0': 'Date'})  # Ensure correct column name

# Set index
data.set_index('Date', inplace=True)

# Feature Engineering
def add_technical_indicators(df):
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean()))
    df['BB_upper'] = df['Close'].rolling(20).mean() + 2 * df['Close'].rolling(20).std()
    df['BB_lower'] = df['Close'].rolling(20).mean() - 2 * df['Close'].rolling(20).std()
    return df

data = add_technical_indicators(data)

# Add time embeddings
data['day_of_week'] = data.index.dayofweek
data['month'] = data.index.month

# Normalize data
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data.drop(columns=['Ticker']))
data_scaled = pd.DataFrame(scaled_features, columns=data.columns[1:], index=data.index)
data_scaled['Ticker'] = data['Ticker']

# Create TimeSeries Dataset
max_prediction_length = 10
max_encoder_length = 60

ts_dataset = TimeSeriesDataSet(
    data_scaled,
    time_idx='Date',
    target='Close',
    group_ids=['Ticker'],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=['day_of_week', 'month'],
    time_varying_unknown_reals=['Close', 'MA_10', 'RSI', 'BB_upper', 'BB_lower'],
    target_normalizer=GroupNormalizer(groups=['Ticker']),
)

# DataLoader
train_dataloader = ts_dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
val_dataloader = ts_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

# Model Definition
trainer = pl.Trainer(max_epochs=50, gpus=1 if torch.cuda.is_available() else 0)
model = TemporalFusionTransformer.from_dataset(ts_dataset, loss=QuantileLoss())

# Train model
trainer.fit(model, train_dataloader, val_dataloader)

# Predict & Generate Trading Signals
def generate_trading_signals(predictions):
    signals = []
    for pred in predictions:
        if pred > 1.01:  # Arbitrary thresholds for Buy/Sell/Hold
            signals.append(('BUY', pred))
        elif pred < 0.99:
            signals.append(('SELL', pred))
        else:
            signals.append(('HOLD', pred))
    return signals

preds = model.predict(val_dataloader)
signals = generate_trading_signals(preds)
print(signals)
