import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

class PricePredictionModel:
    def __init__(self, lookback=60):
        self.lookback = lookback
        self.model = self._build_model()
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _build_model(self):
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.lookback, 1), kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True, kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            LSTM(units=50, kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            Dense(units=1, kernel_regularizer=l2(0.01))
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def prepare_data(self, data):
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        x, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            x.append(scaled_data[i-self.lookback:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(x), np.array(y)

    def train(self, data):
        x, y = self.prepare_data(data)
        tscv = TimeSeriesSplit(n_splits=5)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        for train_index, test_index in tscv.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            
            self.model.fit(x_train, y_train, epochs=100, batch_size=32, 
                           validation_data=(x_test, y_test), callbacks=[early_stopping], verbose=0)

    def predict(self, data):
        x, _ = self.prepare_data(data)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        predicted_prices = self.model.predict(x)
        return self.scaler.inverse_transform(predicted_prices)

def create_and_train_model(data):
    model = PricePredictionModel()
    model.train(data)
    return model

def get_price_prediction(model, data):
    return model.predict(data)[-1][0]