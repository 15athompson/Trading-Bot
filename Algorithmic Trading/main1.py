import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import linregress
import matplotlib.pyplot as plt

class AlgoTradingSystem:
    def __init__(self, data):
        self.data = data
        self.model = None

    def preprocess_data(self):
        # Calculate technical indicators
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['RSI'] = self.calculate_rsi(self.data['Close'], 14)
        self.data['MACD'] = self.calculate_macd(self.data['Close'])
        self.data['Volatility'] = self.data['Close'].rolling(window=20).std()
        
        # Calculate returns
        self.data['Returns'] = self.data['Close'].pct_change()
        
        # Remove NaN values
        self.data.dropna(inplace=True)

    def calculate_rsi(self, prices, period):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line

    def train_model(self):
        features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Volatility']
        X = self.data[features]
        y = self.data['Returns'].shift(-1)  # Predict next day's returns
        
        X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, random_state=42)
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Print model performance
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        print(f"Train R-squared: {train_score:.4f}")
        print(f"Test R-squared: {test_score:.4f}")

    def generate_signals(self):
        features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Volatility']
        X = self.data[features]
        self.data['Predicted_Returns'] = self.model.predict(X)
        self.data['Signal'] = np.where(self.data['Predicted_Returns'] > 0, 1, -1)

    def backtest(self):
        self.data['Strategy_Returns'] = self.data['Signal'].shift(1) * self.data['Returns']
        cumulative_returns = (1 + self.data['Strategy_Returns']).cumprod()
        sharpe_ratio = np.sqrt(252) * self.data['Strategy_Returns'].mean() / self.data['Strategy_Returns'].std()
        
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns.index, cumulative_returns)
        plt.title('Cumulative Returns of Trading Strategy')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.show()
        
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Total Return: {cumulative_returns.iloc[-1]:.4f}")

    def run(self):
        self.preprocess_data()
        self.train_model()
        self.generate_signals()
        self.backtest()

# Usage:
data = pd.read_csv('stock_data.csv', index_col='Date', parse_dates=True)
trading_system = AlgoTradingSystem(data)
trading_system.run()