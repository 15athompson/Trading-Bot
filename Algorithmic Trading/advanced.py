import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import talib
from datetime import datetime, timedelta
import requests
from textblob import TextBlob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from scipy.stats import norm
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import joblib
from functools import wraps
import time
import os
import json
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
import cvxpy as cp

class AdvancedTradingSystem:
    def __init__(self, symbols, start_date, end_date, risk_free_rate=0.02):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.data = {}
        self.regime_model = None
        self.lstm_models = {}
        self.garch_models = {}
        self.alert_thresholds = {
            'drawdown': -0.1,
            'volatility': 0.03,
            'liquidity': 1000000
        }
        self.logger = self._setup_logger()
        self.performance_metrics = {}
        self.config = self._load_config()

    def _setup_logger(self):
        logger = logging.getLogger('AdvancedTradingSystem')
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler('advanced_trading_system.log')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    def _load_config(self):
        with open('config.json', 'r') as f:
            return json.load(f)

    def error_handler(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error in {func.__name__}: {str(e)}")
                self.failsafe()
                raise
        return wrapper

    def failsafe(self):
        self.logger.critical("Failsafe mechanism activated")
        # Implement logic to close all positions, cancel all orders, etc.
        for symbol in self.symbols:
            self.close_position(symbol)
        self.send_alert_email(["CRITICAL: Failsafe mechanism activated"], "emergency@example.com")

    @error_handler
    def close_position(self, symbol):
        # Placeholder for closing a position
        self.logger.info(f"Closing position for {symbol}")

    @error_handler
    def fetch_data(self):
        for symbol in self.symbols:
            self.data[symbol] = yf.download(symbol, start=self.start_date, end=self.end_date)
            self.logger.info(f"Data fetched for {symbol}")

    @error_handler
    def preprocess_data(self):
        for symbol, df in self.data.items():
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            df['RSI'] = talib.RSI(df['Close'])
            df['MACD'], _, _ = talib.MACD(df['Close'])
            df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'])
            df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = talib.BBANDS(df['Close'])
            df.dropna(inplace=True)
            self.logger.info(f"Data preprocessed for {symbol}")

    @error_handler
    def detect_market_regime(self):
        for symbol, df in self.data.items():
            # Simple regime detection based on moving averages
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            df['Market_Regime'] = np.where(df['SMA_50'] > df['SMA_200'], 1, 0)  # 1 for bullish, 0 for bearish
            self.logger.info(f"Market regime detected for {symbol}")

    @error_handler
    def train_regime_model(self):
        X = pd.DataFrame()
        y = pd.Series()
        
        for symbol, df in self.data.items():
            X = pd.concat([X, df[['Returns', 'Volatility', 'RSI', 'MACD']]], axis=1)
            y = pd.concat([y, df['Market_Regime']])
        
        tscv = TimeSeriesSplit(n_splits=5)
        self.regime_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        scores = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            self.regime_model.fit(X_train, y_train)
            y_pred = self.regime_model.predict(X_test)
            scores.append({
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            })
        
        self.logger.info(f"Regime Model Cross-Validation Scores: {scores}")
        self.logger.info(f"Regime Model Mean CV Accuracy: {np.mean([s['accuracy'] for s in scores]):.4f}")

        joblib.dump(self.regime_model, 'regime_model.joblib')

    @error_handler
    def train_lstm_models(self):
        for symbol, df in self.data.items():
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(60, 5)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')

            X = df[['Close', 'Volume', 'RSI', 'MACD', 'ATR']].values
            y = df['Close'].values

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            X_reshaped = []
            y_reshaped = []
            for i in range(60, len(X)):
                X_reshaped.append(X[i-60:i])
                y_reshaped.append(y[i])
            X_reshaped, y_reshaped = np.array(X_reshaped), np.array(y_reshaped)

            model.fit(X_reshaped, y_reshaped, epochs=50, batch_size=32, validation_split=0.2)
            self.lstm_models[symbol] = {'model': model, 'scaler': scaler}
            self.logger.info(f"LSTM model trained for {symbol}")

    @error_handler
    def train_garch_models(self):
        for symbol, df in self.data.items():
            returns = df['Log_Returns'].dropna()
            model = arch_model(returns, vol='Garch', p=1, q=1)
            results = model.fit(disp='off')
            self.garch_models[symbol] = results
            self.logger.info(f"GARCH model trained for {symbol}")

    @error_handler
    def predict_volatility(self, symbol, horizon=5):
        garch_model = self.garch_models[symbol]
        forecast = garch_model.forecast(horizon=horizon)
        return np.sqrt(forecast.variance.values[-1])

    @error_handler
    def optimize_portfolio(self):
        returns = pd.DataFrame({symbol: df['Returns'] for symbol, df in self.data.items()})
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        n = len(self.symbols)
        w = cp.Variable(n)
        gamma = cp.Parameter(nonneg=True)
        ret = mean_returns.values @ w
        risk = cp.quad_form(w, cov_matrix.values)
        
        prob = cp.Problem(cp.Maximize(ret - gamma * risk),
                          [cp.sum(w) == 1, w >= 0])

        risk_aversion = self.config['risk_aversion']
        gamma.value = risk_aversion
        prob.solve()

        if prob.status == 'optimal':
            optimal_weights = w.value
            self.logger.info(f"Optimal portfolio weights: {optimal_weights}")
            return dict(zip(self.symbols, optimal_weights))
        else:
            self.logger.error("Portfolio optimization failed")
            return None

    @error_handler
    def generate_stressed_data(self, num_scenarios=1000):
        stressed_data = {}
        for symbol, df in self.data.items():
            returns = df['Returns']
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Generate scenarios with higher volatility and fat tails
            scenarios = np.random.standard_t(df=3, size=(len(returns), num_scenarios))
            scenarios = scenarios * std_return * 2 + mean_return  # Double the volatility
            
            stressed_returns = pd.DataFrame(scenarios, index=returns.index)
            stressed_prices = df['Close'].iloc[0] * (1 + stressed_returns).cumprod()
            
            stressed_data[symbol] = stressed_prices
        
        return stressed_data

    @error_handler
    def run_strategy_on_data(self, data):
        portfolio_value = self.config['initial_capital']
        positions = {symbol: 0 for symbol in self.symbols}
        
        for date in data[self.symbols[0]].index:
            for symbol in self.symbols:
                price = data[symbol].loc[date]
                signal = self.generate_trading_signal(symbol, date)
                
                if signal > 0 and portfolio_value > price:
                    # Buy
                    shares_to_buy = int(portfolio_value * 0.1 / price)  # Invest 10% of portfolio
                    positions[symbol] += shares_to_buy
                    portfolio_value -= shares_to_buy * price
                elif signal < 0 and positions[symbol] > 0:
                    # Sell
                    portfolio_value += positions[symbol] * price
                    positions[symbol] = 0
            
            portfolio_value += sum(positions[s] * data[s].loc[date] for s in self.symbols)
        
        return (portfolio_value / self.config['initial_capital']) - 1  # Return

    @error_handler
    def generate_trading_signal(self, symbol, date):
        df = self.data[symbol]
        if date not in df.index:
            return 0
        
        # Combine multiple signals
        regime = self.regime_model.predict(df.loc[date, ['Returns', 'Volatility', 'RSI', 'MACD']].values.reshape(1, -1))[0]
        rsi = df.loc[date, 'RSI']
        macd = df.loc[date, 'MACD']
        
        signal = 0
        if regime == 1:  # Bullish regime
            if rsi < 30:  # Oversold
                signal += 1
            if macd > 0:  # MACD above signal line
                signal += 1
        else:  # Bearish regime
            if rsi > 70:  # Overbought
                signal -= 1
            if macd < 0:  # MACD below signal line
                signal -= 1
        
        return signal

    @error_handler
    def stress_test(self, num_simulations=1000):
        self.logger.info("Starting stress testing")
        results = []
        stressed_data = self.generate_stressed_data(num_simulations)
        
        for i in tqdm(range(num_simulations)):
            scenario_data = {symbol: data.iloc[:, i] for symbol, data in stressed_data.items()}
            strategy_performance = self.run_strategy_on_data(scenario_data)
            results.append(strategy_performance)
        
        self.logger.info(f"Stress Test Results: Mean Return = {np.mean(results):.4f}, VaR(95%) = {np.percentile(results, 5):.4f}")
        
        plt.figure(figsize=(10, 6))
        sns.histplot(results, kde=True)
        plt.title("Distribution of Strategy Returns under Stress Scenarios")
        plt.xlabel("Return")
        plt.ylabel("Frequency")
        plt.savefig("stress_test_results.png")
        plt.close()

    @error_handler
    def regulatory_compliance_check(self):
        self.logger.info("Performing regulatory compliance check")
        
        # Check position limits
        for symbol, df in self.data.items():
            position = df['Position'].iloc[-1] if 'Position' in df.columns else 0
            if abs(position) > self.config['max_position_size']:
                self.logger.warning(f"Position limit exceeded for {symbol}")
        
        # Check for wash sales
        for symbol, df in self.data.items():
            if 'Trade' in df.columns:
                last_sell = df[df['Trade'] == 'SELL'].index[-1] if 'SELL' in df['Trade'].values else None
                if last_sell:
                    if 'BUY' in df.loc[last_sell:, 'Trade'].values[:30]:  # Check for buy within 30 days of sell
                        self.logger.warning(f"Potential wash sale detected for {symbol}")
        
        # Check for proper trade reporting
        if not os.path.exists('trade_report.csv'):
            self.logger.error("Trade report file not found")
        
        self.logger.info("Regulatory compliance check completed")

    @error_handler
    def high_performance_execution(self):
        self.logger.info("Initiating high-performance trade execution")
        with ThreadPoolExecutor(max_workers=len(self.symbols)) as executor:
            futures = []
            for symbol in self.symbols:
                futures.append(executor.submit(self.execute_trade, symbol))
            for future in futures:
                future.result()
        self.logger.info("High-performance trade execution completed")

    def execute_trade(self, symbol):
        df = self.data[symbol]
        current_price = df['Close'].iloc[-1]
        position = df['Position'].iloc[-1] if 'Position' in df.columns else 0
        
        signal = self.generate_trading_signal(symbol, df.index[-1])
        
        if signal > 0 and position == 0:
            # Buy logic
            quantity = self.calculate_position_size(symbol)
            self.place_order(symbol, 'BUY', quantity, current_price)
        elif signal < 0 and position > 0:
            # Sell logic
            self.place_order(symbol, 'SELL', position, current_price)

    def calculate_position_size(self, symbol):
        portfolio_value = sum(df['Close'].iloc[-1] * df['Position'].iloc[-1] 
                              for df in self.data.values() if 'Position' in df.columns)
        risk_per_trade = portfolio_value * self.config['risk_per_trade']
        volatility = self.predict_volatility(symbol)
        quantity = int(risk_per_trade / (volatility * self.data[symbol]['Close'].iloc[-1]))
        return min(quantity, self.config['max_position_size'])

    def place_order(self, symbol, order_type, quantity, price):
        # In a real system, this would interact with a broker's API
        self.logger.info(f"Placing {order_type} order for {quantity} shares of {symbol} at {price}")
        # Update position in data
        if order_type == 'BUY':
            self.data[symbol].loc[self.data[symbol].index[-1], 'Position'] += quantity
        else:
            self.data[symbol].loc[self.data[symbol].index[-1], 'Position'] -= quantity
        # Log the trade
        with open('trade_report.csv', 'a') as f:
            f.write(f"{datetime.now()},{symbol},{order_type},{quantity},{price}\n")

    @error_handler
    def continuous_monitoring(self):
        while True:
            self.logger.info("Performing continuous monitoring")
            self.update_market_data()
            self.reevaluate_positions()
            self.check_risk_limits()
            alerts = self.check_alerts()
            if alerts:
                self.send_alert_email(alerts, "analyst@example.com")
            time.sleep(self.config['monitoring_interval'])

    def update_market_data(self):
        end_date = datetime.now().strftime('%Y-%m-%d')
        for symbol in self.symbols:
            new_data = yf.download(symbol, start=self.data[symbol].index[-1] + timedelta(days=1), end=end_date)
            self.data[symbol] = pd.concat([self.data[symbol], new_data])
            self.preprocess_data()  # Rerun preprocessing on new data

    def reevaluate_positions(self):
        for symbol in self.symbols:
            self.execute_trade(symbol)  # This will check for new signals and adjust positions

    def check_risk_limits(self):
        portfolio_value = sum(df['Close'].iloc[-1] * df['Position'].iloc[-1] 
                              for df in self.data.values() if 'Position' in df.columns)
        portfolio_return = (portfolio_value / self.config['initial_capital']) - 1

        if portfolio_return < self.config['max_drawdown']:
            self.logger.warning("Maximum drawdown exceeded")
            self.reduce_overall_exposure()

        for symbol, df in self.data.items():
            position_value = df['Close'].iloc[-1] * df['Position'].iloc[-1]
            if position_value / portfolio_value > self.config['max_position_ratio']:
                self.logger.warning(f"Position limit exceeded for {symbol}")
                self.reduce_position(symbol)

    def reduce_overall_exposure(self):
        for symbol in self.symbols:
            current_position = self.data[symbol]['Position'].iloc[-1]
            if current_position > 0:
                self.place_order(symbol, 'SELL', current_position // 2, self.data[symbol]['Close'].iloc[-1])

    def reduce_position(self, symbol):
        current_position = self.data[symbol]['Position'].iloc[-1]
        if current_position > 0:
            self.place_order(symbol, 'SELL', current_position // 2, self.data[symbol]['Close'].iloc[-1])

    @error_handler
    def check_alerts(self):
        alerts = []
        portfolio_value = sum(df['Close'].iloc[-1] * df['Position'].iloc[-1] 
                              for df in self.data.values() if 'Position' in df.columns)
        portfolio_return = (portfolio_value / self.config['initial_capital']) - 1

        if portfolio_return < self.alert_thresholds['drawdown']:
            alerts.append(f"ALERT: Portfolio drawdown ({portfolio_return:.2%}) exceeds threshold")

        for symbol, df in self.data.items():
            volatility = df['Volatility'].iloc[-1]
            if volatility > self.alert_thresholds['volatility']:
                alerts.append(f"ALERT: {symbol} volatility ({volatility:.2%}) exceeds threshold")

            volume = df['Volume'].iloc[-1]
            if volume < self.alert_thresholds['liquidity']:
                alerts.append(f"ALERT: {symbol} volume ({volume:,.0f}) below threshold")

        return alerts

    def send_alert_email(self, alerts, recipient_email):
        sender_email = self.config['alert_email']['sender']
        password = self.config['alert_email']['password']

        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = recipient_email
        message["Subject"] = "Trading System Alerts"

        body = "\n".join(alerts)
        message.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, recipient_email, message.as_string())

    @error_handler
    def run(self):
        self.fetch_data()
        self.preprocess_data()
        self.detect_market_regime()
        self.train_regime_model()
        self.train_lstm_models()
        self.train_garch_models()
        self.optimize_portfolio()
        self.stress_test()
        self.regulatory_compliance_check()
        self.high_performance_execution()
        
        # Start continuous monitoring in a separate thread
        import threading
        monitoring_thread = threading.Thread(target=self.continuous_monitoring)
        monitoring_thread.start()

        self.logger.info("Trading system is now running")

if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    trading_system = AdvancedTradingSystem(config['symbols'], config['start_date'], config['end_date'])
    trading_system.run()