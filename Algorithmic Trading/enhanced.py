import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
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
from pyfolio import timeseries
import empyrical
from pypfopt import EfficientFrontier, risk_models, expected_returns
from sklearn.cluster import KMeans
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import backtrader as bt
import alpaca_trade_api as tradeapi

class EnhancedTradingSystem:
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
        self.alpaca = tradeapi.REST(self.config['alpaca_api_key'], self.config['alpaca_secret_key'], base_url=self.config['alpaca_base_url'])
        self.sentiment_scores = {}
        self.factor_exposures = {}

    def _setup_logger(self):
        logger = logging.getLogger('EnhancedTradingSystem')
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler('enhanced_trading_system.log')
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
        for symbol in self.symbols:
            self.close_position(symbol)
        self.send_alert_email(["CRITICAL: Failsafe mechanism activated"], self.config['emergency_email'])

    @error_handler
    def close_position(self, symbol):
        try:
            position = self.alpaca.get_position(symbol)
            if int(position.qty) > 0:
                self.alpaca.submit_order(
                    symbol=symbol,
                    qty=position.qty,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
            self.logger.info(f"Closed position for {symbol}")
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {str(e)}")

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
            df['OBV'] = talib.OBV(df['Close'], df['Volume'])
            df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'])
            df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'])
            df.dropna(inplace=True)
            self.logger.info(f"Data preprocessed for {symbol}")

    @error_handler
    def detect_market_regime(self):
        for symbol, df in self.data.items():
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            df['Market_Regime'] = np.where(df['SMA_50'] > df['SMA_200'], 1, 0)  # 1 for bullish, 0 for bearish
            
            # Add Bollinger Band squeeze detection
            df['BB_Squeeze'] = (df['Upper_BB'] - df['Lower_BB']) / df['Middle_BB']
            df['BB_Squeeze_Percentile'] = df['BB_Squeeze'].rolling(window=252).rank(pct=True)
            
            # Add trend strength indicator
            df['ADX_Trend'] = np.where(df['ADX'] > 25, 1, 0)
            
            self.logger.info(f"Market regime and additional indicators detected for {symbol}")

    @error_handler
    def train_regime_model(self):
        X = pd.DataFrame()
        y = pd.Series()
        
        for symbol, df in self.data.items():
            X = pd.concat([X, df[['Returns', 'Volatility', 'RSI', 'MACD', 'ADX', 'MFI', 'BB_Squeeze_Percentile']]], axis=1)
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
                LSTM(64, return_sequences=True, input_shape=(60, 7)),
                Dropout(0.2),
                LSTM(64, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')

            X = df[['Close', 'Volume', 'RSI', 'MACD', 'ATR', 'OBV', 'ADX']].values
            y = df['Close'].values

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            X_reshaped = []
            y_reshaped = []
            for i in range(60, len(X)):
                X_reshaped.append(X[i-60:i])
                y_reshaped.append(y[i])
            X_reshaped, y_reshaped = np.array(X_reshaped), np.array(y_reshaped)

            model.fit(X_reshaped, y_reshaped, epochs=50, batch_size=32, validation_split=0.2, callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ])
            self.lstm_models[symbol] = {'model': model, 'scaler': scaler}
            self.logger.info(f"LSTM model trained for {symbol}")

    @error_handler
    def train_garch_models(self):
        for symbol, df in self.data.items():
            returns = df['Log_Returns'].dropna()
            model = arch_model(returns, vol='Garch', p=1, q=1, dist='t')
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
        mu = expected_returns.mean_historical_return(returns)
        S = risk_models.CovarianceShrinkage(returns).ledoit_wolf()
        
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        
        performance = ef.portfolio_performance(verbose=True)
        self.logger.info(f"Portfolio optimization results: {performance}")
        
        return cleaned_weights

    @error_handler
    def generate_stressed_data(self, num_scenarios=1000):
        stressed_data = {}
        for symbol, df in self.data.items():
            returns = df['Returns']
            mean_return = returns.mean()
            std_return = returns.std()
            
            scenarios = np.random.standard_t(df=3, size=(len(returns), num_scenarios))
            scenarios = scenarios * std_return * 2 + mean_return
            
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
                    shares_to_buy = int(portfolio_value * 0.1 / price)
                    positions[symbol] += shares_to_buy
                    portfolio_value -= shares_to_buy * price
                elif signal < 0 and positions[symbol] > 0:
                    portfolio_value += positions[symbol] * price
                    positions[symbol] = 0
            
            portfolio_value += sum(positions[s] * data[s].loc[date] for s in self.symbols)
        
        return (portfolio_value / self.config['initial_capital']) - 1

    @error_handler
    def generate_trading_signal(self, symbol, date):
        df = self.data[symbol]
        if date not in df.index:
            return 0
        
        regime = self.regime_model.predict(df.loc[date, ['Returns', 'Volatility', 'RSI', 'MACD', 'ADX', 'MFI', 'BB_Squeeze_Percentile']].values.reshape(1, -1))[0]
        rsi = df.loc[date, 'RSI']
        macd = df.loc[date, 'MACD']
        adx = df.loc[date, 'ADX']
        mfi = df.loc[date, 'MFI']
        bb_squeeze = df.loc[date, 'BB_Squeeze_Percentile']
        
        signal = 0
        if regime == 1:  # Bullish regime
            if rsi < 30 and mfi < 20:
                signal += 1
            if macd > 0 and adx > 25:
                signal += 1
            if bb_squeeze < 0.2:  # Potential breakout
                signal += 1
        else:  # Bearish regime
            if rsi > 70 and mfi > 80:
                signal -= 1
            if macd < 0 and adx > 25:
                signal -= 1
            if bb_squeeze < 0.2:  # Potential breakdown
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
        
        var_95 = np.percentile(results, 5)
        cvar_95 = np.mean([r for r in results if r <= var_95])
        
        self.logger.info(f"Stress Test Results: Mean Return = {np.mean(results):.4f}, VaR(95%) = {var_95:.4f}, CVaR(95%) = {cvar_95:.4f}")
        
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
        
        # Check for insider trading patterns
        self.check_insider_trading()
        
        # Check for market manipulation patterns
        self.check_market_manipulation()
        
        self.logger.info("Regulatory compliance check completed")

    @error_handler
    def check_insider_trading(self):
        # This is a simplified check. In a real system, you would need access to insider trading data.
        for symbol, df in self.data.items():
            if 'Trade' in df.columns and 'Volume' in df.columns:
                unusual_volume = df[df['Volume'] > df['Volume'].rolling(window=20).mean() + 2*df['Volume'].rolling(window=20).std()]
                if not unusual_volume.empty:
                    for date in unusual_volume.index:
                        if 'BUY' in df.loc[date:date+timedelta(days=5), 'Trade'].values:
                            self.logger.warning(f"Potential insider trading pattern detected for {symbol} around {date}")

    @error_handler
    def check_market_manipulation(self):
        for symbol, df in self.data.items():
            if 'Trade' in df.columns and 'Close' in df.columns:
                price_changes = df['Close'].pct_change()
                large_changes = price_changes[abs(price_changes) > 0.1]  # 10% price change
                if not large_changes.empty:
                    for date in large_changes.index:
                        if 'SELL' in df.loc[date-timedelta(days=1):date, 'Trade'].values and 'BUY' in df.loc[date:date+timedelta(days=1), 'Trade'].values:
                            self.logger.warning(f"Potential pump and dump pattern detected for {symbol} around {date}")

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
            quantity = self.calculate_position_size(symbol)
            self.place_order(symbol, 'BUY', quantity, current_price)
        elif signal < 0 and position > 0:
            self.place_order(symbol, 'SELL', position, current_price)

    def calculate_position_size(self, symbol):
        portfolio_value = sum(df['Close'].iloc[-1] * df['Position'].iloc[-1] 
                              for df in self.data.values() if 'Position' in df.columns)
        risk_per_trade = portfolio_value * self.config['risk_per_trade']
        volatility = self.predict_volatility(symbol)
        quantity = int(risk_per_trade / (volatility * self.data[symbol]['Close'].iloc[-1]))
        return min(quantity, self.config['max_position_size'])

    def place_order(self, symbol, order_type, quantity, price):
        try:
            if order_type == 'BUY':
                self.alpaca.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='buy',
                    type='limit',
                    time_in_force='day',
                    limit_price=price
                )
            else:
                self.alpaca.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='sell',
                    type='limit',
                    time_in_force='day',
                    limit_price=price
                )
            self.logger.info(f"Placed {order_type} order for {quantity} shares of {symbol} at {price}")
            
            # Update position in data
            if order_type == 'BUY':
                self.data[symbol].loc[self.data[symbol].index[-1], 'Position'] += quantity
            else:
                self.data[symbol].loc[self.data[symbol].index[-1], 'Position'] -= quantity
            
            # Log the trade
            with open('trade_report.csv', 'a') as f:
                f.write(f"{datetime.now()},{symbol},{order_type},{quantity},{price}\n")
        except Exception as e:
            self.logger.error(f"Error placing order for {symbol}: {str(e)}")

    @error_handler
    def continuous_monitoring(self):
        while True:
            self.logger.info("Performing continuous monitoring")
            self.update_market_data()
            self.reevaluate_positions()
            self.check_risk_limits()
            self.update_performance_metrics()
            alerts = self.check_alerts()
            if alerts:
                self.send_alert_email(alerts, self.config['analyst_email'])
            time.sleep(self.config['monitoring_interval'])

    def update_market_data(self):
        end_date = datetime.now().strftime('%Y-%m-%d')
        for symbol in self.symbols:
            new_data = yf.download(symbol, start=self.data[symbol].index[-1] + timedelta(days=1), end=end_date)
            self.data[symbol] = pd.concat([self.data[symbol], new_data])
            self.preprocess_data()

    def reevaluate_positions(self):
        for symbol in self.symbols:
            self.execute_trade(symbol)

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

        # Check VaR limit
        var_95 = self.calculate_var()
        if var_95 > self.config['max_var']:
            self.logger.warning(f"VaR limit exceeded: {var_95:.4f}")
            self.reduce_overall_exposure()

    def reduce_overall_exposure(self):
        for symbol in self.symbols:
            current_position = self.data[symbol]['Position'].iloc[-1]
            if current_position > 0:
                self.place_order(symbol, 'SELL', current_position // 2, self.data[symbol]['Close'].iloc[-1])

    def reduce_position(self, symbol):
        current_position = self.data[symbol]['Position'].iloc[-1]
        if current_position > 0:
            self.place_order(symbol, 'SELL', current_position // 2, self.data[symbol]['Close'].iloc[-1])

    def calculate_var(self, confidence_level=0.95):
        portfolio_returns = sum(df['Returns'] * df['Position'].iloc[-1] 
                                for df in self.data.values() if 'Position' in df.columns)
        var = np.percentile(portfolio_returns, 100 * (1 - confidence_level))
        return -var

    def update_performance_metrics(self):
        portfolio_values = sum(df['Close'] * df['Position'].iloc[-1] 
                               for df in self.data.values() if 'Position' in df.columns)
        returns = portfolio_values.pct_change()
        
        self.performance_metrics['total_return'] = (portfolio_values.iloc[-1] / self.config['initial_capital']) - 1
        self.performance_metrics['sharpe_ratio'] = empyrical.sharpe_ratio(returns, risk_free=self.risk_free_rate)
        self.performance_metrics['sortino_ratio'] = empyrical.sortino_ratio(returns, required_return=0)
        self.performance_metrics['max_drawdown'] = empyrical.max_drawdown(returns)
        self.performance_metrics['calmar_ratio'] = empyrical.calmar_ratio(returns)
        
        self.logger.info(f"Updated performance metrics: {self.performance_metrics}")

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

        var_95 = self.calculate_var()
        if var_95 > self.config['max_var']:
            alerts.append(f"ALERT: VaR ({var_95:.4f}) exceeds maximum threshold")

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
    def run_backtesting(self):
        cerebro = bt.Cerebro()
        
        for symbol, df in self.data.items():
            data = bt.feeds.PandasData(dataname=df)
            cerebro.adddata(data)
        
        cerebro.addstrategy(self.BacktestStrategy)
        cerebro.broker.setcash(self.config['initial_capital'])
        cerebro.broker.setcommission(commission=0.001)
        
        print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
        cerebro.run()
        print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
        
        cerebro.plot()

    class BacktestStrategy(bt.Strategy):
        def __init__(self):
            self.order = None
            self.buyprice = None
            self.buycomm = None
            
            self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=20)
        
        def next(self):
            if not self.position:
                if self.data.close[0] > self.sma[0]:
                    self.order = self.buy()
            else:
                if self.data.close[0] < self.sma[0]:
                    self.order = self.sell()

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
        self.run_backtesting()
        self.high_performance_execution()
        
        # Start continuous monitoring in a separate thread
        import threading
        monitoring_thread = threading.Thread(target=self.continuous_monitoring)
        monitoring_thread.start()

        self.logger.info("Trading system is now running")

if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    trading_system = EnhancedTradingSystem(config['symbols'], config['start_date'], config['end_date'])
    trading_system.run()