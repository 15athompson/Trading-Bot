import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import talib
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
import requests
from textblob import TextBlob
from concurrent.futures import ThreadPoolExecutor

class AdvancedMultiAssetAlgoTradingSystem:
    def __init__(self, symbols, start_date, end_date, risk_free_rate=0.02):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.data = {}
        self.models = {}
        self.scaler = StandardScaler()
        self.portfolio = pd.DataFrame(index=symbols, columns=['Weight', 'Position'])
        self.portfolio['Weight'] = 1 / len(symbols)  # Equal weight initially
        self.portfolio['Position'] = 0

    def fetch_data(self):
        def fetch_single_symbol(symbol):
            stock_data = yf.download(symbol, start=self.start_date, end=self.end_date)
            stock_data['Symbol'] = symbol
            return stock_data

        with ThreadPoolExecutor() as executor:
            self.data = {symbol: data for symbol, data in zip(self.symbols, executor.map(fetch_single_symbol, self.symbols))}

    def preprocess_data(self):
        for symbol, df in self.data.items():
            # Technical indicators
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['RSI'] = talib.RSI(df['Close'])
            df['MACD'], _, _ = talib.MACD(df['Close'])
            df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'])
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'])
            df['OBV'] = talib.OBV(df['Close'], df['Volume'])
            df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'])
            
            # Returns
            df['Returns'] = df['Close'].pct_change()
            
            # Mean reversion features
            df['Price_to_SMA50'] = df['Close'] / df['SMA_50']
            df['Momentum'] = df['Returns'].rolling(window=10).mean()
            
            # Remove NaN values
            df.dropna(inplace=True)

    def fetch_fundamental_data(self):
        for symbol in self.symbols:
            stock = yf.Ticker(symbol)
            info = stock.info
            self.data[symbol]['P/E'] = info.get('trailingPE', np.nan)
            self.data[symbol]['P/B'] = info.get('priceToBook', np.nan)
            self.data[symbol]['Dividend_Yield'] = info.get('dividendYield', np.nan)

    def fetch_sentiment_data(self, api_key):
        def fetch_news(symbol):
            url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={api_key}&language=en&sortBy=publishedAt&pageSize=100"
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()['articles']
            else:
                return []

        for symbol in self.symbols:
            news = fetch_news(symbol)
            sentiments = [TextBlob(article['title']).sentiment.polarity for article in news]
            self.data[symbol]['Sentiment'] = np.mean(sentiments) if sentiments else 0

    def detect_market_regime(self, window=252):
        for symbol, df in self.data.items():
            returns = df['Returns']
            rolling_mean = returns.rolling(window=window).mean()
            rolling_std = returns.rolling(window=window).std()
            
            df['Market_Regime'] = np.where(returns > (rolling_mean + rolling_std), 'Bull',
                                           np.where(returns < (rolling_mean - rolling_std), 'Bear', 'Neutral'))

    def train_models(self):
        features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'ATR', 'BB_upper', 'BB_lower', 'OBV', 'ADX', 
                    'P/E', 'P/B', 'Dividend_Yield', 'Sentiment', 'Price_to_SMA50', 'Momentum']

        for symbol, df in self.data.items():
            X = df[features]
            y = df['Returns'].shift(-1)  # Predict next day's returns
            
            X_scaled = self.scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled[:-1], y[:-1], test_size=0.2, random_state=42)
            
            # Random Forest
            rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
            rf = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5)
            rf.fit(X_train, y_train)
            
            # Gradient Boosting
            gb_params = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
            gb = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_params, cv=5)
            gb.fit(X_train, y_train)
            
            self.models[symbol] = {'RF': rf.best_estimator_, 'GB': gb.best_estimator_}
            
            # Print model performance
            for name, model in self.models[symbol].items():
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                print(f"{symbol} - {name} Train R-squared: {train_score:.4f}")
                print(f"{symbol} - {name} Test R-squared: {test_score:.4f}")

    def generate_signals(self):
        features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'ATR', 'BB_upper', 'BB_lower', 'OBV', 'ADX', 
                    'P/E', 'P/B', 'Dividend_Yield', 'Sentiment', 'Price_to_SMA50', 'Momentum']

        for symbol, df in self.data.items():
            X = df[features]
            X_scaled = self.scaler.transform(X)
            
            df['RF_Pred'] = self.models[symbol]['RF'].predict(X_scaled)
            df['GB_Pred'] = self.models[symbol]['GB'].predict(X_scaled)
            df['Ensemble_Pred'] = (df['RF_Pred'] + df['GB_Pred']) / 2
            
            # Combine model predictions with mean reversion and momentum signals
            df['MR_Signal'] = np.where(df['Price_to_SMA50'] < 0.95, 1, np.where(df['Price_to_SMA50'] > 1.05, -1, 0))
            df['Mom_Signal'] = np.where(df['Momentum'] > 0, 1, -1)
            
            df['Signal'] = np.sign(df['Ensemble_Pred'] + 0.5 * df['MR_Signal'] + 0.5 * df['Mom_Signal'])

    def optimize_portfolio(self):
        def portfolio_volatility(weights, returns):
            return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

        def portfolio_return(weights, returns):
            return np.sum(returns.mean() * weights) * 252

        def portfolio_sharpe(weights, returns, risk_free_rate):
            p_ret = portfolio_return(weights, returns)
            p_vol = portfolio_volatility(weights, returns)
            return (p_ret - risk_free_rate) / p_vol

        returns = pd.DataFrame({symbol: df['Returns'] for symbol, df in self.data.items()})
        n = len(self.symbols)
        args = (returns, self.risk_free_rate)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for asset in range(n))
        
        result = minimize(lambda weights, returns, rf: -portfolio_sharpe(weights, returns, rf),
                          n * [1./n], args=args, method='SLSQP', bounds=bounds, constraints=constraints)

        optimal_weights = result.x
        for symbol, weight in zip(self.symbols, optimal_weights):
            self.portfolio.loc[symbol, 'Weight'] = weight

    def apply_risk_management(self, max_portfolio_var=0.02):
        # Calculate portfolio VaR
        returns = pd.DataFrame({symbol: df['Returns'] for symbol, df in self.data.items()})
        portfolio_returns = (returns * self.portfolio['Weight']).sum(axis=1)
        portfolio_var = np.percentile(portfolio_returns, 5)  # 95% VaR
        
        # Adjust position sizes based on VaR
        if abs(portfolio_var) > max_portfolio_var:
            scaling_factor = max_portfolio_var / abs(portfolio_var)
            self.portfolio['Position'] *= scaling_factor
        
        # Apply correlation-based risk management
        corr_matrix = returns.corr()
        for symbol in self.symbols:
            highly_correlated = corr_matrix[symbol][corr_matrix[symbol] > 0.7].index.tolist()
            if len(highly_correlated) > 1:
                # Reduce position sizes for highly correlated assets
                self.portfolio.loc[highly_correlated, 'Position'] *= 0.8

    def calculate_portfolio_returns(self):
        portfolio_returns = pd.Series(0, index=self.data[self.symbols[0]].index)
        for symbol, df in self.data.items():
            portfolio_returns += df['Returns'] * self.portfolio.loc[symbol, 'Weight'] * self.portfolio.loc[symbol, 'Position']
        return portfolio_returns

    def backtest(self):
        portfolio_returns = self.calculate_portfolio_returns()
        cumulative_returns = (1 + portfolio_returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns.index, cumulative_returns)
        plt.title('Cumulative Returns of Multi-Asset Trading Strategy')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.show()
        
        print(f"Total Return: {total_return:.4f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Max Drawdown: {max_drawdown:.4f}")

    def run(self, sentiment_api_key):
        self.fetch_data()
        self.preprocess_data()
        self.fetch_fundamental_data()
        self.fetch_sentiment_data(sentiment_api_key)
        self.detect_market_regime()
        self.train_models()
        self.generate_signals()
        self.optimize_portfolio()
        self.apply_risk_management()
        self.backtest()

# Usage:
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
start_date = '2020-01-01'
end_date = '2023-12-31'
sentiment_api_key = 'your_newsapi_key_here'
trading_system = AdvancedMultiAssetAlgoTradingSystem(symbols, start_date, end_date)
trading_system.run(sentiment_api_key)