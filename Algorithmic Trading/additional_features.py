import nltk
from textblob import TextBlob
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import yfinance as yf

class EnhancedTradingSystem(EnhancedTradingSystem):
    def __init__(self, symbols, start_date, end_date, risk_free_rate=0.02):
        super().__init__(symbols, start_date, end_date, risk_free_rate)
        nltk.download('punkt')
        self.sentiment_scores = {}
        self.topic_models = {}

    @error_handler
    def fetch_news_data(self):
        for symbol in self.symbols:
            url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={self.config['news_api_key']}"
            response = requests.get(url)
            if response.status_code == 200:
                articles = response.json()['articles']
                self.sentiment_scores[symbol] = self.analyze_sentiment(articles)
                self.topic_models[symbol] = self.perform_topic_modeling(articles)
            else:
                self.logger.error(f"Failed to fetch news for {symbol}")

    def analyze_sentiment(self, articles):
        sentiments = []
        for article in articles:
            blob = TextBlob(article['title'] + " " + article['description'])
            sentiments.append(blob.sentiment.polarity)
        return np.mean(sentiments)

    def perform_topic_modeling(self, articles):
        texts = [article['title'] + " " + article['description'] for article in articles]
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        dtm = vectorizer.fit_transform(texts)
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(dtm)
        return lda

    @error_handler
    def fetch_fundamental_data(self):
        for symbol in self.symbols:
            stock = yf.Ticker(symbol)
            info = stock.info
            self.data[symbol]['PE_Ratio'] = info.get('trailingPE', np.nan)
            self.data[symbol]['Dividend_Yield'] = info.get('dividendYield', np.nan)
            self.data[symbol]['Book_Value'] = info.get('bookValue', np.nan)
            self.data[symbol]['Debt_To_Equity'] = info.get('debtToEquity', np.nan)

    @error_handler
    def calculate_factor_exposures(self):
        for symbol in self.symbols:
            df = self.data[symbol]
            # Market factor (Beta)
            market_returns = yf.download('^GSPC', start=self.start_date, end=self.end_date)['Close'].pct_change()
            df['Market_Factor'] = df['Returns'].rolling(window=252).cov(market_returns) / market_returns.rolling(window=252).var()
            
            # Size factor (Market Cap)
            df['Size_Factor'] = np.log(df['Close'] * df['Volume'])
            
            # Value factor (Book-to-Market ratio)
            df['Value_Factor'] = df['Book_Value'] / (df['Close'] * df['Volume'])
            
            # Momentum factor (12-1 month momentum)
            df['Momentum_Factor'] = df['Close'].pct_change(periods=252) - df['Close'].pct_change(periods=20)
            
            # Volatility factor
            df['Volatility_Factor'] = df['Returns'].rolling(window=20).std()
            
            self.factor_exposures[symbol] = df[['Market_Factor', 'Size_Factor', 'Value_Factor', 'Momentum_Factor', 'Volatility_Factor']]

    @error_handler
    def run(self):
        super().run()
        self.fetch_news_data()
        self.fetch_fundamental_data()
        self.calculate_factor_exposures()

    def generate_trading_signal(self, symbol, date):
        signal = super().generate_trading_signal(symbol, date)
        
        # Incorporate sentiment
        sentiment = self.sentiment_scores.get(symbol, 0)
        signal += np.sign(sentiment)
        
        # Incorporate fundamental data
        pe_ratio = self.data[symbol].loc[date, 'PE_Ratio']
        if not np.isnan(pe_ratio):
            if pe_ratio < 15:  # Assuming PE < 15 is attractive
                signal += 1
            elif pe_ratio > 30:  # Assuming PE > 30 is unattractive
                signal -= 1
        
        # Incorporate factor exposures
        factor_exposures = self.factor_exposures[symbol].loc[date]
        if factor_exposures['Value_Factor'] > factor_exposures['Value_Factor'].rolling(window=252).mean():
            signal += 1
        if factor_exposures['Momentum_Factor'] > factor_exposures['Momentum_Factor'].rolling(window=252).mean():
            signal += 1
        
        return signal

    @error_handler
    def evaluate_strategy(self):
        returns = pd.DataFrame({symbol: df['Returns'] for symbol, df in self.data.items()})
        portfolio_weights = self.optimize_portfolio()
        portfolio_returns = (returns * portfolio_weights).sum(axis=1)
        
        total_return = (portfolio_returns + 1).prod() - 1
        sharpe_ratio = empyrical.sharpe_ratio(portfolio_returns)
        max_drawdown = empyrical.max_drawdown(portfolio_returns)
        
        self.logger.info(f"Strategy Evaluation:")
        self.logger.info(f"Total Return: {total_return:.2%}")
        self.logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        self.logger.info(f"Max Drawdown: {max_drawdown:.2%}")
        
        # Plot cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        plt.figure(figsize=(10, 6))
        cumulative_returns.plot()
        plt.title("Cumulative Portfolio Returns")
        plt.ylabel("Cumulative Returns")
        plt.savefig("cumulative_returns.png")
        plt.close()

if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    trading_system = EnhancedTradingSystem(config['symbols'], config['start_date'], config['end_date'])
    trading_system.run()
    trading_system.evaluate_strategy()