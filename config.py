import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Exchange API credentials
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY')
KRAKEN_API_SECRET = os.getenv('KRAKEN_API_SECRET')

COINBASE_API_KEY = os.getenv('COINBASE_API_KEY')
COINBASE_API_SECRET = os.getenv('COINBASE_API_SECRET')
COINBASE_API_PASSPHRASE = os.getenv('COINBASE_API_PASSPHRASE')

# Supported exchanges
EXCHANGES = ['binance', 'kraken', 'coinbasepro']

# Supported cryptocurrency pairs
SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'XRP/USDT', 'DOT/USDT',
    'BNB/USDT', 'LINK/USDT', 'LTC/USDT', 'BCH/USDT', 'DOGE/USDT'
]

# Trading parameters
TIMEFRAME = '1h'
FAST_SMA = 20
SLOW_SMA = 50

# Risk management
MAX_RISK_PER_TRADE = 0.02
LARGE_ORDER_THRESHOLD = 1000  # In USDT

# Performance tracking
HIGH_VOLATILITY_THRESHOLD = 0.05  # 5% volatility

# Notification settings
NOTIFICATION_EMAIL = os.getenv('NOTIFICATION_EMAIL')
NOTIFICATION_EMAIL_PASSWORD = os.getenv('NOTIFICATION_EMAIL_PASSWORD')
ALERT_EMAIL = os.getenv('ALERT_EMAIL')

# Custom strategies
CUSTOM_STRATEGIES = {}  # Add your custom strategies here

# Backtesting
BACKTEST_START_DATE = '2022-01-01'
BACKTEST_END_DATE = '2023-01-01'

# Web interface
FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY')

# Paper trading
PAPER_TRADING_INITIAL_BALANCE = 10000  # Initial paper trading balance in USDT
PAPER_TRADING_FEE_RATE = 0.001  # 0.1% fee rate for paper trading
PAPER_TRADING_ENABLED = True  # Set to False to disable paper trading

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///crypto_trading_bot.db')

# Adaptive strategy parameters
LOOKBACK_PERIOD = 30
MA_SHORT_WINDOW = 20
MA_LONG_WINDOW = 50
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Model parameters
ENSEMBLE_MODELS = ['RandomForest', 'GradientBoosting', 'XGBoost']
LSTM_LOOKBACK = 10
RL_EPISODES = 1000

# Signal combination weights
ENSEMBLE_WEIGHT = 0.3
LSTM_WEIGHT = 0.2
RL_WEIGHT = 0.2
ADAPTIVE_WEIGHT = 0.3

# Additional risk management parameters
POSITION_SIZE = 0.1  # 10% of available balance
STOP_LOSS_PERCENTAGE = 0.05  # 5% stop loss
TAKE_PROFIT_PERCENTAGE = 0.1  # 10% take profit

# Mobile app configuration
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key')  # Change this in production!
JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1 hour

# Performance Metrics
PERFORMANCE_WINDOW = 30  # Number of days to calculate performance metrics

# Push Notification Configuration
PUSH_NOTIFICATION_ENABLED = True
PUSH_NOTIFICATION_FREQUENCY = 3600  # Send notifications every hour (in seconds)

# Mobile App API URL
MOBILE_APP_API_URL = os.getenv('MOBILE_APP_API_URL', 'http://localhost:5000')

# New configuration parameters
STOP_LOSS_PERCENTAGE = 0.02  # 2% stop loss
RISK_PER_TRADE = 0.01  # 1% risk per trade
MODEL_UPDATE_FREQUENCY = 86400  # 24 hours in seconds
MODEL_RETRAIN_FREQUENCY = 604800  # 7 days in seconds
TRADING_INTERVAL = 3600  # 1 hour in seconds

# Safeguards
MAX_POSITION_SIZE = 0.1  # Maximum 10% of account balance per position
DAILY_LOSS_LIMIT = 0.05  # 5% daily loss limit
CIRCUIT_BREAKER_THRESHOLD = 0.1  # 10% price movement triggers circuit breaker
CIRCUIT_BREAKER_COOLDOWN = 3600  # 1 hour cooldown after circuit breaker is triggered

