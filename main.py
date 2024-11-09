import ccxt
import time
import config
import pandas as pd
import numpy as np
from ml_enhancements import EnsembleTradingModel, LSTMTradingModel, TradingEnvironment, RLTradingAgent
from strategies import MovingAverageCrossover, RSIStrategy
from adaptive_strategy import AdaptiveStrategySelector
import requests
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_socketio import SocketIO, emit
import ssl
import logging
from logging.handlers import RotatingFileHandler
import os
from dotenv import load_dotenv
import re
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json
import threading

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'your_secret_key')
socketio = SocketIO(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

auth = HTTPBasicAuth()

# Set up rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Set up logging
logging.basicConfig(level=logging.INFO)
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)

# User model for Flask-Login
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# Authentication
users = {
    "admin": generate_password_hash(os.getenv('ADMIN_PASSWORD', 'default_password'))
}

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username

def setup_ssl_context():
    ssl_certfile = os.getenv('SSL_CERTFILE')
    ssl_keyfile = os.getenv('SSL_KEYFILE')
    
    if ssl_certfile and ssl_keyfile and os.path.exists(ssl_certfile) and os.path.exists(ssl_keyfile):
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(certfile=ssl_certfile, keyfile=ssl_keyfile)
        return context
    else:
        logger.warning("SSL certificate files not found. Running without SSL.")
        return None

def fetch_data(exchange, symbol, timeframe, limit):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None

def calculate_sma(data, window):
    return data['close'].rolling(window=window).mean()

def calculate_performance_metrics(trades):
    total_profit = sum(trade['profit'] for trade in trades)
    win_rate = sum(1 for trade in trades if trade['profit'] > 0) / len(trades) if trades else 0
    return {
        'totalProfit': total_profit,
        'winRate': win_rate,
        'numberOfTrades': len(trades)
    }

def send_notification(message):
    if config.PUSH_NOTIFICATION_ENABLED:
        try:
            response = requests.post(f"{config.MOBILE_APP_API_URL}/notify", json={'message': message})
            response.raise_for_status()
            logger.info(f"Notification sent: {message}")
        except requests.RequestException as e:
            logger.error(f"Failed to send notification: {e}")

def validate_trade_action(action):
    valid_actions = ['buy', 'sell']
    return action.lower() in valid_actions

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and check_password_hash(users.get(username), password):
            user = User(username)
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/api/trade', methods=['POST'])
@login_required
@limiter.limit("10 per minute")
def trade():
    try:
        data = request.json
        if not data or 'action' not in data:
            return jsonify({'error': 'Invalid input'}), 400
        
        action = data['action']
        if not validate_trade_action(action):
            return jsonify({'error': 'Invalid trade action'}), 400
        
        # Process trade request
        # Implement your trading logic here
        
        logger.info(f"Trade request processed: {action}")
        return jsonify({'status': 'success', 'action': action}), 200
    except Exception as e:
        logger.error(f"Error processing trade request: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    logger.warning(f"Rate limit exceeded: {e}")
    return jsonify(error="Rate limit exceeded"), 429

def adjust_model_weights(performance_metrics, model_weights):
    # Simple weight adjustment based on performance
    if performance_metrics['winRate'] < 0.5:
        model_weights['ENSEMBLE_WEIGHT'] *= 0.9
        model_weights['LSTM_WEIGHT'] *= 1.1
        model_weights['RL_WEIGHT'] *= 1.1
        model_weights['ADAPTIVE_WEIGHT'] *= 1.1
    else:
        model_weights['ENSEMBLE_WEIGHT'] *= 1.1
        model_weights['LSTM_WEIGHT'] *= 0.9
        model_weights['RL_WEIGHT'] *= 0.9
        model_weights['ADAPTIVE_WEIGHT'] *= 0.9
    
    # Normalize weights
    total_weight = sum(model_weights.values())
    for key in model_weights:
        model_weights[key] /= total_weight
    
    return model_weights

def calculate_position_size(account_balance, risk_per_trade, current_price, stop_loss):
    risk_amount = account_balance * risk_per_trade
    position_size = risk_amount / (current_price - stop_loss)
    return min(position_size, account_balance * config.MAX_POSITION_SIZE)

def main():
    # Initialize the exchange
    exchange = ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_API_SECRET'),
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future'  # Use futures market
        }
    })

    symbol = config.SYMBOLS[0]  # Use the first symbol from the config
    timeframe = config.TIMEFRAME
    
    # Fetch historical data for training
    historical_data = fetch_data(exchange, symbol, timeframe, limit=1000)
    if historical_data is None:
        logger.error("Failed to fetch historical data. Exiting.")
        return
    
    # Split data for training and testing
    train_data, test_data = train_test_split(historical_data, test_size=0.2, shuffle=False)
    
    # Initialize and train models
    ensemble_model = EnsembleTradingModel()
    ensemble_model.train(train_data)
    
    lstm_model = LSTMTradingModel(input_shape=(config.LSTM_LOOKBACK, 5))
    lstm_model.train(train_data)
    
    env = TradingEnvironment(train_data)
    rl_agent = RLTradingAgent(env)
    rl_agent.train(episodes=config.RL_EPISODES)

    # Initialize adaptive strategy selector
    ma_crossover = MovingAverageCrossover(short_window=config.MA_SHORT_WINDOW, long_window=config.MA_LONG_WINDOW)
    rsi_strategy = RSIStrategy(period=config.RSI_PERIOD, overbought=config.RSI_OVERBOUGHT, oversold=config.RSI_OVERSOLD)
    adaptive_selector = AdaptiveStrategySelector([ma_crossover, rsi_strategy], lookback_period=config.LOOKBACK_PERIOD)

    trades = []
    last_notification_time = time.time()
    last_model_update_time = time.time()
    last_daily_reset_time = time.time()
    model_weights = {
        'ENSEMBLE_WEIGHT': config.ENSEMBLE_WEIGHT,
        'LSTM_WEIGHT': config.LSTM_WEIGHT,
        'RL_WEIGHT': config.RL_WEIGHT,
        'ADAPTIVE_WEIGHT': config.ADAPTIVE_WEIGHT
    }
    daily_loss = 0
    circuit_breaker_activated = False
    circuit_breaker_time = 0

    while True:
        try:
            current_time = time.time()
            
            # Reset daily loss
            if current_time - last_daily_reset_time >= 86400:  # 24 hours
                daily_loss = 0
                last_daily_reset_time = current_time
            
            # Check circuit breaker
            if circuit_breaker_activated and current_time - circuit_breaker_time >= config.CIRCUIT_BREAKER_COOLDOWN:
                circuit_breaker_activated = False
                logger.info("Circuit breaker deactivated")
            
            # Fetch latest data
            latest_data = fetch_data(exchange, symbol, timeframe, limit=config.MA_LONG_WINDOW)
            if latest_data is None:
                logger.error("Failed to fetch latest data. Skipping this iteration.")
                time.sleep(60)
                continue
            
            # Calculate moving averages
            fast_sma = calculate_sma(latest_data, config.FAST_SMA)
            slow_sma = calculate_sma(latest_data, config.SLOW_SMA)
            
            # Prepare data for predictions
            current_data = latest_data.iloc[-1][['open', 'high', 'low', 'close', 'volume']].values
            lstm_data = latest_data.iloc[-config.LSTM_LOOKBACK:][['open', 'high', 'low', 'close', 'volume']].values
            rl_data = np.append(current_data, [env.balance])
            
            # Get predictions from all models
            ensemble_prediction = ensemble_model.predict(current_data)
            lstm_prediction = lstm_model.predict(lstm_data)
            rl_action = rl_agent.predict(rl_data)
            adaptive_signal = adaptive_selector.generate_signal(latest_data)
            
            # Calculate moving average crossover signal
            ma_signal = 1 if fast_sma.iloc[-1] > slow_sma.iloc[-1] else -1
            
            # Combine predictions using current weights
            combined_signal = (
                model_weights['ENSEMBLE_WEIGHT'] * ensemble_prediction +
                model_weights['LSTM_WEIGHT'] * lstm_prediction +
                model_weights['RL_WEIGHT'] * (rl_action - 1) +
                model_weights['ADAPTIVE_WEIGHT'] * adaptive_signal +
                0.2 * ma_signal  # Add moving average signal
            )
            
            # Trading logic
            current_price = latest_data.iloc[-1]['close']
            account_balance = env.balance  # Assuming env.balance is updated correctly
            
            if not circuit_breaker_activated:
                if combined_signal > 0.5:
                    logger.info("Buy signal!")
                    # Calculate position size
                    stop_loss = current_price * (1 - config.STOP_LOSS_PERCENTAGE)
                    position_size = calculate_position_size(account_balance, config.RISK_PER_TRADE, current_price, stop_loss)
                    
                    # Implement buy logic here
                    trades.append({
                        'type': 'buy',
                        'price': current_price,
                        'timestamp': latest_data.index[-1],
                        'profit': 0,
                        'position_size': position_size,
                        'stop_loss': stop_loss
                    })
                elif combined_signal < -0.5:
                    logger.info("Sell signal!")
                    # Calculate position size
                    stop_loss = current_price * (1 + config.STOP_LOSS_PERCENTAGE)
                    position_size = calculate_position_size(account_balance, config.RISK_PER_TRADE, current_price, stop_loss)
                    
                    # Implement sell logic here
                    trades.append({
                        'type': 'sell',
                        'price': current_price,
                        'timestamp': latest_data.index[-1],
                        'profit': 0,
                        'position_size': position_size,
                        'stop_loss': stop_loss
                    })
                else:
                    logger.info("Hold position")
            else:
                logger.info("Circuit breaker active. No new trades.")
            
            # Calculate profit for the last trade
            if len(trades) >= 2:
                last_trade = trades[-1]
                previous_trade = trades[-2]
                if last_trade['type'] != previous_trade['type']:
                    profit = (last_trade['price'] - previous_trade['price']) * last_trade['position_size'] * (1 if last_trade['type'] == 'sell' else -1)
                    last_trade['profit'] = profit
                    daily_loss += min(profit, 0)
            
            # Log current state
            logger.info(f"Current price: {current_price}")
            logger.info(f"Fast SMA: {fast_sma.iloc[-1]}")
            logger.info(f"Slow SMA: {slow_sma.iloc[-1]}")
            logger.info(f"Ensemble prediction: {ensemble_prediction}")
            logger.info(f"LSTM prediction: {lstm_prediction}")
            logger.info(f"RL action: {rl_action}")
            logger.info(f"Adaptive strategy signal: {adaptive_signal}")
            logger.info(f"Moving Average signal: {ma_signal}")
            logger.info(f"Combined signal: {combined_signal}")
            
            # Implement risk management
            for trade in trades:
                if trade['type'] == 'buy' and current_price <= trade['stop_loss']:
                    logger.warning(f"Stop loss triggered for buy trade at {current_price}")
                    # Implement logic to close the position
                elif trade['type'] == 'sell' and current_price >= trade['stop_loss']:
                    logger.warning(f"Stop loss triggered for sell trade at {current_price}")
                    # Implement logic to close the position
            
            # Check for high volatility
            recent_prices = latest_data['close'].tail(24)  # Last 24 hours
            volatility = (recent_prices.max() - recent_prices.min()) / recent_prices.mean()
            if volatility > config.HIGH_VOLATILITY_THRESHOLD:
                logger.warning("Warning: High market volatility detected")
                send_notification("High market volatility detected")
            
            # Check daily loss limit
            if abs(daily_loss) > account_balance * config.DAILY_LOSS_LIMIT:
                logger.warning("Daily loss limit reached. Stopping trading for the day.")
                send_notification("Daily loss limit reached. Trading stopped for the day.")
                time.sleep(86400 - (current_time - last_daily_reset_time))  # Sleep until next day
                continue
            
            # Check circuit breaker
            if abs(current_price - trades[-1]['price']) / trades[-1]['price'] > config.CIRCUIT_BREAKER_THRESHOLD:
                circuit_breaker_activated = True
                circuit_breaker_time = current_time
                logger.warning("Circuit breaker activated due to large price movement")
                send_notification("Circuit breaker activated. Trading paused.")
            
            # Send periodic notifications
            if current_time - last_notification_time >= config.PUSH_NOTIFICATION_FREQUENCY:
                performance = calculate_performance_metrics(trades)
                notification_message = f"Performance update: Total profit: {performance['totalProfit']:.2f}, Win rate: {performance['winRate']:.2%}"
                send_notification(notification_message)
                last_notification_time = current_time
            
            # Periodically update model weights
            if current_time - last_model_update_time >= config.MODEL_UPDATE_FREQUENCY:
                performance = calculate_performance_metrics(trades)
                model_weights = adjust_model_weights(performance, model_weights)
                logger.info(f"Updated model weights: {json.dumps(model_weights)}")
                last_model_update_time = current_time
            
            # Periodically retrain models
            if current_time - last_model_update_time >= config.MODEL_RETRAIN_FREQUENCY:
                logger.info("Retraining models...")
                ensemble_model.retrain(latest_data)
                lstm_model.retrain(latest_data)
                rl_agent.retrain(latest_data, episodes=config.RL_EPISODES)
                logger.info("Models retrained.")
            
            # Emit updated data to connected clients
            socketio.emit('data_update', {
                'current_price': current_price,
                'fast_sma': fast_sma.iloc[-1],
                'slow_sma': slow_sma.iloc[-1],
                'combined_signal': combined_signal,
                'performance': calculate_performance_metrics(trades)
            })
            
            # Wait for the next iteration
            time.sleep(config.TRADING_INTERVAL)
            
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            time.sleep(60)

# Implement secure headers
@app.after_request
def add_security_headers(response):
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline'; connect-src 'self' ws: wss:;"
    return response

# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('start_bot')
def handle_start_bot():
    # Implement logic to start the bot
    return {'status': 'Bot started'}

@socketio.on('stop_bot')
def handle_stop_bot():
    # Implement logic to stop the bot
    return {'status': 'Bot stopped'}

@socketio.on('get_portfolio')
def handle_get_portfolio():
    # Implement logic to get portfolio data
    return {'portfolio': {'binance': {'BTC': 1.5, 'ETH': 10}}}

@socketio.on('get_performance')
def handle_get_performance():
    # Implement logic to get performance data
    return {'total_value': 10000, 'roi': 5.5, 'sharpe_ratio': 1.2, 'max_drawdown': 10}

@socketio.on('run_backtest')
def handle_run_backtest(data):
    # Implement backtest logic
    return {'backtest_results': 'Backtest completed successfully'}

@socketio.on('run_stress_test')
def handle_run_stress_test(data):
    # Implement stress test logic
    return {'stress_test_results': 'Stress test completed successfully'}

@socketio.on('optimize_parameters')
def handle_optimize_parameters(data):
    # Implement parameter optimization logic
    return {'optimization_results': 'Parameter optimization completed successfully'}

if __name__ == "__main__":
    # Start the main trading logic in a separate thread
    trading_thread = threading.Thread(target=main)
    trading_thread.start()

    # Run the Flask app with SocketIO
    ssl_context = setup_ssl_context()
    if ssl_context:
        socketio.run(app, ssl_context=ssl_context, host='0.0.0.0', port=5000)
    else:
        socketio.run(app, host='0.0.0.0', port=5000)


