from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_socketio import SocketIO, emit
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from advanced_bot import AdvancedTradingBot
from advanced_backtester import AdvancedBacktester
from parameter_optimizer import ParameterOptimizer, GridSearch
from strategies import MovingAverageCrossover, RSIStrategy
from paper_trading import PaperTrading
import threading
import time
import config
from auth import init_auth, authenticate_user, create_user

app = Flask(__name__)
app.config['SECRET_KEY'] = config.FLASK_SECRET_KEY
socketio = SocketIO(app)
bot = AdvancedTradingBot()
backtester = AdvancedBacktester(bot.exchanges[config.EXCHANGES[0]], config.SYMBOLS, config.TIMEFRAME)
paper_trader = PaperTrading(config.PAPER_TRADING_INITIAL_BALANCE, config.PAPER_TRADING_FEE_RATE)
bot_thread = None
update_thread = None

init_auth(app)

def create_initial_user(username, password):
    create_user(username, password)
    print(f"Initial user '{username}' created successfully.")

# Create the initial user
create_initial_user('admin', 'secure_password')

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = authenticate_user(username, password)
        if user:
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@socketio.on('start_bot')
@login_required
def start_bot(data):
    global bot_thread, update_thread
    if bot_thread is None or not bot_thread.is_alive():
        use_paper_trading = data.get('use_paper_trading', False)
        bot_thread = threading.Thread(target=bot.run, args=(use_paper_trading,))
        bot_thread.start()
        update_thread = threading.Thread(target=send_updates)
        update_thread.start()
        return {"status": f"Bot started successfully in {'paper trading' if use_paper_trading else 'live'} mode"}
    else:
        return {"status": "Bot is already running"}

@socketio.on('stop_bot')
@login_required
def stop_bot():
    global bot_thread, update_thread
    if bot_thread and bot_thread.is_alive():
        bot.stop_signal = True
        bot_thread.join()
        bot_thread = None
        if update_thread:
            update_thread.join()
            update_thread = None
        return {"status": "Bot stopped successfully"}
    else:
        return {"status": "Bot is not running"}

def send_updates():
    while bot_thread and bot_thread.is_alive():
        portfolios = {exchange: pm.get_portfolio() for exchange, pm in bot.portfolio_managers.items()}
        performance_metrics = get_performance_metrics()
        socketio.emit('data_update', {'portfolios': portfolios, 'performance': performance_metrics})
        time.sleep(5)  # Send updates every 5 seconds

@socketio.on('get_portfolio')
@login_required
def get_portfolio():
    if bot.use_paper_trading:
        return {'paper_trading': paper_trader.get_positions()}
    else:
        return {exchange: pm.get_portfolio() for exchange, pm in bot.portfolio_managers.items()}

@socketio.on('get_performance')
@login_required
def get_performance():
    return get_performance_metrics()

def get_performance_metrics():
    if bot.use_paper_trading:
        total_value = paper_trader.calculate_portfolio_value(bot.get_current_prices())
        initial_value = config.PAPER_TRADING_INITIAL_BALANCE
    else:
        total_value = sum(pm.get_total_value() for pm in bot.portfolio_managers.values())
        initial_value = bot.initial_portfolio_value
    
    roi = (total_value - initial_value) / initial_value * 100
    
    # Calculate other metrics like Sharpe ratio, max drawdown, etc.
    # This is a simplified example
    metrics = {
        'total_value': total_value,
        'roi': roi,
        'sharpe_ratio': 1.5,  # placeholder
        'max_drawdown': 10,  # placeholder
    }
    return metrics

@socketio.on('run_backtest')
@login_required
def run_backtest(data):
    start_date = data['start_date']
    end_date = data['end_date']
    results = backtester.run(start_date, end_date)
    return results

@socketio.on('run_stress_test')
@login_required
def run_stress_test(data):
    start_date = data['start_date']
    end_date = data['end_date']
    results = backtester.stress_test(start_date, end_date)
    return results

@socketio.on('optimize_parameters')
@login_required
def optimize_parameters(data):
    strategy_name = data['strategy']
    if strategy_name == 'MovingAverageCrossover':
        optimizer = ParameterOptimizer(bot.exchanges[config.EXCHANGES[0]], config.SYMBOLS, config.TIMEFRAME, MovingAverageCrossover)
        bounds = [(10, 50), (20, 200)]  # bounds for short_window and long_window
        best_params, best_return = optimizer.optimize(bounds)
        return {'best_params': best_params.tolist(), 'best_return': best_return}
    elif strategy_name == 'RSIStrategy':
        grid_search = GridSearch(bot.exchanges[config.EXCHANGES[0]], config.SYMBOLS, config.TIMEFRAME, RSIStrategy)
        param_grid = {
            'period': [7, 14, 21],
            'overbought': [70, 75, 80],
            'oversold': [20, 25, 30]
        }
        best_params, best_return = grid_search.search(param_grid)
        return {'best_params': best_params, 'best_return': best_return}
    else:
        return {'error': 'Invalid strategy name'}

@socketio.on('get_paper_trading_history')
@login_required
def get_paper_trading_history():
    return paper_trader.get_trade_history()

@socketio.on('connect')
def handle_connect():
    if not current_user.is_authenticated:
        return False  # reject the connection
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True)