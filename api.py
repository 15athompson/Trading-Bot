from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from main import fetch_data, exchange, symbol, timeframe
import config
import threading
import bcrypt
import ssl

app = Flask(__name__)
CORS(app)
app.config['JWT_SECRET_KEY'] = config.JWT_SECRET_KEY
jwt = JWTManager(app)

# Set up rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

bot_status = 'Idle'
bot_thread = None
users = {}  # In-memory user storage (replace with a database in production)

def run_bot():
    global bot_status
    from main import main
    main()

@app.route('/login', methods=['POST'])
@limiter.limit("10 per minute")
def login():
    username = request.json.get('username', None)
    password = request.json.get('password', None)
    if username in users and bcrypt.checkpw(password.encode('utf-8'), users[username]):
        access_token = create_access_token(identity=username)
        return jsonify(token=access_token), 200
    return jsonify({"msg": "Bad username or password"}), 401

@app.route('/register', methods=['POST'])
@limiter.limit("5 per hour")
def register():
    username = request.json.get('username', None)
    password = request.json.get('password', None)
    if username in users:
        return jsonify({"msg": "Username already exists"}), 400
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    users[username] = hashed
    return jsonify({"msg": "User created successfully"}), 201

@app.route('/status', methods=['GET'])
@jwt_required()
def get_status():
    global bot_status
    latest_data = fetch_data(exchange, config.SYMBOLS[0], config.TIMEFRAME, limit=24)
    current_price = latest_data.iloc[-1]['close']
    price_history = latest_data['close'].tolist()
    
    return jsonify({
        'status': bot_status,
        'currentPrice': current_price,
        'priceHistory': price_history,
        'lastSignal': 'Buy'  # This should be updated with actual last signal
    })

@app.route('/start', methods=['POST'])
@jwt_required()
@limiter.limit("5 per hour")
def start_bot():
    global bot_status, bot_thread
    if bot_status == 'Idle':
        bot_status = 'Running'
        bot_thread = threading.Thread(target=run_bot)
        bot_thread.start()
    return jsonify({'status': bot_status})

@app.route('/stop', methods=['POST'])
@jwt_required()
@limiter.limit("5 per hour")
def stop_bot():
    global bot_status, bot_thread
    if bot_status == 'Running':
        bot_status = 'Idle'
        # Implement a way to safely stop the bot
    return jsonify({'status': bot_status})

@app.route('/performance', methods=['GET'])
@jwt_required()
def get_performance():
    # Implement logic to calculate and return performance metrics
    return jsonify({
        'totalProfit': 1000,
        'winRate': 0.65,
        'numberOfTrades': 50
    })

@app.route('/settings', methods=['GET', 'POST'])
@jwt_required()
def handle_settings():
    if request.method == 'GET':
        # Return current settings
        return jsonify({
            'riskPerTrade': config.MAX_RISK_PER_TRADE,
            'timeframe': config.TIMEFRAME,
            'symbol': config.SYMBOLS[0]
        })
    else:
        # Update settings
        new_settings = request.json
        # Implement logic to update settings
        return jsonify(new_settings)

@app.route('/notify', methods=['POST'])
def send_notification():
    # This endpoint would be called by your bot to send notifications
    # Implement logic to send push notifications
    return jsonify({"msg": "Notification sent"}), 200

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"msg": "Rate limit exceeded"}), 429

def setup_ssl_context():
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    context.load_cert_chain('path/to/fullchain.pem', 'path/to/privkey.pem')
    return context

if __name__ == '__main__':
    ssl_context = setup_ssl_context()
    app.run(host='0.0.0.0', port=5000, ssl_context=ssl_context)

# Security Audit Checklist (to be performed regularly):
# 1. Review and update dependencies
# 2. Check for any new vulnerabilities in used libraries
# 3. Review access logs for suspicious activities
# 4. Ensure all sensitive data is properly encrypted
# 5. Review and test authentication mechanisms
# 6. Check rate limiting effectiveness
# 7. Perform penetration testing
# 8. Review and update security headers
# 9. Ensure HTTPS is properly configured
# 10. Review error handling to avoid information leakage