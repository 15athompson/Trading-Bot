import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
import gym
from gym import spaces

class EnsembleTradingModel:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
    def prepare_data(self, data):
        # Assume data is a pandas DataFrame with columns: 'open', 'high', 'low', 'close', 'volume'
        data['returns'] = data['close'].pct_change()
        data['target'] = (data['returns'].shift(-1) > 0).astype(int)
        
        features = ['open', 'high', 'low', 'close', 'volume', 'returns']
        X = data[features].values[:-1]
        y = data['target'].values[:-1]
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
        
    def train(self, data):
        X_train, X_test, y_train, y_test = self.prepare_data(data)
        
        self.rf_model.fit(X_train, y_train)
        self.gb_model.fit(X_train, y_train)
        
        rf_pred = self.rf_model.predict(X_test)
        gb_pred = self.gb_model.predict(X_test)
        
        ensemble_pred = (rf_pred + gb_pred) / 2
        ensemble_pred = (ensemble_pred > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, ensemble_pred)
        print(f"Ensemble model accuracy: {accuracy}")
        
    def predict(self, current_data):
        rf_pred = self.rf_model.predict_proba(current_data.reshape(1, -1))[0][1]
        gb_pred = self.gb_model.predict_proba(current_data.reshape(1, -1))[0][1]
        ensemble_pred = (rf_pred + gb_pred) / 2
        return 1 if ensemble_pred > 0.5 else 0

    def retrain(self, new_data):
        X_train, X_test, y_train, y_test = self.prepare_data(new_data)
        
        self.rf_model.fit(X_train, y_train)
        self.gb_model.fit(X_train, y_train)
        
        rf_pred = self.rf_model.predict(X_test)
        gb_pred = self.gb_model.predict(X_test)
        
        ensemble_pred = (rf_pred + gb_pred) / 2
        ensemble_pred = (ensemble_pred > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, ensemble_pred)
        print(f"Ensemble model retrained. New accuracy: {accuracy}")

class LSTMTradingModel:
    def __init__(self, input_shape):
        self.model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            LSTM(32, return_sequences=False),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
    def prepare_data(self, data, sequence_length=10):
        # Assume data is a pandas DataFrame with columns: 'open', 'high', 'low', 'close', 'volume'
        features = ['open', 'high', 'low', 'close', 'volume']
        target = (data['close'].pct_change().shift(-1) > 0).astype(int)
        
        X = []
        y = []
        for i in range(len(data) - sequence_length):
            X.append(data[features].values[i:i+sequence_length])
            y.append(target.values[i+sequence_length])
        
        return np.array(X), np.array(y)
        
    def train(self, data, epochs=50, batch_size=32):
        X, y = self.prepare_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
        
        _, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"LSTM model accuracy: {accuracy}")
        
    def predict(self, current_data):
        return self.model.predict(current_data.reshape(1, *current_data.shape))[0][0]

    def retrain(self, new_data, epochs=50, batch_size=32):
        X, y = self.prepare_data(new_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
        
        _, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"LSTM model retrained. New accuracy: {accuracy}")

class TradingEnvironment(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(TradingEnvironment, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.reset()
        
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32)
        
    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        return self._next_observation()
        
    def _next_observation(self):
        obs = self.data.iloc[self.current_step]
        return np.array([
            obs['open'], obs['high'], obs['low'], obs['close'], obs['volume'],
            self.balance
        ])
        
    def step(self, action):
        current_price = self.data.iloc[self.current_step]['close']
        
        if action == 1:  # Buy
            shares_to_buy = self.balance // current_price
            self.position += shares_to_buy
            self.balance -= shares_to_buy * current_price
        elif action == 2:  # Sell
            self.balance += self.position * current_price
            self.position = 0
        
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        next_price = self.data.iloc[self.current_step]['close']
        reward = ((next_price - current_price) / current_price) * self.position
        
        obs = self._next_observation()
        return obs, reward, done, {}

class RLTradingAgent:
    def __init__(self, env, learning_rate=0.001, gamma=0.99):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential([
            Dense(24, input_shape=self.env.observation_space.shape, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.env.action_space.n, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model
        
    def train(self, episodes=1000, batch_size=32, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                if np.random.rand() <= epsilon:
                    action = np.random.randint(0, self.env.action_space.n)
                else:
                    action = np.argmax(self.model.predict(state.reshape(1, -1))[0])
                
                next_state, reward, done, _ = self.env.step(action)
                
                target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1))[0])
                target_f = self.model.predict(state.reshape(1, -1))
                target_f[0][action] = target
                
                self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
                
                state = next_state
                
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            
            if episode % 100 == 0:
                print(f"Episode: {episode}, Epsilon: {epsilon:.2f}")
        
    def predict(self, state):
        return np.argmax(self.model.predict(state.reshape(1, -1))[0])

    def retrain(self, new_data, episodes=1000, batch_size=32, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env.data = new_data
        self.train(episodes, batch_size, epsilon, epsilon_decay, epsilon_min)
        print("RL agent retrained with new data.")

# Usage example:
# data = pd.read_csv('crypto_data.csv')
# ensemble_model = EnsembleTradingModel()
# ensemble_model.train(data)

# lstm_model = LSTMTradingModel(input_shape=(10, 5))
# lstm_model.train(data)

# env = TradingEnvironment(data)
# rl_agent = RLTradingAgent(env)
# rl_agent.train(episodes=1000)