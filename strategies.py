import pandas as pd
import numpy as np

class Strategy:
    def generate_signal(self, data):
        raise NotImplementedError("Subclass must implement abstract method")

class MovingAverageCrossover(Strategy):
    def __init__(self, short_window=20, long_window=50):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signal(self, data):
        if len(data) < self.long_window:
            return 0

        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0

        signals['short_mavg'] = data['close'].rolling(window=self.short_window, min_periods=1, center=False).mean()
        signals['long_mavg'] = data['close'].rolling(window=self.long_window, min_periods=1, center=False).mean()

        signals['signal'][self.short_window:] = np.where(signals['short_mavg'][self.short_window:] 
                                                         > signals['long_mavg'][self.short_window:], 1.0, 0.0)   
        signals['positions'] = signals['signal'].diff()

        return signals['positions'].iloc[-1]

class RSIStrategy(Strategy):
    def __init__(self, period=14, overbought=70, oversold=30):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def generate_signal(self, data):
        if len(data) < self.period:
            return 0

        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        if rsi.iloc[-1] > self.overbought:
            return -1  # Sell signal
        elif rsi.iloc[-1] < self.oversold:
            return 1  # Buy signal
        else:
            return 0  # No signal