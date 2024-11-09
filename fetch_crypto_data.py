import ccxt
import pandas as pd
from datetime import datetime, timedelta

def fetch_crypto_data(symbol, timeframe, start_date, end_date):
    # Initialize the Binance exchange
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })

    # Convert dates to timestamps
    start_timestamp = exchange.parse8601(start_date)
    end_timestamp = exchange.parse8601(end_date)

    all_ohlcv = []

    # Fetch data in chunks
    while start_timestamp < end_timestamp:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, start_timestamp, limit=1000)
        all_ohlcv.extend(ohlcv)
        
        if len(ohlcv) == 0:
            break
        
        start_timestamp = ohlcv[-1][0] + 1  # Next candle's timestamp

    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    return df

if __name__ == "__main__":
    symbol = 'BTC/USDT'
    timeframe = '1h'
    start_date = '2022-01-01T00:00:00Z'
    end_date = datetime.utcnow().isoformat() + 'Z'  # Current date

    data = fetch_crypto_data(symbol, timeframe, start_date, end_date)
    
    # Save to CSV
    data.to_csv('crypto_data.csv')
    print(f"Data saved to crypto_data.csv")
    print(f"Shape of the data: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")