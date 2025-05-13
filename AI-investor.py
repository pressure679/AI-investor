import numpy as np
import pandas as pd
import csv
import os
import random
from datetime import datetime
from io import StringIO
import time

# Constants
DATA_DIR = "/mnt/chromeos/removable/SD Card/Linux-shared-files/crypto and currency pairs"
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
INITIAL_BALANCE = 100.0
TP_PERCENT = 0.013
SL_PERCENT = 0.0085
LEVERAGE_BINANCE = 10
LEVERAGE_DAT = 500
ACTIONS = ["HOLD", "BUY", "SELL"]
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
SEQ_LENGTH = 10
NUM_FEATURES = 3
NUM_ACTIONS = len(ACTIONS)

# W = np.random.randn(NUM_FEATURES, NUM_ACTIONS)
if os.path.exists("weights.npy"):
    W = np.load("weights.npy")
    print("Loaded existing weights from weights.npy")
else:
    W = np.random.randn(NUM_FEATURES, NUM_ACTIONS)


# --- Helper Functions ---

def tail_csv_lines(filename, max_bytes=MAX_FILE_SIZE):
    """Read last few MB of a file (tail-like behavior)"""
    with open(filename, 'rb') as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        seek_start = max(0, size - max_bytes)
        f.seek(seek_start, os.SEEK_SET)
        data = f.read().decode(errors='ignore')
    lines = data.strip().split("\n")
    if "Open time" in lines[0]:  # header inside chunk
        return lines
    else:
        with open(filename, 'r') as f:
            header = f.readline().strip()
        return [header] + lines

def compute_rsi(series, period=14):
    delta = series.diff().dropna()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_state_vector(data, i):
    if i < 20:
        return np.zeros(NUM_FEATURES)
    close = data['close'].iloc[i]
    ma5 = data['close'].iloc[i-5:i].mean()
    ma20 = data['close'].iloc[i-20:i].mean()
    rsi_series = compute_rsi(data['close'].iloc[i-20:i+1])
    rsi = rsi_series.iloc[-1] / 100.0
    ma_diff = (ma5 - ma20) / ma20
    price_ratio = close / ma5
    return np.array([rsi, ma_diff, price_ratio], dtype=np.float32)

def choose_action(state_vec):
    if random.random() < EPSILON:
        return random.randint(0, NUM_ACTIONS - 1)
    q_values = state_vec @ W
    return int(np.argmax(q_values))

def update_weights(state_vec, action, reward, next_state_vec):
    global W
    current_q = state_vec @ W[:, action]
    next_q = np.max(next_state_vec @ W)
    target = reward + GAMMA * next_q
    error = target - current_q
    W[:, action] += ALPHA * error * state_vec

# --- Training Function ---

def train_on_file(filename, W, seq_length=10, leverage=10.0):
    lines = tail_csv_lines(filename, MAX_FILE_SIZE)
    csv_chunk = "\n".join(lines)

    try:
        df = pd.read_csv(StringIO(csv_chunk))
    except Exception as e:
        print(f"Error reading CSV chunk from {filename}: {e}")
        return 0.0, W

    if 'Open time' in df.columns:
        df.rename(columns={'Open time': 'timestamp', 'Close': 'close'}, inplace=True)
    elif 'timestamp' not in df.columns or 'close' not in df.columns:
        print(f"Missing required columns in {filename}")
        return 0.0, W

    df = df[['timestamp', 'close']].dropna()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['close'])
    # try:
    #     df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    # except:
    #     try:
    #         df['timestamp'] = pd.to_datetime(df['timestamp'])
    #     except:
    #         print(f"Could not parse timestamps in {filename}")
    #         return 0.0, W
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    except:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
        except Exception as e:
            print(f"Could not parse timestamps in {filename}: {e}")
            return 0.0, W


    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    if len(df) < seq_length + 1:
        print(f"Still not enough valid rows in {filename}")
        return 0.0, W

    balance = INITIAL_BALANCE
    in_position = False
    entry_price = 0.0
    last_action = 0
    position_size_fraction = 0.1
    last_day = None
    start_of_day_balance = balance

    for epoch in range(5):
        balance = INITIAL_BALANCE
        for i in range(seq_length, len(df) - 1):
            current_price = df['close'].iloc[i]
            next_price = df['close'].iloc[i + 1]
            state_vec = get_state_vector(df[['close']], i)
            timestamp = df.index[i]
            date_str = timestamp.date()

            if last_day != date_str:
                if last_day is not None:
                    profit = balance - start_of_day_balance
                    print(f"{last_day} Profit: {profit:.2f} Balance: {balance:.2f}")
                start_of_day_balance = balance
                last_day = date_str

            action = choose_action(state_vec)
            reward = 0
            position_size = balance * position_size_fraction

            if action == 1:
                if not in_position:
                    in_position = True
                    entry_price = current_price
                    last_action = 1
                elif last_action == 2:
                    profit = (entry_price - current_price) / entry_price * leverage * position_size
                    balance += profit
                    in_position = True
                    entry_price = current_price
                    last_action = 1
                    reward = profit / position_size

            elif action == 2:
                if not in_position:
                    in_position = True
                    entry_price = current_price
                    last_action = 2
                elif last_action == 1:
                    profit = (current_price - entry_price) / entry_price * leverage * position_size
                    balance += profit
                    in_position = True
                    entry_price = current_price
                    last_action = 2
                    reward = profit / position_size

            elif action == 0 and in_position:
                if last_action == 1:
                    unrealized = (current_price - entry_price) / entry_price * leverage
                elif last_action == 2:
                    unrealized = (entry_price - current_price) / entry_price * leverage
                else:
                    unrealized = 0
                reward += np.clip(unrealized, -1.0, 1.0) * 0.5

            next_state_vec = get_state_vector(df[['close']], i + 1)
            update_weights(state_vec, action, reward, next_state_vec)
            # time.sleep(0.1)

    # Final day's profit print
    if last_day is not None:
        profit = balance - start_of_day_balance
        print(f"{last_day} Profit: {profit:.2f} Balance: {balance:.2f}")

    return balance, W

# --- Run Training on All Files ---

files = [
    "ETHUSD_1m_Binance.csv",
    "XRPUSD_1m_Binance.csv",
    "BNBUSD_1m_Binance.csv",
    "BTCUSD_1m_Binance.csv",
]

total_balance = 0.0
for file in files:
    path = os.path.join(DATA_DIR, file)
    print(f"\nTraining on {file}...")
    balance, W = train_on_file(path, W, seq_length=SEQ_LENGTH)
    total_balance += balance

# Save weights after training
np.save("weights.npy", W)
print("Weights saved to weights.npy")

print(f"\nTotal combined balance after training: {total_balance:.2f}")
