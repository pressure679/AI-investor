
# Q-Learning Crypto Training Bot

This Python script is a lightweight backtesting framework for training a Q-learning-based agent to trade cryptocurrency pairs using historical 1-minute interval data.

## Features

- Supports multiple crypto pairs: BTC, ETH, BNB, XRP (Binance format).
- Tail-based CSV loading to minimize memory usage.
- Feature extraction includes:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Band Width
  - Moving Average differences
- Uses a simple linear Q-learning model (Q(s,a) = state_vector @ W).
- Balance updates with leveraged PnL calculations.
- Auto-loads and saves weight matrices (`weights_*.npy`).

## Installation

```bash
pip install numpy pandas
```

## Constants

- `DATA_DIR`: Location of your historical data files
- `MAX_FILE_SIZE`: Max file size read per CSV
- `INITIAL_BALANCE`: Starting virtual balance per run
- `TP_PERCENT`, `SL_PERCENT`: Not used directly but can be integrated for fixed TP/SL logic
- `LEVERAGE_BINANCE`, `LEVERAGE_DAT`: Used in PnL computation
- `SEQ_LENGTH`: Number of candles required for features
- `NUM_FEATURES`: Fixed to 6 (rsi, ma_diff, price_ratio, macd, signal, bb_width)

## Usage

Place Binance-formatted historical CSV files (1m interval) in the folder specified by `DATA_DIR`.

Each file should at least contain the following columns:
- `timestamp` or `Open time`
- `close` or `Close`

Then simply run the script:
```bash
python train_q_learning_bot.py
```

## Output

- Trains on each symbol for 2 epochs
- Saves/loads `weights_<symbol>.npy`
- Prints daily profits during training
- Outputs total balance at the end

## Weight Matrix

The agent uses a matrix `W` of shape `(NUM_FEATURES, NUM_ACTIONS)` where each column corresponds to Q-values for an action:
- `0`: HOLD
- `1`: BUY
- `2`: SELL

The agent follows an epsilon-greedy policy.

## Notes

- Files over 20MB are trimmed using a `tail`-like strategy.
- This is a research/prototyping tool and **not meant for production trading.**
- No fixed TP/SL levels; rewards are proportional to leveraged profit/loss.

## Disclaimer

This project is for educational purposes only and does not constitute financial advice. Use at your own risk.
## 🙏 Credits

Created by Vittus Mikiassen with love for simplicity, performance, and trading experiments.
