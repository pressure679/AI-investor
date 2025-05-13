# AI Investor Trading Bot

A simple reinforcement learning-based cryptocurrency trading bot that learns from historical 1-minute candle data. The bot uses a basic Q-learning approach with a linear model to take BUY, SELL, or HOLD actions based on technical indicators like RSI and moving averages.

## Features

- Tail reads last 5MB of large CSV files for efficient training
- Uses RSI, moving average difference, and price ratio as state features
- Chooses actions using epsilon-greedy Q-learning
- Simulates leveraged trading with reward-based weight updates
- Saves learned weights (`W.npy`) after training
- Prints daily profit and final balance per file

## Data

The historical 1-minute candlestick CSV files used for training are downloaded from [Kaggle](https://www.kaggle.com/), specifically:

- `BTCUSD_1m_Binance.csv`
- `ETHUSD_1m_Binance.csv`
- `BNBUSD_1m_Binance.csv`
- `XRPUSD_1m_Binance.csv`

Place these files in the directory set as `DATA_DIR` in the script.

## Requirements

- Python 3.x
- NumPy
- Pandas

Install dependencies:

```bash
pip install numpy pandas
```

## Usage

To run training on all provided CSV files:

```bash
python AI-investor.py
```

After training, the combined balance and daily profits are printed. The trained weights are saved to `W.npy`.

## File Structure

```
project-folder/
│
├── AI-investor.py         # Main training logic
├── weights.npy            # Saved weights after training
├── README.md              # This file
├── crypto and currency pairs/
│   ├── BTCUSD_1m_Binance.csv
│   ├── ETHUSD_1m_Binance.csv
│   ├── BNBUSD_1m_Binance.csv
│   └── XRPUSD_1m_Binance.csv
```

## License

This project is open-source and free to use under the MIT license.

## Disclaimer

This project is for educational purposes only and does not constitute financial advice. Use at your own risk.
## 🙏 Credits

Created by Vittus Mikiassen with love for simplicity, performance, and trading experiments.
