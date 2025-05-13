# Lightweight AI Trading Bot (No TensorFlow)

A simple, fast, and dependency-light reinforcement learning trading bot using NumPy only — ideal for low-resource environments or quick testing.

This bot uses Q-learning with a linear function approximator to simulate trades on historical cryptocurrency data (e.g., BTCUSD, ETHUSD) from CSV files. It uses basic indicators like RSI, moving averages, and price ratios to determine BUY/SELL/HOLD actions.

---

## ✨ Features

- ✅ No TensorFlow or PyTorch — runs on low-resource devices like Termux or Chromebooks.
- 📉 Uses historical CSV data (Binance 1-minute candles).
- 🔁 Supports training across multiple assets.
- 📊 Daily profit tracking.
- 🎓 Q-learning with linear approximator.
- 💾 Saves and loads weights (`weights.npy`).
- 🔍 Simple indicators: RSI, MA5/MA20, price ratio.
- 📂 Reads only the last 5MB of each CSV (efficient).

---

## 📁 Data Format

The bot expects CSV files with this structure (like Binance exports):

```
Open time,Open,High,Low,Close,Volume,...
```

Ensure the file has at least `"Open time"` and `"Close"` columns.

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/lightweight-ai-trader.git
cd lightweight-ai-trader
pip install -r requirements.txt  # Only needs numpy and pandas
```

> ⚠️ Requires Python 3.7+

---

## ▶️ Usage

1. Put your CSV files (e.g., `BTCUSD_1m_Binance.csv`) into the `DATA_DIR` folder.
2. Run the training script:

```bash
python ai_trader.py
```

Weights are saved to `weights.npy` after training. On next run, it auto-loads them.

---

## ⚙️ Configuration

You can tweak constants at the top of the script:

```python
INITIAL_BALANCE = 100.0
TP_PERCENT = 0.013
SL_PERCENT = 0.0085
LEVERAGE = 10
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
```

---

## 💡 Example Output

```
Training on BTCUSD_1m_Binance.csv...
2025-05-10 Profit: +3.25  Balance: 103.25
2025-05-11 Profit: -1.12  Balance: 102.13

Total combined balance after training: 415.67
Weights saved to weights.npy
```

---

## 📌 Roadmap Ideas

- [ ] Add GUI or visualization (e.g., plot daily balance).
- [ ] Integrate with live data for paper trading.
- [ ] More indicators (MACD, Bollinger Bands).
- [ ] Export detailed trade logs.

---

## 🧠 License

MIT — feel free to use, modify, or distribute this bot.

---

## 🙏 Credits

Created by [Your Name] with love for simplicity, performance, and trading experiments.
