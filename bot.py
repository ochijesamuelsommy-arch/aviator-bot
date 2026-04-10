import os
import telebot
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =========================
# CONFIG
# =========================
DATA_FILE = "aviator_data.csv"
SEQ_LEN = 10
MIN_DATA = 20

# =========================
# TOKEN
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN")
BOT_TOKEN = BOT_TOKEN.strip() if BOT_TOKEN else None

if not BOT_TOKEN:
    raise Exception("BOT_TOKEN missing")

bot = telebot.TeleBot(BOT_TOKEN)

# =========================
# DATA LOAD
# =========================
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.DataFrame(columns=["multiplier"])

scaler = MinMaxScaler()
model = None

# =========================
# CLEAN DATA
# =========================
def clean_value(v):
    v = max(1.01, min(float(v), 20))
    return round(v, 2)

# =========================
# MODEL
# =========================
def build_model():
    m = Sequential()
    m.add(LSTM(96, return_sequences=True, input_shape=(SEQ_LEN, 1)))
    m.add(Dropout(0.2))
    m.add(LSTM(48))
    m.add(Dropout(0.2))
    m.add(Dense(32, activation="relu"))
    m.add(Dense(1))
    m.compile(optimizer="adam", loss="mse")
    return m

# =========================
# TRAIN
# =========================
def train_model():
    global model

    if len(df) < MIN_DATA:
        return

    data = df["multiplier"].values.reshape(-1, 1)
    scaled = scaler.fit_transform(data)

    X, y = [], []

    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i-SEQ_LEN:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    if model is None:
        model = build_model()

    model.fit(X, y, epochs=20, batch_size=8, verbose=0)

# =========================
# ENSEMBLE ENGINE
# =========================
def lstm_predict(seq):
    if model is None:
        return None

    if len(seq) < SEQ_LEN:
        seq = [np.mean(seq)] * (SEQ_LEN - len(seq)) + seq

    arr = np.array(seq[-SEQ_LEN:]).reshape(-1, 1)
    scaled = scaler.transform(arr)

    pred = model.predict(scaled.reshape(1, SEQ_LEN, 1), verbose=0)
    return scaler.inverse_transform(pred)[0][0]


def stat_predict(seq):
    arr = np.array(seq)
    return np.mean(arr) + np.std(arr) * 0.1


def trend_predict(seq):
    arr = np.array(seq)
    if len(arr) < 2:
        return np.mean(arr)
    return np.mean(arr) + (arr[-1] - arr[0]) * 0.3


def ensemble(seq):
    preds = []

    for p in [lstm_predict(seq), stat_predict(seq), trend_predict(seq)]:
        if p is not None:
            preds.append(p)

    return float(np.mean(preds))

# =========================
# PROBABILITY ENGINE
# =========================
def probabilities(value):
    low = max(0, 100 - value * 30)
    mid = max(0, 100 - abs(value - 2) * 40)
    high = max(0, value * 20)

    total = low + mid + high
    if total == 0:
        total = 1

    return (
        round(low / total * 100, 2),
        round(mid / total * 100, 2),
        round(high / total * 100, 2)
    )

# =========================
# COMMANDS
# =========================

@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(
        message,
        "🤖 Expert AI Bot Ready\n"
        "/add value\n/predict numbers\n/count"
    )


@bot.message_handler(commands=['add'])
def add(message):
    global df

    try:
        value = clean_value(message.text.split()[1])

        df = pd.concat([df, pd.DataFrame([{"multiplier": value}])], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)

        train_model()

        bot.reply_to(message, f"✅ Added: {value}")

    except:
        bot.reply_to(message, "❌ Usage: /add 1.5")


@bot.message_handler(commands=['predict'])
def predict(message):
    try:
        seq = list(map(float, message.text.split()[1:]))

        if len(df) < 5:
            bot.reply_to(message, "⚠️ Not enough data yet")
            return

        value = ensemble(seq)
        low, mid, high = probabilities(value)

        label = "LOW" if low > max(mid, high) else "MID" if mid > high else "HIGH"

        bot.reply_to(
            message,
            f"🧠 EXPERT AI SYSTEM\n"
            f"━━━━━━━━━━━━\n"
            f"Prediction: {value:.2f}x\n\n"
            f"📊 Probability:\n"
            f"LOW: {low}%\n"
            f"MID: {mid}%\n"
            f"HIGH: {high}%\n\n"
            f"🎯 Signal: {label}"
        )

    except Exception as e:
        bot.reply_to(message, f"Error: {e}")


@bot.message_handler(commands=['count'])
def count(message):
    bot.reply_to(
        message,
        f"📊 Dataset Info:\n"
        f"Total records: {len(df)}\n"
        f"Average: {df['multiplier'].mean() if len(df)>0 else 0:.2f}\n"
        f"Max: {df['multiplier'].max() if len(df)>0 else 0}\n"
        f"Min: {df['multiplier'].min() if len(df)>0 else 0}"
    )

# =========================
# RUN
# =========================
print("🚀 Final Expert Bot Running...")
bot.infinity_polling()
