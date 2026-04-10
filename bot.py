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
MIN_DATA = 60

BOT_TOKEN = os.getenv("BOT_TOKEN")
BOT_TOKEN = BOT_TOKEN.strip() if BOT_TOKEN else None

bot = telebot.TeleBot(BOT_TOKEN)

# =========================
# DATA
# =========================
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.DataFrame(columns=["multiplier"])

scaler = MinMaxScaler()
model = None


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


def train_model():
    global model

    if len(df) < MIN_DATA:
        return

    arr = df["multiplier"].values.reshape(-1, 1)
    scaled = scaler.fit_transform(arr)

    X, y = [], []

    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i-SEQ_LEN:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    if model is None:
        model = build_model()

    model.fit(X, y, epochs=20, batch_size=8, verbose=0)


# =========================
# ENSEMBLE MODELS
# =========================

def model_lstm(seq):
    if model is None:
        return None

    if len(seq) < SEQ_LEN:
        seq = [np.mean(seq)] * (SEQ_LEN - len(seq)) + seq

    arr = np.array(seq[-SEQ_LEN:]).reshape(-1, 1)
    scaled = scaler.transform(arr)

    pred = model.predict(scaled.reshape(1, SEQ_LEN, 1), verbose=0)
    return scaler.inverse_transform(pred)[0][0]


def model_stat(seq):
    arr = np.array(seq)

    mean = np.mean(arr)
    std = np.std(arr)

    # probabilistic spread model
    return mean + (np.random.randn() * std * 0.1)


def model_trend(seq):
    arr = np.array(seq)
    if len(arr) < 2:
        return np.mean(arr)

    trend = arr[-1] - arr[0]
    return np.mean(arr) + trend * 0.4


# =========================
# PROBABILITY ENGINE
# =========================
def probability_engine(seq):
    lstm = model_lstm(seq)
    stat = model_stat(seq)
    trend = model_trend(seq)

    predictions = [p for p in [lstm, stat, trend] if p is not None]

    final = np.mean(predictions)

    # probability buckets
    low_p = max(0, 100 - final * 35)
    mid_p = max(0, 100 - abs(final - 2.0) * 40)
    high_p = max(0, final * 20)

    total = low_p + mid_p + high_p
    if total == 0:
        total = 1

    low_p = round((low_p / total) * 100, 2)
    mid_p = round((mid_p / total) * 100, 2)
    high_p = round((high_p / total) * 100, 2)

    return final, low_p, mid_p, high_p


# =========================
# COMMANDS
# =========================

@bot.message_handler(commands=['add'])
def add(message):
    global df

    try:
        value = float(message.text.split()[1])
        value = max(1.01, min(value, 20))

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

        if len(df) < 20:
            bot.reply_to(message, "⚠️ Need more data (20+ recommended)")
            return

        final, low_p, mid_p, high_p = probability_engine(seq)

        label = "LOW" if low_p > max(mid_p, high_p) else "MID" if mid_p > high_p else "HIGH"

        bot.reply_to(
            message,
            f"🧠 EXPERT ENSEMBLE AI\n"
            f"━━━━━━━━━━━━━━\n"
            f"Prediction: {final:.2f}x\n\n"
            f"📊 Probabilities:\n"
            f"LOW: {low_p}%\n"
            f"MID: {mid_p}%\n"
            f"HIGH: {high_p}%\n\n"
            f"🎯 Signal: {label}"
        )

    except Exception as e:
        bot.reply_to(message, f"Error: {e}")


@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(
        message,
        "🤖 Expert Ensemble Bot Active\n"
        "/add /predict"
    )


print("🚀 Expert bot running...")
bot.infinity_polling()
