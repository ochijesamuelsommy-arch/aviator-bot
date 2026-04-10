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
MIN_DATA = 50

# =========================
# TOKEN
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN")
BOT_TOKEN = BOT_TOKEN.strip() if BOT_TOKEN else None

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
# 🧼 AUTO CLEAN FUNCTION
# =========================
def clean_value(value):
    # 1. clamp extreme values
    value = max(1.01, min(value, 20))

    # 2. smooth rounding noise
    value = round(value, 2)

    return value


def clean_series(series):
    arr = np.array(series)

    # median smoothing (removes spikes)
    for i in range(len(arr)):
        if i > 0:
            diff = abs(arr[i] - arr[i - 1])
            if diff > 5:  # spike detection
                arr[i] = (arr[i] + arr[i - 1]) / 2

    return arr


# =========================
# DATA PREP
# =========================
def prepare_data(series):
    cleaned = clean_series(series.values)

    values = cleaned.reshape(-1, 1)
    scaled = scaler.fit_transform(values)

    X, y = [], []

    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i-SEQ_LEN:i])
        y.append(scaled[i])

    return np.array(X), np.array(y)


# =========================
# MODEL (IMPROVED)
# =========================
def build_model():
    model = Sequential()

    model.add(LSTM(96, return_sequences=True, input_shape=(SEQ_LEN, 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(48))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    return model


def train_model():
    global model

    if len(df) < MIN_DATA:
        return

    X, y = prepare_data(df["multiplier"])

    if model is None:
        model = build_model()

    model.fit(X, y, epochs=20, batch_size=8, verbose=0)


# =========================
# PREDICTION
# =========================
def ai_predict(seq):
    global model

    if model is None:
        return None

    if len(seq) < SEQ_LEN:
        seq = [np.mean(seq)] * (SEQ_LEN - len(seq)) + seq

    arr = np.array(seq[-SEQ_LEN:]).reshape(-1, 1)
    scaled = scaler.transform(arr)

    pred = model.predict(scaled.reshape(1, SEQ_LEN, 1), verbose=0)
    value = scaler.inverse_transform(pred)[0][0]

    return round(float(value), 2)


def fallback_predict(seq):
    arr = np.array(seq)
    return round(float(np.mean(arr) + (arr[-1] - arr[0]) * 0.3), 2)


# =========================
# COMMANDS
# =========================

@bot.message_handler(commands=['add'])
def add(message):
    global df

    try:
        value = float(message.text.split()[1])

        # 🧼 AUTO CLEAN (NO REJECTION)
        value = clean_value(value)

        df = pd.concat(
            [df, pd.DataFrame([{"multiplier": value}])],
            ignore_index=True
        )

        df.to_csv(DATA_FILE, index=False)

        train_model()

        bot.reply_to(message, f"✅ Added (cleaned): {value}")

    except:
        bot.reply_to(message, "❌ Usage: /add 1.5")


@bot.message_handler(commands=['predict'])
def predict(message):
    try:
        numbers = list(map(float, message.text.split()[1:]))

        if len(df) >= MIN_DATA:
            pred = ai_predict(numbers)
            mode = "AI MODEL"
        else:
            pred = fallback_predict(numbers)
            mode = "FALLBACK"

        if pred < 1.5:
            label = "LOW"
        elif pred < 3:
            label = "MID"
        else:
            label = "HIGH"

        bot.reply_to(
            message,
            f"✈️ {mode}\n"
            f"Result: {label} — {pred}x"
        )

    except Exception as e:
        bot.reply_to(message, f"Error: {e}")


@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "🤖 Auto-Clean AI Bot Ready")


print("Bot running...")
bot.infinity_polling()
