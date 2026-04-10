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
MIN_DATA = 50  # AI activation threshold

# =========================
# TOKEN SAFE LOAD
# =========================
raw_token = os.getenv("BOT_TOKEN")
BOT_TOKEN = raw_token.strip() if raw_token else None

print("TOKEN CHECK:", repr(BOT_TOKEN))

if not BOT_TOKEN:
    raise Exception("BOT_TOKEN missing")

bot = telebot.TeleBot(BOT_TOKEN)

# =========================
# LOAD DATA
# =========================
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.DataFrame(columns=["multiplier"])

# =========================
# SCALER + MODEL
# =========================
scaler = MinMaxScaler()
model = None


# =========================
# DATA PREP
# =========================
def prepare_data(series):
    values = series.values.reshape(-1, 1)
    scaled = scaler.fit_transform(values)

    X, y = [], []

    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i-SEQ_LEN:i])
        y.append(scaled[i])

    return np.array(X), np.array(y)


# =========================
# MODEL (IMPROVED ARCHITECTURE)
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
# TRAIN (CONTROLLED)
# =========================
def train_model():
    global model

    if len(df) < MIN_DATA:
        return

    X, y = prepare_data(df["multiplier"])

    if model is None:
        model = build_model()

    model.fit(
        X, y,
        epochs=20,
        batch_size=8,
        verbose=0
    )


# =========================
# FALLBACK (SMART STATISTICS MODEL)
# =========================
def fallback_predict(seq):
    arr = np.array(seq)

    mean = np.mean(arr)
    std = np.std(arr)

    # trend detection
    trend = (arr[-1] - arr[0]) if len(arr) > 1 else 0

    prediction = mean + (trend * 0.3)

    # clamp unrealistic values
    prediction = max(1.01, min(prediction, 15))

    return round(float(prediction), 2)


# =========================
# AI PREDICTION
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


# =========================
# COMMANDS
# =========================

@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(
        message,
        "🤖 Smart Hybrid AI Bot Ready!\n"
        "/add value\n/predict numbers\n/analyze"
    )


@bot.message_handler(commands=['add'])
def add_data(message):
    global df

    try:
        value = float(message.text.split()[1])

        # filter bad data
        if value <= 1.0 or value > 20:
            bot.reply_to(message, "❌ Invalid value ignored")
            return

        df = pd.concat([df, pd.DataFrame([{"multiplier": value}])], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)

        train_model()

        bot.reply_to(message, f"✅ Added: {value}")

    except:
        bot.reply_to(message, "❌ Usage: /add 1.5")


@bot.message_handler(commands=['analyze'])
def analyze(message):
    try:
        if len(df) < 5:
            bot.reply_to(message, "⚠️ Not enough data")
            return

        bot.reply_to(
            message,
            f"📊 Stats:\n"
            f"Count: {len(df)}\n"
            f"Mean: {df['multiplier'].mean():.2f}\n"
            f"Std: {df['multiplier'].std():.2f}"
        )

    except Exception as e:
        bot.reply_to(message, str(e))


@bot.message_handler(commands=['predict'])
def predict(message):
    try:
        parts = message.text.split()[1:]
        numbers = list(map(float, parts))

        if len(numbers) < 4:
            bot.reply_to(message, "❌ Example: /predict 1.2 2.0 1.5 3.1")
            return

        # =========================
        # HYBRID ENGINE (BEST PART)
        # =========================

        if len(df) >= MIN_DATA:
            pred = ai_predict(numbers)
            mode = "AI MODEL"
        else:
            pred = fallback_predict(numbers)
            mode = "FALLBACK MODE"

        # classification
        if pred < 1.5:
            label = "LOW"
        elif pred < 3:
            label = "MID"
        else:
            label = "HIGH"

        # confidence logic
        confidence = min(
            95,
            max(55, int(100 - abs(pred - np.mean(numbers)) * 8))
        )

        bot.reply_to(
            message,
            f"✈️ Prediction ({mode})\n"
            f"Result: {label} — {pred}x\n"
            f"Confidence: {confidence}%"
        )

    except Exception as e:
        bot.reply_to(message, f"Error: {str(e)}")


# =========================
# RUN
# =========================
print("🤖 Hybrid Bot Running...")
bot.infinity_polling()
