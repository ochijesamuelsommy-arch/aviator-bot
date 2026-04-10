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

# =========================
# TOKEN FIX (PERMANENT)
# =========================
raw_token = os.getenv("BOT_TOKEN")
BOT_TOKEN = raw_token.strip() if raw_token else None

print("TOKEN CHECK:", repr(BOT_TOKEN))

if not BOT_TOKEN:
    raise Exception("BOT_TOKEN missing")

bot = telebot.TeleBot(BOT_TOKEN)

# =========================
# DATA
# =========================
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.DataFrame(columns=["multiplier"])

# =========================
# MODEL
# =========================
scaler = MinMaxScaler()
model = None


def prepare_data(series):
    data = series.values.reshape(-1, 1)
    scaled = scaler.fit_transform(data)

    X, y = [], []

    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i - SEQ_LEN:i])
        y.append(scaled[i])

    return np.array(X), np.array(y)


def build_model():
    m = Sequential()
    m.add(LSTM(64, input_shape=(SEQ_LEN, 1)))
    m.add(Dropout(0.2))
    m.add(Dense(32, activation="relu"))
    m.add(Dense(1))  # regression output

    m.compile(optimizer="adam", loss="mse")
    return m


def train_model():
    global model

    if len(df) < 50:
        return

    X, y = prepare_data(df["multiplier"])

    if model is None:
        model = build_model()

    model.fit(X, y, epochs=15, batch_size=8, verbose=0)


def predict_next(seq):
    global model

    if model is None or len(df) < 50:
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
    bot.reply_to(message, "🤖 Smart Aviator AI Bot is running!\nUse /add /predict /analyze")


@bot.message_handler(commands=['add'])
def add_data(message):
    global df

    try:
        value = float(message.text.split()[1])

        # 🚫 FILTER BAD DATA
        if value <= 1.0 or value > 20:
            bot.reply_to(message, "❌ Unrealistic value ignored")
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
            bot.reply_to(message, "⚠️ Not enough data yet")
            return

        avg = df["multiplier"].mean()
        minimum = df["multiplier"].min()
        maximum = df["multiplier"].max()

        bot.reply_to(
            message,
            f"📊 Analysis:\n"
            f"Count: {len(df)}\n"
            f"Average: {avg:.2f}\n"
            f"Min: {minimum}\n"
            f"Max: {maximum}"
        )

    except Exception as e:
        bot.reply_to(message, f"Error: {str(e)}")


@bot.message_handler(commands=['predict'])
def predict(message):
    try:
        parts = message.text.split()[1:]
        numbers = list(map(float, parts))

        if len(numbers) < 4:
            bot.reply_to(message, "❌ Example: /predict 1.2 2.0 1.5 3.1")
            return

        pred_value = predict_next(numbers)

        if pred_value is None:
            bot.reply_to(message, "⚠️ Not enough data to predict yet (need ~50+ rounds)")
            return

        # CLASSIFY
        if pred_value < 1.5:
            category = "LOW"
        elif pred_value < 3:
            category = "MID"
        else:
            category = "HIGH"

        confidence = min(95, max(50, int(100 - abs(pred_value - np.mean(numbers)) * 10)))

        bot.reply_to(
            message,
            f"✈️ Prediction: {category} — {pred_value}x\n"
            f"Confidence: {confidence}%"
        )

    except Exception as e:
        bot.reply_to(message, f"Error: {str(e)}")


# =========================
# RUN
# =========================
print("🤖 Bot is running...")
bot.infinity_polling()


# ================== RUN BOT ==================
print("Bot is running...")
bot.infinity_polling()
