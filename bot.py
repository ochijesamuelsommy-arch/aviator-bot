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
MEMORY_FILE = "error_memory.csv"

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

if os.path.exists(MEMORY_FILE):
    memory = pd.read_csv(MEMORY_FILE)
else:
    memory = pd.DataFrame(columns=["predicted", "actual", "error"])

scaler = MinMaxScaler()
model = None


# =========================
# CLEANING
# =========================
def clean(v):
    v = max(1.01, min(v, 20))
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


def prepare(series):
    arr = series.values.reshape(-1, 1)
    scaled = scaler.fit_transform(arr)

    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i-SEQ_LEN:i])
        y.append(scaled[i])

    return np.array(X), np.array(y)


def train():
    global model

    if len(df) < MIN_DATA:
        return

    X, y = prepare(df["multiplier"])

    if model is None:
        model = build_model()

    model.fit(X, y, epochs=20, batch_size=8, verbose=0)


# =========================
# 🧠 SELF LEARNING BIAS CORRECTION
# =========================
def get_bias():
    if len(memory) < 5:
        return 0

    return memory["error"].mean()


# =========================
# PREDICT (WITH SELF LEARNING)
# =========================
def predict(seq):
    global model

    if len(df) < MIN_DATA or model is None:
        return np.mean(seq)

    if len(seq) < SEQ_LEN:
        seq = [np.mean(seq)] * (SEQ_LEN - len(seq)) + seq

    arr = np.array(seq[-SEQ_LEN:]).reshape(-1, 1)
    scaled = scaler.transform(arr)

    pred = model.predict(scaled.reshape(1, SEQ_LEN, 1), verbose=0)
    value = scaler.inverse_transform(pred)[0][0]

    # 🔥 APPLY LEARNED BIAS CORRECTION
    bias = get_bias()
    value = value - bias

    return round(float(value), 2)


# =========================
# UPDATE MEMORY (SELF LEARNING CORE)
# =========================
def update_memory(predicted, actual):
    global memory

    error = actual - predicted

    memory = pd.concat([
        memory,
        pd.DataFrame([{
            "predicted": predicted,
            "actual": actual,
            "error": error
        }])
    ], ignore_index=True)

    memory.to_csv(MEMORY_FILE, index=False)


# =========================
# COMMANDS
# =========================

@bot.message_handler(commands=['add'])
def add(message):
    global df

    try:
        value = clean(float(message.text.split()[1]))

        df = pd.concat(
            [df, pd.DataFrame([{"multiplier": value}])],
            ignore_index=True
        )

        df.to_csv(DATA_FILE, index=False)

        train()

        bot.reply_to(message, f"✅ Added: {value}")

    except:
        bot.reply_to(message, "❌ Usage: /add 1.5")


@bot.message_handler(commands=['predict'])
def handle_predict(message):
    try:
        numbers = list(map(float, message.text.split()[1:]))

        pred = predict(numbers)

        label = "LOW" if pred < 1.5 else "MID" if pred < 3 else "HIGH"

        bot.reply_to(
            message,
            f"✈️ Prediction: {label} — {pred}x"
        )

    except Exception as e:
        bot.reply_to(message, str(e))


# =========================
# OPTIONAL: FEEDBACK LEARNING
# =========================
@bot.message_handler(commands=['result'])
def result(message):
    """
    User sends real outcome:
    /result 2.1
    Bot learns from last prediction
    """
    global memory

    try:
        actual = float(message.text.split()[1])
        actual = clean(actual)

        if len(memory) == 0:
            bot.reply_to(message, "⚠️ No prediction to compare")
            return

        last_pred = memory.iloc[-1]["predicted"]

        update_memory(last_pred, actual)

        bot.reply_to(message, "🧠 Learned from mistake!")

    except:
        bot.reply_to(message, "❌ Usage: /result 2.1")


@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(
        message,
        "🤖 Self-Learning AI Bot Active\n"
        "/add /predict /result"
    )


print("🤖 Self-learning bot running...")
bot.infinity_polling()
