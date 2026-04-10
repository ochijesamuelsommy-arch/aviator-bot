import os
import telebot
import pandas as pd

# =========================
# CONFIG
# =========================
DATA_FILE = "aviator_data.csv"

# Fix token permanently
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
# COMMAND: START
# =========================
@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "🤖 Aviator Bot is running!\nUse /add, /predict, /analyze")


# =========================
# COMMAND: ADD DATA
# =========================
@bot.message_handler(commands=['add'])
def add_data(message):
    global df
    try:
        value = float(message.text.split()[1])

        new_row = pd.DataFrame([{"multiplier": value}])
        df = pd.concat([df, new_row], ignore_index=True)

        df.to_csv(DATA_FILE, index=False)

        bot.reply_to(message, f"✅ Added: {value}")

    except:
        bot.reply_to(message, "❌ Usage: /add 1.5")


# =========================
# COMMAND: ANALYZE
# =========================
@bot.message_handler(commands=['analyze'])
def analyze(message):
    try:
        if len(df) < 5:
            bot.reply_to(message, "⚠️ Not enough data yet")
            return

        avg = df["multiplier"].mean()
        minimum = df["multiplier"].min()
        maximum = df["multiplier"].max()

        result = (
            f"📊 Analysis:\n"
            f"Count: {len(df)}\n"
            f"Average: {avg:.2f}\n"
            f"Min: {minimum}\n"
            f"Max: {maximum}"
        )

        bot.reply_to(message, result)

    except Exception as e:
        bot.reply_to(message, f"Error: {str(e)}")


# =========================
# COMMAND: PREDICT
# =========================
@bot.message_handler(commands=['predict'])
def predict(message):
    try:
        parts = message.text.split()[1:]
        numbers = list(map(float, parts))

        if len(numbers) < 4:
            bot.reply_to(message, "❌ Send at least 4 numbers\nExample: /predict 1.2 2.0 1.5 3.1")
            return

        avg = sum(numbers) / len(numbers)

        # Simple prediction logic (can upgrade later)
        if avg < 1.5:
            category = "LOW"
            value = round(max(avg * 0.9, 1.01), 2)
        elif avg < 3:
            category = "MID"
            value = round(avg, 2)
        else:
            category = "HIGH"
            value = round(avg * 1.2, 2)

        result = f"✈️ Prediction: {category} — {value}x"

        bot.reply_to(message, result)

    except Exception as e:
        bot.reply_to(message, f"Error: {str(e)}")


# =========================
# RUN BOT
# =========================
print("🤖 Bot is running...")
bot.infinity_polling()


# ================== RUN BOT ==================
print("Bot is running...")
bot.infinity_polling()
