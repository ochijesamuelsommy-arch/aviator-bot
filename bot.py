import os
import telebot

raw_token = os.getenv("BOT_TOKEN")

# 🔥 HARD FIX: remove hidden spaces + newlines
BOT_TOKEN = raw_token.strip() if raw_token else None

print("TOKEN CHECK:", repr(BOT_TOKEN))

if not BOT_TOKEN:
    raise Exception("BOT_TOKEN missing")

bot = telebot.TeleBot(BOT_TOKEN)
# ================== LOAD DATA ==================
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.DataFrame(columns=["multiplier"])

# ================== CORE LOGIC ==================

def classify(value):
    if value < 1.5:
        return 0  # LOW
    elif value < 3:
        return 1  # MID
    else:
        return 2  # HIGH


def get_distribution(data):
    low = sum(1 for x in data if x < 1.5)
    mid = sum(1 for x in data if 1.5 <= x < 3)
    high = sum(1 for x in data if x >= 3)

    total = len(data)
    return low/total, mid/total, high/total


def get_streak(data):
    if len(data) < 2:
        return classify(data[-1]), 1

    last_class = classify(data[-1])
    streak = 1

    for i in range(len(data)-2, -1, -1):
        if classify(data[i]) == last_class:
            streak += 1
        else:
            break

    return last_class, streak


def get_volatility(data):
    if len(data) < 5:
        return 0
    return np.std(data[-20:])


def get_risk_level(low, mid, high, streak_len):
    if low > 0.6 and streak_len >= 3:
        return "HIGH RISK ⚠️"
    elif mid > 0.5:
        return "MODERATE ⚖️"
    else:
        return "LOW RISK ✅"


def get_recommendation(low, mid, high):
    if low > 0.5:
        return "💡 Safe exit: 1.3x – 1.5x"
    elif mid > 0.4:
        return "💡 Moderate play: 1.5x – 2.5x"
    else:
        return "💡 High risk: Aim 3x+ (risky)"


# ================== COMMAND FUNCTIONS ==================

def analyze_command():
    global df

    if len(df) < 10:
        return "⚠️ Not enough data yet. Add more rounds using /add"

    data = df["multiplier"].tolist()
    recent = data[-50:]

    low, mid, high = get_distribution(recent)
    streak_type, streak_len = get_streak(data)
    volatility = get_volatility(data)
    risk = get_risk_level(low, mid, high, streak_len)
    recommendation = get_recommendation(low, mid, high)

    labels = ["LOW", "MID", "HIGH"]

    return f"""
✈️ Aviator Smart Analysis

📊 Distribution (last {len(recent)} rounds):
LOW: {low*100:.1f}%
MID: {mid*100:.1f}%
HIGH: {high*100:.1f}%

🔥 Current Streak:
{labels[streak_type]} × {streak_len}

📈 Volatility:
{volatility:.2f}

⚠️ Risk Level:
{risk}

{recommendation}
"""


def trend_command():
    global df

    if len(df) < 10:
        return "⚠️ Not enough data."

    recent = df["multiplier"].tolist()[-20:]
    trend = ["L" if x < 1.5 else "M" if x < 3 else "H" for x in recent]

    return "📊 Trend (last 20):\n" + " ".join(trend)


def risk_command():
    global df

    if len(df) < 10:
        return "⚠️ Not enough data."

    data = df["multiplier"].tolist()
    low, mid, high = get_distribution(data[-30:])
    _, streak_len = get_streak(data)

    risk = get_risk_level(low, mid, high, streak_len)

    return f"⚠️ Current Risk Level: {risk}"


def add_command(value):
    global df

    try:
        value = float(value)

        new_row = pd.DataFrame([{"multiplier": value}])
        df = pd.concat([df, new_row], ignore_index=True)

        df.to_csv(DATA_FILE, index=False)

        return f"✅ Added: {value}x"

    except:
        return "❌ Invalid input. Use /add 1.5"


def help_command():
    return """
🤖 Aviator Smart Bot Commands:

/add 1.5 → Add new round
/analyze → Full analysis
/trend → View last 20 rounds
/risk → Quick risk check

💡 This bot analyzes patterns, NOT predictions.
"""


# ================== TELEGRAM HANDLERS ==================

@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "🤖 Aviator Smart Bot is running!\nType /help")


@bot.message_handler(commands=['help'])
def help_cmd(message):
    bot.reply_to(message, help_command())


@bot.message_handler(commands=['add'])
def add_cmd(message):
    try:
        value = message.text.split()[1]
        response = add_command(value)
    except:
        response = "❌ Usage: /add 1.5"

    bot.reply_to(message, response)


@bot.message_handler(commands=['analyze'])
def analyze_cmd(message):
    bot.reply_to(message, analyze_command())


@bot.message_handler(commands=['trend'])
def trend_cmd(message):
    bot.reply_to(message, trend_command())


@bot.message_handler(commands=['risk'])
def risk_cmd(message):
    bot.reply_to(message, risk_command())


# ================== RUN BOT ==================
print("Bot is running...")
bot.infinity_polling()
