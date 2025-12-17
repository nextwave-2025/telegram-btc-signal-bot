from telegram import Bot
import os
import asyncio

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

async def send_demo_signal():
    bot = Bot(token=BOT_TOKEN)

    message = (
        "🚨 DEMO SIGNAL 🚨\n\n"
        "📊 Pair: BTCUSDT\n"
        "📈 Direction: LONG\n"
        "🎯 Entry: 42,500\n"
        "🛑 Stop Loss: 41,900\n"
        "💰 Targets:\n"
        "TP1: 43,200\n"
        "TP2: 44,000\n\n"
        "⚠️ Demo-Signal – keine Anlageberatung"
    )

    await bot.send_message(chat_id=CHAT_ID, text=message)

if __name__ == "__main__":
    asyncio.run(send_demo_signal())
