import os
import time
import asyncio
from telegram import Bot

BOT_TOKEN = os.environ["BOT_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

bot = Bot(token=BOT_TOKEN)

async def send_message(text: str):
    await bot.send_message(chat_id=CHAT_ID, text=text)

def main():
    try:
        asyncio.run(send_message("✅ Bot online. Schritt 1 erfolgreich. (Noch keine Signale)"))
    except Exception as e:
        print(f"Startup send failed: {type(e).__name__}: {e}")

    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()

