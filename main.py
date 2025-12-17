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
    asyncio.run(send_message("✅ Bot online. Schritt 1 erfolgreich. (Noch keine Signale)"))
    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()
