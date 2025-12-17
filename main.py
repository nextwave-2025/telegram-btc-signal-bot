import os
import time
import asyncio
from telegram import Bot
from telegram.error import BadRequest

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

bot = Bot(token=BOT_TOKEN)

async def send_message(text: str):
    try:
        await bot.send_message(chat_id=CHAT_ID, text=text)
        print("Message sent successfully")
    except BadRequest as e:
        print(f"Telegram BadRequest: {e}")
    except Exception as e:
        print(f"Telegram Error: {type(e).__name__}: {e}")

def main():
    print("Bot started. Waiting...")
    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()
