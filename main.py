import os
import time
import asyncio
from telegram import Bot
from telegram.error import BadRequest, Forbidden

BOT_TOKEN = os.environ["BOT_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

bot = Bot(token=BOT_TOKEN)

async def send_message(text: str):
    await bot.send_message(chat_id=CHAT_ID, text=text)

def main():
    try:
        asyncio.run(send_message("✅ Bot online. Schritt 1 erfolgreich. (Noch keine Signale)"))
        print("Startup message sent.")
    except (BadRequest, Forbidden) as e:
        # z.B. Chat not found / fehlende Rechte
        print(f"Startup send failed: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")

    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()
