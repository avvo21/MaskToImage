import asyncio
import os
import argparse
import telegram

class TelegramBot:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.bot = telegram.Bot(token=self.token)
        # Loop to handle asynchronous calls in a synchronous context
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def send_message(self, msg):

        if self.token is None or self.chat_id is None:
            return

        try:
            # Run the asynchronous coroutine synchronously
            self.loop.run_until_complete(self.bot.send_message(chat_id=self.chat_id, text=msg))
        except Exception as e:
            # If it fails (Timeout, No Internet, etc.), it prints the error but do not stop the program
            print(f"TELEGRAM WARNING: Unable to send message. Error: {e}")

    def send_photo(self, photo_path, caption=None):
        if self.loop.is_running():
            asyncio.create_task(self.bot.send_photo(chat_id=self.chat_id, photo=open(photo_path, 'rb'), caption=caption))
        else:
            self.loop.run_until_complete(self.bot.send_photo(chat_id=self.chat_id, photo=open(photo_path, 'rb'), caption=caption))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telegram Bot for notifications")
    parser.add_argument("--token", type=str, default=os.environ.get("TG_BOT_TOKEN"),
                        help="Telegram bot token (or set TG_BOT_TOKEN)")
    parser.add_argument("--chat-id", type=str, default=os.environ.get("TG_CHAT_ID"),
                        help="Telegram chat ID (or set TG_CHAT_ID)")
    parser.add_argument("--message", type=str, default="Test sending message from Python script.",
                        help="Message to send")
    args = parser.parse_args()

    if not args.token or not args.chat_id:
        parser.error("Token and chat-id are required. Pass them as arguments or set TG_BOT_TOKEN and TG_CHAT_ID.")

    bot = TelegramBot(args.token, args.chat_id)
    bot.send_message(args.message)
