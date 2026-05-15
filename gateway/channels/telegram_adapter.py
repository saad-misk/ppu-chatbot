import logging
import telebot

from shared.config.settings import settings
from gateway.core.chat_service import process_chat_message   # Clean service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = telebot.TeleBot(settings.TELEGRAM_BOT_TOKEN)


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, 
        "👋 *Welcome to PPU Assistant!*\n\n"
        "I'm your smart guide for Palestine Polytechnic University.\n"
        "Ask me anything about fees, schedules, registration, departments...",
        parse_mode="Markdown"
    )


import asyncio

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_text = message.text
    chat_id = message.chat.id

    bot.send_chat_action(chat_id, 'typing')

    try:
        response = asyncio.run(
            process_chat_message(
                session_id=f"tg_{chat_id}",
                message=user_text,
                channel="telegram"
            )
        )

        reply_text = response.get(
            "reply",
            "Sorry, I couldn't process that."
        )

        bot.reply_to(message, reply_text)

    except Exception as e:
        logger.error(f"Telegram error: {e}")

        bot.reply_to(
            message,
            "⚠️ Sorry, something went wrong. Please try again."
        )


if __name__ == "__main__":
    if not settings.TELEGRAM_BOT_TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN is missing in .env!")
    else:
        logger.info("🚀 Telegram Bot with Real AI is running!")
        bot.infinity_polling()