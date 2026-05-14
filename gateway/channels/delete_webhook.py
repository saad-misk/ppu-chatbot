import telebot
from shared.config.settings import settings

bot = telebot.TeleBot(settings.TELEGRAM_BOT_TOKEN)

print("Deleting webhook...")
result = bot.delete_webhook(drop_pending_updates=True)
print("Webhook deleted successfully:", result)

print("You can now run the bot normally.")