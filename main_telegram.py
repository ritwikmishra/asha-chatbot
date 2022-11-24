import os
from dotenv import load_dotenv
from main_telegram_service import reply, deletentry
import telebot

load_dotenv()
API_KEY = os.getenv("TOKEN")
bot = telebot.TeleBot(API_KEY)

@bot.message_handler(commands=['start', 'help'])
def gettingStarted(message):
    info = """
    <b>ASHA Bot Commands</b> \n
    /ask: Ask a new question to bot. 
    /detail : Details about the bot. 
    /help : To see this menu again.

    '''Ask Question with the command. 
    Example: /ask bacha dudh na piye toh kya kre'''
    """
    bot.reply_to(message,info, parse_mode='html')

@bot.message_handler(commands=['detail'])
def gettingStarted(message):
    info = """
    <b>ASHA Bot Info</b> \n
    Great bot made for noble cause
    Techical Specifications:
    -
    -
    """
    bot.reply_to(message,info, parse_mode='html')


@bot.message_handler(func=lambda message: message.text is not None and message.text.startswith("/ask"))
def newQna(message):
    cid = message.chat.id
    ques = message.text
    if len(ques.split(" "))>1:
        message = ques[1]
        deletentry(str(cid))
        ans = reply(message, str(cid))
        bot.send_message(cid, ans)
    else:
        bot.send_message(cid, '''Ask Question with the command. 
        Example: /ask bacha dudh na piye toh kya kre''')

@bot.message_handler()
def qna(message):
    cid = message.chat.id
    ans = reply(message.text, str(cid))
    bot.send_message(cid, ans)


bot.polling()
