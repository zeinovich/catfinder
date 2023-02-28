from telebot import types
from bot.utils.commands import COMMANDS

def button_maker():
    markup = types.InlineKeyboardMarkup()
    markup.row_width = 2
    markup.add(types.InlineKeyboardButton('Predict', callback_data=COMMANDS.PREDICT),
               types.InlineKeyboardButton('Breeds', callback_data=COMMANDS.BREEDS),
               types.InlineKeyboardButton('Help', callback_data=COMMANDS.HELP),
               types.InlineKeyboardButton('Info', callback_data=COMMANDS.INFO))
    
    return markup

KEYBOARD = button_maker()