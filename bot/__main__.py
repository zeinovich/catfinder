from bot import bot, LOGGER, CLASSES, OWNER
from bot.predictor import predictor
from bot.utils.commands import COMMANDS
from bot.utils.button_maker import KEYBOARD
from bot.utils.messages import MESSAGES
import sys

class Exit(Exception):
    pass

@bot.message_handler(commands=[COMMANDS.START])
def start(message):
    bot.send_message(message.chat.id, MESSAGES.START, reply_markup=KEYBOARD)
    LOGGER.info(f'{message.chat.id} started the bot')


@bot.message_handler(commands=[COMMANDS.INFO])  
def info(message):
    bot.send_message(message.chat.id, MESSAGES.INFO, reply_markup=KEYBOARD)
    LOGGER.info(f'{message.chat.id} requested info')


@bot.message_handler(commands=[COMMANDS.HELP])
def help(message):
    bot.send_message(message.chat.id, MESSAGES.HELP, reply_markup=KEYBOARD)
    LOGGER.info(f'{message.chat.id} requested help')


@bot.message_handler(commands=[COMMANDS.BREEDS])
def breeds(message):
    LOGGER.info(f'{message.chat.id} -> /breeds')

    reply_msg = ''.join(
        f'{index}. {name.title()}\n'
        for index, name in CLASSES.items()
        if index != '0'
    )

    bot.reply_to(message, reply_msg, reply_markup=KEYBOARD)

@bot.message_handler(commands=[COMMANDS.PREDICT])
def predict(message):
    LOGGER.info(f'{message.chat.id} -> /predict')
    bot.reply_to(message, 'Send me a photo')
    bot.register_next_step_handler(message, predict_step)

def get_image(message):
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    return bot.download_file(file_info.file_path)

def predict_step(message):
    LOGGER.info(f'{message.chat.id} -> predict_step')

    if message.content_type == 'photo':
        image = get_image(message)

        predictor.predict(image)
        reply_msg = predictor.get_message()

        bot.reply_to(message, reply_msg)

    else:
        bot.reply_to(message, 'Not a photo. Send me a photo')
        bot.register_next_step_handler(message, predict_step)


@bot.message_handler(content_types=['photo'])
def got_photo(message):
    LOGGER.info(f'{message.chat.id} -> photo')
    predict_step(message)

@bot.message_handler(commands=['stop'])
def stop(message):
    if int(message.chat.id) == int(OWNER):
        LOGGER.warning('Exiting')
        bot.send_message(OWNER, 'Got stop command. Exiting...')
        sys.tracebacklimit = 0
        raise Exit

@bot.message_handler(content_types=['text'])
def got_text(message):
    LOGGER.info(f'{message.chat.id} -> text')
    bot.reply_to(message, MESSAGES.NOT_A_COMMAND)

if __name__ == '__main__':
    bot.send_message(OWNER, 'Bot started')
    bot.polling()