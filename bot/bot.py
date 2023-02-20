import torch
from torchvision.models import efficientnet_b2
import cv2

import telebot

from datetime import datetime
from time import perf_counter
import json
import os
from types import SimpleNamespace
import logging
import numpy as np

from predictor import Predictor

def get_logg(level=logging.INFO):

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=f'C:/Users/nikit/catfinder/logs/log_{datetime.date(datetime.now())}.log'
    )

    logger = logging.getLogger(name='zeinovich')
    logger.setLevel(level)

    return logger

def load_config() -> dict:
    with open('config.json') as config_file:
        return SimpleNamespace(**json.load(config_file))

def get_classes() -> dict:
    with open(config.CLASSES) as f:
        classes = json.load(f)

    classes = {int(key): value for key, value in classes.items()}
    return classes

def get_state_dict():
    checkpoint = torch.load(config.MODEL_PATH, map_location=device)
    return checkpoint['model_state_dict']

def get_bboxes(input_image, class_name):
    pass

def main():
    logger = get_logg(logging.DEBUG)
    logger.info('Logger initialized')
    
    logger.info(f'Dir: {os.getcwd()}')

    global config
    config = load_config()
    logger.info('Config initialized')
    logger.debug(f'Config: {config.IMG_SIZE, config.MODEL_PATH, config.CLASSES}')

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device: {device}')

    classes = get_classes()
    logging.info(f'Classes are read')
    logging.debug(f'{classes=}')

    bot = telebot.TeleBot(config.BOT_TOKEN)
    logging.info('Bot initialized')

    base_model = efficientnet_b2()
    predictor = Predictor(base_model)

    logging.debug('Model is read')

    state_dict = get_state_dict()
    predictor.model.load_state_dict(state_dict)
    predictor.model.eval()
    
    logging.info('Model initialized')
    logging.debug('Model is set to eval mode')

    #BOT CODE
    @bot.message_handler(commands=['start'])
    def send_welcome(message):
        bot.reply_to(message, "Welcome to the bot")

    @bot.message_handler(commands=['predict'])
    def got_predict(message):
        bot.reply_to(message, "Send a picture")

    @bot.message_handler(commands=['breeds'])
    def print_breeds(message):
        repl_text = ''
        for index, name in classes.items():
            if index == 0:
                continue

            repl_text += f'{index}. {name.title()}\n'

        bot.reply_to(message, repl_text)
        
    @bot.message_handler(content_types=['photo'])
    def predict(message):
        start = perf_counter()
        logging.info(f'Got message in chat {message.chat.id}')
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)

        image = bot.download_file(file_info.file_path)
        image = np.asarray(bytearray(image), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        predictor.predict(image)

        reply_msg = predictor.get_message(classes)
        bot.reply_to(message, reply_msg)
        end = perf_counter()
        logging.info(f'Prediction took {(end - start) * 1000:.0f} ms')
        
    bot.polling()

if __name__ == '__main__':
    main()