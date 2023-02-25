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
import socket
import platform

from predictor import Predictor

def get_logg(level=logging.INFO):

    log_formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
    
    log_file = f'./logs/catfinder_{datetime.date(datetime.now())}.log'

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)   
    file_handler.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(level)

    logger = logging.getLogger(name='zeinovich')
    logger.setLevel(level)

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def load_config() -> dict:
    with open('./config.json') as config_file:
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
    logging.warning('Not implemented')

def main():
    logger = get_logg()
    logger.info('Logger initialized')
    
    logger.info(f'Dir: {os.getcwd()}')

    global config
    config = load_config()
    logger.info('Config initialized')
    logger.debug(f'Config: {config.IMG_SIZE, config.MODEL_PATH, config.CLASSES}')

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'{device=}')

    classes = get_classes()
    logger.info(f'Classes are read')
    logger.debug(f'{classes=}')

    bot = telebot.TeleBot(config.BOT_TOKEN)
    logger.info('Bot initialized')

    base_model = efficientnet_b2()
    predictor = Predictor(base_model)

    logger.debug('Model is read')

    state_dict = get_state_dict()
    predictor.load_state_dict(state_dict)
    predictor.eval()
    logger.info(f'{predictor.input_size=}   {predictor.normalization=}')
    logger.debug('Model is set to eval mode')

    hostname = socket.gethostname()
    ip_addr = socket.gethostbyname(hostname)

    logger.info(f'Started {platform.system()} {platform.release()} IP: {ip_addr}')
    bot.send_message(config.BOT_OWNER, f'{datetime.now()} \n{platform.system()} {platform.release()} \nIP: {ip_addr}')

    #BOT CODE
    @bot.message_handler(commands=['start'])
    def send_welcome(message):
        logger.info(f'{message.chat.id} -> /start')
        bot.reply_to(message, "Welcome to the bot")

    @bot.message_handler(commands=['predict'])
    def got_predict(message):
        bot.reply_to(message, "Send a picture")

    @bot.message_handler(commands=['breeds'])
    def print_breeds(message):
        logger.info(f'{message.chat.id} -> /breeds')
        repl_text = ''
        for index, name in classes.items():
            if index == 0:
                continue

            repl_text += f'{index}. {name.title()}\n'

        bot.reply_to(message, repl_text)
        
    @bot.message_handler(content_types=['photo'])
    def got_photo(message):
        start = perf_counter()
        logger.info(f'{message.chat.id} -> photo')

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

        logger.info(f'ETA={(end - start) * 1000:.0f} ms')

    bot.polling()

if __name__ == '__main__':
    main()