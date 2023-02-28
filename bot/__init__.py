import logging
from os import getenv
import telebot
from dotenv import load_dotenv
from datetime import datetime 
from platform import system, release
from socket import gethostname, gethostbyname
import json

load_dotenv('./config.env')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
                logging.FileHandler(f'./logs/catfinder_{datetime.date(datetime.now())}.log'),
                logging.StreamHandler()
             ]
)

LOGGER = logging.getLogger(__name__)

LOGGER.info('Starting bot')

TOKEN = getenv('TOKEN')
OWNER = getenv('OWNER')
MODEL_PATH = getenv('MODEL_PATH')
CLASSES_PATH = getenv('CLASSES_PATH')
LOGGER.info(f'{MODEL_PATH}')
LOGGER.info(f'{CLASSES_PATH}')

with open(CLASSES_PATH, 'r') as f:
    CLASSES = json.load(f)

ip_addr = gethostbyname(gethostname())

bot = telebot.TeleBot(TOKEN)
bot.send_message(OWNER, f'{datetime.now()} \n{system()} {release()} \nIP: {ip_addr}')

LOGGER.info(f'Initialized {system()} {release()} IP: {ip_addr}')