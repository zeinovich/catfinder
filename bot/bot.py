import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision.transforms import Resize, Normalize
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF

import telebot

from datetime import datetime
import json
import os
from types import SimpleNamespace
import logging
import numpy as np

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

def init_mnet2() -> nn.Module:
    init_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT, progress=True)
    logging.debug('Loaded init_mnetv2')

    init_model.classifier =nn.Sequential(
                            nn.Dropout(p=0.2, inplace=False),
                            nn.Linear(in_features=1280, out_features=1024),
                            nn.ReLU6(inplace=True),
                            nn.Linear(in_features=1024, out_features=37))
    
    logging.debug(f'Classifier set to {init_model.classifier}')

    state_dict = torch.load(config.M_MODEL_PATH, map_location=device)['model_state_dict']
    logging.debug('State_dict loaded')

    init_model.load_state_dict(state_dict=state_dict)
    logging.debug('Model got state_dict')

    for param in init_model.parameters():
        param.requires_grad = False

    logging.debug('Req_grad set to False')

    return init_model

def preprocess_torch(image_path='image_buff/image.jpg') -> torch.Tensor:
    image = Image.open(image_path)
    logging.debug(f'Opened image at {image_path}')

    x = TF.to_tensor(image)
    logging.debug('Converted to tensor')

    x = Resize(config.IMG_SIZE)(x)
    logging.debug('Resized image')
    
    x = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x)
    logging.debug('Normalized image')

    x.unsqueeze_(0)
    logging.debug(f'Unsqueezed shape: {x.shape} ({type(x)})')

    return x

def multiclass_predict(model: nn.Module, input_image: torch.Tensor) -> tuple[float]:

    input_image.to(device)
    logging.info('Got input')
    logging.debug(f'Input img moved to {device}')

    output = model(input_image)
    logging.info('Got output from model')
    logging.debug(f'Output: {output}')
    
    softmax = torch.softmax(output, 1)
    logging.info('Applied softmax on output')
    logging.debug(f'Softmax output: {softmax}')

    proba, pred = torch.max(softmax, 1)
    logging.info('Got proba and pred from torch.max')
    logging.debug(f'proba: {proba}  pred: {pred}')
    logging.debug(f'(*detached) proba: {proba.detach().numpy()[0]}  pred: {pred.detach().numpy()[0]}')
    
    return proba.detach().numpy()[0], pred.detach().numpy()[0]

def get_bboxes(input_image, class_name):
    pass

def get_prediction_message(proba: float, class_name: str) -> str:
    msg = f"I think it's {class_name} ({proba * 100:.2f}%)"
    logging.debug(f'Pred message: {msg}')

    return msg

def main():
    logger = get_logg()
    logger.info('Logger initialized')
    
    logger.info(f'Dir: {os.getcwd()}')

    global config
    config = load_config()
    logger.info('Config initialized')
    logger.debug(f'Config: {config.IMG_SIZE, config.M_MODEL_PATH, config.CLASSES}')

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device: {device}')

    classes = get_classes()
    logging.info(f'Classes are read')
    logging.debug(f'Classes: {classes}')

    bot = telebot.TeleBot(config.BOT_TOKEN)
    logging.info('Bot initialized')

    model_multiclass = init_mnet2()
    logging.debug('Model is read')

    model_multiclass.to(device)
    logging.debug(f'Model is moved to {device}')

    model_multiclass.eval()
    logging.info('Multiclass initialized')
    logging.debug('Model is set to eval mode')

    #yolov8 = object()
    #logging.info('YOLO initialized')

    #BOT CODE
    @bot.message_handler(commands=['start'])
    def send_welcome(message):
        bot.reply_to(message, "Welcome to the bot")

    @bot.message_handler(commands=['predict'])
    def got_predict(message):
        bot.reply_to(message, "Send a picture")

    @bot.message_handler(content_types=['photo'])
    def predict(message):
        logging.info('Bot got image')
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)

        logging.info('Downloading image')
        image = bot.download_file(file_info.file_path)

        with open('image_buff/image.jpg', 'wb') as f:
            f.write(image)
        
        logging.info('Image downloaded')

        image = preprocess_torch()
        logging.info('Image preprocessed (pytorch)')
        logging.debug(f'Image shape: {image.shape}')

        with torch.no_grad():
            logging.debug('no_grad set')

            logging.info('Predicting (pytorch)')
            m_proba, m_pred = multiclass_predict(model_multiclass, image)

            #yolov8.predict & box (label from multiclass)
            class_name = classes[m_pred]
            logging.debug(f'class_name: {class_name}')

            logging.info('Bot sending message')
            bot.reply_to(message, get_prediction_message(m_proba, class_name))
            logging.info('Message sent')

    bot.polling()

if __name__ == '__main__':
    main()