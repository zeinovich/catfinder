from importlib import import_module

from keras.api._v2 import keras as KerasAPI
keras: KerasAPI = import_module('tensorflow.keras')

import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF

import telebot

import json
import os
from types import SimpleNamespace
import numpy as np

def load_config() -> dict:
    with open('config.json') as config_file:
        return SimpleNamespace(**json.load(config_file))

def get_classes() -> dict:
    with open(config.CLASSES) as f:
        return json.load(f)

def init_mnet2() -> nn.Module:
    init_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT, progress=True)

    init_model.classifier =nn.Sequential(
                            nn.Dropout(p=0.2, inplace=False),
                            nn.Linear(in_features=1280, out_features=1024),
                            nn.ReLU6(inplace=True),
                            nn.Linear(in_features=1024, out_features=37))
    
    state_dict = torch.load(config.M_MODEL_PATH)['model_state_dict']
    init_model.load_state_dict(state_dict=state_dict)
    return init_model

def preprocess_keras(image_path='image_buff/image.jpg') -> np.ndarray:
    image = keras.preprocessing.image.load_img(image_path)
    image = keras.preprocessing.image.smart_resize(image, (config.IMG_SIZE, config.IMG_SIZE))
    image = keras.preprocessing.image.img_to_array(image)
    image = image / 255.0
    return image.reshape(1, config.IMG_SIZE, config.IMG_SIZE, 3)

def preprocess_torch(image_path='image_buff/image.jpg') -> torch.Tensor:
    image = Image.open(image_path)
    x = TF.to_tensor(image)
    x.unsqueeze_(0)
    return x

def multiclass_predict(model: nn.Module, input_image: np.ndarray) -> tuple[float]:
    input_image = torch.from_numpy(input_image)
    input_image.to(device)
    output = model(input_image)
    proba, pred = torch.max(torch.log_softmax(output, 1), 1)

    return proba.detach().numpy()[0], pred.detach().numpy()[0]

def get_bboxes(input_image, class_name):
    pass

def get_prediction_message(proba: float, class_name: str) -> str:
    return f"I think it's {class_name} ({proba:.2f}%)"

def get_negative_message(prediction: float) -> str:
    return f"There's no cat with {(prediction) * 100:.2f}% probability\n"

def main():
    print(os.getcwd())

    global config
    config = load_config()

    classes = get_classes()
    bot = telebot.TeleBot(config.BOT_TOKEN)

    model_binary = keras.models.load_model(config.B_MODEL_PATH)
    model_multiclass = init_mnet2()
    yolov8 = object()

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_multiclass.to(device)

    #BOT CODE
    @bot.message_handler(commands=['start'])
    def send_welcome(message):
        bot.reply_to(message, "Welcome to the bot")

    @bot.message_handler(commands=['predict'])
    def got_predict(message):
        bot.reply_to(message, "Send a picture")

    @bot.message_handler(content_types=['photo'])
    def predict(message):
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        image = bot.download_file(file_info.file_path)

        with open('image_buff/image.jpg', 'wb') as f:
            f.write(image)
        
        image = preprocess_keras()

        b_pred = model_binary.predict(image)[0][0]

        if b_pred > 0.5:
            image = preprocess_torch()
            m_proba, m_pred = multiclass_predict(model_multiclass, image)
            #yolov8.predict & box (label from multiclass)
            class_name = classes[m_pred]
            bot.reply_to(message, get_prediction_message(m_proba, class_name))
            #send a photo with box
        else:
            bot.reply_to(message, get_negative_message(1 - b_pred))

    bot.polling()

if __name__ == '__main__':
    main()