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
import numpy as np

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

    init_model.classifier =nn.Sequential(
                            nn.Dropout(p=0.2, inplace=False),
                            nn.Linear(in_features=1280, out_features=1024),
                            nn.ReLU6(inplace=True),
                            nn.Linear(in_features=1024, out_features=37))
    
    state_dict = torch.load(config.M_MODEL_PATH, map_location=device)['model_state_dict']
    init_model.load_state_dict(state_dict=state_dict)

    for param in init_model.parameters():
        param.requires_grad = False
    return init_model

def preprocess_torch(image_path='image_buff/image.jpg') -> torch.Tensor:
    image = Image.open(image_path)
    x = TF.to_tensor(image)
    x = Resize(224)(x)
    x = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x)
    x.unsqueeze_(0)
    return x

def multiclass_predict(model: nn.Module, input_image: np.ndarray) -> tuple[float]:
    #input_image = torch.from_numpy(input_image)
    input_image.to(device)
    output = model(input_image)
    print(f'[INFO] {datetime.now()} Output: {output}')
    softmax = torch.softmax(output, 1)
    print(f'[INFO] {datetime.now()} sotfmax: {softmax}')

    proba, pred = torch.max(softmax, 1)
    print(f'[INFO] {datetime.now()} proba: {proba}  pred: {pred}')
    print(f'[INFO] {datetime.now()} proba: {proba.detach().numpy()[0]}  pred: {pred.detach().numpy()[0]}')
    return proba.detach().numpy()[0], pred.detach().numpy()[0]

def get_bboxes(input_image, class_name):
    pass

def get_prediction_message(proba: float, class_name: str) -> str:
    return f"I think it's {class_name} ({proba * 100:.2f}%)"

def main():
    print(os.getcwd())

    global config
    config = load_config()
    print(f'[INFO] {datetime.now()} Config initialized')

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] {datetime.now()} Device: {device}')

    classes = get_classes()
    print(f'[INFO] {datetime.now()} Classes: {classes}')
    bot = telebot.TeleBot(config.BOT_TOKEN)
    print(f'[INFO] {datetime.now()} Bot initialized')

    model_multiclass = init_mnet2()
    model_multiclass.to(device)     
    model_multiclass.eval()
    print(f'[INFO] {datetime.now()} Multiclass initialized')

    #yolov8 = object()
    #print(f'[INFO] {datetime.now()} YOLO initialized')

    #BOT CODE
    @bot.message_handler(commands=['start'])
    def send_welcome(message):
        bot.reply_to(message, "Welcome to the bot")

    @bot.message_handler(commands=['predict'])
    def got_predict(message):
        bot.reply_to(message, "Send a picture")

    @bot.message_handler(content_types=['photo'])
    def predict(message):
        print(f'[INFO] {datetime.now()} Got image')
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)

        print(f'[INFO] {datetime.now()} Downloading image')
        image = bot.download_file(file_info.file_path)

        with open('image_buff/image.jpg', 'wb') as f:
            f.write(image)
        
        print(f'[INFO] {datetime.now()} Image downloaded')

        image = preprocess_torch()
        print(f'[INFO] {datetime.now()} Image preprocessed (pytorch)')

        with torch.no_grad():
            print(f'[INFO] {datetime.now()} Predicting (multi)')
            m_proba, m_pred = multiclass_predict(model_multiclass, image)
            print(f'[INFO] {datetime.now()} Multiclass prediction: {m_pred} ({m_proba:.2f})')

            #yolov8.predict & box (label from multiclass)
            class_name = classes[m_pred]
            print(f'[INFO] {datetime.now()} class_name: {class_name}')

            print(f'[INFO] {datetime.now()} Bot sending message')
            bot.reply_to(message, get_prediction_message(m_proba, class_name))
            print(f'[INFO] {datetime.now()} Message sent')

    bot.polling()

if __name__ == '__main__':
    main()