from importlib import import_module
from keras.api._v2 import keras as KerasAPI
keras: KerasAPI = import_module('tensorflow.keras')
import telebot
import json
import os
from types import SimpleNamespace
import matplotlib.pyplot as plt
import numpy as np

def load_config():
    with open('config.json') as config_file:
        return SimpleNamespace(**json.load(config_file))

def get_classes():
    with open('classes/classes.json') as f:
        return json.load(f)

def get_random_crops(image, crop_size, num_crops=5):
    crops = []
    for _ in range(num_crops):
        x = np.random.randint(0, image.shape[0] - crop_size)
        y = np.random.randint(0, image.shape[1] - crop_size)
        image_ = image[x:x + crop_size, y:y + crop_size]
        crops.append(keras.preprocessing.image.smart_resize(image_, (config.IMG_SIZE, config.IMG_SIZE)))
    return crops

def preprocess(image_path='image_buff/image.jpg'):
    image = keras.preprocessing.image.load_img(image_path)
    image = keras.preprocessing.image.smart_resize(image, (config.IMG_SIZE, config.IMG_SIZE))
    image = keras.preprocessing.image.img_to_array(image)
    image = image / 255.0
    return image.reshape(1, config.IMG_SIZE, config.IMG_SIZE, 3)

def get_prediction_message(prediction, max_class):
    message = 'Best 5 predictions:\n\n'
    for i in range(len(max_class)):
        message += f'{i+1}. {max_class[i]}: {prediction[i] * 100:.2f}%\n'
    return message

def main():
    print(os.getcwd())

    global config
    config = load_config()

    global classes
    classes = get_classes()
    bot = telebot.TeleBot(config.BOT_TOKEN)

    model = keras.applications.MobileNetV2(input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3), include_top=True, weights='imagenet')

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
        
        image = preprocess()

        prediction = model.predict(image)[0]
        best5_classes  = np.argpartition(prediction, -5)[-5:]
        max_classes = [classes[str(i)] for i in best5_classes]
        prediction = prediction[best5_classes]

        bot.reply_to(message, get_prediction_message(prediction, max_classes))

    bot.polling()

if __name__ == '__main__':
    main()