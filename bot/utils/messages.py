class _Messages():
    def __init__(self):
        self.START = 'Hello! I am a bot that can predict the breed of cat from a photo.'
                        
        self.INFO = 'This bot is created to predict the breed of a dog or a cat from a photo.\
The bot uses a neural network to predict the breed. Neural network was trained on the datasets from Kaggle.\n\n\
For more see: www.github.com/zeinovich/catfinder'
        
        self.HELP = 'To predict the breed of a dog or a cat send a photo to the bot.\
The bot will predict the breed and send it to you. \n\n\
If you want to see the list of breeds send /breeds to the bot. \
If you want to know more about the bot send /info to the bot.'
        
        self.NOT_A_COMMAND = 'This is not a command. Send /help to the bot to see the list of commands.'
        
MESSAGES = _Messages()