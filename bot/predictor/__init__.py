from torchvision.models import efficientnet_b2
from bot import MODEL_PATH, LOGGER, CLASSES
from bot.predictor.model import Predictor

model = efficientnet_b2()
predictor = Predictor(model=model, 
                      model_name='EfficientNet_B2', 
                      model_path=MODEL_PATH,
                      classes=CLASSES)

LOGGER.info(f'{predictor=}')