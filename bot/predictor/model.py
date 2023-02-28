import torch
from torchvision.transforms import Resize, Normalize
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
from bot import LOGGER
from cv2 import cvtColor, COLOR_BGR2RGB, imdecode, IMREAD_COLOR

class Predictor():
    def __init__(self, 
                 model: nn.Module,
                 model_name: str='model',
                 model_path: str='None',
                 yolo: nn.Module=lambda: None, 
                 yolo_version: str=None,
                 input_size=224, 
                 normalization: tuple[list[int]]=None,
                 classes: dict[int, str]=None):
        '''
        Initialize Predictor class.
        model: classification model (nn.Module)
        model_name: name of classification model
        yolo: detection model of type YOLO (nn.Module)
        yolo_version: version of yolo (i.e 'v8')
        input_size: int(default=224)
        Normalization: tuple[list[int]] - list of [mean, std], where mean and std are lists of floats (1 float per channel) 
            By default: ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        '''

        self.model = model

        if self.model is None:
            raise ValueError("Model can't be None")
        
        self.model_name = model_name
        self.model_path = model_path

        if self.model_path is None:
            LOGGER.warning(f'Using {self.model_name} that is untrained')

        self.yolo = yolo
        self.yolo_version = yolo_version
        self.input_size = input_size
        self.normalization = normalization
        self.classes = classes

        self.fan_out = len(classes)

        if normalization is None:
            self.normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.model.classifier = nn.Sequential(
                                              nn.Dropout(p=0.3, inplace=True),
                                              nn.Linear(in_features=1408, out_features=self.fan_out, bias=True)
                                             )
        
        self.load_state_dict()
        LOGGER.info(f'Initialized {self.model_name} on {self.device}')

        self.model.eval()
        
        self._pred, self._proba = None, None

    def load_state_dict(self):
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)['model_state_dict']
            return self.model.load_state_dict(state_dict=state_dict)
        
        except Exception as e:
            LOGGER.error(f'Failed to load state dict: {e}')
            return None
    
    def eval(self):
        return self.model.eval()

    @torch.no_grad()
    def preprocess(self, image: bytes) -> torch.Tensor:
        '''
        Preprocessing input image by applying Resizing and Normalization
        image: torch.Tensor or np.ndarray
        Returns: torch.Tensor 
        '''
        try:
            if isinstance(image, bytes):
                image = np.asarray(bytearray(image), dtype="uint8")
                image = imdecode(image, IMREAD_COLOR)
                image = cvtColor(image, COLOR_BGR2RGB)

            if not isinstance(image, torch.Tensor):
                x = TF.to_tensor(image)

            x = Resize(self.input_size)(x)
            x = Normalize(*self.normalization)(x)

            x.unsqueeze_(0)

            return x
        
        except Exception as e:
            LOGGER.error(f'Failed to preprocess image: {e}')
            return None
    
    @torch.no_grad()
    def predict(self, input_image=(np.ndarray, torch.Tensor)) -> None:
        '''
        Makes prediction based on input_image
        input_image: np.ndarray or torch.Tensor
        Returns: None 

        *to get predictions call get_preds()
        '''
        
        input_image = self.preprocess(input_image)

        if input_image is None:
            LOGGER.error('Input image is None')
            return
        
        input_image.to(self.device)
        output = self.model(input_image)
        LOGGER.debug(f'{output=}')
        
        softmax = torch.softmax(output, 1)
        LOGGER.debug(f'{softmax=}')

        proba_pred = torch.max(softmax, 1)
        self._proba, self._pred = proba_pred[0].detach().cpu().numpy()[0], proba_pred[1].detach().cpu().numpy()[0]
        self._pred = str(self._pred)
        LOGGER.debug(f'{self._proba=}  {self._pred=}')

    def get_preds(self) -> tuple[float, float]:
        '''
        Returns prediction and probability 
        type: float
        '''
        return self._pred, self._proba

    def get_message(self) -> str:
        '''
        Returns message based on predictions and classes
        classes: dict
        Returns: str
        '''

        if self._pred is None:
            LOGGER.warning('Calling get_message on None prediction')
            return

        class_name = self.classes[self._pred]

        if class_name != '0neg':
            msg = f"I think it's {class_name.title()} ({self._proba * 100:.2f}%)"

        else:
            msg = f"I can't see cat or dog of a breed I know ({self._proba * 100:.2f}%)"

        LOGGER.debug(f'Pred message: {msg}')

        self.proba, self.pred = None, None #reset predictions

        return msg
    
    def get_bboxes(self, input_image):
        return self.yolo(input_image)
    
    def __getmodel(self):
        return self.model
    
    def __getyolo(self):
        return self.yolo
    
    def __repr__(self):
        yolo_msg = 'NA' if self.yolo is None else f'YOLO_{self.yolo_version}'
        return f'Predictor({self.model_name=}, {self.input_size=}, {self.normalization=}, yolo={yolo_msg})'