import torch
from torchvision.transforms import Resize, Normalize
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
import logging

class Predictor():
    def __init__(self, model: nn.Module, model_name: str=None, 
                 yolo: nn.Module=lambda: None, yolo_version: str=None,
                 input_size=224, normalization: tuple[list[int]]=None):
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
        self.model_name = model_name
        self.yolo = yolo
        self.yolo_version = yolo_version
        self.input_size = input_size
        self.normalization = normalization

        if normalization is None:
            self.normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        logging.info(f'{self.input_size=}   {self.normalization=}')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Device: {self.device}')

        self.model.classifier = nn.Sequential(
                                              nn.Dropout(p=0.3, inplace=True),
                                              nn.Linear(in_features=1408, out_features=38, bias=True)
                                             )
        
        self._pred, self._proba = None, None

    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict=state_dict)
    
    def eval(self):
        return self.model.eval()

    @torch.no_grad()
    def preprocess(self, image) -> torch.Tensor:
        '''
        Preprocessing input image by applying Resizing and Normalization
        image: torch.Tensor or np.ndarray
        Returns: torch.Tensor 
        '''

        logging.info('Preprocessing')

        if not isinstance(image, torch.Tensor):
            x = TF.to_tensor(image)

        x = Resize(self.input_size)(x)
        x = Normalize(*self.normalization)(x)

        x.unsqueeze_(0)
        logging.debug(f'Unsqueezed shape: {x.shape} ({type(x)})')

        return x
    
    @torch.no_grad()
    def predict(self, input_image=(np.ndarray, torch.Tensor)) -> None:
        '''
        Makes prediction based on input_image
        input_image: np.ndarray or torch.Tensor
        Returns: None 

        *to get predictions call get_preds()
        '''
        logging.info('Predicting')
        input_image = self.preprocess(input_image)

        input_image.to(self.device)
        output = self.model(input_image)
        logging.debug(f'{output=}')
        
        softmax = torch.softmax(output, 1)
        logging.debug(f'{softmax=}')

        proba_pred = torch.max(softmax, 1)
        self._proba, self._pred = proba_pred[0].detach().cpu().numpy()[0], proba_pred[1].detach().cpu().numpy()[0]
        logging.debug(f'{self._proba=}  {self._pred=}')

    def get_preds(self):
        '''
        Returns prediction and probability 
        type: float
        '''
        return self._pred, self._proba

    def get_message(self, classes: dict) -> str:
        '''
        Returns message based on predictions and classes
        classes: dict
        Returns: str
        '''
        if self.pred is None:
            logging.warning('Calling get_message on None prediction')
            return

        class_name = classes[self.pred]

        if class_name != '<NEG>':
            msg = f"I think it's {class_name.title()} ({self.proba * 100:.2f}%)"

        else:
            msg = f"I can't see cat or dog of a breed I know ({self.proba * 100:.2f}%)"

        logging.debug(f'Pred message: {msg}')

        self.proba, self.pred = None, None #reset predictions

        return msg
    
    def get_bboxes(self, input_image):
        return self.yolo(input_image)
    
    def __getmodel(self):
        return self.model
    
    def __getyolo(self):
        return self.yolo
    
    def __repr__(self):
        yolo_msg = f'YOLO_{self.yolo_version}' if self.yolo != (lambda: None) else 'None'
        return f'Predictor({self.input_size=}, {self.normalization=}, {self.model_name=}, yolo={yolo_msg})'