# CATFINDER

Bot defines breed of a cat or dog from photo you send him.

    *In future, it will also detect and draw bounding boxes

![image](https://user-images.githubusercontent.com/114425094/220098595-ca8f44bb-ac40-4c3a-aa51-fa93ed388efd.png)

 

###   MODEL
For now base classification model is EfficientNet_B2 (weights=IMAGENET1K). Classifier is changed to 

    nn.Sequential(
                  nn.Dropout(p=0.3, inplace=True),
                  nn.Linear(in_features=1408, out_features=13, bias=True)
                 )
                 
NB. MobileNet_V2 performed equally.

### DATASET
  Breed classification dataset - OxfordIIITPet. It contains 37 breeds of cats and dogs. We use only 12 cat breeds. Another datasets are added in order to cover as much feature space as possible. Those are StanfordCars, CelebA, AnimalDataset, House_Indoors_Scenes. Their data represent 13th class (NEG). From each 100 datapoints are sampled and then augmented to get total 400 datapoints per dataset. OxfordIIITPet also augmented 3 times for each datapoint. 
  Augmentation is done by albumentations. Steps are 
  
    transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.PiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),            
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ])
  
### TRAINING
  Training is done in GoogleColab. Training consisted of 460 batches (BATCH_SIZE=32). Transformations are applied
  
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  
  Training parameters:
  
    optimizer = optim.AdamW(params=model.parameters(), lr=1e-3, weight_decay=0.1, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(adam, patience=3, factor=0.1, threshold=1e-2, min_lr=1e-9)
    criterion = nn.CrossEntropyLoss(label_smoothing=1e-2)
    callback = EarlyStopper(patience=10, min_delta=1e-2)
  
  Also checkpoint callback is used and best model is occassionally saved. Minimal val_loss delta of 1e-2 is taken.
  
  Training took 37 epochs (~100 mins). Results after training are
  
    min_loss = 0.3468
    best_acc = 0.91
    test_acc = 0.96
    See ./notebooks/training_torch_multiclass.ipynb for detailed classification report
    
   Training and validation accuracies are almost the same, so model neither overfit nor underfit. Logits that classifier outputs are seemingly normally distributed which tell us that our gradients don't die on a way. 
  
### DETECTION

    See detection here in future. YOLO_v8/v7 will be used
    
## BOT

  Bot is built using Telebot. When photo is sent to bot, function got_photo is called and photo is downloaded as bytes. Then Predictor object makes prediction.

### PREDICTOR
  Predictor class is quite simple. It holds model (nn.Module) and yolo (nn.Module) as a variables. Also input_shape and normalization for classification and model names. It's main fucntion is to get input, process and make prediction based on classes you pass. In addition it keeps latest prediction and probability that weren't proceeded to chat. 
  
  For comfort, methods load_state_dict() and eval() are written. 
  
    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict=state_dict)
    
    def eval(self):
        return self.model.eval()
        
  Methods 
  
    def __getmodel(self):
      return self.model
    
    def __getyolo(self):
      return self.yolo
      
  are used to infer models architecture.
 
### COMMANDS
   
   Bot Commands.
   
      /start - To send hello message
      /predict - Helper command just outputs message
      /help - Help
      /info - Info
  
  
## REQUIREMENTS
      
      python>=3.9
      
      BOT REQS
      numpy==1.23.5
      opencv_python==4.7.0.68
      torch==1.13.1
      torchvision==0.14.1

      NOTEBOOK REQS
      matplotlib==3.6.0
      seaborn==0.12.2
      scikit-learn==1.2.1
      tqdm==4.64.1
    
