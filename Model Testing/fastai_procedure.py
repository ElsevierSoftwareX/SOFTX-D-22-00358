
##############################################################3
################ This file is used to train the model.The code is written in Python and models is trianed using FastAIv2 Framework.

#####################################################################################################
###### Import Libraries
import albumentations
from fastai.vision.all import *
from fastai.metrics import accuracy_multi
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.nn.functional as F
import argparse
from fastbook import *
#####################################################################################################

"""Training And Testing Transformations"""
def get_train_aug(): return albumentations.Compose([
     albumentations.RandomResizedCrop(224,224),
            albumentations.Transpose(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.ShiftScaleRotate(p=0.5),
            albumentations.HueSaturationValue(
                hue_shift_limit=0.2,
                sat_shift_limit=0.2,
                val_shift_limit=0.2,
                p=0.5
            ),
            albumentations.CoarseDropout(p=0.2),
            albumentations.Cutout(p=0.2)
])





class AlbumentationsTransform(Transform):
    def __init__(self, aug): self.aug = aug
    def encodes(self, img: PILImage):
        aug_img = self.aug(image=np.array(img))['image']
        return PILImage.create(aug_img)


"""FastAI DataBlock"""
def fake_datablock(tfms):
    db1 = DataBlock(blocks=(ImageBlock, CategoryBlock),
                       get_items=get_image_files,
                       
                      splitter=GrandparentSplitter(),
                       get_y=[parent_label],
                       item_tfms=Resize((224,224)),
                       batch_tfms=tfms
                   )
    return db1

"""Evaluation Metrics"""
def metrics_use():
    Precision = Precision()
    Recall = Recall()
    F1Score = F1Score()
    RocAuc = RocAuc()

    return Precision,RandomVerticalFlip,F1Score,RocAuc

"""Counting Number Of GPU"""
def count_gpus(learn):
    if torch.cuda.device_count() > 1:
        learn.model = nn.DataParallel(learn.model)
    return learn.model



"""Training the Model"""
def train(path,batch_size=32,epochs=5,model_2=None):
    tfms = [AlbumentationsTransform(get_train_aug()),DihedralItem(p=0.15),
            CropPad((224,224),pad_mode=PadMode.Reflection),Rotate(max_deg=15,p=0.25,pad_mode='reflection'),
            Saturation(p=0.25),RandomErasing(p=0.08,sh=0.15,max_count=3)]

    db1 = fake_datablock(tfms)##### Defining the DataBlock
    dls = db1.dataloaders(path,bs=batch_size)##### Loading the DataLoader
    opt_func = partial(Adam, lr=1e-5, wd=0.01, eps=1e-8)####### Defining the Learning Rate and Optimizer
    learn = Learner(dls, model_2, opt_func = opt_func,metrics=[accuracy,Precision,Recall,F1Score])###### Defining the learner
    #lr_min,_ = learn.lr_find()
    lr_min = 0.004
    """Fitting the Model Using One Cycle Method"""
    learn.fit_one_cycle(epochs, lr_max=slice(lr_min), cbs=[SaveModelCallback(monitor="accuracy")])

    model =learn.model
    ###Saving the Model
    torch.save(model, "model_d_94.pth")

"""Defining the Model Architectures"""
class model_architectures():


    def __init__(self,model_id):

        self.model_id=model_id
        self.model=None



    def efficientnet_architectures(self):

        if(self.model_id==1):
            self.model = EfficientNet.from_pretrained("efficientnet-b1", advprop=True)
            self.model._fc = nn.Linear(1280, 2)

        elif(self.model_id==2):
            self.model = EfficientNet.from_pretrained("efficientnet-b2", advprop=True)
            self.model._fc = nn.Linear(1408, 2)

        elif(self.model_id==3):
            self.model = EfficientNet.from_pretrained("efficientnet-b3", advprop=True)
            self.model._fc = nn.Linear(1536, 2)

        elif(self.model_id==4):
            self.model = EfficientNet.from_pretrained("efficientnet-b4", advprop=True)
            self.model._fc = nn.Linear(1792, 2)
        elif(self.model_id==5):
            self.model = EfficientNet.from_pretrained("efficientnet-b5", advprop=True)
            self.model._fc = nn.Linear(2048, 2)

        elif(self.model_id==6):
            self.model = EfficientNet.from_pretrained("efficientnet-b6", advprop=True)
            self.model._fc = nn.Linear(2304, 2)

        else:
            self.model = EfficientNet.from_pretrained("efficientnet-b7", advprop=True)
            self.model._fc = nn.Linear(2560, 2)

        return self.model




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extracts crops from video")
    parser.add_argument("--model-id", help="Efficient-Net Version",nargs="?",type=int,default=3)
    parser.add_argument("--batch-size", help="Batch Size",nargs="?",type=int,default=16)
    parser.add_argument("--epochs", help="Epochs",nargs="?",type=int,default=5)
    parser.add_argument("--path", help="Dataset Path",nargs="?",type=str,default="F")
    parser.add_argument("--model-save-to", help="Directory To Save Model",nargs="?",type=str,default="F")



    args = parser.parse_args()

    if(args.path=="F"):
        print("xxxxxxxxxx---------Path To Dataset Not Given---------xxxxxxxxxx")
        exit()

    if(args.model_save_to=="F"):
        args.model_save_to=os.path.join(str("Saved_Models"))
        os.makedirs(args.model_save_to, exist_ok=True)

    else:
        os.makedirs(args.model_save_to, exist_ok=True)
        args.output_direc=args.model_save_to




    model = model_architectures(args.model_id)
    model = model.efficientnet_architectures()

    train(path=args.path,batch_size=args.batch_size,epochs=args.epochs,model_2=model)
