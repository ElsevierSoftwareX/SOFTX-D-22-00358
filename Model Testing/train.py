

#################################################################
###### Train the Efficient-nets Models on a Fake And Real Faces Dataset

#####################################################################################################
###### Import Libraries
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torchvision.transforms as tt
import PIL

from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np
from fastai.vision.all import *
import albumentations
from augmentations_file import *
import torch
import torchvision
import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from data_parallel import *
from torch.utils.data.dataloader import DataLoader
from augmentations_file import *
import torch.nn as nn
import torch.nn.functional as F
from data_parallel import *
from utils import *
from efficientnet_pytorch import EfficientNet
import torchvision
import argparse
#####################################################################################################










""" Evaluate the Model on the Validation Dataset"""
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [validation_step(model,batch) for batch in val_loader]
    return validation_epoch_end(model,outputs)


""" Fit the Model on the Training Data"""
def fit(epochs, lr, model, train_loader, val_loader,opt_func=torch.optim.SGD,model_save_direc="Saved_Models"):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    d4=1
    PATH=model_save_direc
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        d7=1
        for batch in train_loader:
            images,labels = batch
            out = model(images)
            loss = F.cross_entropy(out, labels)
            train_losses.append(loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if d7 % 100 == 0:
                print(f'Epoch: {epoch}\t Batch: {int(d7/100)} \t[{100. * d7/ len(train_loader):.0f}%]\tLoss: {loss.item():.6f} \t{accuracy(out,labels)}')
            d7=d7+1
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        epoch_end(model,epoch, result)
        history.append(result)
        torch.save(model,os.path.join(PATH,str(d4)+"_d60.pth"))
        d4=d4+1

    return history


random_seed = 34
torch.manual_seed(random_seed);



##### Defining the Pytorch Dataset
def fake_dataset(tfms,data_dir="Datasets"):
    train_dataset = ImageFolder(os.path.join(data_dir,'train'), transform=tfms)
    valid_dataset = ImageFolder(os.path.join(data_dir,'valid'), transform=tfms)
    return train_dataset,valid_dataset

##### Define the Pytorch DataLoaders
def fake_dataloader(train_ds,val_ds,batch_size=1,device="cpu"):


    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size)
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    return train_dl,val_dl




"""Evaluation Metrics Used"""
def metrics_use():
    Precision = Precision()
    Recall = Recall()
    F1Score = F1Score()
    RocAuc = RocAuc()

    return Precision,RandomVerticalFlip,F1Score,RocAuc

"""Count Number of GPUs available"""
def count_gpus(learn):
    if torch.cuda.device_count() > 1:
        learn.model = nn.DataParallel(learn.model)
    return learn.model

"""Fit the Model on the training Dataset"""
def train(path,batch_size=1,epochs=5,model=None,device="cpu",model_save_direc="Saved_Models"):
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transforms = torchvision.transforms.Compose([
        ImgAugTransform(),
        lambda x: PIL.Image.fromarray(x),
        torchvision.transforms.RandomVerticalFlip(),
        tt.RandomHorizontalFlip(),
                             tt.RandomResizedCrop((224,224),interpolation=2),
                             tt.ToTensor(),
                             tt.Normalize(*stats,inplace=True)
                      ])

    train_ds,val_ds = fake_dataset(tfms=transforms,data_dir=path)
    train_dl,val_dl = fake_dataloader(train_ds,val_ds,batch_size=batch_size,device=device)

    num_epochs = epochs
    opt_func = torch.optim.Adam
    lr = 0.0004


    history = fit(epochs=epochs, lr=lr, model=model, train_loader=train_dl, val_loader=val_dl, opt_func=opt_func,model_save_direc=model_save_direc)


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
    parser.add_argument("--batch-size", help="Batch Size",nargs="?",type=int,default=1)
    parser.add_argument("--epochs", help="Epochs",nargs="?",type=int,default=5)
    parser.add_argument("--model-save-to", help="Directory To Save Model",nargs="?",type=str,default="F")



    args = parser.parse_args()


    if(args.model_save_to=="F"):
        args.model_save_to=os.path.join(str("Saved_Models"))
        os.makedirs(args.model_save_to, exist_ok=True)

    else:
        os.makedirs(args.model_save_to, exist_ok=True)



    model = model_architectures(args.model_id)#### Load the Efficient-Net Version
    model = model.efficientnet_architectures()
    device=get_default_device()
    model = to_device(model,device)##### Put model on the GPU if avilable
    model.to(device)
    path="Datasets"

    train(path=path,batch_size=args.batch_size,epochs=args.epochs,model=model,device=device,model_save_direc=args.model_save_to)
