
#####################################################################3
########## Use this file to test whether a given image conatin Fake Faces Or Not


#####################################################################################################
###### Import Libraries

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from facenet_pytorch.models.mtcnn import MTCNN
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tqdm.notebook import tqdm

import PIL.Image as Image
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from torch import functional as Func


from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np

import torchvision


from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as tt

from data_parallel import *
from augmentations_file import *
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import time
import copy
import pickle as pk
import os
import torch.nn.functional as F

#####################################################################################################


###### Testing Transformations


aug_3 = A.Compose({
         A.RandomResizedCrop(224,224)
        })






"""Outputs an image containg red bounding box around fake faces and green around real faces"""
def fake_or_real(model=None,image_path="F",device="cpu",crop=False):
    if(model==None):
        print("Load Model Correctly !!!! ")
        exit()

    d21=0
    frame_4= cv2.imread(os.path.join("test_images",image_path))### Load Image
    ### If cropping is not required
    if(crop==False):

        image = plt.imread(os.path.join("test_images",image_path))#### Load Image
        image = Image.fromarray(image).convert('RGB')#### Convert Image to RGB
        image = aug_3(image=np.array(image))['image']##### Apply Transformations
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float)###### Convert Image to Tensor
        pred = model(image.unsqueeze(0).to(device)).argmax()#### Pass through Model
        

        if(pred==1):
            label="Fake"
        else:
            label="Real"

        color = (0, 0, 255) if label=="Fake" else (0, 255, 0)

        print(label + " "+ "Face Found")
        exit()


    mtcnn = MTCNN(keep_all=True,min_face_size=176,thresholds=[0.85,0.90,0.90],device=device)#### Load Face Detector
    frame = cv2.cvtColor(frame_4, cv2.COLOR_BGR2RGB)
    frame_4 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(frame.astype(np.uint8))


    frame=img
    k=0

    # Detect face
    try:
        boxes,landmarks = mtcnn.detect(frame, landmarks=False)##### Detect Faces
        b1 = img
    #print(boxes)
    except:
        print("Couldn't Find a Face!!!!")##### If mtcnn fails,this print statement is executed
        exit()
   ###### Loop over all faces
    try:
        for i in range(0,len(boxes)):
            #### Crop Faces from Images
            x,y,width,height =boxes[i]
            max_width,max_height = b1.size
            boxes[i][0]-=width/10
            boxes[i][1]-=height/10
            boxes[i][0] = max(0,boxes[i][0])
            boxes[i][1] = max(0,boxes[i][1])
            boxes[i][2] =min(boxes[i][2]+(width/10),max_width)
            boxes[i][3] =min(boxes[i][3]+(height/10),max_height)
            b4 = b1.crop(boxes[i])
            b4.save("test_face_image.jpg")##### Save Cropped Image


            image = plt.imread("test_face_image.jpg")#### Read Cropped Image
            image = Image.fromarray(image).convert('RGB')#### Convet to RGB
            image = aug_3(image=np.array(image))['image']##### Apply Transformations
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image = torch.tensor(image, dtype=torch.float)##### Convert to tensor
            pred = model(image.unsqueeze(0).to(device)).argmax()##### Pass thorugh The Model
            print(pred)

            if(pred==1):
                label="Fake"
            else:
                label="Real"

            color = (0, 0, 255) if label=="Fake" else (0, 255, 0)
            cv2.rectangle(frame_4, (boxes[i][0],boxes[i][1]), (boxes[i][2],boxes[i][3]), color, 2)##### Draw Bounding Box
            cv2.imwrite(os.path.join("test_faces_output",image_path),frame_4)##### Save the Image
            print(label + " "+ "Face Found")
    except:
        print("-----------------***********No Faces Found***********-----------------")
        return










if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Predict Face is Fake Or Real")
    parser.add_argument("--image-name", help="Name of the Image",nargs="?",type=str,default="F")
    parser.add_argument("--crop", help="If cropping is required,input True,else False",nargs="?",type=str,default="F")




    args = parser.parse_args()

    if(args.image_name=="F"):
        print("xxxxxxxxxx---------Path To Image Not Given---------xxxxxxxxxx")
        #exit()
    if(args.crop=="F" or args.crop=="f" or args.crop=="no" or args.crop=="No" or args.crop=="False" or args.crop=="false"):
        args.crop=False
    else:
        args.crop=True

    if(not os.path.isfile(os.path.join("test_images",args.image_name))):
        print('Image does not exist in the folder')
        #exit()     


    model=None
    device=get_default_device()
    print("------------------------Loading Model------------------------")
    model= torch.load(os.path.join("Trained_Models","Fake_Face_Detection_Model.pth"), map_location=device)
    model.to(device)##### Put Model on the CPU/GPU
    model.eval()##### Start the Evaluation Mode for the Pytorch Model
    if(args.crop==False):
        print("No Cropping Done")
    else:
        print("Cropping is Done")


    
    #fake_or_real(model=model,image_path=args.image_name,device=device,crop=args.crop)##### Call the Function
    #load trained model
    

    #trainedModel = torch.load("Resnet_SelfAttentiom.pth", map_location="cpu")
    #for param in trainedModel.parameters():
        #param.requires_grad = False

    image_transforms = transforms.Compose([
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))#(0.5, 0.5, 0.5), (0.5, 0.5, 0.5) 
    ])
def convertImageToTensor1(imagePath):
  image = Image.open(imagePath)
  imageTensor = image_transforms(image).float()
  imageTensor = Variable(imageTensor,requires_grad=False)
  imageTensor = imageTensor.unsqueeze(0)# converted in the form of batch
  return imageTensor

def convertImageToTensor(imagePath):
    image = cv2.imread(imagePath)  
    image = Image.fromarray(image).convert('RGB')#### Convert Image to RGB
    image = aug_3(image=np.array(image))['image']##### Apply Transformations
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image, dtype=torch.float)###### Convert Image to Tensor
    #pred = model(image.unsqueeze(0).to(device)).argmax()
    return image

'''

imageTensor = convertImageToTensor("D:\\Final_Code\\Fake_Images_And_Video_Detection\\DeepFake_Detection_Code_Directory\\Training\\test_images\\real (75).jpg")
#pred = testImage(trainedModel,imageTensor)[0]
#pred = model(imageTensor).argmax()
pred = model(imageTensor.unsqueeze(0).to(device)).argmax()

print(pred.item())
'''
   #load images
fake_images_path = "C:\\Users\\Surbhi\\PycharmProjects\\PCA_Faces\\Faces_5K\\valid\\fake"
fakeImagesList = os.listdir(fake_images_path)
#print("\n Fake:")
#print(fakeImagesList)
fakeTrueLabel = [0]*len(fakeImagesList)
#print(fakeTrueLabel)
real_images_path = "C:\\Users\\Surbhi\\PycharmProjects\\PCA_Faces\\Faces_5K\\valid\\real"
   
realImagesList = os.listdir(real_images_path)
#print("\n Real:")
#print(realImagesList)
realTrueLabel = [1]*len(realImagesList)
#print("\n Real True Label: ", realTrueLabel)


pred_labels=[]
true_labels=[]
image_names=[]
for image in fakeImagesList:
    imageTensor = convertImageToTensor(fake_images_path+"\\"+image)
    #pred = testImage(trainedModel,imageTensor)[0]
    pred = model(imageTensor.unsqueeze(0).to(device)).argmax()
    pred = pred.item()
    ptint(pred)
    true_labels.append(1)
    pred_labels.append(pred)
    image_names.append(image)
for image in realImagesList:  
    imageTensor = convertImageToTensor(real_images_path+"\\"+image)
    #pred = testImage(trainedModel,imageTensor)[0]
    pred = model(imageTensor.unsqueeze(0).to(device)).argmax()
    pred = pred.item()
    print(pred)                                   
    true_labels.append(0)
    pred_labels.append(pred)
    image_names.append(image)
  	
labels = [image_names,true_labels,pred_labels]
print(labels)


with open("D:\\Final_Code\\Fake_Images_And_Video_Detection\\DeepFake_Detection_Code_Directory\\Training\\predictions2.txt", "w") as file:
    file.write("image_name\ttrue_label\tpred_label\n")
    for x in zip(*labels):
        file.write("{0}\t{1}\t{2}\n".format(*x))
    report = classification_report(true_labels, pred_labels)
    print(report)
    file = open('D:\\Final_Code\\Fake_Images_And_Video_Detection\\DeepFake_Detection_Code_Directory\\Training\\classification_report2.txt', 'w')
    file.write(report)
    file.close()


