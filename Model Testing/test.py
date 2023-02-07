import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
from sklearn.metrics import classification_report
import time
import copy
import pickle as pk
import os
import torch.nn.functional as F
'''
class self_attention(nn.Module):
  def __init__(self,in_dim,activation,preTrained_model):
    super(self_attention,self).__init__()
    self.preTrained = preTrainedModel
    #pretrained model added
    self.activation = activation
    self.channels = in_dim
    self.query = nn.Conv2d(in_channels=in_dim, out_channels = in_dim//8, kernel_size=1)
    self.key = nn.Conv2d(in_channels=in_dim, out_channels = in_dim//8, kernel_size=1)
    self.value = nn.Conv2d(in_channels=in_dim, out_channels = in_dim, kernel_size=1)
    self.gamma = nn.Parameter(torch.zeros(1))
    self.softmax = nn.Softmax(dim=-1)
    #till now self attention is applied
    self.fc1 = nn.Linear(100352,256)
    #self.fc2 = nn.Linear(2048,256)
    self.fc3 = nn.Linear(256,2)

  def forward(self,x):
    x = self.preTrained(x)
    "x: Batch*Channel*Width*Height"
    batch_size, channels, width, height = x.size()
    get_query = self.query(x).view(batch_size,-1,width*height).permute(0,2,1)
    get_keys = self.key(x).view(batch_size,-1,width*height)
    attn_scores = torch.bmm(get_query,get_keys)
    attn_scores = self.softmax(attn_scores)
    get_value = self.value(x).view(batch_size,-1,width*height)
    out = torch.bmm(get_value,attn_scores.permute(0,2,1))
    #out = out.view( batch_size, channels, width, height)
    ##till now self attention has been applied
    x1 = torch.flatten(out,1)
    x1 = self.fc1(x1)
    x1 = F.relu(x1)
    #x1 = self.fc2(x1)
    #x1 = F.relu(x1)
    x1 = self.fc3(x1)
    output = F.log_softmax(x1, dim=1)
    return output
    #out = out.view(batch_size, channels, width, height)
'''
aug_3 = A.Compose({
         A.RandomResizedCrop(224,224)
        })
def convertImageToTensor(imagePath):
    image = cv2.imread(imagePath)  
    image = Image.fromarray(image).convert('RGB')#### Convert Image to RGB
    image = aug_3(image=np.array(image))['image']##### Apply Transformations
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image, dtype=torch.float)###### Convert Image to Tensor
    #pred = model(image.unsqueeze(0).to(device)).argmax()
    return image

'''
def convertImageToTensor(imagePath):
  image = Image.open(imagePath)
  imageTensor = image_transforms(image).float()
  imageTensor = Variable(imageTensor,requires_grad=False)
  imageTensor = imageTensor.unsqueeze(0)# converted in the form of batch
  return imageTensor


def testImage(nnModel,imageTensor):
  nnModel.eval()
  with torch.no_grad(): 
    output = nnModel(imageTensor)
    output = output.argmax(1)
  return output
'''
#load trained model
model= torch.load(os.path.join("Trained_Models","Fake_Face_Detection_Model.pth"), map_location="cpu")
#model.to(device)##### Put Model on the CPU/GPU
model.eval()

image_transforms = transforms.Compose([
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))#(0.5, 0.5, 0.5), (0.5, 0.5, 0.5) 
])

#load images
fake_images_path = "C:\\Users\\Surbhi\\PycharmProjects\\PCA_Faces\\Faces_5K\\valid\\fake"
fakeImagesList = os.listdir(fake_images_path)
#print("\n Fake:")
#print(fakeImagesList)
fakeTrueLabel = [1]*len(fakeImagesList)
print(fakeTrueLabel)
real_images_path = "C:\\Users\\Surbhi\\PycharmProjects\\PCA_Faces\\Faces_5K\\valid\\real"
   
realImagesList = os.listdir(real_images_path)
#print("\n Real:")
#print(realImagesList)
realTrueLabel = [0]*len(realImagesList)
print("\n Real True Label: ", realTrueLabel)


pred_labels=[]
true_labels=[]
image_names=[]

for image in fakeImagesList:
    imageTensor = convertImageToTensor(fake_images_path+"\\"+image)
    #pred = testImage(trainedModel,imageTensor)[0]
    pred = model(imageTensor.unsqueeze(0)).argmax()
    pred = pred.item()
    #print(pred)
    true_labels.append(1)
    pred_labels.append(pred)
    image_names.append(image)
for image in realImagesList:  
    imageTensor = convertImageToTensor(real_images_path+"\\"+image)
    #pred = testImage(trainedModel,imageTensor)[0]
    pred = model(imageTensor.unsqueeze(0)).argmax()
    pred = pred.item()                       
    #print(pred)            
    true_labels.append(0)
    pred_labels.append(pred)
    image_names.append(image)

print("Report:")  	
labels = [image_names,true_labels,pred_labels]
print(labels)
print(len(labels))
print((labels[0],"\t",labels[1],"\t",labels[2]))

for x in zip(*labels):
    if x[1] != x[2]:
      print("{0}\t{1}\t{2}\n".format(*x))
    

'''
print("Report1:")

with open("D:\\Final_Code\\Fake_Images_And_Video_Detection\\DeepFake_Detection_Code_Directory\\Training\\predictions3.txt", "w") as file:
    file.write("image_name\ttrue_label\tpred_label\n")
    for x in zip(*labels):
        file.write("{0}\t{1}\t{2}\n".format(*x))
    report = classification_report(true_labels, pred_labels)
    print(report)
    file = open('D:\\Final_Code\\Fake_Images_And_Video_Detection\\DeepFake_Detection_Code_Directory\\Training\\class2.txt', 'w')
    file.write(report)
    file.close()
'''
with open("D:\\Final_Code\\Fake_Images_And_Video_Detection\\DeepFake_Detection_Code_Directory\\Training\\misclassified_img.txt", "w") as file:
    file.write("image_name:\t\ttrue_label\tpred_label\n")
    for x in zip(*labels):
      if x[1] != x[2]:
        file.write("{0}\t{1}\t{2}\n".format(*x))
      #print("{0}\t{1}\t{2}\n".format(*x))      
    report = classification_report(true_labels, pred_labels)
    print(report)
    file = open('D:\\Final_Code\\Fake_Images_And_Video_Detection\\DeepFake_Detection_Code_Directory\\Training\\classification_report2.txt', 'w')
    file.write(report)
    file.close()



