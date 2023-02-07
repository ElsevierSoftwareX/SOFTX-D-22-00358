

########################################################################
######## Use this file to test a directory of Videos

#####################################################################################################
###### Import Libraries
from facenet_pytorch.models.mtcnn import MTCNN
import cv2
import os
import cv2
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
from test_video import *
#####################################################################################################


####### Training And Testing Augmentations
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

train_tfms=transforms
"""Outputs statistics about fake and real videos in a given directory"""
def fake_or_real_direc(model_1=None,model_2=None,ensemble_strat=1,conf_strat=1,per_frame=10,device="cpu"):
    if(model_1==None and model_2==None):
        print("Load Model Correctly !!!! ")
        exit()

    total_frames=0
    total_evaluated_frames=0
    total_fake_frames=0
    total_real_frames=0
    count_fake=0
    count_real=0
    count_zero_faces=0
    count_history=0
    while True:
        count_history=count_history+1
        file_d_3='history'+ "_" + str(count_history)
        if(os.path.exists(os.path.join("Video_Output",file_d_3+'.txt'))==True):
            pass
        else:
            break


    com_file = open(os.path.join("Video_Output",file_d_3+'.txt'),'w')
    content= "This File Contains Information about the videos stored in the directory 'test_video'\n"
    com_file.write(content)
    for video_id in os.listdir("test_video"):
        fake_prob,total_frames,total_evaluated_frames,total_fake_frames,total_real_frames =fake_or_real(model_1=model_1,model_2=model_2,ensemble_strat=ensemble_strat,conf_strat=conf_strat,video_path=video_id,per_frame=per_frame,device=device)
        content = "Video_Name: "+ str(video_id)+ "\n\tMean Fake Score: "+str(fake_prob)[:6]+"\t Total Frames: "+str(total_frames)+ "\t Total Evaluated Frames: "+str(total_evaluated_frames)+ "\t Total Fake Frames\Faces: "+str(total_fake_frames)+"\t Total Real Frames\Faces: "+str(total_real_frames)+"\n"
        com_file.write(content)
        if(fake_prob>=0.5):
            print(f'Video_Name: {video_id}\t Fake Score: {fake_prob} \t Video is Fake')
            count_fake=count_fake+1
        elif(fake_prob==0.0):
            print(f'Video_Name: {video_id}\t ****No Faces Found****')
            count_zero_faces=count_zero_faces+1
        else:
            print(f'Video_Name: {video_id}\t Fake Score: {fake_prob} \t Video is Real')
            count_real=count_real+1
    content = "\n Total Videos: "+ str(count_fake+count_real+count_zero_faces)+ "\n Fake Videos Found: "+ str(count_fake) + "\tReal Videos Found: "+ str(count_real)+ "\t Videos With No Faces Found: "+ str(count_zero_faces)
    com_file.write(content)
    com_file.close()

    print(f'\n Total Videos: {count_fake+count_real+count_zero_faces}\n Fake Videos Found: {count_fake} \tReal Videos Found: {count_real} \tVideos With No Faces Found: {count_zero_faces} \t')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Predict Video is Fake Or Real")

    parser.add_argument("--model-name", help="Name of the Model",nargs="?",type=str,default="F")
    parser.add_argument("--ensemble-strategy", help="Ensemble Strategy To Follow",nargs="?",type=int,default=1)
    parser.add_argument("--confident-strategy", help="Strategy To Decide Whether Video Is Fake Or Not",nargs="?",type=int,default=1)
    parser.add_argument("--per-frame", help="Epochs",nargs="?",type=int,default=10)



    args = parser.parse_args()

    if(args.confident_strategy<1 or args.confident_strategy>4):
        print("xxxxxxxxxx---------Incorrect Value of Confident Strategy Given---------xxxxxxxxxx")
        exit()

    if(args.ensemble_strategy <1 or args.ensemble_strategy>5):
        print("xxxxxxxxxx---------Incorrect Value of Ensemble Strategy Given---------xxxxxxxxxx")
        exit()


        ####### Check if Path to video doesn't exist
        #elif():
        #    print("xxxxxxxxxx---------Path To Video Doesn't Exist---------xxxxxxxxxx")
        #    exit()
    model_1=None
    model_2=None
    device=get_default_device()
    if(args.model_name=="F" or args.model_name=="model_1.pth"):
        print("------------------------Model Path Not Given,Loading Default Model------------------------")
        model_1= torch.load(os.path.join("Trained_Models","model_1.pth"), map_location=device)
        model_1.to(device)##### Put Model on the CPU/GPU
        model_1.eval()##### Start the Evaluation Mode for the Pytorch Model

    elif(args.model_name=="model_2.pth"):
        model_2= torch.load(os.path.join("Trained_Models",args.model_name), map_location=device)
        model_2.to(device)##### Put Model on the CPU/GPU
        model_2.eval()##### Start the Evaluation Mode for the Pytorch Model
        print("------------------------Model Loaded Correctly------------------------")

    elif(args.model_name=="ensemble"):
        print("----------------------------Model Loading----------------------------")
        model_1= torch.load(os.path.join("Trained_Models","model_1.pth"), map_location=device)
        model_1.to(device)##### Put Model on the CPU/GPU
        model_1.eval()##### Start the Evaluation Mode for the Pytorch Model

        model_2= torch.load(os.path.join("Trained_Models","model_2.pth"), map_location=device)
        model_2.to(device)##### Put Model on the CPU/GPU
        model_2.eval()##### Start the Evaluation Mode for the Pytorch Model
        print("------------------------Both Models Loaded Correctly------------------------")
    else:
        print("Model Name not correctly given, check docs!!! ")
        exit()

    ##### Call the Function
    fake_or_real_direc(model_1=model_1,model_2=model_2,ensemble_strat=args.ensemble_strategy,conf_strat=args.confident_strategy,per_frame=args.per_frame,device=device)
