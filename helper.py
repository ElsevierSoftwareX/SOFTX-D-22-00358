from pytube import YouTube  # pip install pytube
from urllib import request  # pip install urllib3
from imgaug import augmenters as iaa
import imgaug as ia
from multiprocessing import context
from newsapi import NewsApiClient
from fileinput import close
from bs4 import BeautifulSoup as bs
import praw
import requests
import cv2
import numpy as np
import os
import pickle
import tweepy
import csv
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import urllib.request
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.transforms as tt
import PIL
from imgaug import augmenters as iaa
import imgaug as ia
import cv2 as cv
import math
import argparse
from facenet_pytorch.models.mtcnn import MTCNN
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.nn.functional as F
from torch import functional as Func
import PIL
from imgaug import augmenters as iaa
import cv2
import glob
import re
import moviepy.editor as moviepy
from facenet_pytorch.models.mtcnn import MTCNN
from tqdm.notebook import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastbook import *
from fastai.vision import *
from fastai import *
import pathlib
from fastai.vision.all import *
from torchvision import transforms as t_2
from matplotlib import pyplot
import warnings
import model
from model import VIT, PatchEmbedding, multiHeadAttention, residual, mlp, TransformerBlock, Transformer, \
    Classification
#import helper
#from helper import ImgAugTransform
from torch.autograd import Variable
import os
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, send_file, render_template
import requests
import helper
import util
from PIL import Image
from PIL.ExifTags import TAGS
import os
import wget
import cv2
from flask import Flask, request, render_template, session, redirect
from flask import Flask, request, jsonify, render_template, Blueprint, render_template, redirect, url_for, request, flash
from flask import Flask, render_template, request
from flask import Flask, render_template,request
import numpy as np
import pandas as pd
import praw
import requests
import cv2
import numpy as np
import os
import pickle
POST_SEARCH_AMOUNT = 10
import urllib.request
from os import listdir
from PIL import Image
import re
import subprocess
import time
import sys
import os.path
import os
import shutil
import pandas as pd
import urllib.request
import urllib
import re
import subprocess
import time
import sys
import os.path
import os
import pandas as pd
import pickle
import time
import re
import numpy as np
import tweepy
import csv
import pandas as pd
import sys
import re



class ImgAugTransform:
    """Test-Time Transformations"""

    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(0.25,
                          iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                     iaa.CoarseDropout(0.1, size_percent=0.5)])),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)
        
"""Predict whether the video is fake or real"""


def predict_on_video(video_path, batch_size, input_size, strategy=np.mean, d6=20, d16=1, d_40='a'):
    v_cap = cv2.VideoCapture(video_path)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Count Number Of Frames
    k = 1
    d23 = 20  # Every nth frame is used for prediction(Here every d23 i.e 10th frame is used for prediction)
    d_53 = 0
    d_53 = d_53 + 1

    parent_direc = os.path.join('static', 'videos')
    image_direc = "video_40"
    video_direc = "boxes_video_40"
    y_preds_2 = []

    d15 = 0
    d21 = 0

    path_d_2 = os.path.join(parent_direc, image_direc)  # Path to single image
    # os.mkdir(path_d_2)
    path_d_3 = os.path.join(parent_direc, video_direc)
    # os.mkdir(path_d_3)
    input_folder = path_d_2
    c67 = []
    c72 = 0
    c73 = 0
    c67.append(0.5)

    # Delete Contents of Image Directory And Video Directory #

    for i in range(v_len):
        # Load frame

        success = v_cap.grab()
        if i % d23 == 0:
            success, frame_4 = v_cap.retrieve()
        else:
            continue
        if not success:
            continue
        mtcnn = MTCNN(keep_all=True, min_face_size=176, thresholds=[0.85, 0.90, 0.90],
                      device=device)  # Initialize Face Detector
        frame = cv2.cvtColor(frame_4, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame.astype(np.uint8))

        frame = img
        # Detect face

        boxes, landmarks = mtcnn.detect(frame, landmarks=False)  # Detect Faces
        try:
            if len(boxes) != None:
                pass
        except:
            continue
        b1 = img
        for i in range(0, len(boxes)):
            x, y, width, height = boxes[i]
            max_width, max_height = b1.size
            # Crop Face From Image ##############
            boxes[i][0] -= width / 10
            boxes[i][1] -= height / 10
            boxes[i][0] = max(0, boxes[i][0])
            boxes[i][1] = max(0, boxes[i][1])
            boxes[i][2] = min(boxes[i][2] + (width / 10), max_width)
            boxes[i][3] = min(boxes[i][3] + (height / 10), max_height)
            b4 = b1.crop(boxes[i])
            
            b4.save("test_face_image.jpg")  # Save Cropped Image

            image = plt.imread("test_face_image.jpg")
            # print("ghnd")

            image = Image.fromarray(image).convert('RGB')
            # print("gfnk")
            image = aug_6(image=np.array(image))['image']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image = torch.tensor(image, dtype=torch.float)

            # img = train_tfms(b4)##### Apply Transformations
            img = image.unsqueeze(
                0)  # Converting the Single Image into a batch of Single Image-----(3,380,380) to(1,3,380,380)
            img = to_device(img, device)  # Put the Image On the GPU if available
            out_3 = model_5(img)  # Pass through the model
            print("Video  probaility:", out_3)
            out_3 = torch.softmax(out_3.squeeze(), 0)
            print("Video predicted probaility value:", out_3)
            out_6 = out_3.cpu().detach().numpy()[1]
            print("numpy value:",out_6)
            y_preds_2.append(out_6)  # Append the Output of the Model to the list

            k = k + 1

            if out_6 > 0.5:
                label = "Fake"
            else:
                label = "Real"

            color = (0, 0, 255) if label == "Fake" else (0, 255, 0)

            # Correct This

            cv2.rectangle(frame_4, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][2]), int(boxes[i][3])), color,
                          2)  # Apply Bounding Box to the Frame,Red -Fake,Green-Real
            #####
            if d21 < 2:
                cv2.imwrite(os.path.join(path_d_2, str(d15) + ".jpg"), frame_4)  # Save Image to later get its shape
            c67.append(frame_4)  # Append the frames to later combine together to form a video
            # c72,c73=frame.shape
            d15 = d15 + 1
            d21 = d21 + 1

    if d21 == 0:
        cv2.imwrite(os.path.join(path_d_2, str(d15) + ".jpg"), frame_4)
        d15 = d15 + 1

    try:
        frame = cv2.imread(os.path.join(input_folder, "0.jpg"))  # Load a frame
        height, width, _ = frame.shape  # Find the shape of the frame
        j = 52
        out = cv2.VideoWriter(os.path.join(path_d_3, d_40), cv2.VideoWriter_fourcc(*'DIVX'), 2,
                              (width, height))  # Initialize a Video
        # c4 = os.listdir(input_folder)### Get the names of the frames that have been used for prediction
        # sort_nicely(c4) ### Sort them Numerically
        i = 0
        for i in range(len(c67)):
            out.write(c67[i])  # Write to the Video
        out.release()  # Release the Video
    except:
        pass

    return np.array(y_preds_2).mean(), path_d_3







def youtube_video_downloader(link, destination):    
    try:
        print(f"YouTube link1: {link}")
        yt = YouTube(link)
        downloaded_video = yt.streams.filter(file_extension='mp4').order_by('resolution')[-1].download(destination)
        print(f"YouTube link2: {link}")
        if "/" in downloaded_video:
            return downloaded_video.split("/")[-1]
        return downloaded_video.split("\\")[-1]
    except Exception as e:
        print(e)
        return -1


def download_video_using_url(url, video_name):
    try:
        request.urlretrieve(url, video_name)
        return 1
    except Exception as e:
        print(e)
        return -1


if __name__ == "__main__":
    result = youtube_video_downloader(
        # link="https://youtu.be/xWOoBJUqlbI",
        link="https://www.youtube.com/watch?v=xWOoBJUqlbI",
        destination="C:\\Users\\sayan\\Documents\\IIT Patna\\Website\\Flask App\\uploaded"
    )
    print(result)

    video_name = download_video_using_url(
        url="http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
        video_name="C:\\Users\\sayan\\Documents\\IIT Patna\\Website\\Flask App\\uploaded\\downloaded.mp4"
    )
    print(video_name)
