#################################################################
###### Test whether a given video is fake or real

#####################################################################################################
###### Import Libraries
from facenet_pytorch.models.mtcnn import MTCNN
import cv2
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import PIL.Image as Image
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from torch import functional as Func
import albumentations as A


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
from strategies import *
#####################################################################################################

####### Training And Testing Time TransformationsS
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

test_tfms=transforms
aug_3 = A.Compose({
         A.RandomResizedCrop(224,224)
        })

##########################################################################################################

input_size=224
##########################################################################################################





##########################################################################################################


"""Some Uitlity Functions"""

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)







"""Predict whether the video is fake or real"""

def predict_on_video(video_path,model_1=None,model_2=None,ensemble_strat=1,conf_strat=1,per_frame=10,input_size=224,video_name='a',device='cpu'):
    v_cap = cv2.VideoCapture(video_path)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))   # Count Number Of Frames

    total_frames=0
    total_evaluated_frames=0
    total_fake_frames=0
    total_real_frames=0

    total_frames=v_len
    k=1
    if(per_frame>=v_len):
        per_frame=1



    parent_direc=""
    image_direc=os.path.join("Video_Output","extra")
    video_direc="Video_Output"
    y_preds=[]
    y_preds.append(0)


    frame_count=0
    face_count=0

    path_d_2 = os.path.join(parent_direc,image_direc)
    #os.mkdir(path_d_2)
    path_d_3 = os.path.join(parent_direc,video_direc)
    #os.mkdir(path_d_3)
    input_folder=path_d_2
    cropped_faces = []

    ##### Delete Contents of Image Directory And Video Directory ######



    for i in range(v_len):
        # Load frame
        face_count=0
        success = v_cap.grab()
        if i % per_frame == 0:
            success, frame_4 = v_cap.retrieve()
        else:
            continue
        if not success:
            continue
        mtcnn = MTCNN(keep_all=True,min_face_size=176,thresholds=[0.85,0.90,0.90],device=device)   # Initialize Face Detector
        frame = cv2.cvtColor(frame_4, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame.astype(np.uint8))
        face_count=0

        frame=img
        # Detect face
        boxes,landmarks = mtcnn.detect(frame, landmarks=False)   ## Detect Faces

        try:
            if(len(boxes)!=None):
                pass
        except:
            continue
        total_evaluated_frames = total_evaluated_frames+1
        b1 = img
        try:
            for i in range(0,len(boxes)):
                x,y,width,height =boxes[i]
                max_width,max_height = b1.size
                ###### Crop Face From Image ##############
                boxes[i][0]-=width/10
                boxes[i][1]-=height/10
                boxes[i][0] = max(0,boxes[i][0])
                boxes[i][1] = max(0,boxes[i][1])
                boxes[i][2] =min(boxes[i][2]+(width/10),max_width)
                boxes[i][3] =min(boxes[i][3]+(height/10),max_height)
                b4 = b1.crop(boxes[i])
                #print(b4)
                ###########################################
                if(model_1 !=None):
                    img = test_tfms(b4)##### Apply Transformations
                    img = img.unsqueeze(0)##### Converting the Single Image into a batch of Sinle Image-----(3,380,380) to(1,3,380,380)
                    img = to_device(img,device)##### Put the Image On the GPU if available
                    out_3 = model_1(img)###### Pass through the model
                    out_3=torch.softmax(out_3.squeeze(),0)
                    out_5 = out_3.cpu().detach().numpy()[0]
                else:
                    out_5=0
                if(model_2 != None):
                    b4.save("test_face_image.jpg")##### Save Cropped Image

                    image = plt.imread("test_face_image.jpg")
                    #print("ghnd")

                    image = Image.fromarray(image).convert('RGB')
                    #print("gfnk")
                    image = aug_3(image=np.array(image))['image']
                    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
                    image = torch.tensor(image, dtype=torch.float)


                    #img = train_tfms(b4)##### Apply Transformations
                    img = image.unsqueeze(0)##### Converting the Single Image into a batch of Sinle Image-----(3,380,380) to(1,3,380,380)
                    img = to_device(img,device)##### Put the Image On the GPU if available
                    out_3 = model_2(img)###### Pass through the model
                    out_3=torch.softmax(out_3.squeeze(),0)
                    out_6 = out_3.cpu().detach().numpy()[1]
                else:
                    out_6=0


                if(model_1==None):
                    out_7=out_6
                elif(model_2==None):
                    out_7=out_5
                else:

                    if ensemble_strat==1:
                        out_7 = ensemble_strategy_1(out_5,out_6)
                        out_7 = (out_5+out_6)/2
                    elif ensemble_strat==2:
                        out_7 = ensemble_strategy_2(out_5,out_6)
                    elif ensemble_strat==3:
                        out_7 = ensemble_strategy_3(out_5,out_6)
                    elif ensemble_strat==4:
                        out_7 = ensemble_strategy_4(out_5,out_6)
                    elif ensemble_strat==5:
                        out_7 = ensemble_strategy_5(out_5,out_6)
                    else:
                        out_7 = (out_5+out_6)/2
                d34 = out_7
                y_preds.append(d34)  ### Apend the Output of the Model to the list

                k=k+1

                if(d34>0.5):
                    label="Fake"
                    total_fake_frames = total_fake_frames+1
                else:
                    label="Real"
                    total_real_frames = total_real_frames+1

                color = (0, 0, 255) if label=="Fake" else (0, 255, 0)


                ##### Correct This
                cv2.rectangle(frame_4, (boxes[i][0], boxes[i][1]),(boxes[i][2], boxes[i][3]),color,2)### Apply Bounding Box to the Frame,Red -Fake,Green-Real
                #####
                if(frame_count<2):
                    cv2.imwrite(os.path.join(path_d_2,str(frame_count)+".jpg"),frame_4)##### Save Image to later get its shape
                cropped_faces.append(frame_4)### Append the frmaes to later combine together to form a video
                #c72,c73=frame.shape
                frame_count=frame_count+1
                face_count=1

        except:
            y_preds.append(0.0)###### Helps in keeping the code from breaking
            pass

    if(face_count==0):
        cv2.imwrite(os.path.join(path_d_2,str(frame_count)+".jpg"),frame_4)
        frame_count=frame_count+1


    try:
        frame = cv2.imread(os.path.join(input_folder,"0.jpg"))   ### Load a frame
        height,width,_=frame.shape  ### Find the shape of the frame
        j=52
        out = cv2.VideoWriter(os.path.join(path_d_3,video_name),cv2.VideoWriter_fourcc(*'DIVX'), 2, (width,height))  #### Initialize a Video
        c4 = os.listdir(input_folder)### Get the names of the frames that have been used for prediction
        sort_nicely(c4) ### Sort them Numerically
        i=0
        print(len(cropped_faces))
        for filename in range(len(cropped_faces)):
            #img = cv2.imread(os.path.join(input_folder,filename))

            out.write(cropped_faces[i])  #### Write to the Video
            i=i+1
        out.release()#### Release the Video
    except:
        pass

    if conf_strat==1:
        y_preds_3 = np.array(y_preds).mean()
    elif conf_strat==2:
        y_preds_3 = confident_strategy_2(np.array(y_preds))
    elif conf_strat==3:
        y_preds_3 = confident_strategy_3(np.array(y_preds))
    else:
        y_preds_3 = confident_strategy_4(np.array(y_preds))



    return y_preds_3,path_d_3,total_frames,total_evaluated_frames,total_fake_frames,total_real_frames












if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Predict Video is Fake Or Real")
    parser.add_argument("--model-name", help="Name of the Model",nargs="?",type=str,default="F")
    parser.add_argument("--video-path", help="Name of the Video",nargs="?",type=str,default="F")
    parser.add_argument("--ensemble-strategy", help="Ensemble Strategy To Follow",nargs="?",type=int,default=1)
    parser.add_argument("--confident-strategy", help="Strategy To Decide Whether Video Is Fake Or Not",nargs="?",type=int,default=1)
    parser.add_argument("--per-frame", help="Epochs",nargs="?",type=int,default=10)



    args = parser.parse_args()

    if(args.video_path=="F"):
        print("xxxxxxxxxx---------Path To Video Not Given---------xxxxxxxxxx")
        exit()
    if(args.confident_strategy<1 or args.confident_strategy>4):
        print("xxxxxxxxxx---------Incorrect Value of Confident Strategy Given---------xxxxxxxxxx")
        exit()

    if(args.ensemble_strategy <1 or args.ensemble_strategy>5):
        print("xxxxxxxxxx---------Incorrect Value of Ensemble Strategy Given---------xxxxxxxxxx")
        exit()
    if(not os.path.isfile(os.path.join("test_video",args.video_path))):
        print('Video does not exist in the folder')
        exit()


    ####### Check if Path to video doesn't exist
    #elif():
    #    print("xxxxxxxxxx---------Path To Video Doesn't Exist---------xxxxxxxxxx")
    #    exit()
    model_1=None
    model_2=None
    device=get_default_device()
    if(args.model_name=="F" or args.model_name=="model_1.pth"):
        print("------------------------Loading model_1(Default model)------------------------")
        model_1= torch.load(os.path.join("Trained_Models","model_1.pth"), map_location=device)
        model_1.to(device)##### Put Model on the CPU/GPU
        model_1.eval()##### Start the Evaluation Mode for the Pytorch Model

    elif(args.model_name=="model_2.pth"):
        model_2= torch.load(os.path.join("Trained_Models",args.model_name), map_location=device)
        model_2.to(device)##### Put Model on the CPU/GPU
        model_2.eval()##### Start the Evaluation Mode for the Pytorch Model
        print("------------------------Model model_2 Loaded Correctly------------------------")

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


    video_path=os.path.join("test_video",args.video_path)
    filename = args.video_path


    strategy=1
    b58,final_video,total_frames,total_evaluated_frames,total_fake_frames,total_real_frames = predict_on_video(video_path=video_path,model_1=model_1,model_2=model_2,ensemble_strat=args.ensemble_strategy,conf_strat=args.confident_strategy,per_frame=args.per_frame,input_size=input_size,video_name=filename[:-4]+'.mp4',device=device)##### Call the Function

    print("Total Frames are: ")
    print(total_frames)
    print("Total Evaluated Frames are: ")
    print(total_evaluated_frames)
    print("Total Fake Frames are: ")
    print(total_fake_frames)
    print("Total Real Frames are: ")
    print(total_real_frames)
    #print(confident_strategy_1(num_d_3))
    #print(confident_strategy_2(num_d_3))
    print("Evaluated/Mean Fake Score for the Video is: ")
    print(b58)
