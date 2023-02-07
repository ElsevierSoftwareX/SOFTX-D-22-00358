<h1>Fake Detection System</h1>

Table of content
1. Demo and Screenshots
2. Overview
3. Motivation 
4. Dataset
5. Directory Tree
6. Installation
7. Deployment
8. Technologiees Used
9. To Do
10. Credits

<h2> Demo and Screenshots</h2>

<h4>Demo of Web App: </h4> 

https://user-images.githubusercontent.com/38296253/217193271-4b18ece8-b2bd-4d9c-8fdb-9cabab3e4313.mp4

<br>
The Demo of Fake Detection tool is available at this link:- 
https://drive.google.com/file/d/1Mx-uEaHOGk7V7KL3QJ_TJLNOS7DhOFjn/view?usp=sharing (for GIF video)<br>
https://drive.google.com/file/d/1BeTUMwci9uNNuU7RxOkGy4SHAM0PJ6u5/view?usp=share_link

<br><br>

 
 The screenshots of the results page for detection of fake facial image, fake image and fake video are:-
 <br>
 <img src="https://user-images.githubusercontent.com/38296253/190884375-aae133aa-e387-481b-95b2-2dae3eb0f77c.PNG" width="300" height="300">
 <img src="https://user-images.githubusercontent.com/38296253/190883469-e41b98e3-e4fb-4d99-bffb-4804358a0b67.PNG" width="300" height="300">
 <img src="https://user-images.githubusercontent.com/38296253/190886226-b6f547bc-2798-4ccd-b218-fcf3dbc31936.PNG" width="300" height="300">
 <img src="https://user-images.githubusercontent.com/38296253/190886193-030e942a-afe4-4d45-a989-849e62ae2c54.PNG" width="300" height="300">
 <img src="https://user-images.githubusercontent.com/38296253/190886273-39d71438-8493-4122-87ad-f20524953696.PNG" width="300" height="300">
 <img src="https://user-images.githubusercontent.com/38296253/190886297-93abce95-139a-4bc2-99dc-441065927a06.PNG" width="300" height="300">
 <br>
 The screenshots of web page of Twitter streaming of images are:- <br><br>
 <img src="https://user-images.githubusercontent.com/38296253/199389137-b2e1ebde-9697-4974-bf43-c34e044d3ec6.PNG" width="300" height="300">
 <img src="https://user-images.githubusercontent.com/38296253/199389191-7d92dc6e-012a-4109-abc0-8f4e0d9b66fd.PNG" width="300" height="300">
 <img src="https://user-images.githubusercontent.com/38296253/199389242-9cc5af0f-2c66-44e6-8e43-b571b053f72b.PNG" width="300" height="300">


<h2> Overview </h2>
FDT web app is a python package toolkit and a single platform for detecting fake media. Users can upload facial image, image or video from their local system or by placing URL link of a image or a video. The tool detects it as a fake with a red bounding box or real with a green bounding box and with a confidence score and its metadata details. It is also integrated with Twitter for streaming image tweets and categorize it into real or fake, impact as an influencer or amateur, sentiments as 1(Positive), 0(Neutral) and -1(Megative), etc, and presents other statistics through pie charts. 

<h2>Motivation</h2>
To deal with the fakes in the wild and in Indian context, an open source and easy-to-use GUI is available for Fake Detection. 
<h2>Directory Structure</h2>

Create new folder under that make following folders.
    
    --> Detection App : In this folder all the cloned files should be placed except Model Testing.
    
    --> Model Testing : In this directory, there is all the guidelines in the instruction file and python files to train and test the different detection models.
    



for testing images, faces and videos on the FDT tool, samples of images, faces and videos are available in the directory and trained weights of models are available as give in the following steps:
Go to avobe drive link- https://drive.google.com/drive/folders/1In68OsTDfS4iaJbaj8jgtEYE-qBaXAfw?usp=sharing, Download folder Fake_Detection_Tool. Go to folder Fake_Detection_Tool > test_samples > Full Images / Facial Images / Videos
For downloading model weights : Go to folder Fake_Detection_Tool > Trained_Models

    
<h2> Dataset </h2>
Dataset is available on this drive link: -
For Fake Face and Deepfake Detection, Dataset directory tree are as follow:
The Dataset should have two folders: train and valid. Both should contain folders named 'fake' and 'real'.

  ![image](https://user-images.githubusercontent.com/38296253/190840044-61ae334d-736c-4260-877d-2327beb1b65f.png)
                                                           
  For Image Manipulation Detection, Dataset directory tree are as follows:
  
  The Dataset should have two folders: train and valid. Both should contain folders named 'images' and 'masks'. The images folder contain manipulated images and their corresponding ground truths are inn masks folder.

  ![image](https://user-images.githubusercontent.com/38296253/190840029-4458e94b-9af3-49f8-a5c1-73c7b1dd285f.png)
  
<h2>Steps to recreate</h2>
<h4>Prerequisites</h4>
- Python 3.8.8
- Flask 1.1.2
- Werkzeug 1.0.1
- conda 4.10.1

<h3> To run the application locally on your machine </h4><br>
Copy your trained model to the models folder.
   - You can download the trained models from the [Google Drive](https://drive.google.com/drive/folders/1UX8jXUXyEjhLLZ38tcgOwGsZ6XFSLDJ-?usp=sharing) and save it in Trained_Models or you can train your models using the steps mentioned in instruction_to_execute_file in Model Testing directory.

#### Step 1 : Clone the repo and go to the Django Application
`git clone https://github.com/hsam-2021/deepfake-generation-and-detection `

#### Step 2: Install the requirements
`pip install -r requirements.txt`

**Note :** Here you may need to do a recursive install of the requirements.txt and depending up compatible versions update and re-install the requirements.txt

#### Step 3: Copy Models

Copy the pre-trained model to the models folder i.e Django Application/models/


#### Step 4: Run the project
Navigate to the path where manage.py file is located `\Deepfake_detection_miniproject_3\Django Application`. Open the command prompt and run 

`python manage.py runserver`

#### Step 5: Accessing the web app in browser
Once the server is up and running copy the following URL in any of your favourite browser.

`http://127.0.0.1:8000/`



The Code is written in Python 3.7. Also, Download and install Anaconda follow steps given in this link:<br>
      https://docs.anaconda.com/anaconda/install/ <br>
To install dependencies required, Download this file - environment.yml <br>
Use this command to create environment from the environment.yml file- <br><b>conda env create -f environment.yml</b>

*Must keep this file facenet_pytorch-2.2.7-py3-none-any.whl in the same folder to install dependency 

 <h4> Deployment </h4>
 To run this web app:
 Step 1- conda activate env<br>
 Step 2: Download the folder <b>Fake_Detection_Tool</b> from drive link where the complete code in folder Fake_Detection_Tool is available.Go to folder Fake_Detection_Tool and run <b> python app.py </b> . It will give localhost address - <b> http://127.0.0.1:5001/ </b> where the tool is hosted at this address.
 Example is shown as: 
 <img src="https://user-images.githubusercontent.com/38296253/190883776-acd3512d-cb37-431b-9195-7b527a77b64a.PNG" width="300" height="200">

 
<h4> Technologies Used </h4>
<img src="https://user-images.githubusercontent.com/38296253/190848411-b39b8984-58fb-4b8d-b193-e2afe43f8b57.png" width="100">
<img src="https://user-images.githubusercontent.com/38296253/190848468-b376733f-8cd1-4d16-91f6-7e553841dba1.png" width="100">

<img src="https://user-images.githubusercontent.com/38296253/190888956-e8002c94-a903-41fc-b01e-16b31366b561.PNG" width="100">
<img src="https://user-images.githubusercontent.com/38296253/190888979-89208eb9-462c-4497-838b-f7ad23e540cd.png" width="100">
<img src="https://user-images.githubusercontent.com/38296253/190889052-b89c3d3d-2698-4f02-9cb2-170bc1c54fd0.png" width="100">
<img src="https://user-images.githubusercontent.com/38296253/190889081-de658f33-d6e3-4568-8dc5-c7fb45871f41.png" width="100">

<h4> To Do </h4>
To integrate it with other platform like Reddit, Instagram etc.
<h4> Note </h4>
Due to restriction in uploading of file size on github, the entire code is available on google drive link. Please refer to google drive link - https://drive.google.com/drive/folders/1In68OsTDfS4iaJbaj8jgtEYE-qBaXAfw?usp=sharing

<h4> Contact </h4>
For any queries and doubts, Please contact us at this email id: surbhiraj786@gmail.com, surbhi_2021cs36@iitp.ac.in
 
