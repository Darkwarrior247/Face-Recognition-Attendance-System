# Face-Recognition-Attendance-System
The Face Recognition Based Attendance System uses face recognition technology to automate student attendance tracking in educational institutions. It captures live video or images, matches them against a registered database of faces, and marks attendance in real-time. By streamlining attendance processes, it reduces paperwork and improves overall efficiency.

# Features
- Face recognition technology to track student attendance
- Real-time matching of captured images or video with a database of registered faces
- Automatic marking of attendance for recognized students
- Database storage for student information and attendance records
- Automation of attendance tracking processes
- Reduction of manual paperwork

# Installations
The installation process for this project is a bit more than usual. First we have to download a C++ compiler. We can do this by installing Visual Studios. You can download the community version for free from their website. Once the intaller we will run it and select the ‘Desktop development with C++’. The download and installation will take some time as it is a few Gbs.

![visualStudio-950x480](https://github.com/Darkwarrior247/Face-Recognition-Attendance-System/assets/103986226/44181d8e-a88a-4131-87ba-799efd88309a)

After completing and restarting the computer, now we will head on to our Pycharm project. Here we will install the required packages. Below is the list.

```bash
   pip opencv-python numpy face_recognition dlib face_recognition --only-binary :all: cmake cmake
```

#Understanding the problem
Although many face recognition algorithms have been developed over the years, their speed and accuracy balance has not been quiet optimal . But some recent advancements have shown promise. A good example is Facebook, where they are able to tag you and your friends with just a few images of training and with accuracy as high as 98%. So how does this work . Today we will try to replicate similar results using a face recognition library developed by Adam Geitgey. Lets look at the 4 problems he explained in his article.

Face recognition is a series of several problems:

1. First, look at a picture and find all the faces in it
2. Second, focus on each face and be able to understand that even if a face is turned in a weird direction or in bad lighting, it is still the same person.
3. Third, be able to pick out unique features of the face that you can use to tell it apart from other people— like how big the eyes are, how long the face is, etc.
4. Finally, compare the unique features of that face to all the people you already know to determine the person’s name.
   
# Attendance Project
Now using the methods we have seen above, we will develop an attendance system where the user is automatically logged when they are detected in the camera. We will store the name along with the time when they appeared.

# Importing Images
As we have imported before we can use the same face_recognition.load_image_file() function to import our images. But when we have multiple images, importing them individually can become messy. Therefore we will write a script to import all images in a given folder at once. For this we will need the os library so we will import that first. We will store all the images in one list and their names in another.

import face_recognition
import cv2
import numpy as np
import os
path = 'ImagesAttendance'
images = []     # LIST CONTAINING ALL THE IMAGES
className = []    # LIST CONTAINING ALL THE CORRESPONDING CLASS Names
myList = os.listdir(path)
print("Total Classes Detected:",len(myList))
for x,cl in enumerate(myList):
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        className.append(os.path.splitext(cl)[0])

# Compute Encodings
Now that we have a list of images we can iterate through those and create a corresponding encoded list for known faces. To do this we will create a function. As earlier we will first convert it into RGB and then find its encoding using the face_encodings() function. Then we will append each encoding to our list.

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
Now we can simply call this function with the images list as the input arguments.

encodeListKnown = findEncodings(images)
print('Encodings Complete')

# The While loop
The while loop is created to run the webcam. But before the while loop we have to create a video capture object so that we can grab frames from the webcam.

cap = cv2.VideoCapture(0)

#Webcam Image
First we will read the image from the webcam and then resize it to quarter the size. This is done to increase the speed of the system. Even though the image being used is 1/4 th of the original, we will still use the original size while displaying. Next we will convert it to RGB.

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
# Webcam Encodings
Once we have the webcam frame we will find all the faces in our image. The face_locations function is used for this purpose. Later we will find the face_encodings as well.

facesCurFrame = face_recognition.face_locations(imgS)
encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

# Find Matches
Now we can match the current face encodings to our known faces encoding list to find the matches. We will also compute the distance. This is done to find the best match in case more than one face is detected at a time.

for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
Once we have the list of face distances we can find the minimum one, as this would be the best match.

matchIndex = np.argmin(faceDis)
Now based on the index value we can determine the name of the person and display it on the original Image.

if matches[matchIndex]:
    name = className[matchIndex].upper()
    y1,x2,y2,x1=faceLoc
    y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
    
# Marking Attendance
Lastly we are going to add the automated attendance code. We will start by writing a function that requires only one input which is the name of the user. First we open our Attendance file which is in csv format. Then we read all the lines and iterate through each line using a for loop. Next we can split using comma ‘,’. This will allow us to get the first element which is the name of the user. If the user in the camera already has an entry in the file then nothing will happen. On the other hand if the user is new then the name of the user along with the current time stamp will be stored. We can use the datetime class in the date time package to get the current time.

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList =[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in  line:
            now = datetime.now()
            dt_string = now.strftime("%H:%M:%S")
            f.writelines(f'n{name},{dt_string}')

# Labeling Unknown faces as well
To find the unknown faces we will replace

if matches[matchIndex]:
    name = classNames[matchIndex].upper()
    #print(name)
    y1,x2,y2,x1 = faceLoc
    y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
    cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    markAttendance(name)
with this

if faceDis[matchIndex]< 0.50:
    name = classNames[matchIndex].upper()
    markAttendance(name)
else: name = 'Unknown'
#print(name)
y1,x2,y2,x1 = faceLoc
y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
All this does is to check if the distance to our min face is less than 0.5 or not. If its not then this means the person is unknown so we change the name to unknown and don’t mark the attendance.
