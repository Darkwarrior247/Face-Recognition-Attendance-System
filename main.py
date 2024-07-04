import cv2
import numpy as np
import face_recognition

imgInput = face_recognition.load_image_file('ImagesFaceRecognition/Jesan.jpg')
imgInput = cv2.cvtColor(imgInput, cv2.COLOR_BGR2RGB)
imgTest1 = face_recognition.load_image_file('ImagesFaceRecognition/Elon Test1.jpg')
imgTest1 = cv2.cvtColor(imgTest1,cv2.COLOR_BGR2RGB)
imgTest2 = face_recognition.load_image_file('ImagesFaceRecognition/Bill Gates.jpg')
imgTest2 = cv2.cvtColor(imgTest2,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgInput)[0]
encodeInput = face_recognition.face_encodings(imgInput)[0]
cv2.rectangle(imgInput, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLoc = face_recognition.face_locations(imgTest1)[0]
encodeTest1 = face_recognition.face_encodings(imgTest1)[0]
cv2.rectangle(imgTest1,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest2)[0]
encodeTest2 = face_recognition.face_encodings(imgTest2)[0]
cv2.rectangle(imgTest2,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeInput], encodeTest1)
faceDis = face_recognition.face_distance([encodeInput], encodeTest1)
print(results,faceDis)
cv2.putText(imgTest1,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

results2 = face_recognition.compare_faces([encodeInput], encodeTest2)
faceDis2 = face_recognition.face_distance([encodeInput], encodeTest2)
print(results2,faceDis2)
cv2.putText(imgTest2,f'{results2}{round(faceDis2[0],2)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

cv2.imshow('Input', imgInput)
cv2.imshow('Result',imgTest1)
cv2.imshow('Result2',imgTest2)

cv2.waitKey(0)