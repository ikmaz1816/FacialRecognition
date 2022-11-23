import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
#Now we have to take images as list form the attendance images
#Intializing the path
path='AttendanceImage'
#Getting the attendance_images
attendance_image=[]
#Filtering it with regex .
attendee_name=[]
#Getting the path to the attendance folder
myAttendee=os.listdir(path)
#Print the name with .jpeg
#print(myAttendee)
#Looping through each particular image and getting its image and attendees name without .jpg
for attendees in myAttendee:
    currentImage=cv2.imread(f'{path}/{attendees}') #AttendanceImage/Ayaan.jpeg
    attendance_image.append(currentImage)
    attendee_name.append(os.path.splitext(attendees)[0])
# Print the name without .jpeg,this will be useful for marking the attendance
#print(attendee_name)


#encoding has to be created for all images
#encoding will help us to identify the image

#we will create a function for getting the encoing of all images
#for this we will pass the list of images to the encoding function

def getEncoding(attendee_name):
    encoding=[]
    for attendees in attendee_name:
        #First convert BGR to RGB
        attendees=cv2.cvtColor(attendees,cv2.COLOR_BGR2RGB)
        encoded_attendees=face_recognition.face_encodings(attendees)[0]
        encoding.append(encoded_attendees)
    return encoding

encodedList=getEncoding(attendance_image)
#To check the attendance list has been appended correctly or not
#print(len(encodedList))

#Now mark Attendance in csv comma separated file
def markAttendance(name):
    with open('Attendance.csv', 'r+') as file:
        dataList=file.readlines()
        nameList=[]
        for data in dataList:
            entry=data.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time=datetime.now()
            date=time.strftime('%H:%M:%S')
            file.writelines(f'\n{name},{date}')
#Now to find the matching with the attendance images through encoding
webcam=cv2.VideoCapture(0)
while True:
    #Now to read the input through the webcam
    success ,image=webcam.read()
    #For fast execution we resize the image
    image_resize=cv2.resize(image,(0,0),None,0.25,0.25)
    #Converting the image from BGR to RGB
    image_resize=cv2.cvtColor(image_resize,cv2.COLOR_BGR2RGB)

    #Through the webcam we get multiple face location and we generate multiple encodings
    FacesInTheFrame=face_recognition.face_locations(image_resize)
    EncodingInTheFrame=face_recognition.face_encodings(image_resize)

    #Run a for loop on both the entites and try to find the minimum face distance for optimal result
    #The encoding with minimum distance will be the detected face

    for encod,face in zip(EncodingInTheFrame,FacesInTheFrame):
        #Will compare the different faces and generate the distances and the minimum distance will be the detected output
        face_match=face_recognition.compare_faces(encodedList,encod)
        face_distance=face_recognition.face_distance(encodedList,encod)
        index=np.argmin(face_distance)
        #Now after getting the index of minimum distance we will try to make a rectangular box around it
        if face_distance[index]<0.5 and face_match[index]:
            nameOfAttendee=attendee_name[index].upper()
            #print(nameOfAttendee)
            y1,x2,y2,x1=face
            #Resizing the image_resize to actual
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(image,(x1,y2-35),(x2,y2),(255,255,255),cv2.FILLED)
            cv2.putText(image,nameOfAttendee,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
            markAttendance(nameOfAttendee)

    cv2.imshow('Webcam',image)
    cv2.waitKey(1)