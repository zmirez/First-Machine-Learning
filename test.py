# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 16:18:13 2018

@author: mire
"""

import numpy as np
import cv2
from keras.models import load_model
import matplotlib.pyplot as plt

def detect_face(image):
    print(image.shape)
    #opencvを使って顔抽出
    image_gs=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cascade=cv2.CascadeClassifier("/Users/mire/Anaconda3/Library/etc/haarcascades/haarcascade_frontalface_alt.xml")
    
    #顔認識
    face_list=cascade.detectMultiScale(image_gs,scaleFactor=1.1,minNeighbors=2,minSize=(64,64))
    
    if len(face_list)>0:
        for rect in face_list:
            x,y,width,height=rect
            cv2.rectangle(image,tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]),(255,0,0),thickness=3)
            img=image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            if image.shape[0]<64:
                print("too small")
                continue
            img=cv2.resize(image,(64,64))
            img=np.expand_dims(img,axis=0)
            name=detect_who(img)
            cv2.putText(image,name,(x,y+height+20),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
    else:
        print("no face")
    return image

def detect_who(img):
    #予測
    name=""
    print(model.predict(img))
    nameNumlabel=np.argmax(model.predict(img))
    if nameNumlabel==0:
        name="zico"
    elif nameNumlabel==1:
        name="u-kwon"
    elif nameNumlabel==2:
        name="taeil"
    elif nameNumlabel==3:
        name="packkyoung"
    
    return name

model=load_model('./my_model.h5')
image=cv2.imread("./")
if image is None:
    print("Not open:")
b,g,r=cv2.split(image)
image=cv2.merge([r,g,b])
whoImage=detect_face(image)

plt.imshow(whoImage)
plt.show()
            