# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 03:51:36 2018

@author: mire
"""

import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from PIL import Image
import glob,os

#64*64にリサイズ
in_dir = "./origin_image/*"
out_dir = "./face_image"
in_jpg = glob.glob(in_dir)
in_fileName = os.listdir("./origin_image/")

#print(in_jpg)
#print(in_fileName)
print(len(in_jpg))
for num in range(len(in_jpg)):
    image = cv2.imread(str(in_jpg[num]))
    if image is None:
        print("Not open:", str(in_jpg[num]))
        continue
    
    image_gs = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier("/Users/mire/Anaconda3/Library/etc/haarcascades/haarcascade_frontalface_alt.xml")
    
    #顔認識
    #cascadeは分類器
    face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1,minNeighbors=2,minSize=(64,64))
    #顔が１つ以上検出
    if len(face_list) > 0:
        for rect in face_list:
            x,y,width,height=rect
            image = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            if image.shape[0]<64:
                continue
            image = cv2.resize(image,(64,64))
            
    #顔が検出されない
    else:
        print("no face")
        continue
    print(image.shape)
    
    #保存
    fileName=os.path.join(out_dir,str(in_fileName[num]))
    cv2.imwrite(str(fileName),image)
    
in_dir = "./face_image/*"
in_jpg = glob.glob(in_dir)
img_file_name_list = os.listdir("./face_image/")
 #img_file_name_listをシャッフル、そのうち2割をtest_imageディレクトリにいれる
random.shuffle(in_jpg)
import shutil
 
for i in range(len(in_jpg) // 5):
     shutil.move(str(in_jpg[i]),"./test_image")
    
    