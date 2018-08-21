# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 18:49:44 2018

@author: mire
"""

#r,g,bのベクトルにして1000個のデータをpcaする

import os
import cv2
import numpy as np
from keras.models import load_model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from keras.layers import Activation,Conv2D,Dense,Flatten,MaxPooling2D,Dropout
from keras.models import Sequential,load_model
from keras.utils.np_utils import to_categorical



pca=PCA(n_components=48)


img_file_name_list=os.listdir("./face_scratch_image/")
print(len(img_file_name_list))

for i in range(len(img_file_name_list)):
    n=os.path.join("./face_scratch_image",img_file_name_list[i])
    img=cv2.imread(n)
    if isinstance(img,type(None))==True:
        img_file_name_list.pop(i)
        continue
#print(img_file_name_list[0:2])


X_train=np.array([])
label_train=np.array([])
image=[]

for j in range(0,len(img_file_name_list)-1):
    n=os.path.join("./face_scratch_image/",img_file_name_list[j])
    img=cv2.imread(n)
    b,g,r=cv2.split(img)
    b_img=np.reshape(b,-1)
    g_img=np.reshape(g,-1)
    b_img=np.append(b_img,g_img)
    r_img=np.reshape(r,-1)
    b_img=np.append(b_img,r_img)
    #print(b_img.shape)
    X_train=np.append(X_train,b_img)
    #print(X_train.shape)
    n=img_file_name_list[j]
    #配列の1,2番目を取得
    label_train=np.append(label_train,int(n[0:2])).reshape(j+1,1)
    
    
#X_train=np.array(X_train)
#print(X_train.shape)
X_train=X_train.reshape(1175,12288)
pca.fit(X_train)
X_pca=pca.transform(X_train)
print(X_pca.shape)
#E=pca.explained_variance_ratio_ #寄与率
#print(E)
#print(sum(E)) #累積寄与率

#test画像の枚数
img_file_name_list=os.listdir("./test_image/")

for i in range(len(img_file_name_list)):
    n=os.path.join("./test_image",img_file_name_list[i])
    img=cv2.imread(n)
    if isinstance(img,type(None))==True:
        img_file_name_list.pop(i)
        continue

#print(len(img[0]))
X_test=np.array([])
label_test=np.array([])


for j in range(0,len(img_file_name_list)):
    n=os.path.join("./test_image",img_file_name_list[j])
    img=cv2.imread(n)
    b,g,r=cv2.split(img)
    b_img=np.reshape(b,-1)
    g_img=np.reshape(g,-1)
    b_img=np.append(b_img,g_img)
    r_img=np.reshape(r,-1)
    b_img=np.append(b_img,r_img)
    #print(b_img.shape)
    X_test=np.append(X_test,b_img)
   # pca.fit(b)
   # b_pca=pca.transform(b)
    #pca.fit(g)
    #g_pca=pca.transform(g)
    #pca.fit(r)
    #r_pca=pca.transform(r)
    n=img_file_name_list[j]
    label_test=np.append(label_test,int(n[0:2])).reshape(j+1,1)
    
#X_test=np.array(X_test)
#E=pca.explained_variance_ratio_
#print(E)
print(X_test.shape)
X_test=X_test.reshape(48,12288)
#pca=PCA(n_components=64)
pca.fit(X_test)
x_pca=pca.transform(X_test)
print(x_pca.shape)

label_train=to_categorical(label_train)
label_test=to_categorical(label_test)


#モデルの定義
model=Sequential()
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dense(4))
model.add(Activation("softmax"))

#コンパイル
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

#print(X_train.shape)
#学習
model.fit(X_pca,label_train,epochs=50)
print("学習したよ！")
#グラフ用
history=model.fit(X_pca,label_train,batch_size=32,epochs=50,verbose=1,validation_data=(x_pca,label_test))

#汎化制度の評価・表示
score=model.evaluate(x_pca,label_test,batch_size=32,verbose=0)
print("評価してみた！")
print(score[0])
print(score[1])
print('validation loss:{0[0]}\nvalidation accuracy:{0[1]}'.format(score))

#acc, val_accのプロット
plt.plot(history.history["acc"], label="acc", ls="-", marker="o")
plt.plot(history.history["loss"], label="loss", ls="-", marker="x")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()
#def detect_face(image):
#    print(image.shape)
#    #opencvを使って顔抽出
#    image_gs=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#    cascade=cv2.CascadeClassifier("/Users/mire/Anaconda3/Library/etc/haarcascades/haarcascade_frontalface_alt.xml")
#    
#    #顔認識
#    face_list=cascade.detectMultiScale(image_gs,scaleFactor=1.1,minNeighbors=2,minSize=(64,64))
#    
#    if len(face_list)>0:
#        for rect in face_list:
#            x,y,width,height=rect
#            cv2.rectangle(image,tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]),(255,0,0),thickness=3)
#            img=image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
#            if image.shape[0]<64:
#                print("too small")
#                continue
#            img=cv2.resize(image,(64,64))
#            img=np.expand_dims(img,axis=0)
#            name=detect_who(img)
#            cv2.putText(image,name,(x,y+height+20),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
#    else:
#        print("no face")
#    return image
#
#def detect_who(img):
#    #予測
#    name=""
#    print(model.predict(img))
#    nameNumlabel=np.argmax(model.predict(img))
#    if nameNumlabel==0:
#        name="zico"
#    elif nameNumlabel==1:
#        name="u-kwon"
#    elif nameNumlabel==2:
#        name="taeil"
#    elif nameNumlabel==3:
#        name="packkyoung"
#    
#    return name
#
#model=load_model('./my_model.h5')
#image=cv2.imread("./origin_image/03.74.jpg")
##print(image)
#if image is None:
#    print("Not open:")
#b,g,r=cv2.split(image)
#image=cv2.merge([r,g,b])
#whoImage=detect_face(image)
#
#plt.imshow(whoImage)
#plt.show()
