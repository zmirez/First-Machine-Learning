# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 01:04:52 2018

@author: mire
"""
import os
import cv2
import numpy as np
from keras.models import load_model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from keras.layers import Activation,Conv2D,Dense,Flatten,MaxPooling2D,Dropout
from keras.models import Sequential,load_model
from keras.utils.np_utils import to_categorical



img_file_name_list=os.listdir("./face_scratch_image/")
print(len(img_file_name_list))

for i in range(len(img_file_name_list)):
    n=os.path.join("./face_scratch_image",img_file_name_list[i])
    img=cv2.imread(n)
    if isinstance(img,type(None))==True:
        img_file_name_list.pop(i)
        continue
#print(img_file_name_list[0:2])


X_train=[]
label_train=[]

for j in range(0,len(img_file_name_list)-1):
    n=os.path.join("./face_scratch_image/",img_file_name_list[j])
    img=cv2.imread(n)
    b,g,r=cv2.split(img)
    img=cv2.merge([r,g,b])
    X_train.append(img)
    n=img_file_name_list[j]
    #配列の1,2番目を取得
    label_train=np.append(label_train,int(n[0:2])).reshape(j+1,1)
    
X_train=np.array(X_train)

#test画像の枚数
img_file_name_list=os.listdir("./test_image/")

for i in range(len(img_file_name_list)):
    n=os.path.join("./test_image",img_file_name_list[i])
    img=cv2.imread(n)
    if isinstance(img,type(None))==True:
        img_file_name_list.pop(i)
        continue

#print(len(img[0]))
X_test=[]
label_test=[]


for j in range(0,len(img_file_name_list)):
    n=os.path.join("./test_image",img_file_name_list[j])
    img=cv2.imread(n)
    b,g,r=cv2.split(img)
    img = cv2.merge([r,g,b])
    X_test.append(img)
    n=img_file_name_list[j]
    label_test=np.append(label_test,int(n[0:2])).reshape(j+1,1)

X_test=np.array(X_test)
    
label_train=to_categorical(label_train)
label_test=to_categorical(label_test)

#モデルの定義
model=Sequential()
model.add(Conv2D(input_shape=(64,64,3),filters=32,kernel_size=(2,2),strides=(1,1),padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=32,kernel_size=(2,2),strides=(1,1),padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=32,kernel_size=(2,2),strides=(1,1),padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("sigmoid"))
model.add(Dense(128))
model.add(Activation("sigmoid"))
model.add(Dense(4))
model.add(Activation("softmax"))

#コンパイル
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
#学習
model.fit(X_train,label_train,batch_size=32,epochs=50)
history = model.fit(X_train, label_train, batch_size=32, epochs=100, verbose=1, validation_data=(X_test, label_test))
#汎化制度の評価・表示
score=model.evaluate(X_test,label_test,batch_size=32,verbose=0)
print('validation loss:{0[0]}\nvalidation accuracy:{0[1]}'.format(score))

#acc,val_accのプロット
plt.plot(history.history["acc"],label="acc",ls="-",marker="o")
plt.plot(history.history["loss"],label="loss-acc",ls="-",marker="x")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()

#modelの保存
#model.save("my_model.h5")





