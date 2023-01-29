# 데이터 확인
import matplotlib.pyplot as plt # 시각화를 위한 matplotlib 모듈
import glob as gb # 파일들의 리스트를 가져오는 모듈
import pandas as pd
import os
import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

path_test  = "/test/test"
path_train = "/train/train"


s = [] # 빈 리스트 생성

files = gb.glob(pathname= str( path_train + '/*.jpg')) # 해당 폴더 안의 jpg로 끝나는 파일 찾기


for file in files:
    image = plt.imread(file) # 파일 리스트 하나하나에 해당하는 이미지를 가져와서 shape을 저장 
    s.append(image.shape)

# 파일들 마다 shape 어떤 구성을 가졌는지 파악
pd.Series(s).value_counts()

import cv2 # 컴퓨터 비전 관련 모듈

X_train = []
Y_train = []
X_Title=[]


size=180 # 공통된 사이즈로 맞춰줌

files = gb.glob(pathname= str( path_train + '/*.jpg'))
# train 폴더 안의 jpg들의 리스트를 가져옴

for file in files: 
    image = cv2.imread(file)
    image_array = cv2.resize(image , (size,size)) # 180 x 180 으로 이미지를 변환
    X_train.append(list(image_array)) 
    
    file=file.split('/')[-1] 
    sep=file.split('.')[0]
    X_Title.append(sep)

    # 만약, 파일 이름에 'dog' 를 가졌다면 1, 아니면 0으로 구분
    if(sep=='dog'):
        Y_train.append(1)
    else:
        Y_train.append(0)

X_test = []

files = gb.glob(pathname= str( path_test + '/*.jpg'))
for file in files: 
    image = cv2.imread(file)
    image_array = cv2.resize(image , (size,size))
    X_test.append(list(image_array))
    
X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)

print(f'X_train shape  is {X_train.shape}')
print(f'X_test shape  is {X_test.shape}')
print(f'y_train shape  is {Y_train.shape}')

print('Size OF Image = ',np.array(X_train[5]).shape)

X_train, Y_train= shuffle(X_train, Y_train, random_state=True)

Y_train = tf.one_hot(Y_train, 2)

X_Train=X_train[:20000]
Y_Train=Y_train[:20000]
X_val=X_train[20000:]
y_val=Y_train[20000:]

# cat 과 dog 가 같은 비율로 있는 것을 확인
X_Title=np.array(X_Title)
pd.Series(X_Title).value_counts().plot.bar()