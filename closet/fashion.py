import matplotlib
from sklearn import metrics
import tensorflow as tf
import matplotlib
import numpy as np

#(inputData, answer),(testx, testy)
(trainX, trainY),(testX, testY) =tf.keras.datasets.fashion_mnist.load_data()
# print(trainY)
# print(trainY.shape)

trainX = trainX / 255.0
testX = testX / 255.0

trainX=trainX.reshape((trainX.shape[0], 28, 28, 1))
testX=testX.reshape((testX.shape[0], 28, 28, 1))


class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

model = tf.keras.Sequential([
     # different feature image 32, kernel size, padding, activation function, inputshape
     # Conv2D는 4차원의 데이터 필요 Ex)(60000,28,28,1),(60000,28,28,3)
     # 기존의 input_shape가 2차원(28,28)이므로 이를 한차원 늘려서 인풋 데이터의 개형을 정의
    tf.keras.layers.Conv2D( 32, (3,3), padding='same', activation = 'relu', input_shape=(28,28,1) ), #1일경우 흑백(명암 데이터만 있으므로), 컬러는 3(rgb값이 있으므로)
    tf.keras.layers.MaxPooling2D((2,2)),#데이터 풀링
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')#10가지 경우의 확률
])

model.summary()
#categorical_crossentropy : 여러 카테고리의 확률을 예측 시 trainY가 원핫 인코딩 되어있을 시 사용
#sparse_categorical_crossentropy : 정수로 인코딩되어 있을 시
model.compile( loss = 'sparse_categorical_crossentropy',  optimizer = 'adam', metrics=['accuracy'])
model.fit(trainX,trainY, validation_data = (testX, testY) , epochs = 50)

score = model.evaluate(testX,testY)
print(score)