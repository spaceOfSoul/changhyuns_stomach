import matplotlib
from sklearn import metrics
import tensorflow as tf
import matplotlib

#(inputData, answer),(testx, testy)
(trainX, trainY),(testX, testY) =tf.keras.datasets.fashion_mnist.load_data()
# print(trainY)
# print(trainY.shape)

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(28,28), activation='relu'),#input_shape : shape of input data
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10,activation='softmax')#10가지 경우의 확률
])

model.summary()
#categorical_crossentropy : 여러 카테고리의 확률을 예측 시 trainY가 원핫 인코딩 되어있을 시 사용
#sparse_categorical_crossentropy : 정수로 인코딩되어 있을 시
model.compile( loss = 'sparse_categorical_crossentropy',  optimizer = 'adam', metrics=['accuracy'])
model.fit(trainX,trainY, epochs = 100)