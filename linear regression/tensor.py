from cv2 import mean
from regex import B
import tensorflow as tf
import random

train_x = [1,2,3,4,5,6,7]
train_y = [3,5,7,9,11,13,15]

a=tf.Variable(0.1)
b=tf.Variable(0.1)

def lostFunction(a,b):
    #meanSquredError
    prediction_y = train_x * a + b
    return tf.keras.losses.mse(train_y, prediction_y)

opt = tf.keras.optimizers.Adam(learning_rate = 0.1)

for i in range(9000):
    opt.minimize(lambda:lostFunction(a,b), var_list = [a,b])
    print(a.numpy(),b.numpy())