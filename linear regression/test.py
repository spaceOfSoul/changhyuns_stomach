from sklearn.model_selection import learning_curve
import tensorflow as tf

#height = [170, 180, 175, 160]
#shoes = [260, 270, 265, 255]

height = 170
shoes = 260

#shoes = height * a + b

a = tf.Variable(0.1)
b = tf.Variable(0.2)

def lostFunction():
    prediction = height * a+b
    return tf.square(260 - (prediction))

opt = tf.keras.optimizers.Adam(learning_rate = 0.1)

for i in range(300):
    opt.minimize(lostFunction, var_list = [a,b])
    print(a,b)
    
print(height*int(a.numpy())+int(b.numpy()))