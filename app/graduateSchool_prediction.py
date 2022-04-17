
import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv('app\gpascore.csv')
print(data)

data = data.dropna()

ydata = data['admit'].values
xdata=[]
for i, rows in data.iterrows():
    xdata.append([ rows['gre'], rows['gpa'], rows['rank'] ])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit( np.array(xdata), np.array(ydata), epochs=1000)

while True:
    a = int(input('toeic score : '))
    b = float(input('credit score : '))
    c = int(input('school rank :'))
    predict = model.predict([[a, b, c]])
    print(predict)
exit()