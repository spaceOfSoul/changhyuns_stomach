import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#study hour, private_class, score
data = [[2,0,81], [4,4,93], [6, 2, 91], [8,3,97]]
x1 = [i[0] for i in data]
x2 = [i[1] for i in data]
y = [i[2] for i in data]

#figure graph
ax = plt.axes(projection='3d')
ax.set_xlabel('study_hours')
ax.set_ylabel('private_class')
ax.set_zlabel('Score')

ax.dist = 11
ax.scatter(x1,x2,y)
plt.show()

x1_data = np.array(x1)
x2_data = np.array(x2)
y_data = np.array(y)

#a1, a2 is slope
#b is intercept
a1=0
a2=0
b=0

lr = 0.05

epochs = 2001

#start linear regression
for i in range(epochs):
    y_pred = a1*x1_data + a2*x2_data + b#function of 
    error = y_data - y_pred#get error value(differ value)
    
    #differentiate errorFunction to a1
    a1_diff = -(1/len(x1_data)) * sum(x1_data * (error)) 
    #differentaite errorFunction to a2
    a2_diff = -(1/len(x2_data)) * sum(x2_data * (error)) 
    #differentiate errorFunction to intercept
    b_diff = -(1/len(x1_data)) * sum(y_data - y_pred)
    
    #update a1 to product learningRate
    a1 = a1-lr*a1_diff
    #update a2 to product learningRate
    a2 = a2-lr*a2_diff
    #update b to product learnigRate
    b = b-lr * b_diff
    
    if i % 100 == 0:
        print("epoch=%.f, slope1=%.04f, slope2=%.04f, intercept=%.04f"%(i,a1,a2,b))

    
    