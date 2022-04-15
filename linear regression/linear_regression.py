import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = [[2,81],[4,93],[6,91],[8,97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

plt.figure(figsize=(8,5))
plt.scatter(x,y)
plt.show()

x_data = np.array(x)
y_data = np.array(y)

#slope and intercept
a = 0
b = 0

#learning rate
lr = 0.03

#loop count
epochs = 2001

#start linear regression
for i in range(epochs):
    y_pred = a* x_data +b
    error = y_data - y_pred#get error value(differ value)
    #differentiate errorFunction to a(slope)
    a_diff = -(2/len(x_data)) * sum(x_data* (error))
    #differentiate errorFunction to b(intercept)
    b_diff = -(2/len(x_data)) * sum(error)
    
    a = a - lr * a_diff
    b = b - lr * b_diff
    print(a)

    if i % 100 == 0:
        print("epoch=%.f, slope=%.3f, intercept=%.3f" % (i,a,b))
        
y_pred = a*x_data+b
plt.scatter(x,y)
plt.plot([min(x_data),max(x_data)], [min(y_pred), max(y_pred)])
plt.show()