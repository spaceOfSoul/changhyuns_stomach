import numpy as np

fake_a_b = [3, 76]#slope and intercept

# x,y data (hour and score)
data = [[2,81],[4,93],[6,91],[8,97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

#return result that replace in graph y = ax + b
def predict(x):
    return (fake_a_b[0]*x + fake_a_b[1])

#MSE
def mse(y, y_hat):
    return ((y-y_hat)**2).mean()

# return final value that replacey to mse function
def mse_val(y, predict_result):
    return mse(np.array(y), np.array(predict_result))

#list for predict value
predict_result = []
for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print("study time = %.f, real score = %.f, predict score = %.f" % (x[i],y[i],predict(x[i])))

print('final mse value(different for initial setting): ', str(mse_val(predict_result, y)))