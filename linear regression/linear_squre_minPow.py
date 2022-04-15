import numpy as np

x = [2, 4, 6, 8]#var
y = [81, 93, 91, 97]#result

#averrage
mx = np.mean(x)
my = np.mean(y)
print("average of x : ", mx)
print("average of y : ", my)

#denominator of slope
divisor = sum([(mx-i)**2 for i in x])

#numerator of slope
def top(x,mx,y,my):
    d=0
    for i in range(len(x)):
        d+= (x[i] - mx)* (y[i] - my)
    return d

dividend = top(x,mx,y,my)
print("denominator : ", divisor)
print("numerator : ", dividend)

a = dividend / divisor
b = my - (mx*a)

print("slppe a = ",a)
print("intercept b = ", b)