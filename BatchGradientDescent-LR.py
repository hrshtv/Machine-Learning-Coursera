from matplotlib import pyplot as plt
import numpy as np

# Random data
x = [1,4,3,2,4,5,5,3,3,4,4,4,4,4,4,1,2]
y = [1,2,3,4,5,6,7,8,9,10,2,3,4,5,4,4,3]

x = np.asarray(x)
y = np.asarray(y)
m = len(x) 

# Parameters for straight line
p0 = 0
p1 = 0

a = 0.001 # Learning rate
    
while(True):
    
    s0 = np.sum(p1*x + p0 - y)
    t0 = p0 - (a/m)*s0
    
    s1 = np.sum((p1*x + p0 - y)*x)
    t1 = p1 - (a/m)*s1
    
    # Simultaneous updating
    p0 = t0
    p1 = t1
    
    if (abs(s0) < 0.01 and abs(s1) < 0.01):
        break 

x_fit = np.linspace(np.amax(x), np.amin(y), 100)
y_fit = p1 * x_fit + p0

plt.scatter(x,y)
plt.plot(x_fit, y_fit)
