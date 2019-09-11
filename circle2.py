import numpy as np
import matplotlib.pyplot as plt
from coeffs import*

import subprocess
import shlex

r1=3
o2=np.array([-1,2])
P=np.array([2,2])
o1=2*P-o2
print(o1)
len=100
theta = np.linspace(0,2*np.pi,len)
x_circ = np.zeros((2,len))
x_circ[0,:] = r1*np.cos(theta)
x_circ[1,:] = r1*np.sin(theta)
x_circ = (x_circ.T + o1).T
plt.plot(x_circ[0,:],x_circ[1,:],label='$circumcircle$')

r2=np.sqrt(o2.T@o2+4)
print(r2)

len=100
theta = np.linspace(0,2*np.pi,len)
x_circ = np.zeros((2,len))
x_circ[0,:] = r2*np.cos(theta)
x_circ[1,:] = r2*np.sin(theta)
x_circ = (x_circ.T + o2).T
plt.plot(x_circ[0,:],x_circ[1,:],label='$circumcircle$')

plt.plot(o1[0],o1[1],'o')
plt.text(o1[0]*(1-0.1),o1[1]*(1-0.2),'o1')

plt.plot(o2[0],o2[1],'o')
plt.text(o2[0]*(1-0.1),o2[1]*(1-0.2),'o2')

plt.axis('equal')
plt.legend(loc='best')
plt.grid()
plt.show()
