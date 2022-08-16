import numpy as np
from matplotlib import pyplot as plt
import math
from sympy import *
# Lagrange Multipliers

# the constraint conditions I h1(x): x^2 + y^2 + z^2 + u^2 = 1
# the constraint conditions II h2(x): x + y + z + u = 1
def h1(x):
    h = sum(x[0:]**2.0) - 1
    return h

def h2(x):
    h = sum(x[0:]) - 1
    return h

# the target function f(x)
def f(x):
    f = np.prod(x[0:])
    return f

# the penalty function p(x)
def p(x):
    p = h1(x)**2
    return p

# construct the new unconstrained target function L(x)
def L(x,lam,sigma):
    L1 = f(x)
    L2 = lam[0]*h1(x)
    L3 = sigma*p(x)
    return L1 + L2 + L3

# initial values
# Nx: number of independent variable
# Nlambda: number of constraint conditions, only considering equality constraint 
Nlambda = 1
Nx = 4
lamin = np.zeros((Nlambda,1))
sigmain = 1.0
judge = 1.0

for i in range(0,Nlambda):
    lamin[i] = 1.0

# for i in range(0,10):
while(np.abs(judge) >= 1e-10):
    # optimization algorithm based on line search: Newton method
    deltaL = 1
    xin = np.array([[1],[1],[1],[1]])
#     for jj in range(0,10):
    while(np.abs(deltaL) >= 1e-12):
        Lin = L(xin,lamin,sigmain)
        # construct the Hesse matrix
        Hesse = np.zeros((Nx,Nx))
        dL = np.zeros((Nx,1))

        Hesse[0,0] = 2*lamin[0] + 8*sigmain*xin[0]**2 + 4*sigmain*(xin[3]**2 + xin[0]**2 + \
                                                                   xin[1]**2 + xin[2]**2 - 1)
        Hesse[0,1] = 8*sigmain*xin[0]*xin[1] + xin[3]*xin[2]
        Hesse[0,2] = 8*sigmain*xin[0]*xin[2] + xin[3]*xin[1]
        Hesse[0,3] = 8*sigmain*xin[3]*xin[0] + xin[1]*xin[2]
        Hesse[1,0] = 8*sigmain*xin[0]*xin[1] + xin[3]*xin[2]
        Hesse[1,1] = 2*lamin[0] + 8*sigmain*xin[1]**2 + 4*sigmain*(xin[3]**2 + xin[0]**2 + xin[1]**2 + \
                                                      xin[2]**2 - 1)
        Hesse[1,2] = 8*sigmain*xin[1]*xin[2] + xin[3]*xin[0]
        Hesse[1,3] = 8*sigmain*xin[3]*xin[1] + xin[0]*xin[2]
        Hesse[2,0] = 8*sigmain*xin[0]*xin[2] + xin[3]*xin[1]
        Hesse[2,1] = 8*sigmain*xin[1]*xin[2] + xin[3]*xin[0]
        Hesse[2,2] = 2*lamin[0] + 8*sigmain*xin[2]**2 + 4*sigmain*(xin[3]**2 + xin[0]**2 + xin[1]**2 + \
                                                      xin[2]**2 - 1)
        Hesse[2,3] = 8*sigmain*xin[3]*xin[2] + xin[0]*xin[1]
        Hesse[3,0] = 8*sigmain*xin[3]*xin[0] + xin[1]*xin[2]
        Hesse[3,1] = 8*sigmain*xin[3]*xin[1] + xin[0]*xin[2]
        Hesse[3,2] = 8*sigmain*xin[3]*xin[2] + xin[0]*xin[1]
        Hesse[3,3] = 2*lamin[0] + 8*sigmain*xin[3]**2 + 4*sigmain*(xin[3]**2 + xin[0]**2 + xin[1]**2 + \
                                                      xin[2]**2 - 1)

        dL[0,0] = 2*lamin[0]*xin[0] + 4*sigmain*xin[0]*(xin[3]**2 + xin[0]**2 + xin[1]**2 + xin[2]**2 - 1)\
        + xin[3]*xin[1]*xin[2]
        dL[1,0] = 2*lamin[0]*xin[1] + 4*sigmain*xin[1]*(xin[3]**2 + xin[0]**2 + xin[1]**2 + xin[2]**2 - 1)\
        + xin[3]*xin[0]*xin[2]
        dL[2,0] = 2*lamin[0]*xin[2] + 4*sigmain*xin[2]*(xin[3]**2 + xin[0]**2 + xin[1]**2 + xin[2]**2 - 1)\
        + xin[3]*xin[0]*xin[1]
        dL[3,0] = 2*lamin[0]*xin[3] + 4*sigmain*xin[3]*(xin[3]**2 + xin[0]**2 + xin[1]**2 + xin[2]**2 - 1)\
        + xin[0]*xin[1]*xin[2]
        xout = xin - np.linalg.inv(Hesse)@dL
        Lout = L(xout,lamin,sigmain)
        deltaL = Lout - Lin
        deltax = xout - xin
        xin = xout*1.0
        print('dx: '+str(np.linalg.norm(deltax)))
        print('dL: '+str(deltaL))
    np.set_printoptions(precision=12,suppress=True)
    lamout = lamin + 2*sigmain*h1(xout)
    sigmaout = sigmain*1.001
    judge = sigmaout*p(xout)
    lamin = lamout*1.0
    sigmain = sigmaout*1.0
    print('judge: '+str(judge))
    
np.set_printoptions(threshold=np.inf)
print('final result: '+str(f(xout)))

#######################################################################
###                   calculate the Hesse matirx                    ###  
#######################################################################

import math
from sympy import *
line = [x,y,z,u]
lam ,sigma ,x ,y ,z ,u = symbols('lam sigma x y z u')
v = x*y*z*u + lam*(x**2 + y**2 + z**2 + u**2 - 1) + sigma*(x**2 + y**2 + z**2 + u**2 - 1)**2
# print(str(x)+str(x)+': '+diff(diff(v,x),x))
# print(str(x)+str(y)+': '+diff(diff(v,x),y))
# print(str(x)+str(z)+': '+diff(diff(v,x),z))
# print(str(x)+str(u)+': '+diff(diff(v,x),u))
# print(str(y)+str(x)+': '+diff(diff(v,y),x))
# print(str(y)+str(y)+': '+diff(diff(v,y),y))
# print(str(y)+str(z)+': '+diff(diff(v,y),z))
# print(str(y)+str(u)+': '+diff(diff(v,y),u))
# print(str(z)+str(x)+': '+diff(diff(v,z),x))
# print(str(z)+str(y)+': '+diff(diff(v,z),y))
# print(str(z)+str(z)+': '+diff(diff(v,z),z))
# print(str(z)+str(u)+': '+diff(diff(v,z),u))
# print(str(u)+str(x)+': '+diff(diff(v,u),x))
# print(str(u)+str(y)+': '+diff(diff(v,u),y))
# print(str(u)+str(z)+': '+diff(diff(v,u),z))
# print(str(u)+str(u)+': '+diff(diff(v,u),u))
for i in line:
    for j in line:
        print(str(i)+str(j)+': '+str(diff(diff(v,i),j)))
for i in line:
    print(str(i)+' : '+str(diff(v,i)))
del x,y,z,u,lam,sigma
