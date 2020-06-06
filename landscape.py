import numpy as np
#import matplotlib.pyplot as plt
from scipy import interpolate

# creates 1st, 2nd derivatives of 4th order
import der_4 as d4

def gaussian(X, sigma, x, y):
    X = np.array(X)
    return np.exp(  -( (x-X[0])**2 + (y-X[1])**2)/(2*0.025)  )

def sum_gaussians(points, sigma, x, y):
    output = np.zeros( x.shape )
    for point in points:
        output += gaussian(point, sigma, x, y)
    return output

def change_bndy(B,s):
    A = B
    A[0,:] = s; A[-1,:] = s; A[:,0] = s; A[:,-1] = s;
    return A

def find_landscape(box_size, terminals, Nx0 = 50, A0 = 5000):

    '''
    Computes landscape function for a set of points. Output is a 2D interpolaiton 
    
    args:
    
    Nx0: grid points on each direction
    L: size of the box
    A0: strength of the potential at point location
    points: list of points where important nodes are
    
    '''
    
    L = box_size; points = terminals;
    
    #1D grid
    xv = np.linspace(0, L, Nx0+1) 
    
    #derivative matrices 
    d2d0, d0d2, xvT, yvT = d4.make_der(Nx0,L)
    
    # solve linear problem with BC's
    C1 = np.ones((Nx0+1,Nx0+1))
    C1Flat = np.diag( change_bndy(C1, 0.).reshape((Nx0+1)*(Nx0+1)) )

    V =  sum_gaussians(points, 0.5, xvT, yvT)
    VFlat = A0*np.diag( change_bndy(V, 1.).reshape((Nx0+1)*(Nx0+1)) )

    J = np.ones((Nx0+1,Nx0+1))
    JFlat = change_bndy(J,0.).reshape((Nx0+1)*(Nx0+1)) 

    LHS = - np.dot(C1Flat, d2d0 + d0d2) + VFlat
    
    soln = np.linalg.solve(LHS, JFlat).reshape(Nx0+1,Nx0+1)
    
    # interpolate
    fLand = interpolate.RectBivariateSpline(xv, xv, soln)
    
    return fLand