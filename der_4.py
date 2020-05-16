import numpy as np

###########################################################
#  derivative matrices to act in 2D as outer prods

def make_der(nx,L):
    d1 = np.zeros((nx+1,nx+1))
    d2 = np.zeros((nx+1,nx+1))

    h = L/(nx)
    
    #calculate 1st and second derivatives with 4th order accuracy
    for i in range(nx+1):
        for j in range(nx+1):
            if j == i-2:
                d1[i,j] =  1/12
                d2[i,j] = -1/12
            elif j == i-1:
                d1[i,j] = -2/3
                d2[i,j] =  4/3
            elif j == i:
                d1[i,j] =  0
                d2[i,j] =  -5/2
            elif j == i+1:
                d1[i,j] =  2/3
                d2[i,j] =  4/3
            elif j == i+2:
                d1[i,j] =  -1/12
                d2[i,j] =  -1/12
            else:
                d1[i,j] = 0
                d2[i,j] = 0
    
    d1[0,0:5] = [-25/12, 4, -3, 4/3, -1/4]
    d1[1,0:5] = [-1/4, -5/6, 3/2, -1/2, 1/12]  
    d1[-2,-5:] = - d1[1,0:5][::-1]
    d1[-1,-5:] = - d1[0,0:5][::-1]
    
    d2[0,0:6] = [15/4, -77/6, 107/6, -13, 61/12, -5/6]
    d2[1,0:6] = [5/6, -5/4, -1/3, 7/6, -1/2, 1/12]  
    d2[-2,-6:] = d2[1,0:6][::-1]
    d2[-1,-6:] = d2[0,0:6][::-1]

    # divide by appropriate grid spacing
    d1 = d1/h
    d2 = d2/h**2

    
###########################################################
  
    Id = np.eye(nx+1)

    d2d0 = np.kron(d2,Id)
    d0d2 = np.kron(Id,d2)
    
###########################################################
    
    # compute grid
    xv = np.linspace(0,L,nx+1) 

    # compute 2D version of x and y
    xvT = np.kron( xv.reshape(nx+1,1) , np.ones(nx+1).reshape(1,nx+1))
    yvT = np.kron( np.ones(nx+1).reshape(nx+1,1), xv.reshape(1,nx+1) )

    #return d1d0, d2d0, d1d1, d0d1, d0d2, xvT, yvT
    return d2d0, d0d2, xvT, yvT

