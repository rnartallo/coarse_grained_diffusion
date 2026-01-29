import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import lsmr, lsqr
import scipy as sp

def OU(theta,eps,x0,tau,samples):
    B = np.array([[2,-theta],[+theta,2]])
    D = np.array([[eps,0],[0,eps]])
    S = np.array([[eps/2,0],[0,eps/2]])
    Sinv = np.array([[2/eps,0],[0,2/eps]])
    expb = expm(-tau*B)
    Q = B@S-D
    C = np.array([[np.exp(2*tau)*eps*np.sinh(2*tau),0],[0,np.exp(2*tau)*eps*np.sinh(2*tau)]])


    N = x0.shape[0]
    X = np.zeros((N,samples))
    
    X[:,0] = x0


    for s in range(1,samples):
        xi = np.random.multivariate_normal(np.reshape(np.zeros((N,1)),N),C)
        X[:,s] = expb@X[:,s-1] + xi
    
    return(X)

def SG_2d_reflecting_boundaries(f,D,dx,xmin,xmax,dy,ymin,ymax):
    
    [f_x,f_y] = f
    [D_xx,D_yy] = D
    
    x_midpoints = np.linspace(xmin+dx/2,xmax-dx/2,int((xmax-xmin)/dx))
    y_midpoints = np.linspace(ymin+dy/2,ymax-dy/2,int((ymax-ymin)/dy))
    x_left = np.linspace(xmin,xmax-dx,int((xmax-xmin)/dx))
    x_right = np.linspace(xmin+dx,xmax,int((xmax-xmin)/dx))
    y_left = np.linspace(xmin,xmax-dx,int((xmax-xmin)/dx))
    y_right = np.linspace(xmin+dx,xmax,int((xmax-xmin)/dx))
    N = x_midpoints.shape[0]
    M = y_midpoints.shape[0]
    states = N*M
    L = np.zeros((states,states))
    states_idx = []
    for i_x in range(N):
        for i_y in range(M):
            states_idx.append((i_x,i_y))
    for idx in range(0,states):
        [i_x,i_y] = states_idx[idx]
        if i_x<N-1:
            right = states_idx.index((i_x+1,i_y))
            if np.abs(np.exp(-f_x(x_right[i_x],y_midpoints[i_y])*dx/D_xx(x_right[i_x],y_midpoints[i_y]))-1)>10**-6:
                L[idx,right] = (1/dx)*f_x(x_right[i_x],y_midpoints[i_y])*np.exp(-f_x(x_right[i_x],y_midpoints[i_y])*dx/D_xx(x_right[i_x],y_midpoints[i_y]))/(1-np.exp(-f_x(x_right[i_x],y_midpoints[i_y])*dx/D_xx(x_right[i_x],y_midpoints[i_y])))
            else:
                L[idx,right] = D_xx(x_right[i_x],y_midpoints[i_y])/(dx**2)                
        if i_y<N-1:
            top = states_idx.index((i_x,i_y+1))
            if np.abs(np.exp(-f_y(x_midpoints[i_x],y_right[i_y])*dy/D_yy(x_midpoints[i_x],y_right[i_y]))-1)>10**-6:
                L[idx,top] = (1/dy)*f_y(x_midpoints[i_x],y_right[i_y])*np.exp(-f_y(x_midpoints[i_x],y_right[i_y])*dy/D_yy(x_midpoints[i_x],y_right[i_y]))/(1-np.exp(-f_y(x_midpoints[i_x],y_right[i_y])*dy/D_yy(x_midpoints[i_x],y_right[i_y])))
            else:
                L[idx,top] = D_yy(x_midpoints[i_x],y_right[i_y])/(dy**2)
        if i_x>0:
            left = states_idx.index((i_x-1,i_y))
            if np.abs(np.exp(-f_x(x_left[i_x],y_midpoints[i_y])*dx/D_xx(x_left[i_x],y_midpoints[i_y]))-1)>10**-6:
                L[idx,left] = (1/dx)*f_x(x_left[i_x],y_midpoints[i_y])/(1-np.exp(-f_x(x_left[i_x],y_midpoints[i_y])*dx/D_xx(x_left[i_x],y_midpoints[i_y])))
            else:
                L[idx,left] = D_xx(x_left[i_x],y_midpoints[i_y])/(dx**2)
        if i_y>0:
            bottom = states_idx.index((i_x,i_y-1))
            if np.abs(np.exp(-f_y(x_midpoints[i_x],y_left[i_y])*dy/D_yy(x_midpoints[i_x],y_left[i_y]))-1)>10**-6:
                L[idx,bottom] = (1/dy)*f_y(x_midpoints[i_x],y_left[i_y])/(1-np.exp(-f_y(x_midpoints[i_x],y_left[i_y])*dy/D_yy(x_midpoints[i_x],y_left[i_y])))
            else:
                L[idx,bottom] = D_yy(x_midpoints[i_x],y_left[i_y])/(dy**2)
                
        # self loop with reflecting boundary conditions
        if i_x<N-1:
            if np.abs(np.exp(-f_x(x_right[i_x],y_midpoints[i_y])*dx/D_xx(x_right[i_x],y_midpoints[i_y]))-1)>10**-6:
                frx = (1/dx)*f_x(x_right[i_x],y_midpoints[i_y])/(1-np.exp(-f_x(x_right[i_x],y_midpoints[i_y])*dx/D_xx(x_right[i_x],y_midpoints[i_y])))
            else:
                frx = D_xx(x_right[i_x],y_midpoints[i_y])/(dx**2)
        if i_x>0:
            if np.abs(np.exp(-f_x(x_left[i_x],y_midpoints[i_y])*dx/D_xx(x_left[i_x],y_midpoints[i_y]))-1)>10**-6:
                flx = (1/dx)*f_x(x_left[i_x],y_midpoints[i_y])*np.exp(-f_x(x_left[i_x],y_midpoints[i_y])*dx/D_xx(x_left[i_x],y_midpoints[i_y]))/(1-np.exp(-f_x(x_left[i_x],y_midpoints[i_y])*dx/D_xx(x_left[i_x],y_midpoints[i_y])))
            else:
                flx = D_xx(x_left[i_x],y_midpoints[i_y])/(dx**2)
        if i_y<N-1:
            if np.abs(np.exp(-f_y(x_midpoints[i_x],y_right[i_y])*dy/D_yy(x_midpoints[i_x],y_right[i_y]))-1)>10**-6:
                fry = (1/dy)*f_y(x_midpoints[i_x],y_right[i_y])/(1-np.exp(-f_y(x_midpoints[i_x],y_right[i_y])*dy/D_yy(x_midpoints[i_x],y_right[i_y])))
            else:
                fry = D_yy(x_midpoints[i_x],y_right[i_y])/(dy**2)
        if i_y>0:
            if np.abs(np.exp(-f_y(x_midpoints[i_x],y_left[i_y])*dy/D_yy(x_midpoints[i_x],y_left[i_y]))-1)>10**-6:
                fly = (1/dy)*f_y(x_midpoints[i_x],y_left[i_y])*np.exp(-f_y(x_midpoints[i_x],y_left[i_y])*dy/D_yy(x_midpoints[i_x],y_left[i_y]))/(1-np.exp(-f_y(x_midpoints[i_x],y_left[i_y])*dy/D_yy(x_midpoints[i_x],y_left[i_y])))
            else:
                fly = D_yy(x_midpoints[i_x],y_left[i_y])/(dy**2)

        if 0<i_x<N-1 and 0<i_y<N-1:
            L[idx,idx] = -(frx+flx) - (fry+fly)
        elif i_x==0 and 0<i_y<N-1:
            L[idx,idx] = -(frx) - (fry+fly)
        elif i_x==(N-1) and 0<i_y<N-1:
            L[idx,idx]= -(flx) - (fry+fly)
        elif 0<i_x<N-1 and i_y==0:
            L[idx,idx]= -(frx+flx) - (fry)
        elif 0<i_x<N-1 and i_y==(N-1):
            L[idx,idx]= -(frx+flx) - (fly)
        elif i_x==0 and i_y==0:
            L[idx,idx]= -(frx) - (fry)
        elif i_x==0 and i_y==(N-1):
            L[idx,idx]= -(frx) - (fly)
        elif i_x==(N-1) and i_y==0:
            L[idx,idx]= -(flx) - (fry)
        elif i_x==(N-1) and i_y==(N-1):
            L[idx,idx] =-(flx) - (fly)
    
    return(L,x_midpoints,y_midpoints,x_left,x_right,y_left,y_right)


def CC_2d_reflecting_boundaries(f,D,dx,xmin,xmax,dy,ymin,ymax):
    
    [f_x,f_y] = f
    [D_xx,D_yy] = D
    
    x_midpoints = np.linspace(xmin+dx/2,xmax-dx/2,int((xmax-xmin)/dx))
    y_midpoints = np.linspace(ymin+dy/2,ymax-dy/2,int((ymax-ymin)/dy))
    x_left = np.linspace(xmin,xmax-dx,int((xmax-xmin)/dx))
    x_right = np.linspace(xmin+dx,xmax,int((xmax-xmin)/dx))
    y_left = np.linspace(xmin,xmax-dx,int((xmax-xmin)/dx))
    y_right = np.linspace(xmin+dx,xmax,int((xmax-xmin)/dx))
    N = x_midpoints.shape[0]
    M = y_midpoints.shape[0]
    states = N*M
    L = np.zeros((states,states))
    states_idx = []
    for i_x in range(N):
        for i_y in range(M):
            states_idx.append((i_x,i_y))
    for idx in range(0,states):
        [i_x,i_y] = states_idx[idx]
        if i_x<N-1:
            right = states_idx.index((i_x+1,i_y))
            if np.abs(f_x(x_right[i_x],y_midpoints[i_y]))>10**-14:
                xi = -f_x(x_right[i_x],y_midpoints[i_y])
                lamda = (dx*(-f_x(x_right[i_x],y_midpoints[i_y])))/D_xx(x_right[i_x],y_midpoints[i_y])
                delta = 1/lamda + 1/(1-np.exp(lamda))
                L[idx,right] = (1/dx)*(xi*(1-delta)+D_xx(x_right[i_x],y_midpoints[i_y])/dx)
            else:
                L[idx,right] = D_xx(x_right[i_x],y_midpoints[i_y])/(dx**2)            
        if i_y<M-1:
            top = states_idx.index((i_x,i_y+1))
            if np.abs(f_y(x_midpoints[i_x],y_right[i_y]))>10**-14:
                xi = -f_y(x_midpoints[i_x],y_right[i_y])
                lamda = (dy*(-f_y(x_midpoints[i_x],y_right[i_y])))/D_yy(x_midpoints[i_x],y_right[i_y])
                delta = 1/lamda + 1/(1-np.exp(lamda))
                L[idx,top] = (1/dy)*(xi*(1-delta)+D_yy(x_midpoints[i_x],y_right[i_y])/dy)
            else:
                L[idx,top] = D_yy(x_midpoints[i_x],y_right[i_y])/(dy**2)
        if i_x>0:
            left = states_idx.index((i_x-1,i_y))
            if np.abs(f_x(x_left[i_x],y_midpoints[i_y]))>10**-14:
                xi = -f_x(x_left[i_x],y_midpoints[i_y])
                lamda = (dx*(-f_x(x_left[i_x],y_midpoints[i_y])))/D_xx(x_left[i_x],y_midpoints[i_y])
                delta = 1/lamda + 1/(1-np.exp(lamda))
                L[idx,left] = -(1/dx)*(xi*delta - D_xx(x_left[i_x],y_midpoints[i_y])/dx)
            else:
                L[idx,left] = D_xx(x_left[i_x],y_midpoints[i_y])/(dx**2)
        if i_y>0:
            bottom = states_idx.index((i_x,i_y-1))
            if np.abs(f_y(x_midpoints[i_x],y_left[i_y]))>10**-14:
                xi = -f_y(x_midpoints[i_x],y_left[i_y])
                lamda = (dy*(-f_y(x_midpoints[i_x],y_left[i_y])))/D_yy(x_midpoints[i_x],y_left[i_y])
                delta = 1/lamda + 1/(1-np.exp(lamda))
                L[idx,bottom] = -(1/dy)*(xi*delta - D_yy(x_midpoints[i_x],y_left[i_y])/dy)
            else:
                L[idx,bottom] = D_yy(x_midpoints[i_x],y_left[i_y])/(dy**2)
                
        # self loop with reflecting boundary conditions
        if i_x<N-1:
            if np.abs(f_x(x_right[i_x],y_midpoints[i_y]))>10**-14:
                xi = -f_x(x_right[i_x],y_midpoints[i_y])
                lamda = (dx*(-f_x(x_right[i_x],y_midpoints[i_y])))/D_xx(x_right[i_x],y_midpoints[i_y])
                delta = 1/lamda + 1/(1-np.exp(lamda))
                frx = (1/dx)*(xi*delta-D_xx(x_right[i_x],y_midpoints[i_y])/dx)
            else:
                frx = -D_xx(x_right[i_x],y_midpoints[i_y])/(dx**2)
        if i_x>0:
            if np.abs(f_x(x_left[i_x],y_midpoints[i_y]))>10**-14:
                xi = -f_x(x_left[i_x],y_midpoints[i_y])
                lamda = (dx*(-f_x(x_left[i_x],y_midpoints[i_y])))/D_xx(x_left[i_x],y_midpoints[i_y])
                delta = 1/lamda + 1/(1-np.exp(lamda))
                flx = (1/dx)*(-xi*(1-delta) - D_xx(x_left[i_x],y_midpoints[i_y])/dx)
            else:
                flx = -D_xx(x_left[i_x],y_midpoints[i_y])/(dx**2)
        if i_y<N-1:
            if np.abs(f_y(x_midpoints[i_x],y_right[i_y]))>10**-14:
                xi = -f_y(x_midpoints[i_x],y_right[i_y])
                lamda = (dy*(-f_y(x_midpoints[i_x],y_right[i_y])))/D_yy(x_midpoints[i_x],y_right[i_y])
                delta = 1/lamda + 1/(1-np.exp(lamda))
                fry = (1/dy)*(xi*delta-D_yy(x_midpoints[i_x],y_right[i_y])/dy)
            else:
                fry = -D_yy(x_midpoints[i_x],y_right[i_y])/(dy**2)
        if i_y>0:
            if np.abs(f_y(x_midpoints[i_x],y_left[i_y]))>10**-14:
                xi = -f_y(x_midpoints[i_x],y_left[i_y])
                lamda = (dy*(-f_y(x_midpoints[i_x],y_left[i_y])))/D_yy(x_midpoints[i_x],y_left[i_y])
                delta = 1/lamda + 1/(1-np.exp(lamda))
                fly = (1/dy)*(-xi*(1-delta) - D_yy(x_midpoints[i_x],y_left[i_y])/dy)
            else:
                fly = -D_yy(x_midpoints[i_x],y_left[i_y])/(dy**2)

        if 0<i_x<N-1 and 0<i_y<N-1:
            L[idx,idx] = (frx+flx) + (fry+fly)
        elif i_x==0 and 0<i_y<N-1:
            L[idx,idx] = (frx) + (fry+fly)
        elif i_x==(N-1) and 0<i_y<N-1:
            L[idx,idx]= (flx) + (fry+fly)
        elif 0<i_x<N-1 and i_y==0:
            L[idx,idx]= (frx+flx) + (fry)
        elif 0<i_x<N-1 and i_y==(N-1):
            L[idx,idx]= (frx+flx) + (fly)
        elif i_x==0 and i_y==0:
            L[idx,idx]= (frx) + (fry)
        elif i_x==0 and i_y==(N-1):
            L[idx,idx]= (frx) + (fly)
        elif i_x==(N-1) and i_y==0:
            L[idx,idx]= (flx) + (fry)
        elif i_x==(N-1) and i_y==(N-1):
            L[idx,idx] = (flx) + (fly)
    
    return(L,x_midpoints,y_midpoints,x_left,x_right,y_left,y_right)

def SG_2d_periodic_boundaries(f,D,dx,xmin,xmax,dy,ymin,ymax):
    
    [f_x,f_y] = f
    [D_xx,D_yy] = D
    
    x_midpoints = np.linspace(xmin+dx/2,xmax-dx/2,int((xmax-xmin)/dx))
    y_midpoints = np.linspace(ymin+dy/2,ymax-dy/2,int((ymax-ymin)/dy))
    x_left = np.linspace(xmin,xmax-dx,int((xmax-xmin)/dx))
    x_right = np.linspace(xmin+dx,xmax,int((xmax-xmin)/dx))
    y_left = np.linspace(xmin,xmax-dx,int((xmax-xmin)/dx))
    y_right = np.linspace(xmin+dx,xmax,int((xmax-xmin)/dx))
    N = x_midpoints.shape[0]
    M = y_midpoints.shape[0]
    states = N*M
    L = np.zeros((states,states))
    states_idx = []
    for i_x in range(N):
        for i_y in range(M):
            states_idx.append((i_x,i_y))
    for idx in range(0,states):
        [i_x,i_y] = states_idx[idx]
        if i_x<N-1:
            right = states_idx.index((i_x+1,i_y))
            if np.abs(np.exp(-f_x(x_right[i_x],y_midpoints[i_y])*dx/D_xx(x_right[i_x],y_midpoints[i_y]))-1)>10**-6:
                L[idx,right] = (1/dx)*f_x(x_right[i_x],y_midpoints[i_y])*np.exp(-f_x(x_right[i_x],y_midpoints[i_y])*dx/D_xx(x_right[i_x],y_midpoints[i_y]))/(1-np.exp(-f_x(x_right[i_x],y_midpoints[i_y])*dx/D_xx(x_right[i_x],y_midpoints[i_y])))
            else:
                L[idx,right] = D_xx(x_right[i_x],y_midpoints[i_y])/(dx**2)
        if i_x == N-1:
            right = states_idx.index((0,i_y))
            if np.abs(np.exp(-f_x(x_right[i_x],y_midpoints[i_y])*dx/D_xx(x_right[i_x],y_midpoints[i_y]))-1)>10**-6:
                L[idx,right] = (1/dx)*f_x(x_right[i_x],y_midpoints[i_y])*np.exp(-f_x(x_right[i_x],y_midpoints[i_y])*dx/D_xx(x_right[i_x],y_midpoints[i_y]))/(1-np.exp(-f_x(x_right[i_x],y_midpoints[i_y])*dx/D_xx(x_right[i_x],y_midpoints[i_y])))
            else:
                L[idx,right] = D_xx(x_right[i_x],y_midpoints[i_y])/(dx**2)
        if i_y<N-1:
            top = states_idx.index((i_x,i_y+1))
            if np.abs(np.exp(-f_y(x_midpoints[i_x],y_right[i_y])*dy/D_yy(x_midpoints[i_x],y_right[i_y]))-1)>10**-6:
                L[idx,top] = (1/dy)*f_y(x_midpoints[i_x],y_right[i_y])*np.exp(-f_y(x_midpoints[i_x],y_right[i_y])*dy/D_yy(x_midpoints[i_x],y_right[i_y]))/(1-np.exp(-f_y(x_midpoints[i_x],y_right[i_y])*dy/D_yy(x_midpoints[i_x],y_right[i_y])))
            else:
                L[idx,top] = D_yy(x_midpoints[i_x],y_right[i_y])/(dy**2)
        if i_y==N-1:
            top = states_idx.index((i_x,0))
            if np.abs(np.exp(-f_y(x_midpoints[i_x],y_right[i_y])*dy/D_yy(x_midpoints[i_x],y_right[i_y]))-1)>10**-6:
                L[idx,top] = (1/dy)*f_y(x_midpoints[i_x],y_right[i_y])*np.exp(-f_y(x_midpoints[i_x],y_right[i_y])*dy/D_yy(x_midpoints[i_x],y_right[i_y]))/(1-np.exp(-f_y(x_midpoints[i_x],y_right[i_y])*dy/D_yy(x_midpoints[i_x],y_right[i_y])))
            else:
                L[idx,top] = D_yy(x_midpoints[i_x],y_right[i_y])/(dy**2)
        if i_x>0:
            left = states_idx.index((i_x-1,i_y))
            if np.abs(np.exp(-f_x(x_left[i_x],y_midpoints[i_y])*dx/D_xx(x_left[i_x],y_midpoints[i_y]))-1)>10**-6:
                L[idx,left] = (1/dx)*f_x(x_left[i_x],y_midpoints[i_y])/(1-np.exp(-f_x(x_left[i_x],y_midpoints[i_y])*dx/D_xx(x_left[i_x],y_midpoints[i_y])))
            else:
                L[idx,left] = D_xx(x_left[i_x],y_midpoints[i_y])/(dx**2)
        if i_x==0:
            left = states_idx.index((N-1,i_y))
            if np.abs(np.exp(-f_x(x_right[N-1],y_midpoints[i_y])*dx/D_xx(x_right[N-1],y_midpoints[i_y]))-1)>10**-6:
                L[idx,left] = (1/dx)*f_x(x_right[N-1],y_midpoints[i_y])/(1-np.exp(-f_x(x_right[N-1],y_midpoints[i_y])*dx/D_xx(x_right[N-1],y_midpoints[i_y])))
            else:
                L[idx,left] = D_xx(x_right[N-1],y_midpoints[i_y])/(dx**2)
        if i_y>0:
            bottom = states_idx.index((i_x,i_y-1))
            if np.abs(np.exp(-f_y(x_midpoints[i_x],y_left[i_y])*dy/D_yy(x_midpoints[i_x],y_left[i_y]))-1)>10**-6:
                L[idx,bottom] = (1/dy)*f_y(x_midpoints[i_x],y_left[i_y])/(1-np.exp(-f_y(x_midpoints[i_x],y_left[i_y])*dy/D_yy(x_midpoints[i_x],y_left[i_y])))
            else:
                L[idx,bottom] = D_yy(x_midpoints[i_x],y_left[i_y])/(dy**2)
        if i_y==0:
            bottom = states_idx.index((i_x,N-1))
            if np.abs(np.exp(-f_y(x_midpoints[i_x],y_right[N-1])*dy/D_yy(x_midpoints[i_x],y_right[N-1]))-1)>10**-6:
                L[idx,bottom] = (1/dy)*f_y(x_midpoints[i_x],y_right[N-1])/(1-np.exp(-f_y(x_midpoints[i_x],y_right[N-1])*dy/D_yy(x_midpoints[i_x],y_right[N-1])))
            else:
                L[idx,bottom] = D_yy(x_midpoints[i_x],y_right[N-1])/(dy**2)
                
        # self loop with periodic boundary conditions
        if np.abs(np.exp(-f_x(x_right[i_x],y_midpoints[i_y])*dx/D_xx(x_right[i_x],y_midpoints[i_y]))-1)>10**-6:
            frx = (1/dx)*f_x(x_right[i_x],y_midpoints[i_y])/(1-np.exp(-f_x(x_right[i_x],y_midpoints[i_y])*dx/D_xx(x_right[i_x],y_midpoints[i_y])))
        else:
            frx = D_xx(x_right[i_x],y_midpoints[i_y])/(dx**2)
        if i_x>0:
            if np.abs(np.exp(-f_x(x_left[i_x],y_midpoints[i_y])*dx/D_xx(x_left[i_x],y_midpoints[i_y]))-1)>10**-6:
                flx = (1/dx)*f_x(x_left[i_x],y_midpoints[i_y])*np.exp(-f_x(x_left[i_x],y_midpoints[i_y])*dx/D_xx(x_left[i_x],y_midpoints[i_y]))/(1-np.exp(-f_x(x_left[i_x],y_midpoints[i_y])*dx/D_xx(x_left[i_x],y_midpoints[i_y])))
            else:
                flx = D_xx(x_left[i_x],y_midpoints[i_y])/(dx**2)
        if i_x==0:
            if np.abs(np.exp(-f_x(x_right[N-1],y_midpoints[i_y])*dx/D_xx(x_right[N-1],y_midpoints[i_y]))-1)>10**-6:
                flx = (1/dx)*f_x(x_right[N-1],y_midpoints[i_y])*np.exp(-f_x(x_right[N-1],y_midpoints[i_y])*dx/D_xx(x_right[N-1],y_midpoints[i_y]))/(1-np.exp(-f_x(x_right[N-1],y_midpoints[i_y])*dx/D_xx(x_right[N-1],y_midpoints[i_y])))
            else:
                flx = D_xx(x_right[N-1],y_midpoints[i_y])/(dx**2)
        if np.abs(np.exp(-f_y(x_midpoints[i_x],y_right[i_y])*dy/D_yy(x_midpoints[i_x],y_right[i_y]))-1)>10**-6:
            fry = (1/dy)*f_y(x_midpoints[i_x],y_right[i_y])/(1-np.exp(-f_y(x_midpoints[i_x],y_right[i_y])*dy/D_yy(x_midpoints[i_x],y_right[i_y])))
        else:
            fry = D_yy(x_midpoints[i_x],y_right[i_y])/(dy**2)
        if i_y>0:
            if np.abs(np.exp(-f_y(x_midpoints[i_x],y_left[i_y])*dy/D_yy(x_midpoints[i_x],y_left[i_y]))-1)>10**-6:
                fly = (1/dy)*f_y(x_midpoints[i_x],y_left[i_y])*np.exp(-f_y(x_midpoints[i_x],y_left[i_y])*dy/D_yy(x_midpoints[i_x],y_left[i_y]))/(1-np.exp(-f_y(x_midpoints[i_x],y_left[i_y])*dy/D_yy(x_midpoints[i_x],y_left[i_y])))
            else:
                fly = D_yy(x_midpoints[i_x],y_left[i_y])/(dy**2)
        if i_y==0:
            if np.abs(np.exp(-f_y(x_midpoints[i_x],y_right[N-1])*dy/D_yy(x_midpoints[i_x],y_right[N-1]))-1)>10**-6:
                fly = (1/dy)*f_y(x_midpoints[i_x],y_right[N-1])*np.exp(-f_y(x_midpoints[i_x],y_right[N-1])*dy/D_yy(x_midpoints[i_x],y_right[N-1]))/(1-np.exp(-f_y(x_midpoints[i_x],y_right[N-1])*dy/D_yy(x_midpoints[i_x],y_right[N-1])))
            else:
                fly = D_yy(x_midpoints[i_x],y_right[N-1])/(dy**2)

        
        L[idx,idx] = -(frx+flx) - (fry+fly)
    
    return(L,x_midpoints,y_midpoints,x_left,x_right,y_left,y_right)

def stationary_dist(L):
    eigenvalues, eigenvectors = eig(L)
    zero_index = np.argmin(np.abs(eigenvalues))
    # The corresponding eigenvector is the stationary distribution
    dist = np.real(eigenvectors[:, zero_index])/np.sum(np.real(eigenvectors[:, zero_index]))
    dist[dist < 0] = 0
    return(dist)

def efficient_stationary_dist(L):
    L_mod = csr_matrix(np.vstack([L, np.ones(L.shape[1])]))

    b = np.zeros(L.shape[0] + 1)
    b[-1] = 1

    # Solve for p_ss using least squares
    p_ss = lsqr(L_mod, b)[0]
    p_ss[p_ss<0]=0
    p_ss = p_ss/np.sum(p_ss)
    
    return(p_ss)

def EPR_MC(L,p_ss):
    N = L.shape[0]
    Phi = 0
    for i in range(0,N):
        if p_ss[i]>10**(-14):
            for j in range(0,N):
                if i!=j and L[i,j]>10**(-14) and p_ss[j]>10**(-14):
                    Phi += 0.5*(L[j,i]*p_ss[i] - L[i,j]*p_ss[j])*np.log((L[j,i]*p_ss[i] + 1e-14)/(L[i,j]*p_ss[j]))
                    # avoid log zero
    return Phi


def discrete_flux_SG_periodic(L,p_ss_flat,N,M):
    J = np.zeros(L.shape)
    states = L.shape[0]
            
    for i in range(0,states-1):
        for j in range(i+1,states):
            J[i,j] = L[j,i]*p_ss_flat[i]-L[i,j]*p_ss_flat[j]
            J[j,i] = -J[i,j]
    return J

def bayesian_transition_initial(no_states,states_idx):
    P = np.zeros((no_states,no_states))
    holding_times_initial = np.zeros(no_states)
    for si in range(0,no_states):
        for sj in range(0,no_states): 
            six, siy = states_idx[si]
            sjx, sjy = states_idx[sj]
    
            if (np.abs(six-sjx)+np.abs(siy-sjy))<=1:
                P[si,sj] += 1
    return(P)

def bayesian_transition_circle_prior(no_states, states_idx, Nx, Ny):
    P = np.zeros((no_states, no_states))
    
    is_in_circle = np.zeros(no_states, dtype=bool)
    for si in range(no_states):
        ix, iy = states_idx[si]
        
        norm_x = (ix / (Nx - 1)) * 2 - 1 if Nx > 1 else 0
        norm_y = (iy / (Ny - 1)) * 2 - 1 if Ny > 1 else 0
        
        if norm_x**2 + norm_y**2 <= 1:
            is_in_circle[si] = True

    for si in range(no_states):
        for sj in range(no_states):
            if is_in_circle[si] and is_in_circle[sj]:
                six, siy = states_idx[si]
                sjx, sjy = states_idx[sj]
                
                if (np.abs(six - sjx) + np.abs(siy - sjy)) <= 1:
                    P[si, sj] = 1
                    
    return P

def bayesian_transition_circle_prior_shuffled(no_states, states_idx, Nx, Ny):
    P = np.zeros((no_states, no_states))
    
    is_in_circle = np.zeros(no_states, dtype=bool)
    for si in range(no_states):
        ix, iy = states_idx[si]
        
        norm_x = (ix / (Nx - 1)) * 2 - 1 if Nx > 1 else 0
        norm_y = (iy / (Ny - 1)) * 2 - 1 if Ny > 1 else 0
        
        if norm_x**2 + norm_y**2 <= 1:
            is_in_circle[si] = True

    for si in range(no_states):
        for sj in range(no_states):
            if is_in_circle[si] and is_in_circle[sj]:
                six, siy = states_idx[si]
                sjx, sjy = states_idx[sj]
                
                P[si, sj] = 1    
    return P

def CT_Markov_Chain_MLE(time_course, states_idx , L_hat, N, dt, out_of_domain):
    holding_time = np.ones(N)
    for t in range(0,len(time_course)-1):
        if t not in out_of_domain and (t+1) not in out_of_domain:
            source_x, source_y = states_idx[time_course[t]]
            sink_x, sink_y = states_idx[time_course[t+1]]
            if (np.abs(source_x-sink_x)+np.abs(source_y-sink_y))<=1:
                L_hat[time_course[t+1],time_course[t]] += 1
                holding_time[time_course[t]] += dt
    for i in range(0,N):
        L_hat[:,i]/=holding_time[i]
    L_hat -= np.diag(np.sum(L_hat,0))
    return L_hat

def CT_Markov_Chain_MLE_non_adjacent(time_course, states_idx , L_hat, N, dt, out_of_domain):
    holding_time = np.ones(N)
    for t in range(0,len(time_course)-1):
        if t not in out_of_domain and (t+1) not in out_of_domain:
            source_x, source_y = states_idx[time_course[t]]
            sink_x, sink_y = states_idx[time_course[t+1]]
            L_hat[time_course[t+1],time_course[t]] += 1
            holding_time[time_course[t]] += dt
    for i in range(0,N):
        L_hat[:,i]/=holding_time[i]
    L_hat -= np.diag(np.sum(L_hat,0))
    return L_hat


def OU_simulation(theta,sigma,x0,tau,samples):
    eps = (sigma**2)/2
    B = np.array([[2,-theta],[+theta,2]])
    expb = expm(-tau*B)
    C = np.array([[np.exp(2*tau)*eps*np.sinh(2*tau),0],[0,np.exp(2*tau)*eps*np.sinh(2*tau)]])

    N = x0.shape[0]
    X = np.zeros((N,samples))
    
    X[:,0] = x0

    for s in range(1,samples):
        xi = np.random.multivariate_normal(np.reshape(np.zeros((N,1)),N),C)
        X[:,s] = expb@X[:,s-1] + xi
    return(X)

def strang_splitting_planar_cycle_theta(theta,eps,h,samples,x0):

    X = np.zeros((2,samples))

    A = np.array([[1,-theta],[theta,1]])
    C = np.array([[eps*np.exp(0.5*h)*np.sinh(h/2),0],[0,eps*np.exp(0.5*h)*np.sinh(h/2)]])
    Eh = sp.linalg.expm(A*0.5*h)

    X[:,0] = x0
    for t in range(0,samples-1):
        xi = np.random.multivariate_normal(mean=np.zeros((2)),cov=C)
        X_temp = Eh@X[:,t] + xi
        x,y = X_temp
        r0 = np.sqrt(x**2 + y**2)
        r = r0/(r0*h+1)
        x = (r/r0)*x
        y = (r/r0)*y
        xi = np.random.multivariate_normal(mean=np.zeros((2)),cov=C)
        X[:,t+1] = Eh@np.array([x,y]) + xi
        
    return(X)


def coarse_grain_trajectory(X,dx,dy):
    xmin = np.min(X[0,:]); xmax = np.max(X[0,:])
    ymin = np.min(X[1,:]); ymax = np.max(X[1,:])

    grid_x = (((X[0,:] - xmin) // dx))
    grid_y = (((X[1,:] - ymin) // dy))

    grid_cells = list(zip(grid_x, grid_y))

    num_cells_x = int(np.floor((xmax - xmin) / dx)) + 1
    num_cells_y = int(np.floor((ymax - ymin) / dy)) + 1

    no_states = num_cells_x*num_cells_y

    states_idx = []
    for i_x in range(num_cells_x):
        for i_y in range(num_cells_y):
            states_idx.append((i_x,i_y))

    time_course = []
    out_of_domain = []
    for t in range(0,len(grid_cells)):
        try:
            idx = states_idx.index(grid_cells[t])
        except ValueError:
            idx = None
        if idx is not None:
            time_course.append(states_idx.index(grid_cells[t]))
        else:
            out_of_domain.append(t)
            time_course.append(np.nan)

    counts = check_max_jump(grid_x,grid_y)

    return(time_course,out_of_domain,counts,states_idx,grid_x,grid_y,no_states,num_cells_x,num_cells_y,xmin, xmax, ymin, ymax)

def check_max_jump(grid_x,grid_y):
    jumps = np.abs(np.diff(grid_x)) + np.abs(np.diff(grid_y))
    max_jump = np.max(jumps)
    counts, bin_edges = np.histogram(jumps, bins=int(max_jump), density=True)
    return counts

def sparse_stationary(L):
    eigvals, eigvecs = sp.sparse.linalg.eigs(L, which='SM')
    idx = np.argmin(np.abs(eigvals))
    return np.real(eigvecs[:,idx])/np.sum(np.real(eigvecs[:,idx]))

def plot_flux_normalized(J, Nx, Ny, x_limits, y_limits, ax, scale):

    N = Nx * Ny

    x = np.linspace(x_limits[0], x_limits[1], Nx)
    y = np.linspace(y_limits[0], y_limits[1], Ny)
    X, Y = np.meshgrid(x, y)

    U = np.zeros((Ny, Nx))
    V = np.zeros((Ny, Nx))

    for i in range(N):
        x_i = i % Nx
        y_i = i // Nx
        for j in range(N):
            x_j = j % Nx
            y_j = j // Nx
            dx = x_j - x_i
            dy = y_j - y_i
            U[y_i, x_i] += dx * J[j, i]
            V[y_i, x_i] += dy * J[j, i]
    magnitude = np.sqrt(U**2 + V**2)
    U_norm = np.zeros_like(U)
    V_norm = np.zeros_like(V)
    
    nonzero_mask = magnitude > 0
    U_norm[nonzero_mask] = U[nonzero_mask] / magnitude[nonzero_mask]
    V_norm[nonzero_mask] = V[nonzero_mask] / magnitude[nonzero_mask]

    for iy in range(Ny):
        for ix in range(Nx):
            norm_x = (ix / (Nx - 1)) * 2 - 1 if Nx > 1 else 0
            norm_y = (iy / (Ny - 1)) * 2 - 1 if Ny > 1 else 0
            


    quiver_plot = ax.quiver(X, Y, U_norm, V_norm, magnitude,
                            cmap='magma',
                            scale=scale,           
                            width=0.004,        
                            headwidth=4,        
                            headlength=5)     


    ax.set_xlim(x_limits[0], x_limits[1])
    ax.set_ylim(y_limits[0], y_limits[1])
    ax.set_aspect('equal', adjustable='box')
    return

def calculate_flux_matrix(L,p):
    J = np.zeros_like(L)
    N = p.shape[0]

    for i in range(0,N-1):
        for j in range(i+1,N):
            J[i, j] = L[j, i]*p[i] - L[i, j]*p[j]
    return J


def CT_Markov_Chain_MLE_shuffling(time_course, states_idx , L_hat, N, dt, out_of_domain):
    holding_time = np.ones(N)
    for t in range(0,len(time_course)-1):
        if t not in out_of_domain and (t+1) not in out_of_domain:
            source_x, source_y = states_idx[time_course[t]]
            sink_x, sink_y = states_idx[time_course[t+1]]
            if (np.abs(source_x-sink_x)+np.abs(source_y-sink_y))<=1:
                u = np.random.uniform()
                if u < 0.5:
                    L_hat[time_course[t+1],time_course[t]] += 1
                    holding_time[time_course[t]] += dt
                else:
                    L_hat[time_course[t], time_course[t+1]] += 1
                    holding_time[time_course[t+1]] += dt
    for i in range(0,N):
        L_hat[:,i]/=holding_time[i]
    L_hat -= np.diag(np.sum(L_hat,0))
    return L_hat