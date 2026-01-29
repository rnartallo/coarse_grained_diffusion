import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.linalg import expm
from scipy.stats import multivariate_normal
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr, lsqr, spsolve, eigsh
from scipy.special import erf, iv
from mpl_toolkits.axes_grid1 import make_axes_locatable
import helpers as h
import importlib
from matplotlib.colors import LogNorm

x0 = np.array([-2,1])
samples = 5000000
tau = 0.001
T= samples*tau


theta = 5; sigma = np.sqrt(2)

iterations = 100
sample_lengths = [1000000,2000000,3000000,4000000,5000000]

EPR_byTime = np.zeros((iterations,len(sample_lengths)))

dx = 0.25

for i in range(0,iterations):

    X = h.OU_simulation(theta,sigma,x0,tau,samples)

    for id, length in enumerate(sample_lengths):

        time_course,out_of_domain,counts,states_idx, grid_x, grid_y, no_states, num_cells_x, num_cells_y, xmin, xmax, ymin, ymax = h.coarse_grain_trajectory(X[:,:length],dx,dy=dx)
        L_init = h.bayesian_transition_initial(no_states, states_idx)
        L_hat = h.CT_Markov_Chain_MLE(time_course, states_idx , L_init, no_states, dt = tau, out_of_domain= out_of_domain)
        p_ss = h.sparse_stationary(L_hat)
        
        EPR_byTime[i,id] = h.EPR_MC(L_hat,p_ss)

np.savetxt("EPR_byTime.csv", EPR_byTime, delimiter=",") 

dx_list = [1,0.5,0.25,0.125]

EPR_byDx = np.zeros((iterations,len(dx_list)))

for i in range(0,iterations):

    X = h.OU_simulation(theta,sigma,x0,tau,samples=5000000)

    for id, dx in enumerate(dx_list):

        time_course,out_of_domain,counts,states_idx, grid_x, grid_y, no_states, num_cells_x, num_cells_y, xmin, xmax, ymin, ymax = h.coarse_grain_trajectory(X,dx,dy=dx)
        L_init = h.bayesian_transition_initial(no_states, states_idx)
        L_hat = h.CT_Markov_Chain_MLE(time_course, states_idx , L_init, no_states, dt = tau, out_of_domain= out_of_domain)
        p_ss = h.sparse_stationary(L_hat)
        
        EPR_byDx[i,id] = h.EPR_MC(L_hat,p_ss)

np.savetxt("EPR_byDx.csv", EPR_byDx, delimiter=",") 





