#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 14:10:35 2018

@author: asadm2

This script carries out the subhalo abundance matching between v_peak and M_r
"""

import matplotlib as mpl
mpl.use('agg')
from scipy.optimize import curve_fit
from progressbar import ProgressBar
from multiprocessing import Pool 
from scipy import interpolate
from matplotlib import rc
import pandas as pd
import numpy as np
import sympy
import math
import csv

###Formatting for plots and animation
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=15)
rc('text', usetex=True)

### Schechter function
def Schechter_func(M,phi_star,M_star,alpha):
    const = 0.4*np.log(10)*phi_star
    first_exp_term = 10**(0.4*(alpha+1)*(M_star-M))
    second_exp_term = np.exp(-10**(0.4*(M_star-M)))
    return const*first_exp_term*second_exp_term

### Differential method
def diff_num_dens(data,nbins,weights,volume,mag_bool):
    if weights is None:
        weights = np.ones(len(data))
    else:
        weights = np.array(weights)
    #Unnormalized histogram and bin edges
    freq,edg = np.histogram(data,bins=nbins,weights=weights)
    bin_centers = 0.5*(edg[1:]+edg[:-1])
    bin_width = edg[1] - edg[0]
    err_poiss = np.sqrt(freq)/(volume*bin_width)
    n_diff = freq/(volume*bin_width)
    return bin_centers,edg,n_diff,err_poiss

### Given data calculate how many bins should be used
def num_bins(data_arr):
    q75, q25 = np.percentile(data_arr, [75 ,25])
    iqr = q75 - q25
    num_points = len(data_arr)
    h =2*iqr*(num_points**(-1/3))
    n_bins = math.ceil((max(data_arr)-min(data_arr))/h) #Round up number   
    return n_bins

### Data file used
eco_obs_catalog = pd.read_csv('../data/gal_Lr_Mb_Re.txt',\
                              delimiter='\s+',header=None,skiprows=2,\
                              names=['M_r','logmbary','Re'])

eco_obs_catalog = eco_obs_catalog.loc[eco_obs_catalog.Re.values >= 0]

Mr_all = eco_obs_catalog.M_r.values
Mr_unique = np.unique(eco_obs_catalog.M_r.values)
v_eco = 192351.36 #Volume of ECO with buffer in (Mpc/h)^3
min_Mr = min(Mr_all)
max_Mr = max(Mr_all)


### Using SF to fit data between -17.33 and -23.5 and extrapolating both ends
Mr_cut = [value for value in Mr_all if value <= -17.33 and value >= -23.5]
max_Mr_cut = max(Mr_cut)
min_Mr_cut = min(Mr_cut)
nbins = 25 #better chi-squared than 35 from num_bins function

### Calculate differential number density using function
bin_centers_cut,bin_edges_cut,n_Mr_cut,err_poiss_cut = diff_num_dens(Mr_cut,\
                                                                     nbins,\
                                                                     None,\
                                                                     v_eco,\
                                                                     mag_bool=\
                                                                     True)

p0 = [10**-2,-20,-1.2] #initial guess for phi_star,M_star,alpha

### Curve fit using Schechter function defined
params_noextrap,pcov = curve_fit(Schechter_func,bin_centers_cut,n_Mr_cut,p0,\
                                 sigma=err_poiss_cut,absolute_sigma=True,\
                                 maxfev=20000,method='lm')

### Best fit parameters from curve fit
Phi_star_bestfit = params_noextrap[0]
M_star_bestfit = params_noextrap[1]
Alpha_bestfit = params_noextrap[2]

### Inverting the Schechter function using sympy so that it will return 
### magnitudes given a number density

### Make all parameters symbols
n,M,phi_star,M_star,alpha = sympy.symbols('n,M,phi_star,M_star,alpha')
const = 0.4*sympy.log(10)*phi_star
first_exp_term = 10**(0.4*(alpha+1)*(M_star-M))
second_exp_term = sympy.exp(-10**(0.4*(M_star-M)))
### Make expression that will be an input for sympy.solve
expr = (const*first_exp_term*second_exp_term)-n
### Get schechter function in terms of M
result = sympy.solve(expr,M,quick=True)
### Make result a lambda function so you can pass parameter and variable values
result_func = sympy.lambdify((n,phi_star,M_star,alpha),result,\
                             modules=["sympy"])

### Get parameter values and change variable names because the original ones
### are now symbols
phi_star_num = params_noextrap[0]
M_star_num = params_noextrap[1]
alpha_num = params_noextrap[2]
### Pick a magnitude range
M_num = np.linspace(-23.5,-17.33,200)
### Calculate n given range of M values
n_num = Schechter_func(M_num,phi_star_num,M_star_num,alpha_num)
### Given the n values just calculated use the lambda function in terms of n to 
### test whether you get back the same magnitude values as M_num
M_test = [result_func(val,phi_star_num,M_star_num,alpha_num) for val in n_num]
### Plot both to make sure they overlap
#fig2 = plt.figure(figsize=(10,10))
#plt.scatter(M_num,n_num,c='b',label='n given M',s=15)
#plt.scatter(M_test,n_num,c='r',label='M given n',s=5)
#plt.legend(loc='best')
#plt.show()


### Halo data
halocat_galcat_merged = pd.read_hdf('../data/halo_gal_Vishnu_Rockstar_macc.h5',\
                                    key='halocat_galcat_merged')
v_sim = 130**3 #(Mpc/h)^3
vpeak = halocat_galcat_merged.halo_vpeak.values
nbins = num_bins(vpeak)
bin_centers_vpeak,bin_edges_vpeak,n_vpeak,err_poiss = diff_num_dens(vpeak,\
                                                                    nbins,\
                                                                    None,\
                                                                    v_sim,\
                                                                    mag_bool=\
                                                                    False)

f_h = interpolate.InterpolatedUnivariateSpline(bin_centers_vpeak,n_vpeak)

pbar = ProgressBar(maxval=len(vpeak))
n_vpeak_arr = [f_h(val) for val in pbar(vpeak)]
pbar = ProgressBar(maxval=len(n_vpeak_arr))

### Parallelising SHAM
def resultfunc_helper(args):
    return result_func(*args)
job_args = [(val, phi_star_num,M_star_num,alpha_num) for val in pbar\
            (n_vpeak_arr)]
pool = Pool(processes=24)
halo_Mr_sham = pool.map(resultfunc_helper,job_args)
halo_Mr_sham = np.ndarray.flatten(np.array(halo_Mr_sham))

### Writing vpeak and Mr values to text file
with open('../data/SHAM_parallel.csv', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(vpeak,halo_Mr_sham))
