#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:37:25 2018

@author: asadm2
"""
from scipy.optimize import curve_fit
from progressbar import ProgressBar
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import integrate
import pandas as pd
import numpy as np
import math
import csv

### Schechter function
def Schechter_func(M,phi_star,M_star,alpha):
    const = 0.4*np.log(10)*phi_star
    first_exp_term = 10**(0.4*(alpha+1)*(M_star-M))
    second_exp_term = np.exp(-10**(0.4*(M_star-M)))
    return const*first_exp_term*second_exp_term

### Given data calculate how many bins should be used
def num_bins(data_arr):
    q75, q25 = np.percentile(data_arr, [75 ,25])
    iqr = q75 - q25
    num_points = len(data_arr)
    h =2*iqr*(num_points**(-1/3))
    n_bins = math.ceil((max(data_arr)-min(data_arr))/h) #Round up number   
    return n_bins

### Cumulative method
def cumu_num_dens(data,nbins,weights,volume):
    if weights is None:
        weights = np.ones(len(data))
    else:
        weights = np.array(weights)
    #Unnormalized histogram and bin edges
    freq,edg = np.histogram(data,bins=nbins,weights=weights)
    bin_centers = 0.5*(edg[1:]+edg[:-1])
    bin_width = edg[1] - edg[0]
    N_cumu = np.cumsum(freq[::-1])[::-1]
    n_cumu = N_cumu/volume
    err_poiss = np.sqrt(N_cumu)/volume
    return bin_centers,edg,n_cumu,err_poiss,bin_width,freq

### Differential method
def diff_num_dens(data,nbins,weights,volume):
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
    return bin_centers,edg,n_diff,err_poiss,bin_width

### Integrate the differential fit schechter function
def integrated_schechter_func(mag_lower,phi_star,M_star,alpha,mag_upper):
    integral = integrate.quad(Schechter_func,mag_lower,mag_upper,\
                              args=(phi_star,M_star,alpha))
    return integral
    
### Data file used
eco_obs_catalog = pd.read_csv('../data/gal_Lr_Mb_Re.txt',\
                              delimiter='\s+',header=None,skiprows=2,\
                              names=['M_r','logmbary','Re'])

eco_obs_catalog = eco_obs_catalog.loc[eco_obs_catalog.Re.values >= 0]
v_eco = 192351.36 #Volume of ECO with buffer in (Mpc/h)^3

Mr_all = eco_obs_catalog.M_r.values
min_Mr = min(Mr_all)
max_Mr = max(Mr_all)

Mr_cut = [value for value in Mr_all if value <= -17.33 and value >= -23.5]
max_Mr_cut = max(Mr_cut)
min_Mr_cut = min(Mr_cut)
nbins = 25 #better chi-squared than 35 from num_bins function

### Calculate cumulative number density using function
bin_centers_cumu,bin_edges_cumu,n_Mr_cumu,err_poiss_cumu,bin_width_cumu = \
cumu_num_dens(Mr_cut,nbins,None,v_eco)

### Calculate differential number density using function for integration
bin_centers_diff,bin_edges_diff,n_Mr_diff,err_poiss_diff,bin_width_diff = \
diff_num_dens(Mr_cut,nbins,None,v_eco)

p0 = [10**-2,-20,-1.2] #initial guess for phi_star,M_star,alpha

### Curve fit using Schechter function defined
params_noextrap,pcov = curve_fit(Schechter_func,bin_centers_diff,n_Mr_diff,p0,\
                                 sigma=err_poiss_diff,absolute_sigma=True,\
                                 maxfev=20000,method='lm')

### Best fit parameters from curve fit
Phi_star = params_noextrap[0]
M_star = params_noextrap[1]
Alpha = params_noextrap[2]

### Integrating fit between -17.33 and -23.5
int_noextrap_arr = []
bin_centers_noextrap = np.arange(min_Mr_cut,max_Mr_cut,0.01)
bin_width_noextrap = bin_centers_noextrap[1] - bin_centers_noextrap[0]
for index,value in enumerate(bin_centers_noextrap):
    mag_lower = -np.inf
    mag_upper = value + (0.5*bin_width_noextrap)
    integral,err = integrated_schechter_func(mag_lower,Phi_star,M_star,Alpha,\
                                             mag_upper)
    int_noextrap_arr.append(integral)
    
int_noextrap_arr = np.array(int_noextrap_arr)   

### Interpolating between Mr and n and reversing it so you can pass an n and
### get an Mr value
Mr_n_interp_func = interpolate.interp1d(int_noextrap_arr, bin_centers_noextrap\
                                        , fill_value="extrapolate")
int_noextrap_arr_new = np.linspace(min(int_noextrap_arr),max(int_noextrap_arr)\
                                   ,10000)
interp_result = Mr_n_interp_func(int_noextrap_arr_new)

### Halo data
halocat_galcat_merged = pd.read_hdf('../data/halo_gal_Vishnu_Rockstar_macc.h5',\
                                    key='halocat_galcat_merged')
v_sim = 130**3 #(Mpc/h)^3
vpeak = halocat_galcat_merged.halo_vpeak.values
nbins = num_bins(vpeak)
bin_centers_vpeak,bin_edges_vpeak,n_vpeak,err_poiss_vpeak,bin_width_vpeak,freq = \
cumu_num_dens(vpeak,nbins,None,v_sim)

fig = plt.figure(figsize=(10,10))
plt.xscale('log')
plt.yscale('log')
plt.errorbar(bin_centers_vpeak,n_vpeak,yerr=err_poiss_vpeak,fmt="ks--",ls='None',\
             elinewidth=0.5,ecolor='k',capsize=5,capthick=0.5,markersize=4)
plt.xlabel(r'$v_{peak}$')
plt.ylabel(r'$\mathrm{(n \geq v_{peak})} [\mathrm{h}^{3}\mathrm{Mpc}^{-3}]$')
plt.title(r'Vishnu Cumulative Peak Velovity Function')
plt.savefig('../reports/figures/vpeak_cumunumdens.png')

### Interpolating 
vpeak_n_interp_func = interpolate.InterpolatedUnivariateSpline\
(bin_centers_vpeak,n_vpeak)

pbar = ProgressBar(maxval=len(vpeak))
n_vpeak_arr = [vpeak_n_interp_func(val) for val in pbar(vpeak)]
pbar = ProgressBar(maxval=len(n_vpeak_arr))
halo_Mr_sham = [Mr_n_interp_func(val) for val in pbar(n_vpeak_arr)]

### Writing vpeak and Mr values to text file
with open('../data/SHAM_parallel.csv', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(vpeak,halo_Mr_sham))
