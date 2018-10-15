#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 00:50:36 2018

@author: asadm2

This script assigns all the other galaxy properties like baryonic mass and 
effective radius to each galaxy in the mock catalog
"""

from progressbar import ProgressBar
import pandas as pd
import numpy as np


### Reading text files
Mr_vpeak_catalog = pd.read_csv('../data/SHAM_parallel.csv', \
                               delimiter='\t', header=None, \
                               names=['vpeak','M_r'])
eco_obs_catalog = pd.read_csv('../data/gal_Lr_Mb_Re.txt',\
                              delimiter='\s+',header=None,skiprows=2,\
                              names=['M_r','logmbary','Re'])
halocat_galcat_merged = pd.read_hdf('../data/halo_gal_Vishnu_Rockstar_macc.h5',\
                                    key='halocat_galcat_merged')
colnames = halocat_galcat_merged.columns

v_eco = 192351.36 #Volume of ECO with buffer in (Mpc/h)^3

Mr_vpeak_catalog = Mr_vpeak_catalog.sort_values('M_r')
eco_obs_catalog = eco_obs_catalog.loc[eco_obs_catalog.Re.values >= 0]
eco_obs_catalog = eco_obs_catalog.sort_values('M_r')

pbar = ProgressBar()
nearest_match_idx_arr = []
mbary_arr = []
re_arr = []
np.random.seed(0)
for mag_value in pbar(Mr_vpeak_catalog.M_r.values):
    diff_arr = np.abs(eco_obs_catalog.M_r.values - mag_value)
    nearest_match_idx = np.where(diff_arr == diff_arr.min())[0]
    if len(nearest_match_idx) > 1:
        nearest_match_idx = np.random.choice(nearest_match_idx)
    else:
        nearest_match_idx = nearest_match_idx[0]
    nearest_match_idx_arr.append(nearest_match_idx)
    mbary_arr.append(eco_obs_catalog.logmbary.values[nearest_match_idx])
    re_arr.append(eco_obs_catalog.Re.values[nearest_match_idx])

Mr_vpeak_catalog['logmbary'] = mbary_arr  
Mr_vpeak_catalog['Re'] = re_arr

mock_catalog = halocat_galcat_merged.merge(Mr_vpeak_catalog,how='inner',\
                           left_on=halocat_galcat_merged.index.values,\
                           right_on=Mr_vpeak_catalog.index.values,\
                           right_index=True)
mock_catalog = mock_catalog.drop(['key_0'],axis=1)
#mock_catalog.columns = colnames + ['v_peak','M_r','logMbary',\
#                                   'Re']

mock_catalog = mock_catalog.drop(['halo_vpeak'],axis=1)
mock_catalog.rename(columns={'vpeak': 'halo_vpeak'}, inplace=True)
mock_catalog = mock_catalog.round({'M_r':3})

mock_catalog.to_hdf('../data/ECO_Vishnu_mock_catalog.h5',key='mock_catalog',\
                    mode='w')


#Mr_vpeak_catalog = Mr_vpeak_catalog.reset_index(drop=True)
#np.savetxt(path_to_interim + 'mock.txt', mock_catalog.values,\
#           fmt='%1.3f',delimiter='\t',header=",".join(mock_catalog.columns))
#fig3 = plt.figure()
#plt.xscale('log')
#plt.gca().invert_yaxis()
#plt.scatter(mock_catalog['v_peak(km/s)'],mock_catalog['M_r'],s=5)
#plt.ylabel(r'$M_{r}$')
#plt.xlabel(r'$v_{peak} /\mathrm{km\ s^{-1}}$')
#plt.show()
#
#r_big = 7470/70 #Mpc
#r_small = 2532/70 #Mpc
#V_sim = (((4/3)*np.pi*(r_big**3)) - ((4/3)*np.pi*(r_small**3)))/(0.7**3) #(Mpc/h)^3
#N_sim = len(mock_catalog[mock_catalog['M_r']<=-17.0])
#n_sim = N_sim/V_sim
#
#N_eco = len(eco_obs_catalog)
#n_eco = N_eco/v_eco
#
#n_frac = n_sim/n_eco
#
#mock = mock_catalog.loc[mock_catalog['M_r']<=-17.0,['M_r','logMbary(logMsun)','Re(kpc)']]
#data = eco_obs_catalog
#mock.columns = ['M_r', 'logmbary', 'Re']
#concatenated = pd.concat([mock.assign(dataset='mock'), data.assign(dataset='data')])
#sns.pairplot(concatenated,hue='dataset')


##############################################################################
#plt.scatter(Mr_unique,n_Mr,c='r',s=5,label='unique bins')
#plt.scatter(bin_centers,n_Mr_2,c='g',s=5,label='cum sum with unique bins')
#plt.scatter(bin_centers_2,n_Mr_3,c='b',s=5,label='cum sum with FD bins')
#plt.fill_between(bin_centers_cut,np.log10(n_Mr_cut-err_poiss_cut),\
#                 np.log10(n_Mr_cut+err_poiss_cut),alpha=0.5,\
#                 label='data until -17.33')
##plt.plot(np.linspace(min_Mr,max_Mr),fit_alldata,'--g',label='all data fit')
#f_h = interpolate.interp1d(n_vpeak,bin_centers_vpeak,fill_value="extrapolate")
#    mbary = eco_obs_catalog.loc[eco_obs_catalog['M_r']==mag_value,'logmbary']
#    mbary_arr.append(mbary)
#    re = eco_obs_catalog.loc[eco_obs_catalog['M_r'] == mag_value, 'Re']
#    re_arr.append(re)

#    if mag_bool:
#        n_diff = np.cumsum(n_diff) 
#    else:
#        n_diff = np.cumsum(np.flip(n_diff,0))
#        n_diff = np.flip(n_diff,0)