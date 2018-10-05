#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 16:13:15 2018

@author: asadm2
"""
import matplotlib as mpl
mpl.use('agg')
from cosmo_utils.utils.stats_funcs import Stats_one_arr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def velocity_func(r,r_s,r_vir,M_vir):
    #r_s and r_vir in Mpc/h in simulation
    r_s = r_s*1000
    r_vir = r_vir*1000
    G = 4.302*10**-6 #kpc M_sun^-1 (km/s)^2
    const = np.sqrt(G*M_vir/r_vir)
    num = np.log(1+(r/r_s)) + ((r/r_s)/(1+(r/r_s)))
    denom = (r/r_vir)*((np.log(1+(r_vir/r_s))) + ((r_vir/r_s)/(1+(r_vir/r_s))))
    result = const*(np.sqrt(num/denom))
    return result

mock_catalog_tiled = pd.read_hdf('../data/ECO_Vishnu_mock_catalog_tiled.h5',\
                                    key='mock_catalog_tiled')

mock_catalog_tiled = mock_catalog_tiled.loc[mock_catalog_tiled.Re.values >= 0]

mock_catalog_tiled_sats = mock_catalog_tiled.loc[mock_catalog_tiled['C_S'] == 0] 
mock_catalog_tiled_cens = mock_catalog_tiled.loc[mock_catalog_tiled['C_S'] == 1] 

velocities_re_sats = velocity_func(mock_catalog_tiled_sats.Re.values,\
                                   mock_catalog_tiled_sats.halo_rs.values,\
                                   mock_catalog_tiled_sats.halo_rvir.values,\
                                   mock_catalog_tiled_sats.halo_mvir.values)

velocities_re_cens = velocity_func(mock_catalog_tiled_cens.Re.values,\
                                   mock_catalog_tiled_cens.halo_rs.values,\
                                   mock_catalog_tiled_cens.halo_rvir.values,\
                                   mock_catalog_tiled_cens.halo_mvir.values)

Stats_one_arr_vre_sats = Stats_one_arr(np.log10(mock_catalog_tiled_sats.\
                                                Re.values),\
                                       np.log10(velocities_re_sats),base=0.4)

Stats_one_arr_vre_cens = Stats_one_arr(np.log10(mock_catalog_tiled_cens.\
                                                Re.values),\
                                       np.log10(velocities_re_cens),base=0.4)  

Stats_one_arr_vpeak_cens = Stats_one_arr(np.log10(mock_catalog_tiled_cens.\
                                                  Re.values),\
                                         np.log10(mock_catalog_tiled_cens.\
                                                  vpeak.values),base=0.4)

Stats_one_arr_vpeak_sats = Stats_one_arr(np.log10(mock_catalog_tiled_sats.\
                                                  Re.values),\
                                         np.log10(mock_catalog_tiled_sats.\
                                                  vpeak.values),base=0.4)
#fig1 = plt.figure()
#plt.errorbar(Stats_one_arr_vre_sats[0],Stats_one_arr_vre_sats[1],\
#             yerr=Stats_one_arr_vre_sats[2],color='b',label='v_re satellites')
#plt.errorbar(Stats_one_arr_vre_cens[0],Stats_one_arr_vre_cens[1],\
#             yerr=Stats_one_arr_vre_cens[2],color='r',label='v_re centrals')
#plt.errorbar(Stats_one_arr_vpeak_cens[0],Stats_one_arr_vpeak_cens[1],\
#             yerr=Stats_one_arr_vpeak_cens[2],color='r',linestyle='-.',\
#             label='v_peak centrals')
#plt.errorbar(Stats_one_arr_vpeak_sats[0],Stats_one_arr_vpeak_sats[1],\
#             yerr=Stats_one_arr_vpeak_sats[2],color='b',linestyle='-.',\
#             label='v_peak satellites')
#plt.xlabel(r'$log_{10}\ R_{e}$[kpc]')
#plt.ylabel(r'$log_{10}\$ v[km/s]')
#plt.legend(loc='best')
#plt.savefig('../reports/figures/vre_vpeak.png')

fig1 = plt.figure(figsize=(10,8))
plt.subplot(221)
plt.plot(Stats_one_arr_vre_sats[0],Stats_one_arr_vre_sats[1],color='b',\
            label='v_re satellites')
plt.fill_between(Stats_one_arr_vre_sats[0],Stats_one_arr_vre_sats[1]+\
                 Stats_one_arr_vre_sats[2],Stats_one_arr_vre_sats[1]-\
                 Stats_one_arr_vre_sats[2],color='b')
plt.plot(Stats_one_arr_vre_cens[0],Stats_one_arr_vre_cens[1],color='r',\
            label='v_re centrals')
plt.fill_between(Stats_one_arr_vre_cens[0],Stats_one_arr_vre_cens[1]+\
                 Stats_one_arr_vre_cens[2],Stats_one_arr_vre_cens[1]-\
                 Stats_one_arr_vre_cens[2],color='r')
plt.xlabel(r'$log_{10}\ R_{e}$[kpc]')
plt.ylabel(r'$log_{10}\ v$[km/s]')
plt.legend(loc='best')

plt.subplot(222)
plt.plot(Stats_one_arr_vpeak_sats[0],Stats_one_arr_vpeak_sats[1],color='b',\
            label='v_peak satellites',linestyle='-.')
plt.fill_between(Stats_one_arr_vpeak_sats[0],Stats_one_arr_vpeak_sats[1]+\
                 Stats_one_arr_vpeak_sats[2],Stats_one_arr_vpeak_sats[1]-\
                 Stats_one_arr_vpeak_sats[2],color='b')
plt.plot(Stats_one_arr_vpeak_cens[0],Stats_one_arr_vpeak_cens[1],color='r',\
            label='v_peak centrals',linestyle='-.')
plt.fill_between(Stats_one_arr_vpeak_cens[0],Stats_one_arr_vpeak_cens[1]+\
                 Stats_one_arr_vpeak_cens[2],Stats_one_arr_vpeak_cens[1]-\
                 Stats_one_arr_vpeak_cens[2],color='r')
plt.xlabel(r'$log_{10}\ R_{e}$[kpc]')
plt.ylabel(r'$log_{10}\ v$[km/s]')
plt.legend(loc='best')

plt.subplot(223)
plt.plot(Stats_one_arr_vpeak_sats[0],Stats_one_arr_vpeak_sats[1],color='b',\
            label='v_peak satellites',linestyle='-.')
plt.fill_between(Stats_one_arr_vpeak_sats[0],Stats_one_arr_vpeak_sats[1]+\
                 Stats_one_arr_vpeak_sats[2],Stats_one_arr_vpeak_sats[1]-\
                 Stats_one_arr_vpeak_sats[2],color='b',alpha=0.4)
plt.plot(Stats_one_arr_vre_sats[0],Stats_one_arr_vre_sats[1],color='b',\
            label='v_re satellites',alpha=0.4)
plt.fill_between(Stats_one_arr_vre_sats[0],Stats_one_arr_vre_sats[1]+\
                 Stats_one_arr_vre_sats[2],Stats_one_arr_vre_sats[1]-\
                 Stats_one_arr_vre_sats[2],color='b',alpha=0.4)
plt.xlabel(r'$log_{10}\ R_{e}$[kpc]')
plt.ylabel(r'$log_{10}\ v$[km/s]')
plt.legend(loc='best')

plt.subplot(224)
plt.plot(Stats_one_arr_vpeak_cens[0],Stats_one_arr_vpeak_cens[1],color='r',\
            label='v_peak centrals',linestyle='-.')
plt.fill_between(Stats_one_arr_vpeak_cens[0],Stats_one_arr_vpeak_cens[1]+\
                 Stats_one_arr_vpeak_cens[2],Stats_one_arr_vpeak_cens[1]-\
                 Stats_one_arr_vpeak_cens[2],color='r')
plt.plot(Stats_one_arr_vre_cens[0],Stats_one_arr_vre_cens[1],color='r',\
            label='v_re centrals',alpha=0.4)
plt.fill_between(Stats_one_arr_vre_cens[0],Stats_one_arr_vre_cens[1]+\
                 Stats_one_arr_vre_cens[2],Stats_one_arr_vre_cens[1]-\
                 Stats_one_arr_vre_cens[2],color='r',alpha=0.4)
plt.xlabel(r'$log_{10}\ R_{e}$[kpc]')
plt.ylabel(r'$log_{10}\ v$[km/s]')
plt.legend(loc='best')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
plt.savefig('../reports/figures/vre_vpeak_cens_sats.png')


