#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:08:10 2018

@author: asadm2

This script calculates the ratio of stellar mass of satellites and stellar
mass of their central per host halo
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import argparse

###Formatting for plots
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=15)
rc('text', usetex=True)

def plot_cens_sats_stelratio(catalog):
    """
    Plots distribution of ratio of stellar mass of satellites and 
    stellar mass of central galaxy per host halo
    
    Parameters
    ----------
    catalog: Pandas dataframe
        Merged dataframe containing halo and galaxy information   
        
    """
    print('    -> Separating centrals and satellites')
    cens = catalog.loc[catalog.C_S.values == 1]
    sats = catalog.loc[catalog.C_S.values == 0]
    
    cens_subset = cens[['halo_hostid','stellar_mass','halo_macc_host_halo']]
    sats = sats.sort_values(by='halo_hostid')      
    cens_subset = cens_subset.sort_values(by='halo_hostid')
    
    print('    -> Grouping satelllites using host halo ID')
    sats_grp_by_halohostid = sats.groupby(['halo_hostid'])
    sats_keys = sats_grp_by_halohostid.groups.keys()
    
    print('    -> Getting stellar masses of all satellites per group')
    sats_per_grp_stellar_mass = []
    for key in sats_keys:
        sats_stellar_mass = sats_grp_by_halohostid.get_group(key).\
                            stellar_mass.values
        sats_per_grp_stellar_mass.append(sats_stellar_mass)
        
    sats_per_grp_stellar_mass_hosthaloid_dict = dict(zip(sats_keys,\
                                                sats_per_grp_stellar_mass))
    
    print('    -> Calculating ratio and checking how many satellites are more'
          'massive than their central')
    stellar_ratio_subs_cens = []
    halo_mass = []
    stellar_mass_cens = []
    stellar_mass_sats = []
    counter_stellar_mass_sats_higher = 0
    counter_ratio_bigger_than_one = 0
    for index,cens_halohostid in enumerate(cens_subset.halo_hostid.values):
        try:
            sats_per_grp_stellar_mass = sats_per_grp_stellar_mass_hosthaloid_dict[cens_halohostid]
            cen_stellar_mass = cens_subset.stellar_mass.values[index]
            host_halo_mass = cens_subset.halo_macc_host_halo.values[index]
            for value in sats_per_grp_stellar_mass:
                halo_mass.append(host_halo_mass)
                stellar_mass_cens.append(cen_stellar_mass)
                stellar_mass_sats.append(value)
                if value > cen_stellar_mass:
                    counter_stellar_mass_sats_higher+=1
                ratio = value/cen_stellar_mass
                if ratio > 1:
                    counter_ratio_bigger_than_one+=1
                stellar_ratio_subs_cens.append(ratio)
        except:
            stellar_ratio_subs_cens.append(0)
    print('    -> Number of satellites more massive than their centrals:{0}'.\
          format(counter_stellar_mass_sats_higher))
    print('    -> Number of satellites whose ratio is bigger than one:{0}'.\
          format(counter_ratio_bigger_than_one))
    
    print('    -> Plotting distribution')
    fig1 = plt.figure()
    plt.hist(stellar_ratio_subs_cens, bins=np.logspace(np.log10(0.01),\
                                                       np.log10(105), 100))
    plt.xlabel(r'$\frac{M_{\star ,satellite}}{M_{\star ,central}}$')   
    plt.gca().set_xscale("log")
    plt.tight_layout()
    print('    -> Saving figure')
    fig1.savefig('../reports/figures/stellar_ratio_dist.png')
    
    print('    -> Plotting SMHM')
    fig2 = plt.figure(figsize=(10,8))
    plt.scatter(np.log10(halo_mass),np.log10(stellar_mass_sats),c='grey',\
                alpha=0.8,label='Satellites',s=5)
    plt.scatter(np.log10(halo_mass),np.log10(stellar_mass_cens),c='red',\
                alpha=0.9,label='Centrals',s=5)
    plt.xlabel(r'$\mathrm{Halo\ mass\ (macc)}/\mathrm{[\frac{M_\odot}{h}]})$') 
    plt.ylabel(r'$\mathrm{Stellar\ mass}/\mathrm{[\frac{M_\odot}{h}]})$')
    plt.legend(loc='best',prop={'size': 6})
    plt.tight_layout()
    print('    -> Saving figure')
    fig2.savefig('../reports/figures/SMHM.png')
    
def args_parser():
    """
    Parsing arguments passed to populate_mock.py script

    Returns
    -------
    args: 
        Input arguments to the script
    """
    print('Parsing in progress')
    parser = argparse.ArgumentParser()
    parser.add_argument('catalog_to_use',type=str,help='Use catalog populated'\
                        ' using halo_macc or halo_mvir')
    args = parser.parse_args()
    return args

def main(args):
    """
    Main function that calls all other functions
    
    Parameters
    ----------
    args: 
        Input arguments to the script

    """
    print('Reading galaxy catalog')
    halocat_galcat_merged = pd.read_hdf('../'+args.catalog_to_use,\
                                        key='halocat_galcat_merged')
    print('Plotting stellar ratio distribution')
    plot_cens_sats_stelratio(halocat_galcat_merged)

if __name__ == '__main__':
    args = args_parser()
    main(args)