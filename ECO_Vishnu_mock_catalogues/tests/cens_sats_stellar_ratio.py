#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:08:10 2018

@author: asadm2
"""

import matplotlib
matplotlib.use('Agg')
from halotools.utils import group_member_generator
from astropy.table import Table
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
    Plots distribution of ratio of total stellar mass of satellites and 
    stellar mass of central galaxy per host halo
    
    Parameters
    ----------
    catalog: Pandas dataframe
        Merged dataframe containing halo and galaxy information   
        
    """
    print('    -> Separating centrals and satellites')
    cens = catalog.loc[catalog.C_S.values == 1]
    sats = catalog.loc[catalog.C_S.values == 0]
    
    cens_subset = cens[['halo_hostid','stellar_mass']]
    sats = sats.sort_values(by='halo_hostid')      
    cens_subset = cens_subset.sort_values(by='halo_hostid')
    
    print('    -> Grouping satelllites using host halo ID')
    sats_grp_by_halohostid = sats.groupby(['halo_hostid'])
    sats_keys = sats_grp_by_halohostid.groups.keys()
    
    print('    -> Calculating total stellar mass of satellites per group')
    total_grp_stellar_mass = []
    for key in sats_keys:
        grp_stellar_mass = sats_grp_by_halohostid.get_group(key).\
                           stellar_mass.sum()
        total_grp_stellar_mass.append(grp_stellar_mass)
        
    grp_stellar_mass_hosthaloid_dict = dict(zip(sats_keys,\
                                                total_grp_stellar_mass))
    
    print('    -> Calculating ratio')
    stellar_ratio_subs_cens = []
    for index,cens_halohostid in enumerate(cens_subset.halo_hostid.values):
        try:
            grp_stellar_mass = grp_stellar_mass_hosthaloid_dict[cens_halohostid]
            cen_stellar_mass = cens_subset.stellar_mass.values[index]
            ratio = grp_stellar_mass/cen_stellar_mass
            stellar_ratio_subs_cens.append(ratio)
        except:
            stellar_ratio_subs_cens.append(0)
    
    print('    -> Plotting distribution')
    fig1 = plt.figure()
    plt.hist(stellar_ratio_subs_cens, bins=np.logspace(np.log10(0.01),\
                                                       np.log10(105), 100))
    plt.xlabel(r'$\frac{\sum M_{\star ,satellite}}{M_{\star ,central}}$')   
    plt.gca().set_xscale("log")
    plt.tight_layout()
    print('    -> Saving figure')
    fig1.savefig('../reports/stellar_ratio_dist.png')
    
#    sats_tb = Table.from_pandas(sats)
#    sats_tb.sort('halo_hostid') 
#    grouping_key = 'halo_hostid'
#    requested_columns = ['stellar_mass']
#    group_gen = group_member_generator(sats_tb, grouping_key, requested_columns)
        
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