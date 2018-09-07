#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 22:06:18 2018

@author: asadm2
"""

import matplotlib
matplotlib.use('Agg')
from cosmo_utils.mock_catalogues.shmr_funcs import Behroozi_relation
from cosmo_utils.utils.stats_funcs import Stats_one_arr
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import argparse

###Formatting for plots
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']},size=15)
rc('text', usetex=True)


def stats_cens_func(cens_df,key):
    if key == 'halo_mvir':
        stats_cens = Stats_one_arr(np.log10(cens_df.halo_mvir_host_halo.\
                                            values),np.log10(cens_df.\
                                                  stellar_mass.values))
    elif key == 'halo_macc':
        stats_cens = Stats_one_arr(np.log10(cens_df.halo_macc_host_halo.\
                                            values),np.log10(cens_df.\
                                                  stellar_mass.values))
    return stats_cens

def plot_SMHM(halocat_galcat_merged,mass_to_plot_key,populate_mock_key):
    print('    -> Separating centrals and satellites')
    cens = halocat_galcat_merged.loc[halocat_galcat_merged.C_S.values == 1]
    sats = halocat_galcat_merged.loc[halocat_galcat_merged.C_S.values == 0]
    print('    -> Overplotting Behroozi 2013 relation for centrals')
    log_mstar_arr = np.linspace(halocat_galcat_merged.stellar_mass.values.min\
                                (),halocat_galcat_merged.stellar_mass.values.\
                                max(),100)
    log_mstar_arr = np.log10(log_mstar_arr)
    log_halo_mass_behroozi = Behroozi_relation(log_mstar_arr,z=0.0186)

    
    if mass_to_plot_key == 'halo_mvir':
        print('    -> Calculating statistics using {0}'.format\
              (mass_to_plot_key))
        stats_cens = stats_cens_func(cens,mass_to_plot_key)
        print('    -> Plotting')
        fig1 = plt.figure()
        plt.scatter(np.log10(sats.halo_mvir_host_halo.values),\
                    np.log10(sats.stellar_mass.values),color='g',s=5,\
                    alpha=0.5,label='Satellites')
        plt.plot(log_halo_mass_behroozi,log_mstar_arr,'-k',\
                 label='Behroozi 2013')
        plt.errorbar(stats_cens[0],stats_cens[1],yerr=stats_cens[2],color='r',\
                     label='Centrals')
        plt.xlabel(r'$\mathrm{Halo\ mass\ (mvir)}/\mathrm{[\frac{M_\odot}{h}]'\
                              '})$')
        plt.ylabel(r'$\mathrm{Stellar\ mass}/\mathrm{[\frac{M_\odot}{h}]})$')
        plt.title('SM-HM relation using {0} to populate mocks'.format\
                  (populate_mock_key))


    elif mass_to_plot_key == 'halo_macc':
        print('    -> Calculating statistics using {0}'.format\
              (mass_to_plot_key))
        stats_cens = stats_cens_func(cens,mass_to_plot_key)
        print('    -> Plotting')
        fig1 = plt.figure()
        plt.scatter(np.log10(sats.halo_macc_host_halo.values),\
                    np.log10(sats.stellar_mass.values),color='g',s=5,\
                    alpha=0.5,label='Satellites')
        plt.plot(log_halo_mass_behroozi,log_mstar_arr,'-k',\
                 label='Behroozi 2013')
        plt.errorbar(stats_cens[0],stats_cens[1],yerr=stats_cens[2],color='r',\
                     label='Centrals')
        plt.xlabel(r'$\mathrm{Halo\ mass\ (mvir)}/\mathrm{[\frac{M_\odot}{h}]'\
                              '})$')
        plt.ylabel(r'$\mathrm{Stellar\ mass}/\mathrm{[\frac{M_\odot}{h}]})$')
        plt.title('SM-HM relation using {0} to populate mocks'.format\
                  (populate_mock_key))

    
    plt.legend(loc='best')
    print('    -> Saving figure')
    fig1.savefig('../reports/SMHM_{0}_hosthalo.png'.format\
                 (mass_to_plot_key.split('_')[1]))
    
def args_parser():
    print('Parsing in progress')
    parser = argparse.ArgumentParser()
    parser.add_argument('host_halo_mass_to_plot',type=str,help='Halo mass '\
                        'type to plot (halo_macc or halo_mvir)')
    parser.add_argument('populate_mock_key',type=str,help='Halo mass type to '\
                        'populate mocks using (halo_macc or halo_mvir)')
    parser.add_argument('catalog_to_use',type=str,help='Use catalog populated'\
                        ' using halo_macc or halo_mvir')
    args = parser.parse_args()
    return args

def main(args):
    print('Reading galaxy catalog')
    halocat_galcat_merged = pd.read_hdf(args.catalog_to_use,\
                                        key='halocat_galcat_merged')
    print('Plotting SMHM')
    plot_SMHM(halocat_galcat_merged,args.host_halo_mass_to_plot,\
              args.populate_mock_key)

if __name__ == '__main__':
    args = args_parser()
    main(args)