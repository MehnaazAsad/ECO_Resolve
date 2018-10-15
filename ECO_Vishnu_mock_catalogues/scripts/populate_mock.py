#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 08:55:51 2018

@author: asadm2

This script creates a halo catalog from Vishnu snapshot, populates the catalog
with galaxies and adds a centrals/satellites (1/0) flag
"""

from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.sim_manager import UserSuppliedHaloCatalog
from halotools.utils import broadcast_host_halo_property
from halotools.sim_manager import CachedHaloCatalog
from progressbar import ProgressBar
import pandas as pd
import numpy as np
import argparse
import os

def create_halocat_from_snapshot(col_ids,col_names,snapshot_file,\
                                 processing_notes):
    """
    Creates a halo catalog from snapshot

    Parameters
    ----------
    col_ids: python array
        array with column IDs to extract from snapshot

    col_names: python array
        array with column names to extract from snapshot
    
    snapshot_file: string
        name of snapshot file
    
    processing_notes: string
        Notes to add to halo catalog cache

    Returns
    ---------
    halocat_file_path: string
        Path to halo catalog
    """
    rockstar_table = pd.read_table(snapshot_file,delimiter='\s+',\
                               compression='gzip',comment='#',\
                               usecols=col_ids,names=col_names)
    
    #Mpc
    rockstar_table.halo_rvir = rockstar_table.halo_rvir/1000 
    rockstar_table.halo_rs = rockstar_table.halo_rs/1000
    
    redshift = 0.0186
    Lbox, particle_mass = 130,3.215e7

    halocat = UserSuppliedHaloCatalog(redshift=redshift,\
                                  halo_rvir=rockstar_table.halo_rvir.values,\
                                  halo_vx=rockstar_table.halo_vx.values,\
                                  halo_vy=rockstar_table.halo_vy.values,\
                                  halo_vz=rockstar_table.halo_vz.values,\
                                  halo_pid=rockstar_table.halo_pid.values,\
                                  halo_upid=rockstar_table.halo_upid.values,\
                                  halo_rs=rockstar_table.halo_rs.values,\
                                  halo_m200=rockstar_table.halo_m200b.values,\
                                  halo_macc=rockstar_table.halo_macc.values,\
                                  halo_mpeak=rockstar_table.halo_mpeak.values,\
                                  Lbox=Lbox,particle_mass=particle_mass,\
                                  halo_x=rockstar_table.halo_x.values,\
                                  halo_y=rockstar_table.halo_y.values,\
                                  halo_z=rockstar_table.halo_z.values,\
                                  halo_id=rockstar_table.halo_id.values,\
                                  halo_mvir=rockstar_table.halo_mvir.values,\
                                  halo_vacc=rockstar_table.halo_vacc.values,\
                                  halo_vpeak=rockstar_table.halo_vpeak.values)

    halocat_file_path = '/home/asadm2/.astropy/cache/halotools/halo_catalogs'\
                    '/vishnu/rockstar/vishnu_rockstar.hdf5'

    halocat.add_halocat_to_cache(halocat_file_path,simname='vishnu',\
                                 halo_finder='rockstar',\
                                 version_name='vishnu_rockstar_ECO_v1.0',\
                                 processing_notes=processing_notes,\
                                 overwrite=True,redshift='0.0186')
    return halocat_file_path

def access_halocat(halocat_file_path):
    """
    Accesses already stored halo catalog

    Parameters
    ----------
    halocat_file_path: string
        Path to halo catalog

    Returns
    ---------
    halocat: hdf5 file
        Mock halo catalog
    """
    halocat = CachedHaloCatalog(fname=halocat_file_path,\
                                update_cached_fname=True)
    broadcast_host_halo_property(halocat.halo_table,'halo_macc')
    return halocat

def populate_mock(key,halocat):
    """
    Populates mock halo catalog with galaxies using Behroozi 2010 relation

    Parameters
    ----------
    key: string
        Halo mass property used to populate mock (halo_mvir or halo_macc)
        
    halocat: hdf5 file
        Mock halo catalog

    Returns
    ---------
    model: 
        subhalo-based composite model
    """
    model = PrebuiltSubhaloModelFactory('behroozi10',redshift=0.0186,\
                                        prim_haloprop_key=key) 
    model.populate_mock(halocat)
    return model

def create_temp_hdf5(halo_table,gal_table):
    """
    Creates temporary hdf5 copies of halo and galaxy tables

    Parameters
    ----------
    halo_table: Astropy table
        Halo catalog
        
    gal_table: Astropy table
        Galaxy catalog

    Returns
    ---------
    halo_table_fname: string
        Filename of temporary halo table
    
    galaxy_table_fname: string
        Filename of temporary galaxy table
    """
    halo_table_fname = 'halo_table_temp.hdf5'
    galaxy_table_fname = 'galaxy_table_temp.hdf5'
    halo_table.write(halo_table_fname,path='updated_data',\
                     compression=True)
    gal_table.write(galaxy_table_fname,path='updated_data',\
                     compression=True)
    return halo_table_fname,galaxy_table_fname
    
def merge_halocat_galcat(halo_catalog_fname,galaxy_catalog_fname):
    """
    Merges halo and galaxy catalogs

    Parameters
    ----------
    halo_catalog_fname: string
        Filename of temporary halo table
    
    galaxy_catalog_fname: string
        Filename of temporary galaxy table

    Returns
    ---------
    halocat_galcat_merged: Pandas dataframe
        Merged dataframe containing halo and galaxy information    
    """
    halo_pd = pd.read_hdf(halo_catalog_fname)
    gal_pd = pd.read_hdf(galaxy_catalog_fname)
    
    cols_to_use = list(set(halo_pd.columns) - set(gal_pd.columns))
    cols_to_use.append('halo_id')
    
    halocat_galcat_merged = halo_pd[cols_to_use].merge(gal_pd,on='halo_id')
    return halocat_galcat_merged

def centrals_satellites_flag(catalog):
    """
    Adds centrals and satellites flag to merged catalog using ID information

    Parameters
    ----------
    catalog: Pandas dataframe
        Merged catalog of halos and galaxies
    
    Returns
    ---------
    catalog: Pandas dataframe
        Merged dataframe containing halo and galaxy information plus a 
        centrals/satellites flag column
    """
    C_S = []
    pbar = ProgressBar()
    
    for idx in pbar(range(7450441)):
        if catalog['halo_hostid'][idx] == catalog['halo_id'][idx]:
            C_S.append(1)
        else:
            C_S.append(0)
    
    C_S = np.array(C_S)
    catalog['C_S'] = C_S
    return catalog
    
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
    parser.add_argument('populate_mock_key',type=str,help='Halo mass type to '\
                        'populate mocks using (halo_macc or halo_mvir)')
    parser.add_argument('-halocat_from_snapshot',type=bool, \
                        help='True if creating halocat from snapshot.'\
                        'False if accessing stored halo catalog.'\
                        'Default is False.',default=False)
    parser.add_argument('-column_ids',nargs='+',type=int,help='Column IDs to '\
                        'extract from snapshot',default=None)
    parser.add_argument('-column_names',nargs='+',type=str,help='Names of '\
                        'columns to extract',default=None)
    parser.add_argument('-processing_notes',type=str,help='Note to add to '\
                        'halo catalog cache. Must be one string.',default=None)
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
    cols_to_use = [1,5,6,10,11,12,17,18,19,20,21,22,39,60,61,62,63]
    col_names = ['halo_id','halo_pid','halo_upid','halo_mvir','halo_rvir',\
             'halo_rs','halo_x','halo_y','halo_z','halo_vx','halo_vy',\
             'halo_vz','halo_m200b','halo_macc','halo_mpeak','halo_vacc',\
             'halo_vpeak']
    snapshot_file = 'hlist_0.98169.list.gz'
    processing_notes = 'Positions, IDs (HID,PID,UPID),velocities '\
                       '(vpeak and vacc), radius (rvir,rs) and mass '\
                       'information extracted (mvir,macc,m200,mpeak) '\
                       'from original snapshot'
                       

    if args.halocat_from_snapshot:
        for val in args.column_ids:
            cols_to_use.append(val)
        for name in args.column_names:
            col_names.append(name)
        processing_notes+=args.processing_notes
        
        print('Creating halo catalog from snapshot')
        halocat_file_path = create_halocat_from_snapshot(cols_to_use,\
                                                         col_names,\
                                                         snapshot_file,\
                                                         processing_notes)
    
    else:
        halocat_file_path = '/home/asadm2/.astropy/cache/halotools/'\
                            'halo_catalogs/vishnu/rockstar/vishnu_rockstar.hdf5'
    
    print('Accessing stored halo catalog')
    halocat = access_halocat(halocat_file_path)
    print('Populating mock using {0}'.format(args.populate_mock_key))
    model = populate_mock(args.populate_mock_key,halocat)

    #Bypass big and little endian compiler issues when converting astropy 
    #tables to pandas dataframes directly
    print('Making temporary copies of halo and galaxy catalogs')
    halo_catalog_fname,galaxy_catalog_fname = \
    create_temp_hdf5(halocat.halo_table,model.mock.galaxy_table)
    print('Merging both catalogs')
    halocat_galcat_merged = merge_halocat_galcat(halo_catalog_fname,\
                                                 galaxy_catalog_fname)
    print('Adding centrals/satellites flag')
    halocat_galcat_merged = centrals_satellites_flag(halocat_galcat_merged)
    print('Saving galaxy catalog as hdf5 file')
    halocat_galcat_merged.to_hdf('../halo_gal_Vishnu_Rockstar_{0}.h5'.format\
                                 (args.populate_mock_key.split('_')[1]),\
                                 key='halocat_galcat_merged',mode='w')
    
    print('Removing temporary copies of catalogs')
    os.remove(halo_catalog_fname)
    os.remove(galaxy_catalog_fname)
    
# Main function
if __name__ == '__main__':
    args = args_parser()
    main(args)