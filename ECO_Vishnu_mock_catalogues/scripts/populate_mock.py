#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 08:55:51 2018

@author: asadm2
"""

from cosmo_utils.mock_catalogues.shmr_funcs import Behroozi_relation
from halotools.empirical_models import PrebuiltSubhaloModelFactory
from halotools.sim_manager import UserSuppliedHaloCatalog
from halotools.utils import broadcast_host_halo_property
from cosmo_utils.utils.stats_funcs import Stats_one_arr
from halotools.sim_manager import CachedHaloCatalog
from progressbar import ProgressBar
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

cols_to_use = [1,5,6,10,11,12,17,18,19,20,21,22,39,60,61,62,63]
col_names = ['halo_id','halo_pid','halo_upid','halo_mvir','halo_rvir',\
             'halo_rs','halo_x','halo_y','halo_z','halo_vx','halo_vy',\
             'halo_vz','halo_m200b','halo_macc','halo_mpeak','halo_vacc',\
             'halo_vpeak']
rockstar_table = pd.read_table('hlist_0.98169.list.gz',delimiter='\s+',\
                               compression='gzip',comment='#',\
                               usecols=cols_to_use,names=col_names)

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
                    '/bolshoi/rockstar/bolshoi_test_v1.hdf5'

halocat.add_halocat_to_cache(halocat_file_path,simname='bolshoi',\
                             halo_finder='rockstar',\
                             version_name='my_rockstar_catalog_v1.0',\
                             processing_notes='Positions, IDs (HID,PID,UPID),'\
                             'velocities (vpeak and vacc), radius (rvir,rs)'\
                             'and mass information extracted'\
                             '(mvir,macc,m200,mpeak) from original snapshot'\
                             ,overwrite=True,redshift='0.0186')

halocat = CachedHaloCatalog(fname=halocat_file_path, update_cached_fname=True)
model = PrebuiltSubhaloModelFactory('behroozi10',redshift=0.0186,\
                                    prim_haloprop_key='halo_macc') 
model.populate_mock(halocat)

broadcast_host_halo_property(halocat.halo_table, 'halo_macc')

halo_pd = halocat.halo_table.to_pandas()
gal_pd = model.mock.galaxy_table.to_pandas()

cols_to_use = list(set(halo_pd.columns) - set(gal_pd.columns))
cols_to_use.append('halo_id')

halocat_galcat_merged = halo_pd[cols_to_use].merge(gal_pd,on='halo_id')

C_S = []
pbar = ProgressBar()

for idx in pbar(range(7450441)):
    if halocat_galcat_merged['halo_hostid'][idx] == halocat_galcat_merged\
    ['halo_id'][idx]:
        C_S.append(1)
    else:
        C_S.append(0)

C_S = np.array(C_S)
halocat_galcat_merged['C_S'] = C_S


cens = halocat_galcat_merged.loc[halocat_galcat_merged.C_S.values == 1]
sats = halocat_galcat_merged.loc[halocat_galcat_merged.C_S.values == 0]

stats_cens = Stats_one_arr(np.log10(cens.halo_mvir_host_halo.values),\
                           np.log10(cens.stellar_mass.values))

## Overplotting Behroozi 2013 relation for centrals
log_mstar_arr = np.linspace(halocat_galcat_merged.stellar_mass.values.min(),\
                          halocat_galcat_merged.stellar_mass.values.max(),100)
log_mstar_arr = np.log10(log_mstar_arr)
log_halo_mass_behroozi = Behroozi_relation(log_mstar_arr,z=0.0186)

fig1 = plt.figure()
plt.scatter(np.log10(sats.halo_mvir_host_halo.values),\
            np.log10(sats.stellar_mass.values),color='g',label='Satellites')
plt.plot(log_halo_mass_behroozi,log_mstar_arr,'-k',label='Behroozi 2013')
plt.errorbar(stats_cens[0],stats_cens[1],yerr=stats_cens[2],color='r',\
             label='Centrals')
plt.legend(loc='best')
plt.xlabel('Halo mass')
plt.ylabel('Stellar mass')
plt.show()

        



