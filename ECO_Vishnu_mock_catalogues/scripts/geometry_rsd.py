#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 16:22:26 2018

@author: asadm2
"""

from astropy.cosmology import FlatLambdaCDM
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np

def tile_sim_box(mock_catalog):
    ## Box 1
    mock_catalog_original = mock_catalog.loc[mock_catalog['M_r'] <= -17.33]
    mock_catalog_original = mock_catalog_original.reset_index()
    
    ## Box 2
    row_start_number = mock_catalog_original.index[-1]+1
    mock_catalog_temp = mock_catalog_original.append(mock_catalog_original,\
                                                     ignore_index=True)
    mock_catalog_temp.x.values[row_start_number:] -= 130
    mock_catalog_temp.halo_x.values[row_start_number:] -= 130
    
    ## Box 3
    row_start_number = mock_catalog_temp.index[-1]+1
    mock_catalog_temp = mock_catalog_temp.append(mock_catalog_original,\
                                                 ignore_index=True)
    mock_catalog_temp.y.values[row_start_number:] -= 130
    mock_catalog_temp.halo_y.values[row_start_number:] -= 130
    
    ## Box 4
    row_start_number = mock_catalog_temp.index[-1]+1
    mock_catalog_temp = mock_catalog_temp.append(mock_catalog_original,\
                                                 ignore_index=True)
    mock_catalog_temp.x.values[row_start_number:] -= 130
    mock_catalog_temp.halo_x.values[row_start_number:] -= 130
    mock_catalog_temp.y.values[row_start_number:] -= 130
    mock_catalog_temp.halo_y.values[row_start_number:] -= 130
    
    ## Box 5
    row_start_number = mock_catalog_temp.index[-1]+1
    mock_catalog_temp = mock_catalog_temp.append(mock_catalog_original,\
                                                 ignore_index=True)
    mock_catalog_temp.z.values[row_start_number:] -= 130
    mock_catalog_temp.halo_z.values[row_start_number:] -= 130
    
    ## Box 6
    row_start_number = mock_catalog_temp.index[-1]+1
    mock_catalog_temp = mock_catalog_temp.append(mock_catalog_original,\
                                                 ignore_index=True)
    mock_catalog_temp.x.values[row_start_number:] -= 130
    mock_catalog_temp.halo_x.values[row_start_number:] -= 130
    mock_catalog_temp.z.values[row_start_number:] -= 130
    mock_catalog_temp.halo_z.values[row_start_number:] -= 130
    
    ## Box 7
    row_start_number = mock_catalog_temp.index[-1]+1
    mock_catalog_temp = mock_catalog_temp.append(mock_catalog_original,\
                                                 ignore_index=True)
    mock_catalog_temp.y.values[row_start_number:] -= 130
    mock_catalog_temp.halo_y.values[row_start_number:] -= 130
    mock_catalog_temp.z.values[row_start_number:] -= 130
    mock_catalog_temp.halo_z.values[row_start_number:] -= 130
    
    ## Box 8
    row_start_number = mock_catalog_temp.index[-1]+1
    mock_catalog_temp = mock_catalog_temp.append(mock_catalog_original,\
                                                 ignore_index=True)
    mock_catalog_temp.x.values[row_start_number:] -= 130
    mock_catalog_temp.halo_x.values[row_start_number:] -= 130
    mock_catalog_temp.y.values[row_start_number:] -= 130
    mock_catalog_temp.halo_y.values[row_start_number:] -= 130
    mock_catalog_temp.z.values[row_start_number:] -= 130
    mock_catalog_temp.halo_z.values[row_start_number:] -= 130
    
    return(mock_catalog_temp)

def cart_to_spherical_coords(cart_arr, dist):
    """
    Computes the right ascension and declination for the given 
    point in (x,y,z) position
    Parameters
    -----------
    cart_arr: numpy.ndarray, shape (3,)
        array with (x,y,z) positions
    dist: float
        dist to the point from observer's position
    Returns
    -----------
    ra_val: float
        right ascension of the point on the sky
    dec_val: float
        declination of the point on the sky
    """
    ## Reformatting coordinates
    # Cartesian coordinates
    (   x_val,
        y_val,
        z_val) = cart_arr/float(dist)
    # Distance to object
    dist = float(dist)
    ## Declination
    dec_val = 90. - np.degrees(np.arccos(z_val))
    ## Right ascension
    if x_val == 0:
        if y_val > 0.:
            ra_val = 90.
        elif y_val < 0.:
            ra_val = -90.
    else:
        ra_val = np.degrees(np.arctan(y_val/x_val))
    ##
    ## Seeing on which quadrant the point is at
    if x_val < 0.:
        ra_val += 180.
    elif (x_val >= 0.) and (y_val < 0.):
        ra_val += 360.

    return ra_val, dec_val

mock_catalog = pd.read_hdf('../data/ECO_Vishnu_mock_catalog.h5',\
                           key='mock_catalog')    
mock_catalog_tiled = tile_sim_box(mock_catalog)

ngal = len(mock_catalog_tiled)
speed_c = 3*10**5 #km/s
z_min = 0
z_max = 0.5
dz = 10**-3
H0 = 70
omega_m = 0.25
omega_b = 0.04
redshift_median = 0.021723 #median redshift of ECO

redshift_arr = np.arange(z_min,z_max,dz)
cosmo = FlatLambdaCDM(H0,omega_m,omega_b)
como_dist = cosmo.comoving_distance(redshift_arr)
comodist_z_interp = interp1d(como_dist,redshift_arr)

cart_gals = mock_catalog_tiled[['x','y','z']].values
vel_gals = mock_catalog_tiled[['vx','vy','vz']].values

dist_from_obs_arr = np.zeros(ngal)
ra_arr = np.zeros(ngal)
dec_arr = np.zeros(ngal)
cz_arr = np.zeros(ngal)
cz_nodist_arr = np.zeros(ngal)
vel_tan_arr = np.zeros(ngal)
vel_tot_arr = np.zeros(ngal)
vel_pec_arr = np.zeros(ngal)
for x in tqdm(range(ngal)):
    dist_from_obs = (np.sum(cart_gals[x]**2))**.5
    cz_local = speed_c * comodist_z_interp(dist_from_obs)
    cz_val = cz_local
    ra,dec = cart_to_spherical_coords(cart_gals[x],dist_from_obs)
    vr = np.dot(cart_gals[x], vel_gals[x])/dist_from_obs
    cz_val += vr*(1+redshift_median)
    vel_tot = (np.sum(vel_gals[x]**2))**.5
    vel_tan = (vel_tot**2 - vr**2)**.5
    vel_pec  = (cz_val - cz_local)/(1 + redshift_median)
    dist_from_obs_arr[x] = dist_from_obs
    ra_arr[x] = ra
    dec_arr[x] = dec
    cz_arr[x] = cz_val
    cz_nodist_arr[x] = cz_local
    vel_tot_arr[x] = vel_tot
    vel_tan_arr[x] = vel_tan
    vel_pec_arr[x] = vel_pec

mock_catalog_tiled['r_dist'] = dist_from_obs_arr
mock_catalog_tiled['ra'] = ra_arr
mock_catalog_tiled['dec'] = dec_arr
mock_catalog_tiled['cz'] = cz_arr
mock_catalog_tiled['cz_nodist'] = cz_nodist_arr
mock_catalog_tiled['vel_tot'] = vel_tot_arr
mock_catalog_tiled['vel_tan'] = vel_tan_arr
mock_catalog_tiled['vel_pec'] = vel_pec_arr

mock_catalog_tiled = mock_catalog_tiled.loc[(mock_catalog_tiled['cz'] >= 2530) \
                                            & (mock_catalog_tiled['cz'] <= 8000)]
mock_catalog_tiled = mock_catalog_tiled.reset_index()

mock_catalog_tiled.to_hdf('../data/ECO_Vishnu_mock_catalog_tiled.h5',\
                          key='mock_catalog_tiled',mode='w')

#mock_catalog_tiled = pd.read_hdf('../data/ECO_Vishnu_mock_catalog_tiled.h5',\
#                                    key='mock_catalog_tiled')
#
#fig1 = plt.figure()
#ax = fig1.add_subplot(111, projection='3d')
#ax.scatter(mock_catalog_tiled['x'], mock_catalog_tiled['y'], mock_catalog_tiled['z'])
#ax.set_xlabel('x (Mpc/h)')
#ax.set_ylabel('y (Mpc/h)')
#ax.set_zlabel('z (Mpc/h)')
#ax.set_xlim(-130,130)
#ax.set_ylim(-130,130)
#ax.set_zlim(-130,130)
##ax.view_init(0, 90)
#plt.grid(None,'minor')
#plt.savefig('../reports/figures/3d_tiled_sim_slice_2.png')
