#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 16:07:26 2018

@author: asadm2
"""
import matplotlib as mpl
mpl.use('agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

mock_catalog_tiled = pd.read_hdf('../data/ECO_Vishnu_mock_catalog_tiled.h5',\
                                    key='mock_catalog_tiled')

H0 = 70
d = mock_catalog_tiled.cz.values/H0
RA = mock_catalog_tiled.ra.values 
DEC = mock_catalog_tiled.dec.values 
x = d*np.cos(DEC)*np.cos(RA)
y = d*np.cos(DEC)*np.sin(RA)
z = d*np.sin(DEC) 

mock_catalog_tiled['x'] = x
mock_catalog_tiled['y'] = y
mock_catalog_tiled['z'] = z

mock_catalog_tiled_slice = mock_catalog_tiled.\
                           loc[mock_catalog_tiled.z.values >= -5]
mock_catalog_tiled_slice = mock_catalog_tiled.\
                           loc[mock_catalog_tiled.z.values <= 5]
                           
colour_r = (1 + (mock_catalog_tiled_slice.x.values/130))/2                   
colour_g = (1 + (mock_catalog_tiled_slice.y.values/130))/2                   
colour_b = (1 + (mock_catalog_tiled_slice.z.values/130))/2     

colour_rgb = [[colour_r[index],colour_g[index],colour_b[index]] for index in\
              range(len(mock_catalog_tiled_slice))]
colour_rgb = np.array(colour_rgb)

fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(mock_catalog_tiled_slice['x'],mock_catalog_tiled_slice['y'],\
           mock_catalog_tiled_slice['z'],c=colour_rgb)
ax.set_xlabel('x (Mpc/h)')
ax.set_ylabel('y (Mpc/h)')
ax.set_zlabel('z (Mpc/h)')
ax.set_xlim(-130,130)
ax.set_ylim(-130,130)
ax.set_zlim(-130,130)
ax.set_aspect("equal")
#ax.view_init(0, 90)
plt.grid(None,'minor')
plt.savefig('../reports/figures/3d_tiled_sim_slice_z.png')