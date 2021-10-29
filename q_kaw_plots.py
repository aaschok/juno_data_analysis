import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as optimize
import spiceypy as spice
from scipy.stats import binned_statistic
from datetime import datetime
import spiceypy as spice
from juno_functions import time_in_lat_window
  

def q_test():
    r = re.compile('_data_')
    for year in ['2016', '2017', '2018', '2019', '2020']:
                spice.furnsh(f'/data/juno_spacecraft/data/meta_kernels/juno_{year}.tm')
    local_time_array = np.array([])
    radial_array = np.array([])
    latitude_array = np.array([])
    q_kaw_array = np.array([])
    mean_q_kaw_array = np.array([])
    rad_range_dict = {}
    orb_start  =np.array([])
    rad_range = []
    
    #Using the crossin masterlist, finds date ranges when Juno is within the sheah to remove this data
    crossings_df = pd.read_csv('/data/juno_spacecraft/data/crossings/crossingmasterlist/jno_crossings_master_fixed.dat')
    crossings_df = crossings_df.drop('NOTES', axis=1)
    in_sheath = False
    sheath_windows_df = pd.DataFrame({'START': [], 'END': []}) 
    for index, row in crossings_df.iterrows():
        if row.BOUNDARYID.lower() == 'sheath':
            if not in_sheath:
                start = datetime.fromisoformat(f'{row.DATE}T{row.TIME}')
                in_sheath = True
            
        if row.BOUNDARYID.lower() == 'magnetosphere':
            if in_sheath:
                end = datetime.fromisoformat(f'{row.DATE}T{row.TIME}')
                in_sheath = False
                sheath_windows_df = sheath_windows_df.append({'START':start,
                                                            'END': end},
                                                            ignore_index=True)

    for root, dirs, files in os.walk('/home/aschok/Documents/data/heating_data_+-10'):
        for file_name in files:
            if r.search(file_name):
                with open(os.path.join(root, file_name), 'rb') as data_file:
                    pkl = pickle.load(data_file)
                    
                    
                    for index, row in sheath_windows_df.iterrows():
                        if (pkl.index.min() < row.START < pkl.index.max()) or\
                           (pkl.index.min() < row.END < pkl.index.max()):
                            pkl = pkl[(pkl.index < row.START) | (pkl.index > row.END)]
                            print(pkl)
                    
                    orb_start = np.append(orb_start, pkl.index[0])
                    et_array = [spice.utc2et(i) for i in pkl.index.strftime('%Y-%m-%dT%H:%M:%S')]
                    positions, lt = spice.spkpos('JUNO', et_array,
                                                'JUNO_JSS', 'NONE', 'JUPITER')
                    x = positions.T[0]
                    y = positions.T[1]
                    z = positions.T[2]
                    rad = [np.sqrt(np.sum(np.power(vector, 2))) for vector in positions]
                    lat = np.arcsin(z / rad) * (180/np.pi)
                    long = np.arctan2(y, x) *(180/np.pi)
                    local_time = ((long + 180)*24/360)%24
                    
                    lat_mask = lat<=0
                    lat = np.array(lat)[lat_mask]
                    rad = np.array(rad)[lat_mask]
                    local_time = np.array(local_time)[lat_mask]
                    q_kaw = np.array(pkl.q_kaw)[lat_mask]
                    
                    local_time_array = np.append(local_time_array, np.mean(local_time))
                    radial_array = np.append(radial_array, np.array(rad)/69911)
                    latitude_array = np.append(latitude_array, lat)
                    q_kaw_array = np.append(q_kaw_array, q_kaw)
                    mean_q_kaw_array = np.append(mean_q_kaw_array, np.mean(q_kaw))
                    rad_range.append([np.min(rad)/69911, np.max(rad)/69911])
                            
    orb_start, rad_range = zip(*sorted(zip(orb_start, rad_range), key=lambda x: x[1]))
    spice.kclear()  
    
    fig, (ax1, ax4, ax2, ax3) = plt.subplots(4, 1, figsize=(10,8))
    ax1.scatter(local_time_array,
                mean_q_kaw_array,
                color='blue')
    ax1.set_title('$q_{kaw}$ vs. Local Time')
    ax1.set_ylabel('$q_{kaw}$')
    ax1.set_xlabel('Local Time')
    ax1.set_yscale('log')
    
    radial_array = radial_array[np.isfinite(q_kaw_array)]
    latitude_array = latitude_array[np.isfinite(q_kaw_array)]
    q_kaw_array = q_kaw_array[np.isfinite(q_kaw_array)]
    
    bins = np.arange(np.floor(np.min(radial_array)) - 0.5,
                              np.ceil(np.max(radial_array)) + 1.5)
    means, edges, num = binned_statistic(radial_array, q_kaw_array,
                                         'mean', bins)
    bin_std, std_edges, num = binned_statistic(radial_array, q_kaw_array,
                                               'std', bins)
    edges += 0.5
    ax2.plot(edges[:-1], means, label='avg $q_{kaw}$')
    ax2.plot(edges[:-1], means + bin_std, label='avg $q_{kaw}$ + 1$\sigma$')
    ax2.set_xlabel('R ($R_j$)')
    ax2.set_ylabel('$q_{kaw}$')
    ax2.set_title('$q_{kaw}$ vs. Radial distance')
    ax2.set_yscale('log')
    ax2.legend(loc='upper right')
    ax2.set_xlim(edges[0], edges[-1])

    i = 0
    for rads in rad_range:
        i += 1
        ax4.plot(rads, [i, i], label=f'{i}', linewidth=2)
    ax4.set_title('Radial distance range per orbit')
    ax4.set_xlim(edges[0], edges[-1])
    ax4.set_xlabel('R ($R_{j}$)')
    ax4.set_ylabel('Orbit number')
    ax4.set_yticks(np.arange(0, 35, 5))

    bins = np.arange(np.floor(np.min(latitude_array)) - 0.5,
                              np.ceil(np.max(latitude_array)) + 1.5)
    means, edges, num = binned_statistic(latitude_array, q_kaw_array,
                                         'mean', bins)
    bin_std, std_edges, num = binned_statistic(latitude_array, q_kaw_array,
                                               'std', bins)
    edges += 0.5
    ax3.plot(edges[:-1], means, label='avg $q_{kaw}$')
    ax3.plot(edges[:-1], means + bin_std, label='avg $q_{kaw}$ + 1$\sigma$')
    ax3.set_xlabel('Latitude (deg)')
    ax3.set_ylabel('$q_{kaw}$')
    ax3.set_title(f'$q_{{kaw}}$ vs. Latitude')
    ax3.set_yscale('log')
    ax3.legend(loc='upper right')        
                
    plt.tight_layout(h_pad=0.25)      
    plt.show()
    

    
def orbits_plot():
    
    start = '2016-07-31T00:00:00'
    end = '2020-12-04T00:00:00'
    time_series = pd.date_range(start, end, freq='1H')
    for year in ['2016', '2017', '2018', '2019', '2020']:
            spice.furnsh(f'/data/juno_spacecraft/data/meta_kernels/juno_{year}.tm')
    et_range = [spice.utc2et(i) for i in time_series.strftime('%Y-%m-%dT%H:%M:%S')]

    positions, lt = spice.spkpos('JUNO', et_range, 'JUNO_JSS', 'NONE', 'JUPITER')
    
    x = positions.T[0]/69911
    y = positions.T[1]/69911
    z = positions.T[2]/69911
    
    plt.plot(x, y)
        
    plt.grid()
    plt.show()
    
    lat_df = time_in_lat_window(start, end)
    
    for index, window in lat_df.iterrows():
        time_series = pd.date_range(window['START'].isoformat(), window['END'].isoformat(), freq='1H')
        et_range = [spice.utc2et(i) for i in time_series.strftime('%Y-%m-%dT%H:%M:%S')]

        positions, lt = spice.spkpos('JUNO', et_range, 'JUNO_JSS', 'NONE', 'JUPITER')
        
        x = positions.T[0]/69911
        y = positions.T[1]/69911
        z = positions.T[2]/69911
        
        plt.plot(x, y)
        
    plt.grid()
    plt.show()
    spice.kclear()
    
if __name__ == '__main__':
  q_test()  