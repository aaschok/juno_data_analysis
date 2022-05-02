from juno_classes import MagData, CWTData, Turbulence, JadeData
from juno_functions import time_in_lat_window, _get_files, find_orb_num
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from datetime import datetime, timedelta
import pandas as pd

def save():
    start = '2016-07-31T00:00:00'
    end = '2017-01-07T00:00:00'

    lat_df = time_in_lat_window(start, end)
    for index, row in lat_df.iterrows():
        mag_class = MagData(row['START'].isoformat(), row['END'].isoformat())
        mag_class.downsample_data(60)
        mag_class.mean_field_align(60)
        time_series = mag_class.data_df.index
        b_perp = np.sqrt(mag_class.data_df.B_PERP2.to_numpy()**2
                        + mag_class.data_df.B_PERP1.to_numpy()**2)
        cwt = CWTData(time_series, b_perp, 60)
        file_name = f'{row["START"].date()}_{row["END"].date()}_cwt.pickle'
        file_loc = f'/home/aschok/Documents/data/{file_name}'
        with open(file_loc, 'wb') as file:
            
            pickle.dump({'notes': 'mag data downsampled to 60 second inervals and MFA using a 60 minute window',
                        'time': cwt.time_series,
                        'freqs': cwt.freqs,
                        'coi': cwt.coi,
                        'power': cwt.power}, file, protocol=4)
            file.close()
        print('Saved data')
        
def save_heating():
    start = '2016-07-31T00:00:00'
    end = '2020-11-10T00:00:00'
    
    lat_df = time_in_lat_window(start, end)
    for index, row in lat_df.iterrows():
        start_datetime = row['START']
        end_datetime = row['END']
        file = f'q_data_{start_datetime.date()}-{end_datetime.date()}.pickle'
    
        turb = Turbulence(start_datetime.isoformat(),
                          end_datetime.isoformat(),
                          1, 60, 30)
        file_path = f'/home/aschok/Documents/data/heating_data/{file}'
        with open(file_path, 'wb') as pickle_file:
            pickle.dump(turb.q_data, pickle_file)
            print(f'Saved data from {start_datetime} to {end_datetime}')
            pickle_file.close()

def save_jade_ion():
    
    orbs = pd.read_fwf('/data/juno_spacecraft/data/orbits/juno_rec_orbit_v08.orb')
    orbs = orbs.drop(index=[0])
    for index in orbs.index[:-1]:
        orb_start = datetime.strptime(orbs['Event UTC APO'][index], '%Y %b %d %H:%M:%S')
        orb_mid = datetime.strptime(orbs['OP-Event UTC PERI'][index], '%Y %b %d %H:%M:%S')
        orb_end = datetime.strptime(orbs['Event UTC APO'][index + 1], '%Y %b %d %H:%M:%S')
        orb_num = orbs['No.'][index]
    
        orb_dict = {'num': orb_num, 'inbound': [orb_start, orb_mid], 'outbound': [orb_mid, orb_end]}
    
        for in_out in ['inbound', 'outbound']:

            start = orb_dict[in_out][0].strftime('%Y-%m-%dT%H:%M:%S')
            end = orb_dict[in_out][1].strftime('%Y-%m-%dT%H:%M:%S')

            data_folder = '/data/juno_spacecraft/data/jad'
            file_path = f'/home/aschok/Documents/data/pickled_jad/jad_ion_orb{orb_num}_{in_out}.pkl'
            dat_files = _get_files(start, end,'.DAT',data_folder,'JAD_L30_LRS_ION_ANY_CNT') 
            jade_ion = JadeData(dat_files, start, end)
            jade_ion.getIonData()
        
            data = jade_ion.ion_df.to_numpy().transpose()[: -1, ]
            time_array = jade_ion.ion_df.index.strftime('%Y-%m-%dT%H:%M:%S').to_numpy()
            dim1_array = jade_ion.ion_dims
            with open(file_path, 'wb') as pickle_file:
                pickle.dump({'UTC': time_array, 'Data': data, 'Energy': dim1_array}, pickle_file)
                pickle_file.close()

def save_jade_elec():
    
    orbs = pd.read_fwf('/data/juno_spacecraft/data/orbits/juno_rec_orbit_v08.orb')
    orbs = orbs.drop(index=[0])
    for index in orbs.index[:-1]:
        
        orb_start = datetime.strptime(orbs['Event UTC APO'][index], '%Y %b %d %H:%M:%S')
        orb_mid = datetime.strptime(orbs['OP-Event UTC PERI'][index], '%Y %b %d %H:%M:%S')
        orb_end = datetime.strptime(orbs['Event UTC APO'][index + 1], '%Y %b %d %H:%M:%S')
        orb_num = orbs['No.'][index]
    
        orb_dict = {'num': orb_num, 'inbound': [orb_start, orb_mid], 'outbound': [orb_mid, orb_end]}
        skip_orbs = np.arange(0, 10)

        for in_out in ['inbound', 'outbound']:

            start = orb_dict[in_out][0].strftime('%Y-%m-%dT%H:%M:%S')
            end = orb_dict[in_out][1].strftime('%Y-%m-%dT%H:%M:%S')

            data_folder = '/data/juno_spacecraft/data/jad'
            file_path = f'/home/aschok/Documents/data/pickled_jad/jad_elec_orb{orb_num}_{in_out}.pkl'
            dat_files = _get_files(start, end,'.DAT',data_folder,'JAD_L30_LRS_ION_ANY_CNT') 
            jade_elec = JadeData(dat_files, start, end)
            jade_elec.getElecData()
        
            data = jade_elec.elec_df.to_numpy().transpose()[: -1, ]
            time_array = jade_elec.elec_df.index.strftime('%Y-%m-%dT%H:%M:%S').to_numpy()
            dim1_array = jade_elec.elec_dims
            with open(file_path, 'wb') as pickle_file:
                pickle.dump({'UTC': time_array, 'Data': data, 'Energy': dim1_array}, pickle_file)
                pickle_file.close()
            print(f'Orbit {orb_num} {in_out} made')


    
if __name__ == '__main__':
    save_jade_ion()
    save_jade_elec()
