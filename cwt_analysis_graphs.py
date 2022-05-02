#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 10:30:04 2021

@author: aschok
"""
from datetime import date, datetime, timedelta
import pickle
from re import T

import matplotlib.dates as mdates
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycwt as wavelet
import spiceypy as spice
from matplotlib.colors import LogNorm
from scipy import integrate, signal

from juno_classes import CWTData, MagData, PlotClass, Turbulence, PosData
from juno_functions import time_in_lat_window, get_sheath_intervals, find_orb_num, get_orbit_data


def _plots(start_datetime, end_datetime, window_size, days_per_graph,
                 min_frequency, max_frequency, save_location):
        
        print(start_datetime)
        print(end_datetime)
        mag_class = MagData(start_datetime.isoformat(), end_datetime.isoformat())
        mag_class.downsample_data(60)
        mag_class.cart_to_sphere()
        mag_class.mean_field_align(window_size=window_size)
        mag_data = mag_class.data_df[['BX', 'BY', 'BZ', 'B_PERP1', 'B_PERP2', 'Br', 'Btheta', 'Bphi']]
        sheath_windows_df = get_sheath_intervals('/data/juno_spacecraft/data/crossings/crossingmasterlist/jno_crossings_master_v6.txt')
        position_class = PosData(mag_data.index)
        position_class.sys_3()
        sys_3_df = position_class.sys_3_df
        
        
        del(mag_class)
        
        start_datetime = mag_data.index[0]
        end_datetime = mag_data.index[-1]        
        num_graphs = (end_datetime - start_datetime)/pd.Timedelta(days=days_per_graph)
        end_datetime = start_datetime
        for i in range(0, int(np.floor(num_graphs)) + 1):
            start_datetime = end_datetime
            end_datetime = start_datetime + pd.Timedelta(days=days_per_graph)
            graph_data = mag_data[start_datetime.isoformat(): end_datetime.isoformat()]
            end_datetime = graph_data.index[-1]
            graph_data = graph_data.fillna(0)
            if len(graph_data) <= 2 or (end_datetime - start_datetime) < timedelta(days=1):
                break
            
            b_perp = np.sqrt(graph_data.B_PERP1**2 + graph_data.B_PERP2**2).to_numpy()
            cwt = CWTData(graph_data.index, b_perp, 60, min_frequency, max_frequency)
            cwt.remove_sheath()
            
            sheath_window = False
            data = graph_data
            for index, row in sheath_windows_df.iterrows():
                if (graph_data.index.min() < row.START < graph_data.index.max()) or\
                (graph_data.index.min() < row.END < graph_data.index.max()):
                    sheath_window = True
                    data = data[(data.index < row.START) | (data.index > row.END)]
            if sheath_window:
                graph_data = data                
            
            grid = gs.GridSpec(2, 2, wspace=0.265)
            fig = plt.figure(figsize=(20, 10))
            fig.suptitle(f'{start_datetime.date()} to {end_datetime.date()}', fontsize=20)
            ax0 = fig.add_subplot(grid[0,0])
            plot_class = PlotClass(ax0, ylabel='Magnetic Field')
            plot_class.plot(graph_data.index, [graph_data.Br, graph_data.Btheta, graph_data.Bphi],
                            True, ['Br', 'Btheta', 'Bphi'])
            
            # Z_centrifugal plotted
            sys_3_graph = sys_3_df[start_datetime.isoformat(): end_datetime.isoformat()]
            ax_sys3 = ax0.twinx()
            ax_sys3.plot(sys_3_graph.index, sys_3_graph.eq_dist, label='sys_3')
            ax_sys3.set_ylabel('Z_cent (R_j)')
            ax_sys3.set_title(f'Distance Range {round(sys_3_df.radial_3.loc[start_datetime.isoformat()],2)} $R_j$ - {round(sys_3_df.radial_3.loc[end_datetime.isoformat()],2)} $R_j$')
            plot_class.xaxis_datetime_tick_labels(True)
            
            # Jade Mean Counts Plotted
            orb_num = find_orb_num(start_datetime)
            try:
                with open('/home/delamere/Juno/jad_mean_orbit_{orb_num}.pkl', 'rb') as jad_mean_file:
                    mean_data = pickle.load(jad_mean_file)
                    jad_mean_file.close()
                plot_index = np.where((mean_data.jad_tm >= start_datetime) & (mean_data.jad_tm <= end_datetime))
                ax0.plot(mean_data.jad_tm[plot_index], mean_data.jad_mean[plot_index], label='mean_counts')
            except:
                pass
            ax0.legend()
            
            ax1 = fig.add_subplot(grid[1,0])
            cwt.cwt_plot(ax1, False, True, x_ticks_labeled=True)
            ax2 = fig.add_subplot(grid[:,1])
            cwt.psd_plot(ax2, 'hour')
            ax2.axvline(1/timedelta(hours=5).total_seconds(), color='gray', linestyle='--')
            ax2.axvline(1/timedelta(hours=10).total_seconds(), color='gray', linestyle='--')
            plt.savefig(f'{save_location}/PSD_{start_datetime.date()}_{end_datetime.date()}',
                        bbox_inches='tight', pad_inches=0.05, dpi=150)
            print(f'{start_datetime} to {end_datetime} plotted')
            plt.close(fig)
            
            cwt_data = {'Time': cwt.time_series, 'Data': cwt.power, 'Freq':cwt.freqs}
            cwt_pickle_file = f'/home/aschok/Documents/data/pickled_cwt_psd/cwt_{start_datetime.strftime("%Y-%jT%H:%M:%S")}-{end_datetime.strftime("%Y-%jT%H:%M:%S")}.pkl'
            with open(cwt_pickle_file, 'wb') as pkl:
                pickle.dump(cwt_data, pkl)
                pkl.close()
                
            psd_data = {'Freqs': cwt.freqs, 'PSD': cwt.psd}
            psd_pickle_file = f'/home/aschok/Documents/data/pickled_cwt_psd/psd_{start_datetime.strftime("%Y-%jT%H:%M:%S")}-{end_datetime.strftime("%Y-%jT%H:%M:%S")}.pkl'
            with open(psd_pickle_file, 'wb') as pkl:
                pickle.dump(psd_data, pkl)
                pkl.close()

def graph_peaks_in_lat(start_iso, end_iso, save_location, window_size,
                days_per_graph, max_frequency, min_frequency,
                major='1D', minor='12H'):

    lat_windows_df = time_in_lat_window(start_iso, end_iso, 0, -10)
    print(lat_windows_df)
    for index, row in lat_windows_df.iterrows():
        start = row['START']
        end = row['END']
        _plots(start, end, window_size, days_per_graph, major, minor,
               min_frequency, max_frequency, save_location)    

def graph_peaks_in_orb(orbits, save_loc, window_size,
                days_per_graph, max_frequency, min_frequency,
                major='1D', minor='12H'):
    orbit_df = get_orbit_data()
    for orb_num in orbits:
        start = orbit_df['Start'][orb_num]
        end = orbit_df['Start'][orb_num+1]
        save_location = f'{save_loc}/orbit_{orb_num}'
        _plots(start, end, window_size, days_per_graph, major, minor,
               min_frequency, max_frequency, save_location) 
            
def find_disturbed(start, end, window_size, major, minor, min_freq, max_freq, save):            
    
    def _disturbed_plot(start_datetime, end_datetime, window_size, major, minor, min_frequency, max_frequency,
              save):

        print('Getting Data')
        mag_class = MagData(start_datetime.isoformat(), end_datetime.isoformat())
        mag_class.downsample_data()
        mag_class.mean_field_align(window_size)
        time_series = mag_class.data_df.index
        b_perp = np.sqrt(mag_class.data_df.B_PERP2.to_numpy()**2
                        + mag_class.data_df.B_PERP1.to_numpy()**2)
        del(mag_class)
        
        cwt = CWTData(time_series, b_perp, 60, min_frequency, max_frequency)
        cwt.remove_coi()
        print('Plotting')
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(20,10))
        plot_class = PlotClass(ax0, ylabel='Field (nT)', title='Perpendicular Field')
        plot_class.plot(time_series, b_perp)
        plot_class.xaxis_datetime_tick_labels(False)

        cwt.cwt_plot(ax1, False, False)
        power = cwt.power
        freq_range = cwt.freqs
        del(cwt)

        per_freq = np.array([])
        for i in range(power.shape[1]):
            avg = integrate.trapz(power[:,i], freq_range)
            per_freq = np.append(per_freq, avg)

        above_threshhold = 0
        threshhold = np.mean(per_freq)
        for i in per_freq:
            if i >= threshhold:
                above_threshhold += 1

        disturbed = above_threshhold/len(per_freq)
        
        disturbed_plot = PlotClass(ax2, ylabel='Bandpower',
                            title=f'Frequency range bandpower {round(disturbed*100, 2)}% Disturbed')
        disturbed_plot.plot(time_series, per_freq)
        ax2.axhline([np.mean(per_freq)], linestyle='--', color='black', linewidth=0.25)
        disturbed_plot.xaxis_datetime_tick_labels(True)
        save_name = f'{save}/Disturbed_{start_datetime.date()}_{end_datetime.date()}'
        plt.savefig(save_name, bbox_inches='tight', pad_inches=0.05, dpi=150)
        print(f'Disturbed regions for {start_datetime.date()} to {end_datetime.date()} plotted')
        plt.close(fig)
        
        return
    
    lat_df = time_in_lat_window(start, end)
    for index, row in lat_df.iterrows():
        start = row['START']
        end = row['END']
        print(start.date())
        print(end.date())
        _disturbed_plot(start, end, window_size, major, minor, min_freq, max_freq, save)
        
def psd_compare(start, end, window_size, min_freq, max_freq, save):
    lat_df = time_in_lat_window(start, end)
    for index, row in lat_df.iterrows():
        start_datetime = row['START']
        end_datetime = row['END']
        mag = MagData(start_datetime, end_datetime)
        mag.downsample_data(60)
        mag.mean_field_align(window_size)
        print(mag.data_df.describe(), '\n')
        pd.options.mode.use_inf_as_na = True
        print(mag.data_df[mag.data_df.isna().any(axis=1)], '\n')
        
        b_perp = np.sqrt(mag.data_df.B_PERP1**2 + mag.data_df.B_PERP2**2).to_numpy()
        time_series = mag.data_df.index
        
        cwt = CWTData(time_series, b_perp, 60, min_freq, max_freq)
        cwt.remove_coi()

        fig, ((cax1, pax1), (cax2, pax2), (cax3, pax3)) = plt.subplots(3, 2, figsize=(18,10),
                                                                       sharex='col')
        fig.suptitle('Comparison of PSDs above, below, and without a threshhold')
        
        cwt.cwt_plot(cax1, False, False, colorbar=True)
        cwt.psd_plot(pax1, 'hour', ylabel='PSD', title='No Threshhold')
        
        
        mean= np.mean(cwt.power)
        above_peak_matrix = np.ma.masked_less_equal(cwt.power, mean).filled(fill_value=0)
        below_peak_matrix = np.ma.masked_greater_equal(cwt.power, mean).filled(fill_value=0)
        cwt.power = above_peak_matrix
        
        cwt.cwt_plot(cax2, False, False, colorbar=True)
        cwt.psd_plot(pax2, 'hour', ylabel='PSD', title='Above')
        
        cwt.power = below_peak_matrix
        cwt.cwt_plot(cax3, False, False, colorbar=True)
        cwt.psd_plot(pax3, 'hour', ylabel='PSD', title='Below ')
        
        fig.tight_layout()
        plt.savefig(f'{save}/psd_comparisons{start_datetime.date()}_{end_datetime.date()}.png',
                    bbox_inches='tight', pad_inches=0.01, dpi=150)
        plt.close(fig)
        print(f'Plotted {start_datetime.date()} to {end_datetime.date()}')

def heating_distributions(start_iso, end_iso):
    
    def _calc_func(start_datetime, end_datetime):
        print('Getting Data')
        print(start_datetime, end_datetime)
        heating = Turbulence(start_datetime.isoformat(),
                             end_datetime.isoformat(),
                             1, 60, 30, instrument=['fgm_jno', 'r1s'])
        
        time_series = heating.q_kaw.index
        for year in ['2016', '2017', '2018', '2019', '2020']:
            spice.furnsh(f'/data/juno_spacecraft/data/meta_kernels/juno_{year}.tm')
        et_range = [spice.utc2et(i) for i in time_series.strftime('%Y-%m-%dT%H:%M:%S')]

        positions, lt = spice.spkpos('JUNO', et_range, 'JUNO_JSS', 'NONE', 'JUPITER')
        x = positions.T[0]
        y = positions.T[1]
        z = positions.T[2]
        rad = [np.sqrt(np.sum(np.power(vector, 2))) for vector in positions]
        lat = np.arcsin(z / rad) * (180/np.pi)
        long = np.arctan2(y, x) *(180/np.pi) + 360
        spice.kclear()
        return heating.q_kaw, np.array(rad)/69911, long, lat
    
    
    lat_df = time_in_lat_window(start_iso, end_iso)
    calc_func = _calc_func
    mean_q = np.array([])
    local_time = np.array([])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,15))
    for index, row in lat_df.iterrows():
        start = row['START']
        end = row['END']
        q_kaw, radial_distance, longitude, latitude = calc_func(start, end)
        local_time = np.append(local_time, (np.mean(longitude) + 180)*24/360%24)
        mean_q = np.append(mean_q, np.mean(q_kaw))
        
        ax2.scatter(np.array(radial_distance), q_kaw, marker='+', color='black')
        ax2.set_xlabel('R ($R_j$)')
        ax2.set_ylabel('$q_{kaw}$')
        ax2.set_title('$q_{kaw}$ vs. Radial distance')
        ax2.set_yscale('log')
        
        ax3.scatter(np.array(latitude), q_kaw, marker='+', color='black')
        ax3.set_xlabel('Latitude (deg)')
        ax3.set_ylabel('$q_{kaw}$')
        ax3.set_title('$q_{kaw}$ vs. Latitude')
        ax3.set_yscale('log')
        
    ax1.scatter(local_time, mean_q, marker='+', color='black')
    ax1.set_title('$q_{kaw}$ vs. Local Time')
    ax1.set_ylabel('$q_{kaw}$')
    ax1.set_xlabel('Local Time')
    ax1.set_yscale('log')
    plt.savefig('/home/aschok/Documents/figures/test3.png', bbox_inches='tight')

if __name__ == '__main__':

    # start_time = '2017-04-10T00:00:00'
    # end_time = '2020-11-10T23:59:59'
    # save_loc = r'/home/aschok/Documents/figures/cwt/testing'
    # max_f = 1/timedelta(minutes=1).total_seconds()
    # min_f = 1/timedelta(hours=1).total_seconds()
    # find_disturbed(start_time, end_time, 60, '5d', '6H', min_f, max_f, save_loc)
    
    # start_time = '2016-07-31T00:00:00'
    # end_time = '2016-08-31T23:59:59'
    # save_loc = r'/home/aschok/Documents/figures/cwt/testing'
    # max_f = 1/timedelta(hours=1).total_seconds()
    # min_f = 1/timedelta(hours=10).total_seconds()
    # graph_peaks(start_time, end_time, save_loc, days_per_graph=1, window_size=60,
    #             max_frequency=max_f, min_frequency=min_f, major='5d', minor='12H')

    orbits = np.arange(12, 18)
    save_loc = r'/home/aschok/Documents/figures/cwt/cwt-3day'
    max_f = 1/timedelta(hours=1).total_seconds()
    min_f = 1/timedelta(hours=20).total_seconds()
    graph_peaks_in_orb(orbits, save_loc, days_per_graph=3, window_size=30,
                max_frequency=max_f, min_frequency=min_f, major='5d', minor='12H')

    # start_time = '2019-12-31T00:00:00'
    # end_time = '2020-11-10T00:00:00'
    # max_f = 1/timedelta(hours=1).total_seconds()
    # min_f = 1/timedelta(hours=20).total_seconds()
    # save_loc = '/home/aschok/Documents/figures/cwt/testing'
    # psd_compare(start_time, end_time, 24, min_f, max_f, save_loc)

    # start_time = '2016-07-25T00:00:00'
    # end_time = '2020-11-10T00:00:00'
    # heating_distributions(start_time, end_time)
    