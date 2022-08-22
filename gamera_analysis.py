import os
import pickle
import struct
import re
from datetime import datetime, timedelta

import h5py
import matplotlib
import matplotlib.dates as mdates
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.signal as signal
from matplotlib import gridspec
from matplotlib.colors import LogNorm

from juno_classes import CWTData, PlotClass

SMALL_SIZE = 25
MEDIUM_SIZE = SMALL_SIZE
BIGGER_SIZE = SMALL_SIZE

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def gam_h5_to_df(file_path):
    h5 = h5py.File(file_path, 'r')
    h5_df = pd.DataFrame()
    for key in h5.keys():
        if key == 'time_steps':
            h5_df.index = pd.TimedeltaIndex(
                h5['time_steps'][()], 's').astype('timedelta64[s]')
        else:
            h5_df[key] = h5[key][()]
    return h5_df


def mean_field_align(data_df, window_size=24):
    mag_data = data_df[['BX', 'BY', 'BZ']]
    window_size = timedelta(minutes=window_size).total_seconds()
    dt = mag_data.index[1] - mag_data.index[0]
    values_per_window = int(np.ceil(window_size/dt))

    raw_magnitudes = np.sqrt((mag_data**2).sum(axis=1))
    avg_fields = mag_data.rolling(values_per_window,
                                  center=True,
                                  closed="both",
                                  min_periods=2).mean()

    avg_magnitudes = np.sqrt((avg_fields**2).sum(axis=1))
    mean_diff = mag_data - avg_fields
    z_hat = avg_fields.divide(avg_magnitudes, axis=0).to_numpy()
    unit_vector = mag_data.divide(raw_magnitudes, axis=0)

    cross = pd.DataFrame(
        [np.cross(z_hat[i], unit_vector.iloc[i])
            for i in range(0, len(z_hat))]
    )
    cross_magnitude = np.sqrt((cross**2).sum(axis=1))
    y_hat = cross.divide(cross_magnitude, axis=0).to_numpy()
    x_hat = np.cross(y_hat, z_hat)

    bx = np.array(
        [np.dot(mean_diff.iloc[i], x_hat[i]) for i in range(0, len(x_hat))]
    )
    by = np.array(
        [np.dot(mean_diff.iloc[i], y_hat[i]) for i in range(0, len(y_hat))]
    )
    bz = np.array(
        [np.dot(mean_diff.iloc[i], z_hat[i]) for i in range(0, len(z_hat))]
    )

    mfa_df = pd.DataFrame(
        {"B_PAR": bz, "B_PERP1": by, "B_PERP2": bx}, index=mag_data.index
    )

    avg_fields = avg_fields.rename(
        columns={"BX": "MEAN_BX", "BY": "MEAN_BY", "BZ": "MEAN_BZ"}
    )
    data_df = pd.concat([data_df, avg_fields, mfa_df], axis=1)
    return data_df


def mag_cwt_psd_plot(grid, fig, mag, time, cwt):
        # Function to be used to format the time axis to be readable in plots
    def format_timedelta(x, pos):
        hours = int(x//3600)
        minutes = int(x % 3600//60)
        seconds = int(x % 60)
        return f'{hours:02}:{minutes:02}:{seconds:02}'

    formatter = matplotlib.ticker.FuncFormatter(format_timedelta)
    ax0 = fig.add_subplot(grid[0, 0])
    ax0.plot(time, mag)
    ax0.xaxis.set_major_formatter(formatter)

    ax1 = fig.add_subplot(grid[1, 0])
    cwt.cwt_plot(ax1, False)
    ax1.set_yscale('log')
    ax1.xaxis.set_major_formatter(formatter)

    ax2 = fig.add_subplot(grid[1:])
    cwt.psd_plot(ax2, 'hour')

    periods_of_interest = np.array([10, 5, 3])
    freqs_of_interest = 1 / (periods_of_interest * 3600)
    for freq in freqs_of_interest:
        ax2.axvline(freq, c="grey", linestyle="--", alpha=0.75)

    return ax0, ax1, ax2


def gam_cwt():
    data_folder = '/data/gamera/jupiter/05-19-22'

    bx_df = gam_h5_to_df(f'{data_folder}/Gam_Bx_TS.h5')[230*3600: 300*3600]
    by_df = gam_h5_to_df(f'{data_folder}/Gam_By_TS.h5')[230*3600:300*3600]
    bz_df = gam_h5_to_df(f'{data_folder}/Gam_Bz_TS.h5')[230*3600:300*3600]

    print('What size window of data for the cwt in hours?')
    print('If left blank the whole length of data is used')
    cwt_window_size = input('Window Size (hrs): ').strip()
    for point in bx_df.columns:
        bx = bx_df[point].rename('BX')
        by = by_df[point].rename('BY')
        bz = bz_df[point].rename('BZ')
        mag_df = pd.concat([bx, by, bz], axis=1)
        mag_df = mean_field_align(mag_df, 45)

        if cwt_window_size == '':
            num_windows = 2
            cwt_window = mag_df.index[-1]
            cwt_window_name = 'full'
        else:
            cwt_window = int(cwt_window_size)*3600
            num_windows = int(np.ceil(mag_df.index[-1]/(cwt_window)))
            cwt_window_name = f'{cwt_window_size}hrs'

        # GAMERA startup ends at around 60hrs, data analysis starts at the 60hr timestep
        start_index = 230 * 3600
        for _ in range(1, num_windows):
            end_index = start_index + cwt_window

            analysis_data = mag_df[start_index: end_index]
            b_perp = np.sqrt(analysis_data.B_PERP1**2 +
                             analysis_data.B_PERP2**2).to_numpy()
            max_f = 1 / timedelta(hours=1).total_seconds()
            min_f = 1 / timedelta(hours=20).total_seconds()
            cwt = CWTData(analysis_data.index, b_perp, 700, min_f, max_f)
            cwt.remove_coi()

            """
            A Figure is created and a gridspec is created on the figure
            This gridspec is for organizing the position of multiple cwt analysis plot sets
            For instance if you want a full cwt analysis set with signal, cwt, and psd plotted for
            mag data on top and density on bottom. Gridspec would be created with (2, 1, figure=fig).
            """
            fig = plt.figure(figsize=(20, 10))
            grid = gs.GridSpec(1, 1, figure=fig)
            fig.suptitle(f'Gamera {point}')

            """
            A gridspec nested in the first is created in the 0,0 position
            This will allow for dynamically adding any other plots around the 3 cwt plots
            without having to change the plotting function to fit 
            If a densty cwt analysis were desired undeneath its data would need
            to be sent through the plot function with a grid created similarly 
            as blow but with subplot_spec=grid[1,0]
            """
            grid1 = gs.GridSpecFromSubplotSpec(2, 2, subplot_spec=grid[0, 0])
            mag_cwt_psd_plot(grid1, fig, b_perp, analysis_data.index, cwt)

            if cwt_window_name == 'full':
                save_name = f'gamera_cwt_{point}_full.png'
                psd_name = f'gamera_psd_{point}_full.pkl'
            else:
                if cwt_window < 24*3600:
                    def hours(x): return int(x//3600)
                    def minutes(x): return int(x % 3600//60)
                    def seconds(x): return int(x % 60)
                    start_name = f'{hours(start_index)}:{minutes(start_index)}:{seconds(start_index)}'
                    end_name = f'{hours(end_index)}:{minutes(end_index)}:{seconds(end_index)}'
                    save_name = f'gamera_cwt_{point}_{start_name}-{end_name}.png'
                    psd_name = f'gamera_psd_{point}_{start_name}-{end_name}.pkl'
                else:
                    def days(x): return int(x//(3600*24))
                    start_name = f'{days(start_index)}'
                    end_name = f'{days(end_index)}days'
                    save_name = f'gamera_cwt_{point}_{start_name}-{end_name}.png'
                    psd_name = f'gamera_psd_{point}_{start_name}-{end_name}.pkl'

            save_folder = f'/home/aschok/Documents/figures/gamera/{cwt_window_name}_cwts'
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            plt.savefig(f'{save_folder}/{save_name}')
            print(save_name)
            plt.close(fig)

            if not os.path.exists(f'{save_folder}/psd_files'):
                os.mkdir(f'{save_folder}/psd_files')
            psd_df = pd.DataFrame({'freqs': cwt.freqs, 'psd': cwt.psd})
            with open(f'{save_folder}/psd_files/{psd_name}', 'wb') as pkl:
                pickle.dump(psd_df, pkl)
            start_index = end_index


# def gam_psd_hist():
#     psd_directory = '/home/aschok/Documents/figures/gamera/full_cwts_200-300hrs/psd_files'
#     data_folder = '/data/gamera/jupiter/05-19-22'
#     pos_df = gam_h5_to_df(f'{data_folder}/Gam_pos.h5')
#     rho_df = pos_df.pow(2).sum(axis=0).pow(1/2)
#     points = rho_df[(0 < rho_df) & (rho_df < 60)].index.astype(str).to_numpy()
#     bins = 18
#     max_f = 1 / timedelta(hours=1).total_seconds()
#     min_f = 1 / timedelta(hours=20).total_seconds()
#     peaks_arr = np.array([])
#     point_re = re.compile('point\d{0,2}')
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     for parent, child, files in os.walk(psd_directory):
#         for file in files:
#             if file.endswith('.png'):
#                 continue
#             file_point = point_re.search(file).group()
#             point_in_list = file_point in points
#             if (point_in_list):
#                 print(file)
#                 print(rho_df[file_point])
#                 with open(f'{parent}/{file}', 'rb') as pkl:
#                     psd = pickle.load(pkl)

#                 psd = psd[(min_f < psd.freqs)
#                           & (max_f > psd.freqs)].rolling(3).mean()
#                 peaks, _ = signal.find_peaks(
#                     np.log10(psd.psd), prominence=0.01)
#                 widths = signal.peak_widths(psd.psd, peaks)
#                 prominence = signal.peak_prominences(psd.psd, peaks)[0]
#                 print(f"Width = {widths[0]}")
#                 print(f"Prominences = {prominence}\n")
#                 peaks_arr = np.append(
#                     peaks_arr, psd.freqs.to_numpy()[peaks])
#                 ax1.plot(psd.freqs, psd.psd)
#                 ax1.scatter(
#                     psd.freqs.to_numpy()[peaks], psd.psd.to_numpy()[
#                         peaks]
#                 )

#     ax1.set_yscale("log")
#     ax1.xaxis.set_visible(False)
#     with np.errstate(divide='ignore'):
#         ax2_per = ax1.secondary_xaxis(
#             "bottom", functions=(lambda x: 1 / (x * 3600), lambda x: 3600 / x)
#         )
#     ax2_per.set_xlabel("Period (hrs)")
#     ax2_per.set_xticks(np.array([10, 7, 5, 3, 2, 1]))
#     periods_of_interest = np.array([10, 5, 3])
#     freqs_of_interest = 1 / (periods_of_interest * 3600)
#     for freq in freqs_of_interest:
#         ax2.axvline(freq, c="grey", linestyle="--", alpha=0.75)

#     ax2.hist(peaks_arr, bins=bins)
#     ax2.xaxis.set_visible(False)
#     ax2_per = ax2.secondary_xaxis(
#         "bottom", functions=(lambda x: 1 / (x * 3600), lambda x: 3600 / x)
#     )
#     ax2_per.set_xlabel("Period (hrs)")
#     ax2_per.set_xticks(np.array([10, 7, 5, 3, 2, 1]))
#     periods_of_interest = np.array([10, 5, 3])
#     freqs_of_interest = 1 / (periods_of_interest * 3600)
#     for freq in freqs_of_interest:
#         ax2.axvline(freq, c="grey", linestyle="--", alpha=0.75)
#     plt.show()

#     fig, ax = plt.subplots()
#     ax2_per = ax.secondary_xaxis(
#         "bottom", functions=(lambda x: 1 / (x * 3600), lambda x: 3600 / x)
#     )
#     ax2_per.set_xlabel("Period (hrs)")
#     ax2_per.set_xticks(np.array([10, 7, 5, 3, 2, 1]))
#     ax.hist(peaks_arr, bins=bins)
#     ax.xaxis.set_visible(False)
#     ax2_per = ax2.secondary_xaxis(
#         "bottom", functions=(lambda x: 1 / (x * 3600), lambda x: 3600 / x)
#     )
#     ax2_per.set_xlabel("Period (hrs)")
#     ax2_per.set_xticks(np.array([10, 7, 5, 3, 2, 1]))
#     plt.show()


def gam_density_cwt():
    data_folder = '/data/gamera/jupiter/05-19-22'
    density_df = gam_h5_to_df(f'{data_folder}/Gam_D_TS.h5')
    density_df = density_df.rolling(6, min_periods=2).mean()
    density_df = density_df[230*3600: 300*3600]
    pos_df = gam_h5_to_df(f'{data_folder}/Gam_pos.h5')
    rho_df = pos_df.pow(2).sum(axis=0).pow(1/2)
    points = rho_df[(0 < rho_df) & (rho_df < 40)].index.astype(str).to_numpy()
    max_f = 1 / timedelta(hours=1).total_seconds()
    min_f = 1 / timedelta(hours=20).total_seconds()
    save_folder = '/home/aschok/Documents/figures/gamera/density_cwt'
    for column in points:
        data = density_df[column]
        cwt = CWTData(data.index, data, 700, min_f, max_f)

        fig = plt.figure(figsize=(20, 10))
        grid = gs.GridSpec(1, 1, figure=fig)
        fig.suptitle(f'Gamera Density {column}')

        grid1 = gs.GridSpecFromSubplotSpec(2, 2, subplot_spec=grid[0, 0])
        ax0, ax1, ax2 = mag_cwt_psd_plot(grid1, fig, data, data.index, cwt)
        ax0.set_yscale('log')

        psd_df = pd.DataFrame({'freqs': cwt.freqs, 'psd': cwt.psd})
        psd_save = f'gam_dense_psd_{column}.pkl'
        with open(f'{save_folder}/psd_files/{psd_save}', 'wb') as pkl:
            pickle.dump(psd_df, pkl)

        save_name = f'gam_dense_cwt_{column}.png'
        plt.savefig(f'{save_folder}/{save_name}')
        plt.close(fig)
        print(save_name)


def gam_density_hist():
    data_folder = '/data/gamera/jupiter/05-19-22'
    density_df = gam_h5_to_df(f'{data_folder}/Gam_D_TS.h5')
    density_df = density_df[230*3600: 300*3600]
    pos_df = gam_h5_to_df(f'{data_folder}/Gam_pos.h5')
    rho_df = pos_df.pow(2).sum(axis=0).pow(1/2)
    points = rho_df[(0 < rho_df) & (rho_df < 60)].index.astype(str).to_numpy()

    # chosen_points = np.arange(0, 44)
    # points = np.array([f'point{i}' for i in chosen_points])
    del_ts = np.array([2, 4, 6, 8, 10])
    change_df = pd.DataFrame({'del_n': [], 'del_t': []})
    tot_change = np.array([])
    for del_t in del_ts:
        for point, data in density_df.iteritems():
            if point in points:
                indx_per_delt = int(np.floor(del_t*3600/700))
                data = np.log(data)
                del_n = data.iloc[::indx_per_delt].pct_change()*100
                # del_n = del_n[(-300 < del_n) & (del_n < 300)]
                temp_df = pd.DataFrame({'del_n': del_n, 'del_t': del_t})
                change_df = pd.concat([change_df, temp_df], ignore_index=True)

        #         fig, ax1 = plt.subplots(1, 1)
        #         ax1.plot(data.index/3600, data)

        #         for i in data.index[::indx_per_delt]:
        #             ax1.axvline(i/3600, linestyle='--', color='grey', alpha=0.5)
        #         ax1.set_title(point)
        # print(change_df[change_df.del_t == del_t])
        # print(change_df[change_df.del_t == del_t].describe())
        # plt.show()
    del_n_less = (change_df.del_n < 100)
    del_n_great = (change_df.del_n > -100)
    change_df = change_df[del_n_less & del_n_great]
    fig, ax = plt.subplots(1, 1)
    hist_data = change_df.del_n
    bins_amt = 70
    print(hist_data.describe(), '\n')
    ax.hist(hist_data, bins=bins_amt)

    def gaussian(x, *p):
        A, mu, sigma = p
        return A * np.exp(-(x - mu)**2 / (sigma**2))

    hist, bin_edges = np.histogram(change_df.del_n.to_numpy(), bins=bins_amt)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    p0 = [1, 0, 1]
    coeff, var_matrix = scipy.optimize.curve_fit(
        gaussian, bin_centers, hist, p0=p0)
    print(f'A: {coeff[0]}, mu: {coeff[1]}, sigma: {coeff[2]}\n')
    normal_dist = gaussian(bin_centers, *coeff)
    ax.plot(bin_centers, normal_dist)
    peaks, _ = signal.find_peaks(normal_dist)
    ax.axvline(bin_centers[peaks], linestyle='--', color='C3', alpha=0.5)
    widths = signal.peak_widths(normal_dist, peaks, rel_height=0.5)
    print(widths[:2])

    # ax.hlines(widths[1], bin_centers[int(np.ceil(widths[2]))],
    #           bin_centers[int(np.ceil(widths[3]))], color='C2')
    ax.set_xlabel('$\Delta$ log(n) (%)')
    ax.set_ylabel('Counts')
    ax.axvline(0, color='grey', linestyle='--', alpha=0.5)
    plt.show()

def gam_psd_hist():
    gamera_psd = '/home/aschok/Documents/figures/gamera/full_cwts_200-300hrs/psd_files'
    gamera_dense_psd = '/home/aschok/Documents/figures/gamera/density_cwt/psd_files'
    max_f = 1 / timedelta(hours=1).total_seconds()
    min_f = 1 / timedelta(hours=20).total_seconds()
    bins = 24
    import scipy.signal as signal
    pos_df = gam_h5_to_df(f'/data/gamera/jupiter/05-19-22/Gam_pos.h5')
    rho_df = pos_df.pow(2).sum(axis=0).pow(1/2)
    gamera_points_inner = rho_df[(0 < rho_df) & (rho_df < 60)].index.astype(str).to_numpy()
    gamera_points_out = rho_df[(60 < rho_df) & (rho_df < 150)].index.astype(str).to_numpy()

    type_re = re.compile(r"\d\.pkl")
    point_re = re.compile('point\d{0,2}')
    gamera_psd_dict = {1: np.array([]), 2: np.array([]), 3: np.array([])}
    dense_psd_dict = {1: np.array([]), 2: np.array([]), 3: np.array([])}
                
    for point in gamera_points_inner:
        with open(f'{gamera_psd}/gamera_psd_{point}_full.pkl', 'rb') as pkl:
                    psd = pickle.load(pkl)
        psd = psd[(min_f < psd.freqs)
                          & (max_f > psd.freqs)].rolling(3).mean()
        peaks, _ = signal.find_peaks(
            np.log10(psd.psd), prominence=0.01)
        gamera_psd_dict[3] = np.append(gamera_psd_dict[3], psd.freqs.to_numpy()[peaks])
        
        with open(f'{gamera_dense_psd}/gam_dense_psd_{point}.pkl', 'rb') as pkl:
            psd = pickle.load(pkl)
        psd = psd[(min_f < psd.freqs)
                          & (max_f > psd.freqs)].rolling(3).mean()
        peaks, _ = signal.find_peaks(
            np.log10(psd.psd), prominence=0.01)
        dense_psd_dict[3] = np.append(dense_psd_dict[3], psd.freqs.to_numpy()[peaks])
                
    for point in gamera_points_out:
        with open(f'{gamera_psd}/gamera_psd_{point}_full.pkl', 'rb') as pkl:
                    psd = pickle.load(pkl)
        psd = psd[(min_f < psd.freqs)
                          & (max_f > psd.freqs)].rolling(3).mean()
        peaks, _ = signal.find_peaks(
            np.log10(psd.psd), prominence=0.01)
        gamera_psd_dict[1] = np.append(gamera_psd_dict[1], psd.freqs.to_numpy()[peaks])

        with open(f'{gamera_dense_psd}/gam_dense_psd_{point}.pkl', 'rb') as pkl:
                    psd = pickle.load(pkl)
        psd = psd[(min_f < psd.freqs)
                          & (max_f > psd.freqs)].rolling(3).mean()
        peaks, _ = signal.find_peaks(
            np.log10(psd.psd), prominence=0.01)
        dense_psd_dict[1] = np.append(dense_psd_dict[1], psd.freqs.to_numpy()[peaks])
    
    SMALL_SIZE = 14
    MEDIUM_SIZE = SMALL_SIZE
    BIGGER_SIZE = SMALL_SIZE

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6,8))
    ax1.hist(gamera_psd_dict[3],bins=bins)
    ax1.set_ylabel('Counts')
    ax1.set_title('R < 60Rj')
    ax1.xaxis.set_visible(False)
    ax1_per = ax1.secondary_xaxis(
        "bottom", functions=(lambda x: 1 / (x * 3600), lambda x: 3600 / x)
    )
    ax1_per.set_xticks(np.array([10, 7, 5, 3, 2, 1]))
    ax1_per.set_xticklabels([])
    
    ax2.hist(gamera_psd_dict[1], bins=bins)
    ax2.set_ylabel('Counts')
    ax2.set_title('R > 60Rj')
    ax2.xaxis.set_visible(False)
    ax2_per = ax2.secondary_xaxis(
        "bottom", functions=(lambda x: 1 / (x * 3600), lambda x: 3600 / x)
    )
    ax2_per.set_xticks(np.array([10, 7, 5, 3, 2, 1]))
    ax2_per.set_xlabel('Period (hrs)')

    plt.tight_layout()
    plt.show()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6,8))
    ax1.hist(dense_psd_dict[3], bins=bins)
    ax1.set_ylabel('Counts')
    ax1.set_title('R < 60Rj')
    ax1.xaxis.set_visible(False)
    ax1_per = ax1.secondary_xaxis(
        "bottom", functions=(lambda x: 1 / (x * 3600), lambda x: 3600 / x)
    )
    ax1_per.set_xticks(np.array([10, 7, 5, 3, 2, 1]))
    ax1_per.set_xticklabels([])
    
    ax2.hist(dense_psd_dict[1], bins=bins)
    ax2.set_ylabel('Counts')
    ax2.set_title('R > 60Rj')
    ax2_per = ax2.secondary_xaxis(
        "bottom", functions=(lambda x: 1 / (x * 3600), lambda x: 3600 / x)
    )
    ax2_per.set_xlabel("Period (hrs)")
    ax2_per.set_xticks(np.array([10, 7, 5, 3, 2, 1]))
    ax2.xaxis.set_visible(False)
    plt.tight_layout()
    plt.show()

def gam_testing():
    data_folder = '/data/gamera/jupiter/05-19-22'
    density_df = gam_h5_to_df(f'{data_folder}/Gam_D_TS.h5')
    bx_df = gam_h5_to_df(f'{data_folder}/Gam_Bx_TS.h5')
    by_df = gam_h5_to_df(f'{data_folder}/Gam_By_TS.h5')
    bz_df = gam_h5_to_df(f'{data_folder}/Gam_Bz_TS.h5')

    vphi_df = gam_h5_to_df(f'{data_folder}/Gam_Vphi_TS.h5')
    vtheta_df = gam_h5_to_df(f'{data_folder}/Gam_Vtheta_TS.h5')
    pos_df = gam_h5_to_df(f'{data_folder}/Gam_pos.h5')

    points = np.array([3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19])
    points = [f'point{i}' for i in points]
    # for point in bx_df.columns:
    #     if point not in points:
    #         continue
    #     bx = bx_df[point].rename('BX')
    #     by = by_df[point].rename('BY')
    #     bz = bz_df[point].rename('BZ')
    #     mag_df = pd.concat([bx, by, bz], axis=1)
    #     mag_df = mean_field_align(mag_df, 24)
    #     b_mag = np.sqrt((mag_df[['BX', 'BY', 'BZ']]**2).sum(axis=1))
    #     b_perp = np.sqrt((mag_df[['B_PERP1', 'B_PERP2']]**2).sum(axis=1))
    #     plot_density = density_df[point].rolling(15).mean()
    #     v_phi = vphi_df[point].rolling(15).mean()
    #     v_theta = vtheta_df[point].rolling(15).mean()

    #     fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
    #     fig.suptitle(point)
    #     ax1.plot(mag_df.index/3600, b_perp, color='C1')
    #     ax1.set_ylabel('$B_{\perp}$')

    #     ax2.plot(vphi_df.index/3600, v_phi, color='C2')
    #     ax2.axhline(0, linestyle='--', color='grey', alpha=0.5)
    #     ax2.set_ylabel('$V_{\phi}$')
    #     # ax2_theta = ax2.twinx()
    #     # ax2_theta.plot(vtheta_df.index/3600, v_theta)

    #     ax3.plot(plot_density.index/3600, plot_density, color='C3')
    #     ax3.set_yscale('log')
    #     ax3.set_ylabel('Density')

    #     ax4.plot(mag_df.index/3600, b_perp, color='C1')
    #     ax4_perp = ax4.twinx()
    #     ax4_perp.plot(mag_df.index/3600, v_phi, color='C2')
    #     ax4_dense = ax4.twinx()
    #     ax4_dense.plot(plot_density.index/3600, plot_density, color='C3')
    #     ax4_dense.set_yscale('log')
    #     plt.show()
    print(pos_df['point1'])
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
    for point in bx_df.columns:
        if point not in points:
            continue
        bx = bx_df[point].rename('BX')
        by = by_df[point].rename('BY')
        bz = bz_df[point].rename('BZ')
        mag_df = pd.concat([bx, by, bz], axis=1)
        mag_df = mean_field_align(mag_df, 24)
        b_mag = np.sqrt(mag_df[['BX', 'BY', 'BZ']].pow().sum(axis=1))
        b_perp = np.sqrt(mag_df[['B_PERP1', 'B_PERP2']].pow(2).sum(axis=1))
        plot_density = density_df[point].rolling(15).mean()
        v_phi = vphi_df[point].rolling(15).mean()
        rad_dist = np.sqrt((pos_df[point]**2).sum())

        fig.suptitle(point)
        ax1.plot(mag_df.index/3600, b_perp, color='C1')
        ax1.set_ylabel('$B_{\perp}$')

        ax2.plot(vphi_df.index/3600, v_phi, color='C2')
        ax2.set_ylabel('$V_{\phi}$')

        ax3.plot(plot_density.index/3600, plot_density, color='C3')
        ax3.set_yscale('log')
        ax3.set_ylabel('Density')

        ax4.plot(mag_df.index/3600, b_perp, color='C1')
        ax4_perp = ax4.twinx()
        ax4_perp.plot(mag_df.index/3600, v_phi, color='C2')
        ax4_dense = ax4.twinx()
        ax4_dense.plot(plot_density.index/3600, plot_density, color='C3')
        ax4_dense.set_yscale('log')
    plt.show()


if __name__ == '__main__':
    l = np.array([])
    functions = np.array([])
    copy_dict = dict(locals())
    func_dict = {}
    for key, value in copy_dict.items():
        if callable(value) and value.__module__ == __name__:
            if not str(key).startswith('_'):
                l = np.append(l, key)
                functions = np.append(functions, value)
    func_df = pd.DataFrame(l)
    print('Enter the number for the desired function')
    for index, row in func_df.iterrows():
        print(f'{index+1}: {row.values[0]}')
    func_num = int(input('Choose Function:'))
    functions[func_num-1]()
