import importlib
import pickle
from datetime import datetime, timedelta
from tracemalloc import start
import warnings
import os
import re
from matplotlib import projections

import matplotlib.dates as mdates
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
import scipy.signal as signal

from juno_classes import (
    CWTData,
    JAD_MOM_Data,
    JadClass,
    JadeData,
    MagData,
    PlotClass,
    PosData,
    WavData,
)
from juno_functions import (
    _get_files,
    find_orb_num,
    get_orbit_data,
    get_sheath_intervals,
)

SMALL_SIZE = 15
MEDIUM_SIZE = 23
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def mag_jad_wav_plot():
    desired_orbit = int(input("Which orbit to look at?\n Orbit: "))
    orb_data = get_orbit_data()
    start_time = orb_data.loc[desired_orbit].Start
    end_time = orb_data.loc[desired_orbit].Perijove - timedelta(days=2)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(15, 10))
    fig.suptitle(f'Orbit {desired_orbit}')
    mag = MagData(
        start_time.strftime('%Y-%m-%dT%H:%M:%S'),
        end_time.strftime('%Y-%m-%dT%H:%M:%S'),
        "/home/aschok/Documents/data/mag_pos_pickle",
        instrument=["jno", "v01"],
        datafile_type=".pkl",
    )
    mag.downsample_data(60)
    mag.mean_field_align(24)
    time_series = mag.data_df.index
    pos = PosData(time_series)
    sheath_df = pos.in_sheath()
    mag.data_df.in_sheath = sheath_df.in_sheath
    mag.data_df = mag.data_df[mag.data_df.in_sheath == False]

    # Mag data plotted
    plot_class = PlotClass(ax1, ylabel="B (nT)")
    mag.plot(ax1, start_time, end_time, ["Br", "Btheta", "Bphi"])
    b_mag = np.sqrt(mag.data_df[['Br', 'Btheta', 'Bphi']].pow(2).sum(axis=1))
    ax1.plot(mag.data_df.index, b_mag, "black")
    ax1.plot(mag.data_df.index, -b_mag, "black", label='|B|')
    ax1.set_ylabel('B (nT)')
    ax1.legend(loc='upper left')
    print("Mag data")

    windows_file = (
        "/home/aschok/Documents/figures/cwt_windows_of_interest/windows_for_cwt_v01.txt"
    )
    def shade_window(window_start, window_end, wind_type):
        wind_type = int(wind_type)
        color_dict = {1: 'grey', 2: 'plum', 3: 'salmon'}
        ax1.axvspan(window_end, window_start,
                    facecolor=color_dict[wind_type], alpha=0.25)
        ax2.axvspan(window_end, window_start,
                    facecolor=color_dict[wind_type], alpha=0.25)

    
    windows_df = pd.read_csv(windows_file)
    for index, row in windows_df.iterrows():
        window_start = datetime.fromisoformat(row.Start)
        window_end = datetime.fromisoformat(row.End)
        orb_num = row.Orbit
        start_in_index = window_start in mag.data_df.index
        end_in_index = window_end in mag.data_df.index
        is_orb = orb_num == desired_orbit
        if start_in_index or end_in_index or is_orb:
            shade_window(window_start, window_end, row.Type)


    try:
        # Mean jade plotted
        jade_mean_file = (
            f"/data/juno_spacecraft/data/jad/pickled_jad/jad_mean_orbit_{desired_orbit}.pkl"
        )
        print(f"Opening {jade_mean_file}")
        with open(jade_mean_file, "rb") as pickled_jad:
            jade_mean = pickle.load(pickled_jad)
        pickled_jad.close()
        jad_mean_df = pd.DataFrame(
            {"jad_mean": jade_mean.jad_mean,
                "z_cent": jade_mean.z_cent, "R": jade_mean.R},
            index=jade_mean.t,
        )

        plotted_mean = jad_mean_df[start_time:end_time]
        ax2.plot(plotted_mean.index, plotted_mean.jad_mean, label="Mean Jade")
        ax2.set_yscale("log")
        ax2.set_ylabel('Jade Mean Counts')

        # # Densities plotted
        # heavy_file = f"/home/delamere/Juno/jad_heavies_orbit_{desired_orbit}.pkl"
        # proton_file = f"/home/delamere/Juno/jad_protons_orbit_{desired_orbit}.pkl"
        # with open(proton_file, "rb") as prt:
        #     protons = pickle.load(prt)

        # protons_df = protons.data_df
        # mask = protons_df.n_sig < 1000
        # plotted_protons = protons_df[mask][start_time:end_time]
        # with open(heavy_file, "rb") as hvy:
        #     heavies = pickle.load(hvy)

        # heavy_df = heavies.data_df
        # mask = heavy_df.n_sig < 10
        # plotted_heavy = heavy_df[mask][start_time:end_time]

        # ax2.scatter(
        #     plotted_protons.index, plotted_protons.n, s=0.5, c="orange", label="Protons"
        # )
        # ax2.scatter(plotted_heavy.index, plotted_heavy.n,
        #             s=0.5, c="green", label="Heavies")
        # ax2.legend(
        #     frameon=False,
        #     loc="lower center",
        #     bbox_to_anchor=(0.5, 1),
        #     ncol=3,
        #     markerscale=4,
        # )

        # Z_cent plotted
        # ax_sys3 = ax1.twinx()
        # ax_sys3.plot(plotted_mean.index, plotted_mean.z_cent, alpha=0.5)
        # ax_sys3.axhline(0, alpha=0.5, color="grey", ls="--")
        # ax_sys3.set_ylabel("$Z_{cent} (R_{j})$")
        # plot_class.xaxis_datetime_tick_labels(True)
    except:
        pass

    # Waves data is plotted
    wav_class = WavData(start_time, end_time)
    arr = wav_class.data_df.to_numpy()
    arr = arr.transpose()
    vmin = 1e-14
    vmax = 1e-10
    lev = np.linspace(np.log(vmin), np.log(vmax), 10)
    im = ax3.pcolormesh(
        wav_class.t,
        wav_class.freq[1:40],
        arr[1:40, :],
        norm=LogNorm(vmin=5e-15, vmax=1e-10),
        shading="auto",
    )
    ax3.set_yscale("log")
    ax3.set_ylabel("Freq (Hz)")
    ax3.set_xlabel("Time")
    fig.autofmt_xdate()
    
    def time2dist(arr):
        new_ticks = np.zeros(len(arr))
        for i, x in enumerate(arr):
            stamp = mdates.num2date(x)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                index = mag.data_df.index.get_loc(stamp, method="nearest")
            new_ticks[i] = round(mag.data_df.iloc[index].radial_3, 2)
        return new_ticks

    # Top x axis fo radial distance created
    ax1_dist = ax1.twiny()
    ax1_dist.set_xticks(ax1.get_xticks())
    ax1_dist.set_xticklabels(time2dist(ax1.get_xticks()))
    ax1_dist.set_xbound(ax1.get_xbound())
    ax1_dist.set_xlabel('Radial Distance ($R_j$)')
    plt.tight_layout()
    wind_array = pd.DataFrame({'start':[], 'end':[], 'orbit':[], 'type':[]})
    import matplotlib.backend_bases as mbb
    while True:
        pnts = plt.ginput(2, timeout=-1,
                                  mouse_pop=mbb.MouseButton.MIDDLE,
                                  mouse_stop=mbb.MouseButton.RIGHT)
        if len(pnts) <= 1:
            save = f'/home/aschok/Documents/figures/cwt_windows_of_interest/mag_jad_wav_orbit{int(desired_orbit)}.png'
            plt.savefig(save, bbox_inches='tight')
            print_df = wind_array.astype(str).apply(','.join, axis=1)
            for row in print_df:
                print(row)
            break
        else:
            (pt1, pt2) = pnts
            datetime_array = np.array([mdates.num2date(pt1[0]), mdates.num2date(pt2[0])])
            start_datetime = np.min(datetime_array)
            end_datetime = np.max(datetime_array)
            window_type = int(input('Window Type:'))
            temp = pd.DataFrame({'start': start_datetime.strftime('%Y-%m-%dT%H:%M:%S'),
                                 'end': end_datetime.strftime('%Y-%m-%dT%H:%M:%S'),
                                 'orbit': str(desired_orbit),
                                 'type': str(window_type)},
                                index=[0])
            wind_array = pd.concat([wind_array, temp], ignore_index=True)
            shade_window(start_datetime, end_datetime, window_type)
            plt.draw()
                      

# ------------------------------------------------------


def _mag_cwt_psd_plot(grid, mag, time, cwt):

    # Plot Perp Mag Field
    ax0 = plt.subplot(grid[0, 0])
    plot_class = PlotClass(ax0, ylabel="$B_{\perp}$ (nT)")
    plot_class.plot(time, mag, False)
    plot_class.xaxis_datetime_tick_labels(True)

    ax1 = plt.subplot(grid[1, 0])
    cwt.cwt_plot(ax1, False, True, x_ticks_labeled=True, xlabel='Time')

    ax2 = plt.subplot(grid[0:2, 1])
    cwt.psd_plot(ax2, "hour", ylabel='Total Wave Power')
    

    periods_of_interest = np.array([10, 10/2, 10/3, 10/4])
    freqs_of_interest = 1 / (periods_of_interest * 3600)
    ax2.axvline(freqs_of_interest[0], c="C1", linestyle="--", alpha=0.75, label='m = 1')
    ax2.axvline(freqs_of_interest[1], c="C2", linestyle="--", alpha=0.75, label='m = 2')
    ax2.axvline(freqs_of_interest[2], c="C3", linestyle="--", alpha=0.75, label='m = 3')
    ax2.axvline(freqs_of_interest[3], c="C4", linestyle="--", alpha=0.75, label='m = 4')
    ax2.legend()
    return ax0, ax1, ax2


def mag_cwt():
    max_f = 1 / timedelta(hours=1).total_seconds()
    min_f = 1 / timedelta(hours=20).total_seconds()
    chosen_orbit = int(input('Which orbit: '))

    save_location = f"/home/aschok/Documents/figures/cwt_windows_of_interest/mag_cwt"
    windows_file = (
        "/home/aschok/Documents/figures/cwt_windows_of_interest/windows_for_cwt_v01.txt"
    )
    windows_df = pd.read_csv(windows_file)
    for index, row in windows_df.iterrows():
        start_datetime = datetime.fromisoformat(row.Start)
        end_datetime = datetime.fromisoformat(row.End)
        orbit = row.Orbit
        if chosen_orbit is None:
            pass
        elif orbit != chosen_orbit:
            continue
        elif orbit == chosen_orbit:
            pass

        mag = MagData(
            row.Start,
            row.End,
            "/data/juno_spacecraft/data/pickled_mag_pos",
            instrument=["jno", "v01"],
            datafile_type=".pkl",
        )
        mag.downsample_data(60)
        mag.mean_field_align(24)

        total_size = mag.data_df.index[-1] - mag.data_df.index[0]
        num_3day_wind = total_size.total_seconds()/(3*24*3600)
        if num_3day_wind < 1:
            int_3day_wind = 1
        elif round(num_3day_wind % 1, 5) > 0.66:
            int_3day_wind = int(np.ceil(num_3day_wind))
        else:
            int_3day_wind = int(np.floor(num_3day_wind))

        start_dt = mag.data_df.index[0]
        for i in range(1, int_3day_wind+1):
            if i == int_3day_wind:
                end_dt = mag.data_df.index[-1]
            else:
                end_dt = start_dt + timedelta(days=3)

            analysis_data = mag.data_df[start_dt: end_dt]
            b_perp = np.sqrt(analysis_data.B_PERP1**2 +
                             analysis_data.B_PERP2**2).to_numpy()
            if any(np.isnan(b_perp)):
                b_perp = np.nan_to_num(b_perp)
            cwt = CWTData(analysis_data.index, b_perp, 60, min_f, max_f)
            cwt.remove_coi()
            cwt.psd_calc()
            psd_series = pd.DataFrame({"psd": cwt.psd, "freqs": cwt.freqs})

            # Setup figure
            fig = plt.figure(figsize=(20, 10), dpi=300)
            grid = gs.GridSpec(2, 2, figure=fig, wspace=0.27, hspace=0.2)
            ax0, ax1, ax2 = _mag_cwt_psd_plot(
                grid, b_perp, analysis_data.index, cwt)

            def _time2dist(arr):
                new_ticks = np.zeros(len(arr))
                for i, x in enumerate(arr):
                    stamp = mdates.num2date(x)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        index = mag.data_df.index.get_loc(
                            stamp, method="nearest")
                    new_ticks[i] = round(mag.data_df.iloc[index].radial_3, 2)
                return new_ticks

            # Top x axis fo radial distance created
            ax1_dist = ax0.twiny()
            ax1_dist.set_xticks(ax0.get_xticks())
            ax1_dist.set_xticklabels(_time2dist(ax0.get_xticks()))
            ax1_dist.set_xbound(ax0.get_xbound())
            ax1_dist.set_xlabel("Radial Distance (Rj)")
            ax1_dist.xaxis.labelpad = 10
            ax1_dist.tick_params(axis='x', labelsize=13)

            if not os.path.exists(f'{save_location}'):
                os.makedirs(f'{save_location}')
            
            if not np.isnan(row.Type):
                folder_name = {1: 'boundary_psd', 2: 'transition_psd', 3: 'disk_psd'}
                save_name = f'cwt_orbit{orbit}_{start_dt.strftime("%Y%j")}-{end_dt.strftime("%Y%j")}_{row.Type}.png'
                psd_save = f'{folder_name[row.Type]}/psd_orbit{orbit}_{start_dt.strftime("%Y%j")}-{end_dt.strftime("%Y%j")}_{row.Type}.pkl'
                for folder in folder_name.values():
                    if not os.path.exists(f'{save_location}/{folder}'):
                        os.makedirs(f'{save_location}/{folder}')
            else:
                save_name = f'cwt_orbit{orbit}_{start_dt.strftime("%Y%j")}-{end_dt.strftime("%Y%j")}.png'
                psd_save = f'psd_pickled/psd_orbit{orbit}_{start_dt.strftime("%Y%j")}-{end_dt.strftime("%Y%j")}.pkl'
                if not os.path.exists(f'{save_location}/psd_pickled'):
                        os.makedirs(f'{save_location}/psd_pickled')

            plt.savefig(f"{save_location}/{save_name}")
            plt.close("all")

            with open(f"{save_location}/{psd_save}", "wb") as pickle_file:
                pickle.dump(psd_series, pickle_file)
            pickle_file.close()
            print(save_name)
            start_dt = end_dt


def cwt_jad_mean():
    max_f = 1 / timedelta(hours=1).total_seconds()
    min_f = 1 / timedelta(hours=20).total_seconds()
    chosen_orbit = None
    save_location = "/home/aschok/Documents/figures/cwt_windows_of_interest/jade_mean_cwt"
    windows_file = (
        "/home/aschok/Documents/figures/cwt_windows_of_interest/windows_for_cwt_v01.txt"
    )
    windows_df = pd.read_csv(windows_file)
    for index, row in windows_df.iterrows():
        start_datetime = datetime.fromisoformat(row.Start)
        end_datetime = datetime.fromisoformat(row.End)
        orbit = find_orb_num(end_datetime)

        if chosen_orbit is None:
            pass
        elif orbit != chosen_orbit:
            continue
        elif orbit == chosen_orbit:
            pass

        # proton_file = f"/home/delamere/Juno/jad_protons_orbit_{orbit}.pkl"
        # with open(proton_file, "rb") as prt:
        #     protons = pickle.load(prt)
        # prt.close()

        # proton_df = protons.data_df[start_datetime:end_datetime]
        # proton_df = proton_df.resample(
        #     "60s", origin="start").mean().shift(30, freq="s")
        # proton_df = proton_df.rolling("30min").mean()

        # cwt = CWTData(proton_df.index, proton_df.n.to_numpy(),
        #               60, min_f, max_f)

        mag = MagData(
                    row.Start,
                    row.End,
                    "/home/aschok/Documents/data/mag_pos_pickle",
                    instrument=["jno", "v01"],
                    datafile_type=".pkl",
                )

        # Mean jade plotted
        jade_mean_file = (
            f"/data/juno_spacecraft/data/jad/pickled_jad/jad_mean_orbit_{orbit}.pkl"
        )
        try:
            with open(jade_mean_file, "rb") as pickled_jad:
                print(f"Opening {jade_mean_file}")
                jade_mean = pickle.load(pickled_jad)
            pickled_jad.close()

            jad_mean_df = pd.DataFrame(
                {"jad_mean": jade_mean.jad_mean,
                    "z_cent": jade_mean.z_cent, "R": jade_mean.R},
                index=jade_mean.t,
            )
        except:
            continue
        jad_mean_df = jad_mean_df[start_datetime:end_datetime]
        jad_mean_df = jad_mean_df.resample(
            "60s", origin="start").mean().shift(30, freq="s")
        jad_mean_df = jad_mean_df.rolling("30min").mean()
        total_size = jad_mean_df.index[-1] - jad_mean_df.index[0]
        num_3day_wind = total_size.total_seconds()/(3*24*3600)
        if num_3day_wind < 1:
            int_3day_wind = 1
        elif round(num_3day_wind % 1, 5) > 0.66:
            int_3day_wind = int(np.ceil(num_3day_wind))
        else:
            int_3day_wind = int(np.floor(num_3day_wind))
        start_dt = jad_mean_df.index[0]
        for i in range(1, int_3day_wind+1):
            if i == int_3day_wind:
                end_dt = jad_mean_df.index[-1]
            else:
                end_dt = start_dt + timedelta(days=3)
            analysis_data = jad_mean_df[start_dt: end_dt]
            cwt = CWTData(analysis_data.index, analysis_data.jad_mean, 60, min_f, max_f)
            cwt.remove_coi()
            cwt.psd_calc()
            psd_series = pd.DataFrame({"psd": cwt.psd, "freqs": cwt.freqs})

            # Setup figure
            fig = plt.figure(figsize=(20, 10))
            grid = gs.GridSpec(2, 2, figure=fig, wspace=0.265)
            ax0, ax1, ax2 = _mag_cwt_psd_plot(
                grid, analysis_data.jad_mean, analysis_data.index, cwt)

            def _time2dist(arr):
                new_ticks = np.zeros(len(arr))
                for i, x in enumerate(arr):
                    stamp = mdates.num2date(x)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        index = mag.data_df.index.get_loc(stamp, method="nearest")
                    new_ticks[i] = round(mag.data_df.iloc[index].radial_3, 2)
                return new_ticks

            # Top x axis fo radial distance created
            ax1_dist = ax0.twiny()
            ax1_dist.set_xticks(ax0.get_xticks())
            ax1_dist.set_xticklabels(_time2dist(ax0.get_xticks()))
            ax1_dist.set_xbound(ax0.get_xbound())
            ax1_dist.set_xlabel("Radial Distance (Rj)")

            if not os.path.exists(f'{save_location}'):
                os.makedirs(f'{save_location}')
            
            if not np.isnan(row.Type):
                folder_name = {1: 'boundary_psd', 2: 'transition_psd', 3: 'disk_psd'}
                save_name = f'cwt_orbit{orbit}_{start_dt.strftime("%Y%j")}-{end_dt.strftime("%Y%j")}_{row.Type}.png'
                psd_save = f'{folder_name[row.Type]}/psd_orbit{orbit}_{start_dt.strftime("%Y%j")}-{end_dt.strftime("%Y%j")}_{row.Type}.pkl'
                for folder in folder_name.values():
                    if not os.path.exists(f'{save_location}/{folder}'):
                        os.makedirs(f'{save_location}/{folder}')
            else:
                save_name = f'cwt_orbit{orbit}_{start_dt.strftime("%Y%j")}-{end_dt.strftime("%Y%j")}.png'
                psd_save = f'psd_pickled/psd_orbit{orbit}_{start_dt.strftime("%Y%j")}-{end_dt.strftime("%Y%j")}.pkl'
                if not os.path.exists(f'{save_location}/psd_pickled'):
                        os.makedirs(f'{save_location}/psd_pickled')

            plt.savefig(f"{save_location}/{save_name}")
            plt.close("all")

            with open(f"{save_location}/{psd_save}", "wb") as pickle_file:
                pickle.dump(psd_series, pickle_file)
            pickle_file.close()
            print(save_name)
            start_dt = end_dt
        # cwt = CWTData(plotted_mean.index, plotted_mean.jad_mean.to_numpy(),
        #               60, min_f, max_f)
        
        # # Setup figure
        # grid = gs.GridSpec(2, 2, wspace=0.265)
        # fig = plt.figure(figsize=(20, 10))

        # # Plot Perp Mag Field
        # ax0 = plt.subplot(grid[0, 0])
        # plot_class = PlotClass(ax0, ylabel="Jade Mean Counts")
        # plot_class.plot(plotted_mean.index, plotted_mean.jad_mean, False)
        # ax0.set_yscale("log")

        # mag = MagData(
        #     row.Start,
        #     row.End,
        #     "/home/aschok/Documents/data/mag_pos_pickle",
        #     instrument=["jno", "v01"],
        #     datafile_type=".pkl",
        # )

        # def time2dist(arr):
        #     new_ticks = np.zeros(len(arr))
        #     for i, x in enumerate(arr):
        #         stamp = mdates.num2date(x)
        #         with warnings.catch_warnings():
        #             warnings.simplefilter("ignore")
        #             index = mag.data_df.index.get_loc(stamp, method="nearest")
        #         new_ticks[i] = round(mag.data_df.iloc[index].radial_3, 2)
        #     return new_ticks

        # # Top x axis fo radial distance created
        # ax1_dist = ax0.twiny()
        # ax1_dist.set_xticks(ax0.get_xticks())
        # ax1_dist.set_xticklabels(time2dist(ax0.get_xticks()))
        # ax1_dist.set_xbound(ax0.get_xbound())
        # ax1_dist.set_xlabel("Radial Distance (Rj)")

        # ax1 = plt.subplot(grid[1, 0])
        # cwt.cwt_plot(ax1, False, True, x_ticks_labeled=True)

        # ax2 = plt.subplot(grid[0:2, 1])
        # cwt.psd_plot(ax2, "hour")
        # periods_of_interest = np.array([10, 5, 3])
        # freqs_of_interest = 1 / (periods_of_interest * 3600)
        # for freq in freqs_of_interest:
        #     ax2.axvline(freq, c="grey", linestyle="--", alpha=0.75)

        # if not os.path.exists(f'{save_location}'):
        #     os.makedirs(f'{save_location}')
        
        # if not np.isnan(row.Type):
        #     folder_name = {1: 'boundary_psd', 2: 'transition_psd', 3: 'disk_psd'}
        #     save_name = f'cwt_orbit{orbit}_{start_dt.strftime("%Y%j")}-{end_dt.strftime("%Y%j")}_{row.Type}.png'
        #     psd_save = f'{folder_name[row.Type]}/psd_orbit{orbit}_{start_dt.strftime("%Y%j")}-{end_dt.strftime("%Y%j")}_{row.Type}.pkl'
        #     for folder in folder_name.values():
        #         if not os.path.exists(f'{save_location}/{folder}'):
        #             os.makedirs(f'{save_location}/{folder}')
        # else:
        #     save_name = f'cwt_orbit{orbit}_{start_dt.strftime("%Y%j")}-{end_dt.strftime("%Y%j")}.png'
        #     psd_save = f'psd_pickled/psd_orbit{orbit}_{start_dt.strftime("%Y%j")}-{end_dt.strftime("%Y%j")}.pkl'
        #     if not os.path.exists(f'{save_location}/psd_pickled'):
        #             os.makedirs(f'{save_location}/psd_pickled')

        # plt.savefig(f"{save_location}/{save_name}")
        # plt.close("all")

        # with open(f"{save_location}/{psd_save}", "wb") as pickle_file:
        #     pickle.dump(psd_series, pickle_file)
        # pickle_file.close()
        # print(save_name)
        # start_dt = end_dt


def protons_mag_cwt():
    max_f = 1 / timedelta(hours=1).total_seconds()
    min_f = 1 / timedelta(hours=20).total_seconds()
    chosen_orbit = None
    windows_file = (
        "/home/aschok/Documents/data/cwt_windows_of_interest/windows_for_cwt_v01.txt"
    )
    windows_df = pd.read_csv(windows_file)
    for index, row in windows_df.iterrows():
        start_datetime = datetime.fromisoformat(row.Start)
        end_datetime = datetime.fromisoformat(row.End)
        orbit = find_orb_num(end_datetime)
        if chosen_orbit is None:
            pass
        elif orbit != chosen_orbit:
            continue
        elif orbit == chosen_orbit:
            pass
        mag = MagData(
            row.Start,
            row.End,
            "/data/juno_spacecraft/data/pickled_mag_pos",
            instrument=["jno", "v01"],
            datafile_type=".pkl",
        )
        mag.check_gaps()
        mag.downsample_data(60)
        mag.mean_field_align(24)
        b_perp = np.sqrt(mag.data_df.B_PERP1**2 +
                         mag.data_df.B_PERP2**2).to_numpy()
        b_perp = np.nan_to_num(b_perp)
        cwt = CWTData(mag.data_df.index, b_perp, 60, min_f, max_f)
        cwt.psd_calc()
        psd_series = pd.DataFrame({"psd": cwt.psd, "freqs": cwt.freqs})

        # Setup figure
        grid = gs.GridSpec(4, 2, wspace=0.265)
        fig = plt.figure(figsize=(20, 10))

        # Plot Perp Mag Field
        ax0 = plt.subplot(grid[0, 0])
        perp_plot = ax0.plot(mag.data_df.index, b_perp, label='$B_{\perp}$')

        def time2dist(arr):
            new_ticks = np.zeros(len(arr))
            for i, x in enumerate(arr):
                stamp = mdates.num2date(x)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    index = mag.data_df.index.get_loc(stamp, method="nearest")
                new_ticks[i] = round(mag.data_df.iloc[index].radial_3, 2)
            return new_ticks

        # Top x axis fo radial distance created
        ax0_dist = ax0.twiny()
        ax0_dist.set_xticks(ax0.get_xticks())
        ax0_dist.set_xticklabels(time2dist(ax0.get_xticks()))
        ax0_dist.set_xbound(ax0.get_xbound())
        ax0_dist.set_xlabel("Radial Ditsance (Rj)")

        ax1 = plt.subplot(grid[1, 0])
        cwt.cwt_plot(ax1, False, True, x_ticks_labeled=True)

        ax2 = plt.subplot(grid[0:2, 1])
        cwt.psd_plot(ax2, "hour")
        periods_of_interest = np.array([10, 5, 3])
        freqs_of_interest = 1 / (periods_of_interest * 3600)
        for freq in freqs_of_interest:
            ax2.axvline(freq, c="grey", linestyle="--", alpha=0.75)
        psd_10 = min(range(len(cwt.freqs)),
                     key=lambda i: abs(cwt.freqs[i] - 1/36000))
        power_fit = ((cwt.freqs/27e-6)**(-5/3))*cwt.psd[psd_10]
        ax2.plot(cwt.freqs, power_fit, '--', color='C0')

        # Proton Data Plotting
        ax3 = plt.subplot(grid[2, 0])
        ax4 = plt.subplot(grid[3, 0])
        ax5 = plt.subplot(grid[2:4, 1])

        proton_file = f"/home/delamere/Juno/jad_protons_orbit_{orbit}.pkl"
        with open(proton_file, "rb") as prt:
            protons = pickle.load(prt)

        mask = protons.data_df.n <= 100
        proton_df = protons.data_df[start_datetime: end_datetime][mask]
        proton_df = proton_df.resample("60s", origin="start").mean()
        proton_df = proton_df.shift(30, freq="s")
        proton_df = proton_df.rolling("30min").mean()
        cwt = CWTData(proton_df.index, proton_df.n, 60, min_f, max_f)

        proton_plot = ax3.plot(proton_df.index, proton_df.n, label='Protons')
        ax3.set_yscale("log")
        ax3.set_ylabel('Proton Density')
        cwt.cwt_plot(ax4, False, True, x_ticks_labeled=True)
        cwt.psd_plot(ax5, "hour", label='Protons')
        periods_of_interest = np.array([10, 5, 3])
        freqs_of_interest = 1 / (periods_of_interest * 3600)
        for freq in freqs_of_interest:
            ax5.axvline(freq, c="grey", linestyle="--", alpha=0.75)
        ax5.legend(frameon=False,
                   loc='upper right',
                   bbox_to_anchor=(1, 1),)

        psd_10 = min(range(len(cwt.freqs)),
                     key=lambda i: abs(cwt.freqs[i] - 1/36000))
        power_fit = ((cwt.freqs/27e-6)**(-5/3))*cwt.psd[psd_10]
        ax5.plot(cwt.freqs, power_fit, '--', color='C0')

        # Jade mean counts data
        jade_file = f'/data/juno_spacecraft/data/jad/pickled_jad/jad_mean_orbit_{orbit}.pkl'
        with open(jade_file, 'rb') as jmc:
            pkl = pickle.load(jmc)
            jade_df = pd.DataFrame({
                "jade_mean": pkl.jad_mean,
                "z_cent": pkl.z_cent,
                "R": pkl.R}, index=pkl.t,
            )

        jade_df = jade_df[start_datetime: end_datetime]
        jade_df = jade_df.resample('60s').mean()
        jade_df = jade_df.shift(30, freq='s')
        avg_sample_rate = jade_df.index.to_series().diff().mean()
        if avg_sample_rate <= pd.Timedelta(seconds=120):
            jade_df = jade_df.rolling('30min').mean()
            dt = 60
        else:
            window_size = int(avg_sample_rate.seconds*4)
            jade_df = jade_df.rolling(f'{window_size}s').mean()
            dt = window_size

        cwt = CWTData(jade_df.index, jade_df.jade_mean.to_numpy(),
                      dt, min_f, max_f)

        ax0_jade = ax0.twinx()
        jad_mag = ax0_jade.plot(jade_df.index, jade_df.jade_mean,
                                label='Jade Mean', color='orange',
                                alpha=0.65)
        ax0_jade.set_yscale('log')
        lns = jad_mag + perp_plot
        labels = [l.get_label() for l in lns]
        ax0.legend(lns, labels)

        ax3_jade = ax3.twinx()
        jade_plot = ax3_jade.plot(jade_df.index, jade_df.jade_mean,
                                  c='orange', label='Jade Mean')
        ax3_jade.set_yscale('log')
        ax3_jade.set_ylabel('Jade Mean Counts')
        lns = jade_plot + proton_plot
        labels = [l.get_label() for l in lns]
        ax3.legend(lns, labels)

        ax5_jade = ax5.twinx()
        jade_psd = cwt.psd_plot(
            ax5_jade, 'hour', label='Jade Mean', color='orange')
        ax5_jade.set_yscale('log')
        ax5_jade.legend(frameon=False,
                        loc='upper right',
                        bbox_to_anchor=(1, 0.95),)
        psd_10 = min(range(len(cwt.freqs)),
                     key=lambda i: abs(cwt.freqs[i] - 1/36000))
        power_fit = ((cwt.freqs/27e-6)**(-5/3))*cwt.psd[psd_10]
        ax5_jade.plot(cwt.freqs, power_fit, '--', c='orange')

        save_location = (
            "/home/aschok/Documents/data/cwt_windows_of_interest/mag_proton_cwt/tst"
        )

        save_name = f'cwt_proton_orbit{orbit}_{start_datetime.strftime("%Y%j")}_{end_datetime.strftime("%Y%j")}.png'
        grid.tight_layout(fig)
        plt.savefig(f"{save_location}/{save_name}")
        plt.close("all")
        print(save_name)


def hist_psd():
    orbits = np.arange(0, 16, 1)
    psd_files = "/home/aschok/Documents/figures/cwt_windows_of_interest/jade_mean_cwt/psd_pickled"
    max_f = 1 / timedelta(hours=1).total_seconds()
    min_f = 1 / timedelta(hours=20).total_seconds()
    bins = 24
    orbit_re = re.compile(r"orbit\d{1,2}")
    orbit_num_re = re.compile(r"\d{1,2}")
    type_re = re.compile(r"\d\.pkl")
    peaks_arr = np.array([])
    psd_array = np.array([])
    peaks_dict = {1: np.array([]), 2: np.array([]), 3: np.array([])}
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for parent, child, files in os.walk(psd_files):
        for file in files:
            if not file.endswith('.pkl'):
                continue
            orb_num = int(orbit_num_re.search(
                orbit_re.search(file).group()).group())
            wind_type = int(type_re.search(file).group().replace('.pkl',''))
            if orb_num in orbits:
                
                file_path = f"{psd_files}/{file}"
                with open(file_path, "rb") as psd_pkl:
                    psd_data = pickle.load(psd_pkl)
                psd_data = psd_data[(min_f < psd_data.freqs)
                                    & (max_f > psd_data.freqs)]

                peaks, _ = signal.find_peaks(psd_data.psd)
                widths = signal.peak_widths(psd_data.psd, peaks)
                prominence = signal.peak_prominences(psd_data.psd, peaks)[0]
                print(orb_num)
                print(f"Width = {widths[0]}")
                print(f"Prominences = {prominence}\n")
                peaks_arr = np.append(
                    peaks_arr, psd_data.freqs.to_numpy()[peaks])
                psd_array = np.append(psd_array, psd_data.psd.to_numpy()[peaks])
                peaks_dict[wind_type] = np.append(peaks_dict[wind_type], psd_data.freqs.to_numpy()[peaks])

                ax1.plot(psd_data.freqs, psd_data.psd, label=orb_num)
                ax1.scatter(
                    psd_data.freqs.to_numpy()[peaks], psd_data.psd.to_numpy()[
                        peaks]
                )

    ax1.set_yscale("log")
    ax1.xaxis.set_visible(False)
    ax2_per = ax1.secondary_xaxis(
        "bottom", functions=(lambda x: 1 / (x * 3600), lambda x: 3600 / x)
    )
    ax2_per.set_xlabel("Period (hrs)")
    ax2_per.set_xticks(np.array([10, 7, 5, 3, 2, 1]))
    ax2.hist(peaks_arr, bins=bins)
    ax2.xaxis.set_visible(False)
    ax2_per = ax2.secondary_xaxis(
        "bottom", functions=(lambda x: 1 / (x * 3600), lambda x: 3600 / x)
    )
    ax2_per.set_xlabel("Period (hrs)")
    ax2_per.set_xticks(np.array([10, 7, 5, 3, 2, 1]))
    plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(7, 12))
    ax1.hist(peaks_dict[1], bins=bins)
    ax1.set_ylabel('Counts')
    ax1.set_title('Flux pileup region')
    ax2_per = ax1.secondary_xaxis(
        "bottom", functions=(lambda x: 1 / (x * 3600), lambda x: 3600 / x)
    )
    ax2_per.set_xlabel("Period (hrs)")
    ax2_per.set_xticks(np.array([10, 7, 5, 3, 2, 1]))
    ax2_per.xaxis.set_visible(False)
    
    ax2.hist(peaks_dict[2], bins=bins)
    ax2.set_ylabel('Counts')
    ax2.set_title('Intermediate region')
    ax2_per = ax2.secondary_xaxis(
        "bottom", functions=(lambda x: 1 / (x * 3600), lambda x: 3600 / x)
    )
    ax2_per.set_xlabel("Period (hrs)")
    ax2_per.set_xticks(np.array([10, 7, 5, 3, 2, 1]))
    ax2_per.xaxis.set_visible(False)
    
    ax3.hist(peaks_dict[3], bins=bins)
    ax3.set_ylabel('Counts')
    ax3.set_title('Plasmadisc region')
    ax3.xaxis.set_visible(False)
    ax2_per = ax3.secondary_xaxis(
        "bottom", functions=(lambda x: 1 / (x * 3600), lambda x: 3600 / x)
    )
    ax2_per.set_xlabel("Period (hrs)")
    ax2_per.set_xticks(np.array([10, 7, 5, 3, 2, 1]))
    plt.show()

    fig, ax = plt.subplots()
    bins=32
    ax2_per = ax.secondary_xaxis(
        "bottom", functions=(lambda x: 1 / (x * 3600), lambda x: 3600 / x)
    )
    ax2_per.set_xlabel("Period (hrs)")
    ax2_per.set_xticks(np.array([10, 7, 5, 3, 2, 1]))
    ax.hist(peaks_arr, bins=bins)
    ax.set_ylabel('Counts')
    ax.xaxis.set_visible(False)
    ax2_per = ax2.secondary_xaxis(
        "bottom", functions=(lambda x: 1 / (x * 3600), lambda x: 3600 / x)
    )
    ax2_per.set_xlabel("Period (hrs)")
    ax2_per.set_xticks(np.array([10, 7, 5, 3, 2, 1]))
    plt.show()

def peaks_hist():
    max_f = 1 / timedelta(hours=1).total_seconds()
    min_f = 1 / timedelta(hours=20).total_seconds()
    chosen_orbit = 5

    windows_file = (
        "/home/aschok/Documents/data/cwt_windows_of_interest/windows_for_cwt_v01.txt"
    )
    windows_df = pd.read_csv(windows_file)
    for index, row in windows_df.iterrows():
        start_datetime = datetime.fromisoformat(row.Start)
        end_datetime = datetime.fromisoformat(row.End)
        orbit = find_orb_num(end_datetime)

        if chosen_orbit is None:
            pass
        elif orbit != chosen_orbit:
            continue
        elif orbit == chosen_orbit:
            pass

        mag = MagData(
            row.Start,
            row.End,
            "/data/juno_spacecraft/data/pickled_mag_pos",
            instrument=["jno", "v01"],
            datafile_type=".pkl",
        )
        mag.check_gaps()
        mag.downsample_data(60)
        mag.mean_field_align(24)
        b_perp = np.sqrt(mag.data_df.B_PERP1**2 +
                         mag.data_df.B_PERP2**2).to_numpy()
        b_perp = np.nan_to_num(b_perp)
        cwt = CWTData(mag.data_df.index, b_perp, 60, min_f, max_f)

        # Setup figure
        grid = gs.GridSpec(2, 2, wspace=0.265)
        fig = plt.figure(figsize=(20, 10))

        ax0 = plt.subplot(grid[0, 0])
        plot_class = PlotClass(ax0, ylabel="Mag")
        plot_class.plot(mag.data_df.index, b_perp, False)

        def time2dist(arr):
            new_ticks = np.zeros(len(arr))
            for i, x in enumerate(arr):
                stamp = mdates.num2date(x)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    index = mag.data_df.index.get_loc(stamp, method="nearest")
                new_ticks[i] = round(mag.data_df.iloc[index].radial_3, 2)
            return new_ticks

        # Top x axis fo radial distance created
        ax1_dist = ax0.twiny()
        ax1_dist.set_xticks(ax0.get_xticks())
        ax1_dist.set_xticklabels(time2dist(ax0.get_xticks()))
        ax1_dist.set_xbound(ax0.get_xbound())
        ax1_dist.set_xlabel("Radial Ditsance (Rj)")

        ax1 = plt.subplot(grid[1, 0])
        cwt.cwt_plot(ax1, False, True, x_ticks_labeled=True)

        ax2 = plt.subplot(grid[0:2, 1])
        cwt.peaks_hist(ax2, min_f, max_f, x_units="hour")
        cwt.psd_plot(ax2, "hour")
        periods_of_interest = np.array([10, 5, 3])
        freqs_of_interest = 1 / (periods_of_interest * 3600)
        for freq in freqs_of_interest:
            ax2.axvline(freq, c="grey", linestyle="--", alpha=0.75)

        save_location = "/home/aschok/Documents/data/cwt_windows_of_interest/"

        save_name = f'proton_cwt_orbit{orbit}_{start_datetime.strftime("%Y%j")}_{end_datetime.strftime("%Y%j")}.png'
        plt.show()


def dense_v_vphi():
    # Generates plots comparing density and v_phi
    chosen_orbit = None
    windows_file = (
        "/home/aschok/Documents/data/cwt_windows_of_interest/windows_for_cwt_v01.txt"
    )
    windows_df = pd.read_csv(windows_file)
    for index, row in windows_df.iterrows():
        start_datetime = datetime.fromisoformat(row.Start)
        end_datetime = datetime.fromisoformat(row.End)
        orbit = find_orb_num(end_datetime)
        if chosen_orbit is None:
            pass
        elif orbit != chosen_orbit:
            continue
        elif orbit == chosen_orbit:
            pass

        proton_file = f"/home/delamere/Juno/jad_protons_orbit_{orbit}.pkl"
        with open(proton_file, "rb") as prt:
            protons = pickle.load(prt)
        prt.close()

        protons.data_df['R'] = protons.R
        protons.data_df['z_cent'] = protons.z_cent
        z_cent_df = protons.data_df.z_cent
        proton_df = protons.data_df[start_datetime:end_datetime]
        mask = ((proton_df.n_sig / abs(proton_df.n)) < 10) &\
               ((proton_df.vphi_sig / abs(proton_df.vphi)) < 10)
        proton_df = proton_df[mask]
        # proton_df = proton_df.resample(
        #     "60s", origin="start").mean().shift(30, freq="s")
        proton_df = proton_df.rolling("60min").mean()

        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))
        fig.suptitle(f'Orbit {orbit}')
        dvv = ax1.scatter(proton_df.index, proton_df.vphi,
                          c=proton_df.n, norm=LogNorm())
        ax1_z = ax1.twinx()
        ax1_z.plot(z_cent_df[start_datetime:end_datetime].index,
                   z_cent_df[start_datetime:end_datetime])
        ax1.set_ylabel('$v_{\phi}$')
        cbr = plt.colorbar(dvv, ax=ax1)
        cbr.set_label('Density')
        ax1.axhline(0, linestyle='--', c='grey', alpha=0.6)

        mag = MagData(
            row.Start,
            row.End,
            "/home/aschok/Documents/data/mag_pos_pickle",
            instrument=["jno", "v01"],
            datafile_type=".pkl",
        )
        mag.downsample_data(60)
        mag.mean_field_align(24)
        b_perp = np.sqrt(mag.data_df.B_PERP1**2 +
                         mag.data_df.B_PERP2**2).to_numpy()
        b_perp = np.nan_to_num(b_perp)

        def time2dist(arr):
            new_ticks = np.zeros(len(arr))
            for i, x in enumerate(arr):
                stamp = mdates.num2date(x)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    index = mag.data_df.index.get_loc(stamp, method="nearest")
                new_ticks[i] = round(mag.data_df.iloc[index].radial_3, 2)
            return new_ticks

        # Top x axis fo radial distance created
        ax1_dist = ax1.twiny()
        ax1_dist.set_xticks(ax1.get_xticks())
        ax1_dist.set_xticklabels(time2dist(ax1.get_xticks()))
        ax1_dist.set_xbound(ax1.get_xbound())
        ax1_dist.set_xlabel("Radial Ditsance (Rj)")

        ax1_mag = ax1.twinx()
        ax1_mag.plot(mag.data_df.index, b_perp, c='orange', alpha=0.6)

        # V_hi vs. Density plotted
        ax2.scatter(proton_df.n, proton_df.vphi)
        ax2.set_ylabel('$v_{\phi}$')
        ax2.set_xscale('log')
        ax2.set_xlabel('Density')
        ax2.axhline(0, linestyle='--', c='grey', alpha=0.6)

        save_location = "/home/aschok/Documents/data/cwt_windows_of_interest/dense_vphi"
        save_name = f'dense_vphi_orbit{orbit}_{start_datetime.strftime("%Y%j")}_{end_datetime.strftime("%Y%j")}.png'
        plt.savefig(f'{save_location}/{save_name}')
        plt.close(fig)
        print(save_name)


def mag_perp_jade_plot():
    desired_orbit = int(input("Which orbit to look at?\n Orbit: "))
    orb_data = get_orbit_data()
    start_time = orb_data.loc[desired_orbit].Start
    end_time = orb_data.loc[desired_orbit].Perijove - timedelta(days=2)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(10, 4))
    # fig.suptitle(f"Orbit {desired_orbit}")

    mag = MagData(
        start_time.strftime('%Y-%m-%dT%H:%M:%S'),
        end_time.strftime('%Y-%m-%dT%H:%M:%S'),
        "/home/aschok/Documents/data/mag_pos_pickle",
        instrument=["jno", "v01"],
        datafile_type=".pkl",
    )
    mag.downsample_data(60)
    mag.mean_field_align(24)
    mag.data_df = mag.data_df[mag.data_df.in_sheath == False]

    # Mag data plotted
    plot_class = PlotClass(ax1, ylabel="B (nT)")
    b_mag = np.sqrt((mag.data_df[['Br', 'Btheta', 'Bphi']]**2).sum(axis=1))
    b_perp = np.sqrt((mag.data_df[['B_PERP1', 'B_PERP2']]**2).sum(axis=1)).to_numpy()
    ax1.plot(mag.data_df.index, b_mag, color='C1')
    ax1.set_ylabel('|B|')
    ax2.plot(mag.data_df.index, b_perp, color='C2')
    ax2.set_ylabel('$B_{\perp}$')
    
    print("Mag data")    
    
    # Mean jade plotted
    jade_mean_file = (
        f"/data/juno_spacecraft/data/jad/pickled_jad/jad_mean_orbit_{desired_orbit}.pkl"
    )

    with open(jade_mean_file, "rb") as pickled_jad:
        print(f"Opening {jade_mean_file}")
        jade_mean = pickle.load(pickled_jad)
    pickled_jad.close()

    jad_mean_df = pd.DataFrame(
        {"jad_mean": jade_mean.jad_mean,
            "z_cent": jade_mean.z_cent, "R": jade_mean.R},
        index=jade_mean.t,
    )

    plotted_mean = jad_mean_df[start_time:end_time]
    plotted_mean = plotted_mean.resample(
        "60s", origin="start").mean().shift(30, freq="s")
    plotted_mean = plotted_mean.rolling("30min").mean()
    ax3.plot(plotted_mean.index, plotted_mean.jad_mean, color='C3')
    ax3.set_yscale('log')
    ax3.set_ylabel('Jade Mean')
    
    ax4.plot(mag.data_df.index, b_mag, color='C1')
    ax4.plot(mag.data_df.index, b_perp, color='C2')
    ax4_jade = ax4.twinx()
    ax4_jade.plot(plotted_mean.index, plotted_mean.jad_mean, color='C3')
    ax4_jade.set_yscale('log')
    plt.show()
    
    proton_file = f"/home/delamere/Juno/jad_protons_orbit_{desired_orbit}.pkl"
    with open(proton_file, "rb") as prt:
        protons = pickle.load(prt)
    prt.close()

    proton_df = protons.data_df[start_time:end_time]
    mask = ((proton_df.n_sig / abs(proton_df.n)) < 10) &\
           ((proton_df.vphi_sig / abs(proton_df.vphi)) < 10)
    proton_df = proton_df[mask]
    proton_df = proton_df.resample(
        "60s", origin="start").mean().shift(30, freq="s")
    proton_df = proton_df.rolling("30min").mean()    

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
    ax1.plot(mag.data_df.index, b_perp, color='C2')
    ax1.set_ylabel('$B_{\perp}$')
    ax2.plot(proton_df.index, proton_df.n, color='C3')
    ax2.set_yscale('log')
    ax2.set_ylabel('Density')
    ax3.plot(proton_df.index, proton_df.vphi, color='C4')
    ax3.set_ylabel('v_phi')
    
    ax4.plot(mag.data_df.index, b_perp, color='C2', label='B_perp')
    ax4_dense = ax4.twinx()
    ax4_dense.plot(proton_df.index, proton_df.n, color='C3', label='Protons')
    ax4_dense.set_yscale('log')
    ax4_v = ax4.twinx()
    ax4_v.plot(proton_df.index, proton_df.vphi, color='C4', label='v_phi')
    plt.show()


def windows_plot():
    orbits = np.arange(1, 16)
    fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
    for orbit in orbits:
        orbit_df = get_orbit_data()
        datetime_array = pd.date_range(orbit_df.Start[orbit],
                                        orbit_df.Start[orbit + 1],
                                        freq='30min')
        pos = PosData(datetime_array)
        pos_df = pos.JSS_position()
        rad = (pos_df[['X', 'Y', 'Z']].pow(2).sum(axis=1).pow(1/2))/7.14e4
        long = np.arctan2(pos_df.Y, pos_df.X)
        ax.plot(long, rad, color='grey', alpha=0.75, ls='--', linewidth=1.25)
        max_rad_idx = rad.index.get_loc(orbit_df.Start[orbit], method='nearest')
        max_rad = rad.iloc[max_rad_idx]
        long_max_r = long.iloc[max_rad_idx]
        ax.annotate(orbit, (long_max_r, max_rad), (long_max_r, max_rad + 5.5))
        
    windows_file = (
        "/home/aschok/Documents/figures/cwt_windows_of_interest/windows_for_cwt_v01.txt"
    )
    windows_df = pd.read_csv(windows_file)
    color_dict = {1: 'black', 2: 'orange', 3: 'brown'}
    for i, (index, row) in enumerate(windows_df.iloc[:-1].iterrows()):
        start_datetime = datetime.fromisoformat(row.Start)
        end_datetime = datetime.fromisoformat(row.End)
        orbit = row.Orbit
        if orbit in orbits:
            datetime_series = pd.date_range(start_datetime, end_datetime, freq='30min')
            pos = PosData(datetime_series)
            pos_df = pos.JSS_position()
            rad = (pos_df[['X', 'Y', 'Z']].pow(2).sum(axis=1).pow(1/2))/7.14e4
            long = np.arctan2(pos_df.Y, pos_df.X)
            ax.plot(long, rad, color=color_dict[row.Type], linewidth=2)
    local_time_ticks = np.round(((np.array([np.pi, 3*np.pi/2]) + np.pi)*12/np.pi)%24)
    local_time_ticks = [f'{i.astype(np.int64)} LT' for i in local_time_ticks]
    ax.set_xticks([np.pi, 3*np.pi/2])
    ax.set_xticklabels(local_time_ticks)
    ax.set_xlim(np.pi, 280*np.pi/180)
    import matplotlib.patches as mpatch
    black_patch = mpatch.Patch(color='black', label='Flux pileup', lw=1)
    orange_patch = mpatch.Patch(color='orange', label='Intermediate')
    brown_patch = mpatch.Patch(color='brown', label='Plasmadisc')
    ax.legend(loc='lower left', handles=[black_patch, orange_patch, brown_patch])
    plt.show()


if __name__ == "__main__":
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
    func_num = int(input('Choose Function: '))
    functions[func_num-1]()
          
    