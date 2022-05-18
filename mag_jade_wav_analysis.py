import importlib
import pickle
from datetime import datetime, timedelta
import warnings
import os
import re

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


def mag_jad_wav_plot():
    desired_orbit = int(input("Which orbit to look at?\n Orbit: "))
    orb_data = get_orbit_data()
    start_time = orb_data.loc[desired_orbit].Start
    end_time = orb_data.loc[desired_orbit].Perijove - timedelta(days=2)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 4))
    fig.suptitle(f"Orbit {desired_orbit}")

    mag = MagData(
        start_time,
        end_time,
        "/home/aschok/Documents/data/mag_pos_pickle",
        instrument=["jno", "v01"],
        datafile_type=".pkl",
    )
    time_series = mag.data_df.index
    mag.data_df = mag.data_df[mag.data_df.in_sheath == False]

    # Mag data plotted
    plot_class = PlotClass(ax1, ylabel="B (nT)")
    mag.plot(ax1, start_time, end_time, ["Br", "Btheta", "Bphi"])
    b_mag = np.sqrt(
        mag.data_df.Br**2 + mag.data_df.Btheta**2 + mag.data_df.Bphi**2
    ).to_numpy()
    ax1.plot(mag.data_df.index, b_mag, "black")
    ax1.plot(mag.data_df.index, -b_mag, "black")
    print("Mag data")

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

    # Mean jade plotted
    jade_mean_file = (
        f"/data/juno_spacecraft/data/jad/pickled_jad/jad_mean_orbit_{desired_orbit}.pkl"
    )
    print(f"Opening {jade_mean_file}")
    with open(jade_mean_file, "rb") as pickled_jad:
        jade_mean = pickle.load(pickled_jad)
    pickled_jad.close()
    jad_mean_df = pd.DataFrame(
        {"jad_mean": jade_mean.jad_mean, "z_cent": jade_mean.z_cent, "R": jade_mean.R},
        index=jade_mean.t,
    )

    plotted_mean = jad_mean_df[start_time:end_time]

    ax2.plot(plotted_mean.index, plotted_mean.jad_mean, label="Mean Jade")
    ax2.set_yscale("log")

    # Densities plotted
    heavy_file = f"/home/delamere/Juno/jad_heavies_orbit_{desired_orbit}.pkl"
    proton_file = f"/home/delamere/Juno/jad_protons_orbit_{desired_orbit}.pkl"
    with open(proton_file, "rb") as prt:
        protons = pickle.load(prt)
    
    protons_df = protons.data_df
    mask = protons_df.n_sig < 1000
    plotted_protons = protons_df[mask][start_time:end_time]
    with open(heavy_file, "rb") as hvy:
        heavies = pickle.load(hvy)
    
    heavy_df = heavies.data_df
    mask = heavy_df.n_sig < 1000
    plotted_heavy = heavy_df[mask][start_time:end_time]

    ax2.scatter(
        plotted_protons.index, plotted_protons.n, s=0.5, c="orange", label="Protons"
    )
    ax2.scatter(plotted_heavy.index, plotted_heavy.n,
                s=0.5, c="green", label="Heavies")
    ax2.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=3,
        markerscale=4,
    )

    # Z_cent plotted
    ax_sys3 = ax1.twinx()
    ax_sys3.plot(plotted_mean.index, plotted_mean.z_cent, alpha=0.75)
    ax_sys3.axhline(0, alpha=0.5, color="grey", ls="--")
    ax_sys3.set_ylabel("Z_cent (R_j)")
    plot_class.xaxis_datetime_tick_labels(True)

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

    plt.show()


def cwt():
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
        orbit = find_orb_num(start_datetime)
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
        b_perp = np.sqrt(mag.data_df.B_PERP1**2 +
                         mag.data_df.B_PERP2**2).to_numpy()
        if any(np.isnan(b_perp)):
            b_perp = np.nan_to_num(b_perp)
        cwt = CWTData(mag.data_df.index, b_perp, 60, min_f, max_f)
        cwt.psd_calc()
        psd_series = pd.DataFrame({"psd": cwt.psd, "freqs": cwt.freqs})

        # Setup figure
        grid = gs.GridSpec(2, 2, wspace=0.265)
        fig = plt.figure(figsize=(20, 10))

        # Plot Perp Mag Field
        ax0 = plt.subplot(grid[0, 0])
        plot_class = PlotClass(ax0, ylabel="Magnetic Field")
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
        cwt.psd_plot(ax2, "hour")

        periods_of_interest = np.array([10, 5, 3])
        freqs_of_interest = 1 / (periods_of_interest * 3600)
        for freq in freqs_of_interest:
            ax2.axvline(freq, c="grey", linestyle="--", alpha=0.75)

        save_location = "/home/aschok/Documents/data/cwt_windows_of_interest"

        save_name = f'mag_cwt/cwt_orbit{orbit}_{start_datetime.strftime("%Y%j")}_{end_datetime.strftime("%Y%j")}.png'
        plt.savefig(f"{save_location}/{save_name}")
        plt.close("all")

        pickle_name = f'psd_orbit{orbit}_{start_datetime.strftime("%Y%j")}_{end_datetime.strftime("%Y%j")}.pkl'
        with open(f"{save_location}/psd_pickled/{pickle_name}", "wb") as pickle_file:
            pickle.dump(psd_series, pickle_file)
        pickle_file.close()
        print(save_name)


def cwt_protons():
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

        proton_file = f"/home/delamere/Juno/jad_protons_orbit_{orbit}.pkl"
        with open(proton_file, "rb") as prt:
            protons = pickle.load(prt)
        prt.close()

        proton_df = protons.data_df[start_datetime:end_datetime]
        proton_df = proton_df.resample(
            "60s", origin="start").mean().shift(30, freq="s")
        proton_df = proton_df.rolling("30min").mean()

        cwt = CWTData(proton_df.index, proton_df.n.to_numpy(),
                      60, min_f, max_f)

        # Setup figure
        grid = gs.GridSpec(2, 2, wspace=0.265)
        fig = plt.figure(figsize=(20, 10))

        # Plot Perp Mag Field
        ax0 = plt.subplot(grid[0, 0])
        plot_class = PlotClass(ax0, ylabel="Proton Density")
        plot_class.plot(proton_df.index, proton_df.n, False)
        ax0.set_yscale("log")

        mag = MagData(
            row.Start,
            row.End,
            "/home/aschok/Documents/data/mag_pos_pickle",
            instrument=["jno", "v01"],
            datafile_type=".pkl",
        )

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
        cwt.psd_plot(ax2, "hour")
        periods_of_interest = np.array([10, 5, 3])
        freqs_of_interest = 1 / (periods_of_interest * 3600)
        for freq in freqs_of_interest:
            ax2.axvline(freq, c="grey", linestyle="--", alpha=0.75)

        save_location = "/home/aschok/Documents/data/cwt_windows_of_interest/"

        save_name = f'proton_cwt_orbit{orbit}_{start_datetime.strftime("%Y%j")}_{end_datetime.strftime("%Y%j")}.png'
        plt.savefig(f"{save_location}/{save_name}")
        plt.close("all")
        print(save_name)


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
        psd_10 = min(range(len(cwt.freqs)), key=lambda i: abs(cwt.freqs[i] - 1/36000))
        power_fit = ((cwt.freqs/27e-6)**(-5/3))*cwt.psd[psd_10]
        ax2.plot(cwt.freqs, power_fit, '--', color='C0')

        # Proton Data Plotting
        ax3 = plt.subplot(grid[2, 0])
        ax4 = plt.subplot(grid[3, 0])
        ax5 = plt.subplot(grid[2:4, 1])

        proton_file = f"/home/delamere/Juno/jad_protons_orbit_{orbit}.pkl"
        with open(proton_file, "rb") as prt:
            protons = pickle.load(prt)

        proton_df = protons.data_df[start_datetime: end_datetime]
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
                   loc = 'upper right',
                   bbox_to_anchor=(1, 1),)

        psd_10 = min(range(len(cwt.freqs)), key=lambda i: abs(cwt.freqs[i] - 1/36000))
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
        if  avg_sample_rate <= pd.Timedelta(seconds=120):
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
        lns =  jad_mag + perp_plot
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
        jade_psd = cwt.psd_plot(ax5_jade, 'hour', label='Jade Mean', color='orange')
        ax5_jade.set_yscale('log')
        ax5_jade.legend(frameon=False,
                   loc = 'upper right',
                   bbox_to_anchor=(1, 0.95),)
        psd_10 = min(range(len(cwt.freqs)), key=lambda i: abs(cwt.freqs[i] - 1/36000))
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
    psd_files = "/home/aschok/Documents/data/cwt_windows_of_interest/psd_pickled"
    max_f = 1 / timedelta(hours=1).total_seconds()
    min_f = 1 / timedelta(hours=12).total_seconds()
    orbit_re = re.compile(r"orbit\d{1,2}")
    orbit_num_re = re.compile(r"\d{1,2}")
    peaks_arr = np.array([])
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for parent, child, files in os.walk(psd_files):
        for file in files:
            orb_num = int(orbit_num_re.search(
                orbit_re.search(file).group()).group())
            if orb_num in orbits:
                pass
                file_path = f"{psd_files}/{file}"
                with open(file_path, "rb") as psd_pkl:
                    psd_data = pickle.load(psd_pkl)
                psd_pkl.close()
                psd_data = psd_data[(min_f < psd_data.freqs)
                                    & (max_f > psd_data.freqs)]

                peaks, _ = signal.find_peaks(psd_data.psd, width=[0, 15])
                widths = signal.peak_widths(psd_data.psd, peaks)
                prominence = signal.peak_prominences(psd_data.psd, peaks)[0]
                print(orb_num)
                print(f"Width = {widths[0]}")
                print(f"Prominences = {prominence}\n")
                peaks_arr = np.append(
                    peaks_arr, psd_data.freqs.to_numpy()[peaks])
                ax1.plot(psd_data.freqs, psd_data.psd, label=orb_num)
                ax1.scatter(
                    psd_data.freqs.to_numpy()[peaks], psd_data.psd.to_numpy()[
                        peaks]
                )

    ax1.set_yscale("log")
    ax1.legend()
    ax1.xaxis.set_visible(False)
    ax2_per = ax1.secondary_xaxis(
        "bottom", functions=(lambda x: 1 / (x * 3600), lambda x: 3600 / x)
    )
    ax2_per.set_xlabel("Period (hrs)")
    periods = np.arange(0, 12, 1)
    freqs = 1 / (periods * 3600)
    ax2.hist(peaks_arr, bins=12)
    ax2.xaxis.set_visible(False)
    ax2_per = ax2.secondary_xaxis(
        "bottom", functions=(lambda x: 1 / (x * 3600), lambda x: 3600 / x)
    )
    ax2_per.set_xlabel("Period (hrs)")
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

def jade_mean_mag_compare():
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
        perp_df = pd.DataFrame({'b_perp':b_perp}, index=mag.data_df.index)

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
        jade_df = jade_df.rolling('30min').mean()
        mean_df = jade_df.drop(columns=['z_cent', 'R'])

        mj_df = perp_df.merge(mean_df, how='outer', left_index=True, right_index=True)
        mj_df = mj_df.interpolate().reindex(perp_df.index)
        corr = mj_df.corr(method='kendall')
        print(corr)

        differential = np.diff(jade_df.jade_mean)
        print(differential)
        fig, (ax1,ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(jade_df.index[1:], differential)

        ax2.plot(perp_df.index, perp_df.b_perp, color='orange')
        plt.show()    

def choose_run_func():
    func_dict = {
        1: mag_jad_wav_plot,
        2: cwt,
        3: cwt_protons,
        4: protons_mag_cwt,
        5: hist_psd,
        6: peaks_hist,
        7: jade_mean_mag_compare
    }

    print("Which function to run?")
    print("1:mag_jad_wav_pots\n2:cwt\n3:cwt_protons\n4:protons_mag_cwt")
    print("5:hist_psd\n6:peaks_hist\n7:jade_mean_mag_compare")
    func = int(input("Enter number for cooresponding function: "))
    func_dict[func]()


if __name__ == "__main__":
    choose_run_func()
