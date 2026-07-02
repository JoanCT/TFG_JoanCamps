from numpy import *
from scipy.integrate import odeint
import math
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from PyLTSpice import SimCommander, RawRead, SpiceEditor, SimRunner, LTspice
import numpy as np
import os

output_folder = "simulation_3"

runner = SimRunner(
    output_folder=output_folder,
    simulator=LTspice,
    parallel_sims=1
)

PP_coarse = np.arange(0.4, 0.6, 0.005)
PP=PP_coarse

#PP_fine = np.arange(2.1, 2.301, 0.001)

#PP = np.concatenate([PP_coarse, PP_fine])

transient_peaks = 5

params = [0.3] 

c = 0.1
a = -1

for eps in params:

    r_s = (1/eps) * 100

    y_local_max = []
    y_local_min = []

    # CHANGED: Swapped AscEditor(".asc") for SpiceEditor(".net")
    netlist = SpiceEditor("prova_mmo.net")

    netlist.set_parameter("R_s", r_s)
    netlist.set_parameter("a", a)

    for P in PP:

        # only update the changing parameter
        netlist.set_parameter("P_amp", P)

        # run simulation
        raw_file, log_file = runner.run_now(netlist)

        # read results
        raw = RawRead(raw_file)
        il =  raw.get_trace("V(b)").get_wave()

        # -------- peak analysis --------
        peaks_max, _ = find_peaks(il, distance=10)
        peak_max_values = il[peaks_max[transient_peaks:]]
        distinct_levels_max = np.unique(np.round(peak_max_values, decimals=7))
        y_local_max.append(distinct_levels_max)
        print(f"P = {P:.2f} | Array Length: {len(il)} | Peaks Found: {len(peaks_max)}")

        peaks_min, _ = find_peaks(-il, distance=10)
        peak_min_values = il[peaks_min[transient_peaks:]]
        distinct_levels_min = np.unique(np.round(peak_min_values, decimals=7))
        y_local_min.append(distinct_levels_min)

        # DELETE FILES AFTER USE
        for f in [raw_file, log_file]:
            if os.path.exists(f):
                os.remove(f)

    # -------- plotting --------
    fig = plt.figure()

    for i, P in enumerate(PP):

        plt.scatter(
            [P]*len(y_local_max[i]),
            y_local_max[i],
            s=3,
            c='b',
            edgecolors='none',
            label='Local max' if i == 0 else ""
        )

        plt.scatter(
            [P]*len(y_local_min[i]),
            y_local_min[i],
            s=3,
            c='r',
            edgecolors='none',
            label='Local min' if i == 0 else ""
        )

    plt.xlabel('$P$')
    plt.ylabel(rf"$v$")
    plt.title(rf'$\varepsilon$={eps}')
    plt.legend()

plt.show()