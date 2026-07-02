from numpy import *
from scipy.integrate import odeint
import math
from scipy.signal import find_peaks
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from PyLTSpice import SimCommander, RawRead,AscEditor,SimRunner,LTspice
import numpy as np

output_folder = "simulation_5"

runner = SimRunner(
    output_folder=output_folder,          # ← all .raw/.log go here
    simulator=LTspice,                    # or just omit if LTspice is default
    parallel_sims=1                       # set >1 if you want parallelism later
)


# values you want to test
E_pass = .1 #.001
EE = arange(-1, -0.5, E_pass)

# empty lists to append the values later
y_local_max_i = []
y_local_max_v = []

y_local_min_i = []
y_local_min_v = []

for E in EE:
    #Write the actual path to the .asc file
    netlist = AscEditor("circuit_conf_model_definitiu_reduït_tl071_teoric_sup.asc")
    netlist.set_parameter("E", E) 
    raw_file, log_file = runner.run_now(netlist)
    raw = RawRead(raw_file)

    il1 = 100 * raw.get_trace("I(L1)").get_wave()
    v = raw.get_trace("V(n001)").get_wave()

    # 1. Discard the first 40% of the simulation to KILL ALL transients
    settle_index = int(len(il1) * 0.4)
    il1_steady = il1[settle_index:]
    v_steady = v[settle_index:]

    oscillation_threshold = 1e-4

    # ==========================================
    # PROCESS CURRENT (iL)
    # ==========================================
    if np.max(il1_steady) - np.min(il1_steady) < oscillation_threshold:
        # Flat line (DC)
        distinct_levels_max_i = np.array([np.round(np.mean(il1_steady), decimals=4)])
        distinct_levels_min_i = distinct_levels_max_i
    else:
        # Zero-Crossing Method for Max/Min
        # Shift signal so its average is 0, making it easy to find crossings
        il1_centered = il1_steady - np.mean(il1_steady)
        
        # Find indices where the signal goes from negative to positive (starts a new cycle)
        crossings = np.where(np.diff(np.sign(il1_centered)) > 0)[0]
        
        cycle_maxes = []
        cycle_mins = []
        
        # Loop through each individual cycle
        for i in range(len(crossings) - 1):
            cycle = il1_steady[crossings[i]:crossings[i+1]]
            cycle_maxes.append(np.max(cycle))
            cycle_mins.append(np.min(cycle))
            
        # Clean up microscopic differences
        distinct_levels_max_i = np.unique(np.round(cycle_maxes, decimals=3))
        distinct_levels_min_i = np.unique(np.round(cycle_mins, decimals=3))

    y_local_max_i.append(distinct_levels_max_i)
    y_local_min_i.append(distinct_levels_min_i)

    # ==========================================
    # PROCESS VOLTAGE (v)
    # ==========================================
    if np.max(v_steady) - np.min(v_steady) < oscillation_threshold:
        distinct_levels_max_v = np.array([np.round(np.mean(v_steady), decimals=4)])
        distinct_levels_min_v = distinct_levels_max_v
    else:
        v_centered = v_steady - np.mean(v_steady)
        crossings_v = np.where(np.diff(np.sign(v_centered)) > 0)[0]
        
        cycle_maxes_v = []
        cycle_mins_v = []
        
        for i in range(len(crossings_v) - 1):
            cycle_v = v_steady[crossings_v[i]:crossings_v[i+1]]
            cycle_maxes_v.append(np.max(cycle_v))
            cycle_mins_v.append(np.min(cycle_v))
            
        distinct_levels_max_v = np.unique(np.round(cycle_maxes_v, decimals=3))
        distinct_levels_min_v = np.unique(np.round(cycle_mins_v, decimals=3))

    y_local_max_v.append(distinct_levels_max_v)
    y_local_min_v.append(distinct_levels_min_v)
    

# --- FIRST GRAPH: Current i_L ---

fig1 = plt.figure()
plt.xlabel('$E$')
plt.ylabel(rf"$i'_L$")
plt.grid(True, linestyle='--', alpha=0.5)

for i, E in enumerate(EE):
    if len(y_local_max_i[i]) > 0:
        plt.plot([E]*len(y_local_max_i[i]), y_local_max_i[i], 'k.', markersize=2)
    if len(y_local_min_i[i]) > 0:
        plt.plot([E]*len(y_local_min_i[i]), y_local_min_i[i], 'k.', markersize=2)

 
# --- SECOND GRAPH: Voltage v (Separate Window) ---

fig1 = plt.figure()
plt.xlabel('$E$')
plt.ylabel(rf'$v$')
plt.grid(True, linestyle='--', alpha=0.5)

for i, E in enumerate(EE):
    if len(y_local_max_v[i]) > 0:
        plt.plot([E]*len(y_local_max_v[i]), y_local_max_v[i], 'k.', markersize=2)
    if len(y_local_min_v[i]) > 0:
        plt.plot([E]*len(y_local_min_v[i]), y_local_min_v[i], 'k.', markersize=2)


plt.show() 
