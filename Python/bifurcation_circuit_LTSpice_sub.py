from numpy import *
from scipy.integrate import odeint
import math
from scipy.signal import find_peaks
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from PyLTSpice import SimCommander, RawRead,AscEditor,SimRunner,LTspice
import numpy as np


output_folder = "simulation_4"

runner = SimRunner(
    output_folder=output_folder,          
    simulator=LTspice,                    
    parallel_sims=1                       
)

# Run ONE simulation
netlist = AscEditor("circuit_conf_model_definitiu_reduït_tl071_teoric_bif_sub.asc")
raw_file, log_file = runner.run_now(netlist)
raw = RawRead(raw_file)
t_turnaround = 0.100  # 50ms - Time E reverses direction
tol_x = 0.01          # Tolerance for E
tol_v = 0.05          # Tolerance for Voltage
tol_y = 0.05          # Tolerance for Current
rounding_decimals = 4


t = raw.get_trace("time").get_wave()
v = raw.get_trace("V(n001)").get_wave()
y = 100 * raw.get_trace("I(L1)").get_wave()
e_val = raw.get_trace("V(n006)").get_wave() 

# --- 2. DATA PROCESSING FUNCTIONS ---
def get_directional_extrema(signal, time_array, e_array, t_split,
                            distance=10,
                            amp_tol=0.02,
                            consecutive=3):

    peaks_max, _ = find_peaks(signal, distance=distance)
    peaks_min, _ = find_peaks(-signal, distance=distance)

    extrema = np.sort(np.concatenate((peaks_max, peaks_min)))

    def process(mask, equilibrium_first):

        samples = np.where(mask)[0]
        ext = extrema[mask[extrema]]

        # No oscillation
        if len(ext) < 4:
            return e_array[samples], signal[samples]

        # Peak-to-peak amplitudes
        amps = np.abs(np.diff(signal[ext]))

        osc = amps > amp_tol

        # Remove isolated detections
        for i in range(len(osc)-consecutive+1):
            if np.all(osc[i:i+consecutive]):
                osc[:i] = False
                break

        if equilibrium_first:
            # --------------------------------------------------
            # equilibrium ---> oscillation
            # --------------------------------------------------

            start = None
            for i in range(len(osc)-consecutive+1):
                if np.all(osc[i:i+consecutive]):
                    start = i
                    break

            if start is None:
                return e_array[samples], signal[samples]

            first_ext = ext[start]

            before = samples[samples < first_ext]

            x = np.concatenate([
                e_array[before],
                e_array[ext[start:]]
            ])

            y = np.concatenate([
                signal[before],
                signal[ext[start:]]
            ])

            return x, y

        else:
            # --------------------------------------------------
            # oscillation ---> equilibrium
            # --------------------------------------------------

            stop = None
            for i in range(len(osc)-consecutive, -1, -1):
                if np.all(osc[max(0,i-consecutive+1):i+1]):
                    stop = i
                    break

            if stop is None:
                return e_array[samples], signal[samples]

            last_ext = ext[min(stop+1, len(ext)-1)]

            after = samples[samples >= last_ext]

            x = np.concatenate([
                e_array[ext[:stop+2]],
                e_array[after]
            ])

            y = np.concatenate([
                signal[ext[:stop+2]],
                signal[after]
            ])

            return x, y

    forward = time_array < t_split
    backward = time_array >= t_split

    # Forward sweep:
    # equilibrium -> oscillation
    xf, yf = process(forward, equilibrium_first=True)

    # Backward sweep:
    # oscillation -> equilibrium
    xb, yb = process(backward, equilibrium_first=False)

    return xf, yf, xb, yb

def match_hysteresis_data(x_fwd, y_fwd, x_bwd, y_bwd, tx, ty):
    c_x, c_y, exc_x_fwd, exc_y_fwd = [], [], [], []
    bwd_points = list(zip(x_bwd, y_bwd))
    for x_f, y_f in zip(x_fwd, y_fwd):
        match = False
        for i, (x_b, y_b) in enumerate(bwd_points):
            if abs(x_f - x_b) <= tx and abs(y_f - y_b) <= ty:
                c_x.append(x_f); c_y.append(y_f); bwd_points.pop(i); match = True; break
        if not match: exc_x_fwd.append(x_f); exc_y_fwd.append(y_f)
    return c_x, c_y, exc_x_fwd, exc_y_fwd, [p[0] for p in bwd_points], [p[1] for p in bwd_points]

# --- 3. EXECUTE ---
v_x_f, v_y_f, v_x_b, v_y_b = get_directional_extrema(v, t, e_val, t_turnaround)
y_x_f, y_y_f, y_x_b, y_y_b = get_directional_extrema(y, t, e_val, t_turnaround)

v_data = match_hysteresis_data(v_x_f, v_y_f, v_x_b, v_y_b, tol_x, tol_v)
y_data = match_hysteresis_data(y_x_f, y_y_f, y_x_b, y_y_b, tol_x, tol_y)

# --- 4. PLOTTING ---
def plot_bifurcation(data, title, ylabel):
    c_x, c_y, f_x, f_y, b_x, b_y = data
    plt.figure()
    
    # Use len() > 0 to safely check if the array has data
    if len(c_x) > 0: 
        plt.plot(c_x, c_y, 'k.', markersize=2, label='Coinciding')
    if len(f_x) > 0: 
        plt.plot(f_x, f_y, 'r.', markersize=2, label='Exclusive: Left to Right')
    if len(b_x) > 0: 
        plt.plot(b_x, b_y, 'b.', markersize=2, label='Exclusive: Right to Left')
        
    plt.xlabel('$E$')
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

# --- ENSURE PLOT FUNCTION IS ROBUST ---
def plot_bifurcation(data, title, ylabel):
    c_x, c_y, f_x, f_y, b_x, b_y = data
    plt.figure()
    if len(c_x) > 0: plt.plot(c_x, c_y, 'k.', markersize=2, label='Coinciding')
    if len(f_x) > 0: plt.plot(f_x, f_y, 'r.', markersize=2, label='Exclusive: Left to Right')
    if len(b_x) > 0: plt.plot(b_x, b_y, 'b.', markersize=2, label='Exclusive: Right to Left')
    plt.xlabel('$E$')
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

# --- CALL PLOT ---
plot_bifurcation(v_data, 'Voltage Bifurcation Map', '$v$')
plot_bifurcation(y_data, 'Current Bifurcation Map', "$i'_L$")
plt.show()