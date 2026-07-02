from numpy import *
from scipy.integrate import odeint
import math
from scipy.signal import find_peaks
from matplotlib.pyplot import *
import matplotlib.pyplot as plt

def g_piecewise(x, c, gam):

    m=(2/3-gam*sqrt(c))/(1-sqrt(c))

    return np.piecewise(
        x,
        [x < -1-math.sqrt(c),
         (x >= -1-math.sqrt(c)) & (x < -1+math.sqrt(c)),
         (x >= -1+math.sqrt(c)) & (x < 1-math.sqrt(c)),
         (x >= 1-math.sqrt(c)) & (x < 1+math.sqrt(c)),
         x >= 1+math.sqrt(c)],
        [
            lambda x: -2*m*(x+1+sqrt(c))-gam*sqrt(c)-2/3,
            lambda x: gam*(x + 1)-(2/3),
            lambda x: m*x,
            lambda x: gam*(x - 1)+(2/3),
            lambda x: -2*m*(x-1-sqrt(c))+gam*sqrt(c)+2/3
        ]
    )

def FHN(y, t, a, c, eps, gam, P, f):
    g = g_piecewise(y[0], c, gam)
    return array([
        g - eps*y[0] - y[1] ,
        c*(y[0] - a - P*math.sin(2*math.pi*f*t))
    ])



y0 = [0,0]

## this block calculates solutions for many K's, it should take some time
# empty lists to append the values later
y_local_max = []
y_local_min = []

#params for graphs
P_pass = .001 #.005
t_pass = .001    #.001
transient_peaks = 5 #30

PP = arange(0.5, 1, P_pass)
t = arange(0, 300, t_pass)

gam=0
a=-1.3 #1.3
f=0.02 #0.02
c=0.1

epss = [0.3] #[0, 0.15, 0.25]

for eps in epss:

        y_local_max = []
        y_local_min = []

        for P in PP:
            pars = (a, c, eps, gam, P, f)

            y = odeint(FHN, y0, t, pars, rtol=1e-6, atol=1e-6)

            peaks_max, _ = find_peaks(y[:,0])
            peak_max_values = y[peaks_max[transient_peaks:],0]
            distinct_levels_max = np.unique(np.round(peak_max_values, decimals=4))
            y_local_max.append(distinct_levels_max)

            peaks_min, _ = find_peaks(-y[:,0])
            peak_min_values = y[peaks_min[transient_peaks:],0]
            distinct_levels_min = np.unique(np.round(peak_min_values, decimals=4))
            y_local_min.append(distinct_levels_min)

        # Create a new window for this parameter combination
        fig = plt.figure()

        for i, P in enumerate(PP):

            plt.scatter(
                [P]*len(y_local_max[i]),
                y_local_max[i],
                s=3,
                c='b',
                edgecolors='none',
                label='Local maxima' if i == 0 else ""
            )

            plt.scatter(
                [P]*len(y_local_min[i]),
                y_local_min[i],
                s=3,
                c='r',
                edgecolors='none',
                label='Local minima' if i == 0 else ""
            )

        plt.xlabel('$P$')
        plt.ylabel('x')
        plt.title(rf'$\varepsilon$ = {eps}')
        plt.legend()

plt.show()