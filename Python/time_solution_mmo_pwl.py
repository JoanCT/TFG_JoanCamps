import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math

# ---------------------------------------------------------
# PWL nullcline
# ---------------------------------------------------------
def g_piecewise(x, c, gam):

    s = np.sqrt(c)
    m = (2/3 - gam*s) / (1 - s)

    return np.piecewise(
        x,
        [
            x < -1 - s,
            (x >= -1 - s) & (x < -1 + s),
            (x >= -1 + s) & (x < 1 - s),
            (x >= 1 - s) & (x < 1 + s),
            x >= 1 + s
        ],
        [
            lambda x: -2*m*(x + 1 + s) - gam*s - 2/3,
            lambda x: gam*(x + 1) - 2/3,
            lambda x: m*x,
            lambda x: gam*(x - 1) + 2/3,
            lambda x: -2*m*(x - 1 - s) + gam*s + 2/3
        ]
    )

# ---------------------------------------------------------
# FitzHugh–Nagumo-like system
# ---------------------------------------------------------
def FHN(t, y, a, c, eps, gam, P, f):

    x, z = y

    g = g_piecewise(x, c, gam)

    dxdt = g - eps*x - z
    dzdt = c*(x - a - P*np.sin(2*np.pi*f*t - np.pi/2))

    return [dxdt, dzdt]

# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------
f = 0.002 #0.002
c = 0.1   #0.1
a = -0.9 #0.7
P = 1
eps = 0.25
gam = 0

# ---------------------------------------------------------
# Initial condition
# ---------------------------------------------------------
y0 = [-1.8, 1.14]#[-1.2, -.36]

# ---------------------------------------------------------
# Time interval
# ---------------------------------------------------------
t0 = 0.0
tf = 900.0
dt = 0.01

t_eval = np.arange(t0, tf + dt, dt)

# ---------------------------------------------------------
# High-precision integration
# ---------------------------------------------------------
sol = solve_ivp(
    FHN,
    (t0, tf),
    y0,
    args=(a, c, eps, gam, P, f),
    method='DOP853',
    t_eval=t_eval,
    rtol=1e-8,
    atol=1e-8
)

# ---------------------------------------------------------
# Extract solution
# ---------------------------------------------------------
t = sol.t
x = sol.y[0]
z = sol.y[1]

forcing = a + P*np.sin(2*np.pi*f*t - np.pi/2)


# ---------------------------------------------------------
# Times when x(t) = Hopflike
# ---------------------------------------------------------
target = -1 + np.sqrt(c)

vertical_times = []

h = x - target

for i in range(len(h) - 1):

    # exact hit
    if h[i] == 0:
        vertical_times.append(t[i])

    # crossing
    elif h[i] * h[i+1] < 0:

        # linear interpolation
        alpha = -h[i] / (h[i+1] - h[i])

        tcross = t[i] + alpha * (t[i+1] - t[i])

        vertical_times.append(tcross)        
   

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))

plt.plot(t, x, color='blue', label='x')

plt.plot(t, z, color='green', label='y')

plt.plot(
    t,
    forcing,
    color='red',
    linestyle='dashed',
    label='forcing'
)

# ---------------------------------------------------------
# Vertical red lines
# ---------------------------------------------------------

for tv in vertical_times:

    plt.axvline(
        x=tv,
        color='red',
        linestyle=':',
        linewidth=1.5
    )
    

plt.title(
    rf"$\epsilon$ = {eps}, $\gamma$ = {gam}, $f$ = {f}, $P$ = {P}, $a$ = {a}"
)

plt.xlabel("t")
plt.ylabel("x , y")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()