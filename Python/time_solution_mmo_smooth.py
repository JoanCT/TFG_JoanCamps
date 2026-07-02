import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math

# ---------------------------------------------------------
# FitzHugh–Nagumo-like system
# ---------------------------------------------------------

def FHN(t, y, a, c, eps, P, f):
    
    x, z = y

    dxdt = x*(1-eps) - (1/3)*x**3 - z
    dzdt = c*(x - a - P*np.sin(2*np.pi*f*t - np.pi/2))

    return [dxdt, dzdt]
    

# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------
f = 0.002
c = 0.1
a = -0.9
P = 1
eps = 0.25

# ---------------------------------------------------------
# Initial condition
# ---------------------------------------------------------
y0 =  [-1.8, 1.14]

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
    args=(a, c, eps, P, f),
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
# Times when x(t) = Hopf
# ---------------------------------------------------------
target = -0.866

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
    rf"$\epsilon$ = {eps}, a={a}, $f$ = {f}, $P$ = {P}"
)

plt.xlabel("t")
plt.ylabel("x , y")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()