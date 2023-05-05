import numpy as np
from matplotlib import pyplot as plt

## dydt = c(-y+sin(t))

c = 50

start = 0
stop = 5
n_steps = 300

t = np.linspace(start=start, stop=stop, num = n_steps)
dt = t[1] - t[0]


def explicit_solve():
    y = np.zeros(n_steps)
    dydt = np.zeros(n_steps)
    y[0] = 0
    for i in range(1, n_steps):       
        y[i] = y[i-1] + dt * dydt[i-1]
        dydt[i] = c * (-y[i] + np.sin(t[i]))

    return y, dydt


def implicit_solve():
    y = np.zeros(n_steps)
    dydt = np.zeros(n_steps)
    y[0] = 0
    for i in range(1, n_steps):       
        y[i] =  (y[i-1] + dt * c * np.sin(t[i])) / (1 + dt*c)
        dydt[i] = c * (-y[i] + np.sin(t[i]))

    return y, dydt


def gt():
    y = np.zeros(n_steps)
    dydt = np.zeros(n_steps)
    y[0] = 0
    for i in range(1, n_steps):       
        y[i] = c * np.exp(-c*t[i]) + (c ** 2) * np.sin(t[i]) - c * np.cos(t[i])
        y[i] /= (1 + c**2)
        dydt[i] = c * (-y[i] + np.sin(t[i]))
    return y, dydt



ey ,edydt = explicit_solve()
iy ,idydt = implicit_solve()
y ,dydt = implicit_solve()



fig, axes = plt.subplots(2,2)

axes[0,0].plot(t, ey, "r")
axes[0,0].plot(t, y, "g")
axes[0,0].set_title("Explicit position")

axes[0,1].plot(t, edydt, "r")
axes[0,1].plot(t, dydt, "g")
axes[0,1].set_title("Explicit derivative")

axes[1,0].plot(t, iy, "r")
axes[1,0].plot(t, y, "g")
axes[1,0].set_title("Implicit position")

axes[1,1].plot(t, idydt, "r")
axes[1,1].plot(t, dydt, "g")
axes[1,1].set_title("Implicit derivative")


plt.show()







