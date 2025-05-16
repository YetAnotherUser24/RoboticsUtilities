import matplotlib
import os
import platform


from serialrobot import SerialRobot

import numpy as np
import matplotlib.pyplot as plt

# Longitudes
l1 = 1.0
l2 = 1.0
l3 = 1.0
l4 = 0.5
# Parámetros de Denavit-Hartenberg para cada articulación
# Orden de los parámetros: d, th, a, alpha. Articulaciones: 'r' (revolución), 'p' (prismática)
L = [
    [l1, np.pi / 2, l2, 0, "r"],
    [0, -np.pi / 2, l3, 0, "r"],
    [-l4, 0, 0, 0, "p"],
    [0, np.pi / 2, 0, np.pi, "r"],
]
# Creación del "robot" usando los parámetros DH
scara = SerialRobot(L, name="scara")
T = scara.fkine([0, 0, 0, 0], verbose=False)
print(np.round(T, 3))
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
alims = [[-2, 2], [-2, 2], [-0.2, 1.3]]  # Límites
scara.plot(ax, [np.radians(0), 0, 0, 0], axlimits=alims, radius=0.1, color="r")
# Cll the function with your preferred mode ('gui' or 'png')
scara.plot_robot(display_mode="file")
