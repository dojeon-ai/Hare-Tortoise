import imageio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from typing import List, Optional

matplotlib.use('Agg') # Workaround on qt 'xcb' loading problem.


def figure_to_array(fig):
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)


def visualize_plot(
    x: List, 
    y: List, 
    title: Optional[str] = None, 
    x_label: Optional[str] = None, 
    y_label: Optional[str] = None, 
    ):    
    fig, ax = plt.subplots()
    ax.plot(x,y)
    if title:
        ax.set_title(title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    plot = figure_to_array(fig)
    plt.close()
    
    return plot


def visualize_trisurf(
    x: List, 
    y: List, 
    z: List, 
    title: Optional[str] = None, 
    x_label: Optional[str] = None, 
    y_label: Optional[str] = None, 
    z_label: Optional[str] = None, 
    ):    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(x, y, z, alpha=0.8, cmap='viridis')
    
    if title:
        ax.set_title(title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if z_label:
        ax.set_zlabel(z_label)

    return ax



