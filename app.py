import streamlit as st
import time
from itertools import cycle
from gp_animation import GaussianProcessAnimation
from GPy.kern import Matern32, Brownian, RBF, Cosine, Exponential, \
    Linear, GridRBF, MLP, PeriodicMatern32, Spline, White, StdPeriodic

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# def animate_multi_plots(data, colors=None, title=None, save_file=None, interval=100):

colors = None


@st.cache
def get_data(kernel_name, variance_value=1.0):
    n_dims = 100
    n_frames = 20
    n_traces = 3

    x = np.linspace(0, 10, n_dims)[:, np.newaxis]

    if kernel_name == "RBF":
        kernel = RBF(input_dim=1, variance=variance_value)
    elif kernel_name == "Brownian":
        kernel = Brownian(input_dim=1, variance=variance_value)
    elif kernel_name == "Matern32":
        kernel = Matern32(input_dim=1, variance=variance_value)
    elif kernel_name == "Cosine":
        kernel = Cosine(input_dim=1, variance=variance_value)
    elif kernel_name == "Exponential":
        kernel = Exponential(input_dim=1, variance=variance_value)
    elif kernel_name == "Linear":
        kernel = Linear(input_dim=1)
    elif kernel_name == "GridRBF":
        kernel = GridRBF(input_dim=1, variance=variance_value)
    elif kernel_name == "MLP":
        kernel = MLP(input_dim=1, variance=variance_value)
    elif kernel_name == "PeriodicMatern32":
        kernel = PeriodicMatern32(input_dim=1, variance=variance_value)
    elif kernel_name == "Spline":
        kernel = Spline(input_dim=1, variance=variance_value)
    elif kernel_name == "White":
        kernel = White(input_dim=1, variance=variance_value)
    else:
        raise ValueError("Unknown Kernel name")

    kernel_matrix = kernel.K(x, x)

    gaussian_process_animation = GaussianProcessAnimation(kernel_matrix, n_dims=n_dims, n_frames=n_frames)
    frames = gaussian_process_animation.get_traces(n_traces)
    data = np.stack(frames).transpose((2, 0, 1))
    return data


##################################################################



kernel_name = st.sidebar.selectbox("Kernel:", ["RBF",
                                               "Brownian",
                                               "Matern32",
                                               "Cosine",
                                               "Exponential",
                                               "Linear",
                                               "GridRBF",
                                               "MLP",
                                               "PeriodicMatern32",
                                               "Spline",
                                               "White"])

variance_value = st.sidebar.text_input("Variance:", 1.0)

if st.sidebar.button("Start"):
    data = get_data(kernel_name, variance_value=variance_value)

    traces = list(range(data.shape[1]))

    fig = plt.figure()
    ax = plt.axes(xlim=(0, np.shape(data)[0]), ylim=(np.min(data), np.max(data)))
    ax.set_title(kernel_name)

    lines = []
    for index, lay in enumerate(traces):
        if colors:
            lobj = ax.plot([], [], lw=2, color=colors[index])[0]
        else:
            lobj = ax.plot([], [], lw=2)[0]
        lines.append(lobj)

    the_plot = st.pyplot(plt)


    def init():
        for line in lines:
            line.set_data([], [])
        return lines


    def animate(i):
        x = np.array(range(1, data.shape[0] + 1))
        for lnum, line in enumerate(lines):
            line.set_data(x, data[:, traces[lnum] - 1, i])

        the_plot.pyplot(plt)


    for i in cycle(range(np.shape(data)[2])):
        animate(i)
        print(i)
        # time.sleep(0.01)
