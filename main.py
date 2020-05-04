import numpy as np
from gp_animation import GaussianProcessAnimation
from visualization import animate_multi_plots
from kernels import compute_kernel, periodic_cov, exponentiated_quadratic, mlp_cov, relu_cov, ratquad_cov, rbf_cov, \
    brownian_cov, sinc_cov


n_dims = 150
n_frames = 100
n_traces = 3

x = np.linspace(0, 10, n_dims).reshape(-1, 1)
kernel = compute_kernel(x, x, kernel=rbf_cov)

gaussian_process_animation = GaussianProcessAnimation(kernel, n_dims=n_dims, n_frames=n_frames)
frames = gaussian_process_animation.get_traces(n_traces)
frames = np.stack(frames).transpose((2, 0, 1)) #should be in the format of (length, n_traces, n_frame))
animate_multi_plots(frames, interval=10, title="RBF")
