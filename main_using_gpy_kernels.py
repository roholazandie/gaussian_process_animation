import numpy as np
from gp_animation import GaussianProcessAnimation
from visualization import animate_multi_plots
from GPy.kern import Matern32, Brownian, RBF, Cosine, Exponential, \
    Linear, GridRBF, MLP, PeriodicMatern32, Spline, White,\
    StdPeriodic, DomainKernel, LogisticBasisFuncKernel, Matern52, Symmetric, Prod

n_dims = 150
n_frames = 100
n_traces = 3

x = np.linspace(0, 10, n_dims)[:, np.newaxis]

#kernel = Matern32(input_dim=1, variance=2.0)
#kernel = Brownian(input_dim=1, variance=2.0)
#kernel = RBF(input_dim=1, variance=2.0)
#kernel = Cosine(input_dim=1)
#kernel = Exponential(input_dim=1, variance=1.0)
#kernel = Linear(input_dim=1)
#kernel = GridRBF(input_dim=1, variance=2)
#kernel = MLP(input_dim=1, variance=2)
#kernel = PeriodicMatern32(input_dim=1)
#kernel = Spline(input_dim=1)
#kernel = White(input_dim=1)
#kernel = StdPeriodic(input_dim=1)
#kernel = DomainKernel(input_dim=1, start=0, stop=5)

kernel1 = LogisticBasisFuncKernel(input_dim=1, centers=[4])
kernel2 = Matern52(input_dim=1)
kernel = Prod(kernels=[kernel1, kernel2])

kernel_matrix = kernel.K(x, x)

gaussian_process_animation = GaussianProcessAnimation(kernel_matrix, n_dims=n_dims, n_frames=n_frames)
frames = gaussian_process_animation.get_traces(n_traces)
frames = np.stack(frames).transpose((2, 0, 1)) #should be in the format of (length, n_traces, n_frame))
animate_multi_plots(frames, interval=10, title=kernel.name)
