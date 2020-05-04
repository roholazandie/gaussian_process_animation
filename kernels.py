import numpy as np


def sinc_cov(x, x_prime, variance=1., w=1.):
    "Sinc covariance function."
    r = np.linalg.norm(x - x_prime, 2)
    return variance * np.sinc(np.pi * w * r)


def periodic_cov(x, x_prime, variance=1., lengthscale=1., w=1.):
    "Periodic covariance function"
    r = np.linalg.norm(x - x_prime, 2)
    return variance * np.exp(-2. / (lengthscale ** 2) * np.sin(np.pi * r * w) ** 2)


def ratquad_cov(x, x_prime, variance=1., lengthscale=1., alpha=1.):
    "Rational quadratic covariance function"
    r = np.linalg.norm(x - x_prime, 2)
    return variance * (1. + r * r / (2 * alpha * lengthscale ** 2)) ** -alpha


def exponentiated_quadratic(x, x_prime, variance=1., lengthscale=1.):
    "Exponentiated quadratic covariance function."
    r = np.linalg.norm(x - x_prime, 2)
    return variance * np.exp((-0.5 * r * r) / lengthscale ** 2)

def rbf_cov(x, x_prime, sigma_f=1.0, l=1.0):
    sqdist = np.sum((x-x_prime) ** 2)
    return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * sqdist)

def mlp_cov(x, x_prime, variance=1., w=1., b=5., alpha=0.):
    "Covariance function for a MLP based neural network."
    soft = 0.0001
    inner = np.dot(x, x_prime) * w + b
    norm = np.sqrt(np.dot(x, x) * w + alpha + soft) * np.sqrt(np.dot(x_prime, x_prime) * w + b + alpha)
    arg = np.clip(inner / norm, -1, 1)  # clip as numerically can be > 1
    theta = np.arccos(arg)
    return variance * 0.5 * (1. - theta / np.pi)


def relu_cov(x, x_prime, scale=1., w=1., b=5., alpha=0.):
    """Covariance function for a ReLU based neural network.
    :param x: first input
    :param x_prime: second input
    :param scale: overall scale of the covariance
    :param w: the overall scale of the weights on the input.
    :param b: the overall scale of the bias on the input
    :param alpha: the smoothness of the relu activation"""

    def h(costheta, inner, s, a):
        "Helper function"
        cos2th = costheta * costheta
        return (1 - (2 * s * s - 1) * cos2th) / np.sqrt(a / inner + 1 - s * s * cos2th) * s

    variance = 1

    inner = np.dot(x, x_prime) * w + b
    inner_1 = np.dot(x, x) * w + b
    inner_2 = np.dot(x_prime, x_prime) * w + b
    norm_1 = np.sqrt(inner_1 + alpha)
    norm_2 = np.sqrt(inner_2 + alpha)
    norm = norm_1 * norm_2
    s = np.sqrt(inner_1) / norm_1
    s_prime = np.sqrt(inner_2) / norm_2
    arg = np.clip(inner / norm, -1, 1)  # clip as numerically can be > 1
    arg2 = np.clip(inner / np.sqrt(inner_1 * inner_2), -1, 1)  # clip as numerically can be > 1
    theta = np.arccos(arg)
    return variance * 0.5 * (
                (1. - theta / np.pi) * inner + h(arg2, inner_2, s, alpha) / np.pi + h(arg2, inner_1, s_prime,
                                                                                      alpha) / np.pi)


def prod_cov(x, x_prime, kerns, kern_args):
    "Product covariance function."
    k = 1.
    for kern, kern_arg in zip(kerns, kern_args):
        k *= kern(x, x_prime, **kern_arg)
    return k


def brownian_cov(t, t_prime, variance=1.):
    "Brownian motion covariance function."
    if t >= 0 and t_prime >= 0:
        return variance * np.min([t, t_prime])
    else:
        raise ValueError("For Brownian motion covariance only positive times are valid.")


def compute_kernel(X, X2=None, kernel=None, **kwargs):
    """Compute the full covariance function given a kernel function for two data points."""
    if X2 is None:
        X2 = X
    K = np.zeros((X.shape[0], X2.shape[0]))
    for i in np.arange(X.shape[0]):
        for j in np.arange(X2.shape[0]):
            K[i, j] = kernel(X[i, :], X2[j, :], **kwargs)

    return K