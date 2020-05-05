import numpy as np


class GaussianProcessAnimation:

    def __init__(self, kernel, n_dims=150, n_frames=100):
        '''
        :param kernel:
        :param n_dims:
        :param n_frames:
        '''
        self.kernel = kernel
        self.n_dims = n_dims
        self.n_frames = n_frames
        self.epsilon = 1.0e-8

    def get_traces(self, n_traces=1):
        '''
        :param n_traces: number of traces
        :return: return a list of traces of each animation
        '''
        L = np.linalg.cholesky(self.kernel + self.epsilon * np.eye(len(self.kernel)))

        traces = []
        for _ in range(n_traces):
            s = self._animate(self.n_dims, self.n_frames)
            frames = [np.dot(L, s[:, f]) for f in range(s.shape[1])]
            traces.append(frames)

        return traces

    def _animate(self, n_dim, n_frames):
        '''

        :param n_dim: number of dimension of the gaussian
        :param n_frames: number of frames we need
        :return:
        '''
        x = np.random.randn(n_dim, 1) # draw a sample
        r = np.linalg.norm(x) #
        x = x / r # projec onto sphere
        t = np.random.randn(n_dim, 1) # sample tangent direction
        t = t - np.dot(t.T, x) * x  # orthogonalise by Gram-Schmidt.
        t = t / np.linalg.norm(t) # standardise
        s = np.linspace(0, 2 * np.pi, n_frames) # space to span
        # s = s[0: -2]
        t = np.dot(t, s[np.newaxis]) # span linspace in direction of t
        X = r * self._exp_map(x, t) # project onto sphere, re-scale
        return X

    def _exp_map(self, mu, E):
        '''
        Computes exponential map on a sphere (by Soren Hauberg)
        :param mu:
        :param E:
        :return:
        '''
        n_dim = E.shape[0]
        theta = np.sqrt(np.sum(E ** 2, axis=0))
        map = mu * np.cos(theta) + E * np.tile(np.sin(theta) / theta, (n_dim, 1))
        if np.any(np.abs(theta) < 1e-7):
            for i in np.where(np.abs(theta) < self.epsilon)[0]:
                map[:, i] = mu.ravel()

        return map


if __name__ == "__main__":
    gp_animation = GaussianProcessAnimation(2)
    gp_animation._animate(5, 10)
