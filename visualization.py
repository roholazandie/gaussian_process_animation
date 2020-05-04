import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_multi_plots(data, colors=None, title=None, save_file=None, interval=100):
    traces = list(range(data.shape[1]))

    fig = plt.figure()
    ax = plt.axes(xlim=(0, np.shape(data)[0]), ylim=(np.min(data), np.max(data)))
    if title:
        ax.set_title(title)

    lines = []
    for index, lay in enumerate(traces):
        if colors:
            lobj = ax.plot([], [], lw=2, color=colors[index])[0]
        else:
            lobj = ax.plot([], [], lw=2)[0]
        lines.append(lobj)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(i):
        x = np.array(range(1, data.shape[0] + 1))
        for lnum, line in enumerate(lines):
            line.set_data(x, data[:, traces[lnum] - 1, i])
        return tuple(lines)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=np.shape(data)[2], interval=interval, blit=True)

    if save_file:
        # save animation at 30 frames per second
        anim.save(save_file + '.gif', writer='imagemagick', fps=30)

    plt.show()


if __name__ == "__main__":
    npdata = np.random.randint(100, size=(10, 5, 100))  # (length, n_layers, n_frame)
    animate_multi_plots(npdata)
