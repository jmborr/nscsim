import matplotlib.pyplot as plt


def single_curve(x, y, xlabel='x', ylabel='y', xscale='linear',
                 yscale='linear', **plotkwargs):
    r"""
    matplotlib.axes.Axes.plot figure

    Parameters
    ----------
    x: numpy.ndarray
        Abcisa values
    y: numpy.ndarray
        Ordinate values
    plotkwargs: dict
        Additional arguments to matplotlib.axes.Axes.plot()
    """
    if plotkwargs is None:
        plotkwargs = dict()
    fig, ax = plt.subplots(1,1)
    ax.plot(x, y, **plotkwargs)
    ax.set_xlabel(xlabel, size=25)
    ax.set_ylabel(ylabel, size=25)
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    ax.legend()
    plt.tight_layout()
    plt.show()
