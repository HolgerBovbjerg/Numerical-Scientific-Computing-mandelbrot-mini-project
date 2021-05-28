import h5py
import matplotlib.pyplot as plt
import numpy as np


def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.
    From some guy on StackOverflow
    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for index, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (index - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[index % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())


if __name__ == "__main__":
    # Barchart
    f = h5py.File('mandelbrot_data', 'r')
    group = f['times']

    data_bar_chart = {
        'Cython_naive': [np.array(f['times']['Cython_implementation_using_naive_function'])],
        'Cython_vector': [np.array(f['times']['Cython_implementation_using_vector_function'])],
        'Distributed_vector': [np.array(f['times']['Distributed_vector_implementation'])],
        'GPU': [np.array(f['times']['GPU_implementation'])],
        'Multiprocessing_vector': [np.array(f['times']['Multiprocessing_implementation_using_vector_function'])],
        'Naive': [np.array(f['times']['Naive_implementation'])],
        'Numba': [np.array(f['times']['Numba_implementation'])],
        'Vectorized': [np.array(f['times']['Vectorized_implementation'])],
    }
    fig, ax = plt.subplots()
    bar_plot(ax, data_bar_chart, total_width=.8, single_width=.9)
    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    for i, v in enumerate(data_bar_chart.values()):
        ax.text(i * 0.1 - 0.39, v[0]+1.25, str(round(v[0], 2)), color='blue', fontweight='bold')
    ax.set_ylabel('Average execution time, seconds')
    ax.set_title('Average execution time for 4096x4096 input')
    fig.show()
    fig.savefig("bar_plot_times.pdf", dpi=200)

    # Size compare plot
    sizes = [4096, 2048, 1024, 512, 256]
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for (key, value) in group.items():
        if key == 'Naive':
            ax2.plot(sizes, np.array(value), label=key)
        elif key == 'Vectorized':
            ax2.plot(sizes, np.array(value), label=key)
            ax1.plot(sizes, np.array(value), label=key)
        else:
            ax1.plot(sizes, np.array(value), label=key)
    ax1.legend()
    ax1.set_xlabel('Size of input in points')
    ax1.set_ylabel('Average execution time, seconds')
    ax1.set_title('Average execution time vs. size of input mesh')
    fig1.show()
    fig1.savefig('line_plot_without_naive.pdf', dpi=200)
    ax2.legend()
    ax2.set_xlabel('Size of input in points')
    ax2.set_ylabel('Average execution time, seconds')
    ax2.set_title('Average execution time vs. size of input mesh')
    fig2.show()
    fig2.savefig('line_plot_with_naive.pdf', dpi=200)

    f.close()
