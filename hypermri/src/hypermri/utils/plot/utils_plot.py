"""
Most of the utils functions are used for plotting inside jupyter-notebooks.

Available functions:

1. plot_colorbar

2. plot_image_series

3. plot_4d
"""
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_colorbar(figure, axis, data, **kwargs):
    """Appends colorbar to axis and scales it according to min and max of data.

    Requires the following imports:

        import matplotlib.cm as cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable

    Which cmap the colorbar picks is determined in the following order:
        1. In case the user specifies a cmap with 'cmap=...', that cmap is used.
        2. We try to extract the used cmap from the parent axis object.
        3. Fallback to default colormap, which is 'viridis'.
    """
    # sort out arrangement of colorbar and plot
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # normalize internal data form 0 to 1
    norm = plt.Normalize(np.nanmin(data), np.nanmax(data))

    # try to get corresponding cmap from axis in case none was specified
    cmap = kwargs.get("cmap", False)
    if not cmap:
        try:
            cmap = axis.get_children()[0].get_cmap()
        except AttributeError as e:
            print("Failed to retrieve cmap from axis.get_children()[0]")
            cmap = "viridis"

    # plot the colorbar
    figure.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="vertical",
        **kwargs,
    )

    return cax


def plot_image_series(
    arrays,
    label_list,
    nrows=1,
    cmap="plasma",
    plot_func=None,
    normalize=False,
    **subplot_kwrags,
):
    """Plots series of images into subplots, optionally into multiple rows"""

    def pad_or_truncate(some_list, target_len):
        return some_list[:target_len] + [""] * (target_len - len(some_list))

    ncols = len(arrays) // nrows

    if len(label_list) != nrows * ncols:
        label_list = pad_or_truncate(label_list, nrows * ncols)

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, **subplot_kwrags)

    if normalize:
        global_min, global_max = np.min(arrays), np.max(arrays)

    for ax, arr, label in zip(axs.flat, arrays, label_list):
        if plot_func:
            plot_func(ax, arr)
            plot_colorbar(fig, ax, arr)
        else:
            if normalize:
                ax.imshow(arr, cmap=cmap, vmin=global_min, vmax=global_max)
                plot_colorbar(fig, ax, [global_min, global_max])
            else:
                ax.imshow(arr, cmap=cmap)
                plot_colorbar(fig, ax, arr)
        ax.axis("off")
        ax.set_title(label)

    fig.tight_layout()


def plot_3d(data, **scatter_kwargs):
    """3D colorcoded matrix plot.

    https://stackoverflow.com/questions/14995610/how-to-make-a-4d-plot-with-matplotlib-using-arbitrary-data
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    mask = data < 1e5
    idx = np.arange(int(np.prod(data.shape)))
    x, y, z = np.unravel_index(idx, data.shape)
    p = ax.scatter(
        x,
        y,
        z,
        c=data.flatten(),
        s=10.0 * mask,
        edgecolor="face",
        alpha=0.5,
        marker="o",
        linewidth=0,
        **scatter_kwargs,
    )
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    cbar = fig.colorbar(p, shrink=0.5, label="Frequency [Hz]")
    cbar.solids.set(alpha=1)
    fig.tight_layout()
    return ax


def subplot_auto_size(num_subplots=0):
    """
    Automatically caclculates the number of rows and columns for a number of subplots so that the outlie is as close as
    possible to a square.
    Parameters
    ----------
    num_subplots: int
        Number of subplots

    Returns
    -------
    Number of Rows, Number of Colums

    Examples
    --------
    >>> grid_rows, grid_cols = subplot_auto_size(num_subplots = 10)
    >>> fig, axs = plt.subplots(grid_rows, grid_cols)
    """
    # Calculate subplot grid dimensions
    if num_subplots == 0:
        return 0, 0
    grid_size = int(np.ceil(np.sqrt(num_subplots)))
    grid_cols = (
        grid_size - 1 if grid_size * (grid_size - 1) >= num_subplots else grid_size
    )
    grid_rows = grid_size
    return grid_rows, grid_cols


def get_colors_from_cmap(cmap_name, N):
    """
        Generates a list of `N` colors from a given colormap.

        Parameters:
        ----------
        cmap_name : str
            The name of the colormap to sample colors from. This can be any colormap available in Matplotlib (e.g., 'viridis', 'plasma', 'inferno', 'magma', etc.).

        N : int
            The number of colors to generate from the colormap.

        Returns:
        -------
        colors : list of RGBA tuples
            A list of `N` colors sampled from the specified colormap. Each color is represented as an RGBA tuple.

        Example:
        -------
        >>> colors = get_colors_from_cmap('viridis', 5)
        >>> print(colors)
        [array([0.267004, 0.004874, 0.329415, 1.      ]),
         array([0.190631, 0.407061, 0.556089, 1.      ]),
         array([0.20803 , 0.718701, 0.472873, 1.      ]),
         array([0.993248, 0.906157, 0.143936, 1.      ]),
         array([0.993248, 0.906157, 0.143936, 1.      ])]
    # Generated using ChatGPT
        """
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, N))
    return colors