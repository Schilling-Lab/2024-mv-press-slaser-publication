import pathlib

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import PyPDF2  # has to be installed!
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import ScaledTranslation
from matplotlib_scalebar.scalebar import ScaleBar

################################################################################
#                                     PREAMLE                                  #
################################################################################
# # PREAMBLE FOR SEXY PLOTS
# import matplotlib as mpl
# mpl.style.use('/Users/andre/MRI/MR/auswertungen/master_thesis.rc')
#
# from hypermri.utils.plot.thesisPlottingHelp import *
################################################################################


# Master Thesis Text Field Dimensions
T_WIDTH = 5.78851  # in
T_HEIGHT = 8.1866  # in

# custom fieldmap colormaps (based on seismic, but with green around zero):
colors = [
    (0.0, "#0000C2"),  # Dark blue
    (0.4, "#B1B1FF"),  # Blue
    # (0.4, '#5555FF'),  # Red
    (0.5, "#B1FFB1"),  # White in the middle
    # (0.6, '#FF5555'),  # Red
    (0.6, "#FFB1B1"),  # Red
    (1.0, "#C20000"),  # Dark red
]
colors = [
    (0.0, "#000099"),  # Dark blue
    (0.2, "#0000CC"),  # Blue
    (0.4, "#7D7DFF"),  # Blue
    (0.5, "#97FF97"),  # Green in the middle
    (0.6, "#FF8181"),  # Red
    (0.8, "#CC0000"),  # Red
    (1.0, "#990000"),  # Dark red
]
# Create the custom colormap
custom_cmap = LinearSegmentedColormap.from_list("custom_seismic", colors)


# usage:
# fig, ax = plt.subplots()
# scalebar = ScaleBar(**SCALEBAR_ARGS)
# ax.add_artist(scalebar)
#
# if you want to overwrite something use ScaleBar(**{**SCALEBAR_ARGS, **{'pad': 0}})
SCALEBAR_ARGS = dict(
    dx=1,
    units="mm",
    location="lower left",
    color="white",
    height_fraction=0.02,
    pad=0.05,
    length_fraction=0.19,
    scale_formatter=lambda value, unit: r"\textbf{" + f"{value} {unit}" + "}",
    sep=3,
    frameon=False,
)


def add_subplotlabel(
    label, fig, ax, inside=False, color="black", fontsize=14, wrap=None, **kwargs
):
    """
    Adds a label to a Matplotlib subplot.

    This function adds a formatted label to the specified subplot, using a consistent
    positioning based on the provided `inside` flag.

    Parameters
    ----------
    label : str
        The text to display as the label.
    fig : plt.Figure
        The Matplotlib figure containing the subplot.
    ax : plt.Axes
        The subplot to add the label to.
    inside : bool, optional
        Whether to place the label inside the subplot (True) or outside (False).
        Defaults to False.
    color : str, optional
        The color of the label text. Defaults to "black".
    fontsize : float, optional
        The font size of the label text. Defaults to 14.
    wrap : Callable[[str], str], optional
        A function that wraps the label text. Defaults to None, which applies bold
        formatting using r"\textbf{...}".
    **kwargs : Any
        Additional keyword arguments to be passed to the `plt.Text` constructor.

    Returns
    -------
    plt.Text
        The created label text object.

    Raises
    -------
    ValueError
        If the `wrap` argument is not a callable function.
    """
    if wrap is None:
        wrap = lambda text: r"\textbf{" + str(text) + r"}"
    # Define the label position in inches, to achive a consistent look
    if inside:
        pad = 0.05
        x_inches = pad
        y_inches = -(
            fontsize * 1e-2 + pad
        )  # lettersize in inches + padding, but shift it downwards
    else:
        pad = 0.05
        x_inches = 0
        y_inches = pad
    # used to convert inches to figure coordinates
    trans = ScaledTranslation(x_inches, y_inches, fig.dpi_scale_trans)

    ax_text_out = ax.text(
        0,
        1,
        wrap(label),
        color=color,
        fontsize=fontsize,
        transform=ax.transAxes + trans,
    )

    return ax_text_out


def get_pdf_dimensions(pdf_file):
    with open(pdf_file, "rb") as file:
        pdf = PyPDF2.PdfReader(file)
        page = pdf.pages[0]  # Fetching the first page (index 0)
        width = page.mediabox[2] / 72  # Width in inches (72 points = 1 inch)
        height = page.mediabox[3] / 72  # Height in inches (72 points = 1 inch)
    return float(width), float(height)


def get_adjusted_figure_size(current_fig_size, actual_fig_size, goal_fig_width):
    """
    Calculates the adjusted figure size to reach a target width.

    This function calculates the new figure size needed to achieve a desired
    `goal_fig_width` while maintaining the aspect ratio of the existing figure.

    Parameters
    ----------
    current_fig_size : tuple[float, float]
        The current figure size (width, height) in inches.
    actual_fig_size : tuple[float, float]
        The actual dimensions of the saved PDF figure (width, height) in inches.
    goal_fig_width : float
        The desired final width of the figure in inches.

    Returns
    -------
    tuple[float, float]
        The adjusted figure size (width, height) in inches.
    """
    # Get current figure size
    fig_width, fig_height = current_fig_size
    # Get dimensions of the saved PDF figure
    pdf_width, pdf_height = actual_fig_size
    # Get wanted final dimensions
    goal_width = goal_fig_width

    diff_width = goal_width - pdf_width

    # update width
    fig_width_new = fig_width + diff_width
    # update height
    fig_height_new = fig_width_new * (fig_height / fig_width)  # keep aspect ratio

    return (fig_width_new, fig_height_new)


def bruteforce_correct_size(
    current_fig, fig_name, goal_width=T_WIDTH, tolerance=0.001, iterations=10
):
    """
    Adjusts the figure size of a Matplotlib figure to match a specified target width.

    This function iteratively adjusts the figure size and saves temporary PDF files
    until the actual width of the saved PDF matches the `goal_width` within the
    specified `tolerance`.

    Parameters
    ----------
    current_fig : plt.Figure
        The Matplotlib figure to adjust.
    fig_name : str
        The filename for the final PDF output.
    goal_width : float, optional
        The desired width of the final PDF in inches. Defaults to `T_WIDTH`.
    tolerance : float, optional
        The maximum allowed difference between the actual and desired width in inches.
        Defaults to 0.001.
    iterations : int, optional
        The maximum number of iterations to attempt before giving up. Defaults to 10.

    Returns
    -------
    None

    Raises
    -------
    AssertionError
        If the `fig_name` does not end with ".pdf".

    Notes
    -----
    This function uses a low DPI (80) when saving temporary PDF files to improve
    performance. The final figure is saved with a higher DPI (1000).

    Examples
    -------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> # ... create your plot ...
    >>> bruteforce_correct_size(fig, "my_figure.pdf", goal_width=5.0)
    """
    # quickly make sure the input is not super stupid:
    assert fig_name.endswith(".pdf"), "Not ending with '.pdf' -> Only works for PDF"

    success = False

    w, h = current_fig.get_size_inches()
    ratio = h / w
    print("                   WIDTH     HEIGHT")
    print(f"         Wanted:  {goal_width:.5f}   {goal_width * ratio:.5f}\n")

    to_be_deleted_fig_files = []
    with plt.ioff():
        for i in range(iterations):
            # get the current figure size
            current_fig_size = current_fig.get_size_inches()

            print(
                f"[{i}/{iterations}]  Digital:  {current_fig_size[0]:.5f}   {current_fig_size[1]:.5f}"
            )

            # save a temporary figure version to get the actual pdf size
            fname_tmp = (
                ".".join(fig_name.split(".")[:-1])
                + f"_TMP_{i}."
                + fig_name.split(".")[-1]
            )
            to_be_deleted_fig_files.append(fname_tmp)

            # NOTE: VERY LOW DPI TO SAVE TIME
            current_fig.savefig(fname_tmp, bbox_inches="tight", pad_inches=0.01, dpi=80)

            actual_fig_size = get_pdf_dimensions(fname_tmp)

            print(
                f"         Actual:  {actual_fig_size[0]:.5f}   {actual_fig_size[1]:.5f}"
            )

            if np.abs(actual_fig_size[0] - goal_width) < tolerance:
                success = True
                print("------------------------------------------")
                print("Tolerance Reached --> Stop and save final figure.")
                break

            # calculate the new proposed figsize
            new_fig_size = get_adjusted_figure_size(
                current_fig_size, actual_fig_size, goal_width
            )

            print(f"       Proposed:  {new_fig_size[0]:.5f}   {new_fig_size[1]:.5f}")
            current_fig.set_size_inches(new_fig_size)

            # Redraw the plot with the new figure size
            plt.draw()

        # save final image if successfull
        if success:
            plt.savefig(fig_name, bbox_inches="tight", pad_inches=0.01, dpi=1000)
        else:
            print("Failed to achieve desired figsize. Nothing will be saved.")

    # clean up all tmp images
    for tmp_name in to_be_deleted_fig_files:
        file_to_rem = pathlib.Path(tmp_name)
        file_to_rem.unlink()
