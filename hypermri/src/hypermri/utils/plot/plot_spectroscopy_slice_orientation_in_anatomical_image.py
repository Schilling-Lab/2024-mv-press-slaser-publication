###########################################################################################################################################################

# Plot boundaries of spectroscopy slice in an anatomical image - compatible with spec and CSI slice

###########################################################################################################################################################


# =========================================================================================================================================================
# Import of necessary packages
# =========================================================================================================================================================

import os, glob

from tqdm.notebook import tqdm
import ipywidgets as widgets

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as tick
from matplotlib import colors

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib_scalebar.scalebar import ScaleBar

from PIL import Image
import imageio
import pandas as pd


# =========================================================================================================================================================
# Definition of necessary functions
# =========================================================================================================================================================


def get_extent(BrukerExp, specimen='rodent'):
    read_offset = BrukerExp.method['PVM_SPackArrReadOffset']
    phase_offset = BrukerExp.method['PVM_SPackArrPhase1Offset']

    mm_read = BrukerExp.method['PVM_Fov'][0]
    mm_phase = BrukerExp.method['PVM_Fov'][1]

    # mm_read_pixel_size = BrukerExp.method['PVM_SpatResol'][0]
    # mm_phase_pixel_size = BrukerExp.method['PVM_SpatResol'][1]

    if BrukerExp.method['PVM_SPackArrSliceOrient'] == 'axial':
        if specimen == 'rodent':
            # Conventional image display: turn by 90 degrees counterclockwise
            extent = [
                -mm_read / 2 - read_offset,
                mm_read / 2 - read_offset,
                -mm_phase / 2 - phase_offset,
                mm_phase / 2 - phase_offset,
            ]
        elif specimen == 'other':
            # Conventional image display: turn by 90 degrees counterclockwise
            extent = [
                -mm_read / 2 + read_offset,
                mm_read / 2 + read_offset,
                -mm_phase / 2 + phase_offset,
                mm_phase / 2 + phase_offset,
            ]

    elif BrukerExp.method['PVM_SPackArrSliceOrient'] == 'coronal':
        if specimen == 'rodent':
            # Conventional image display: turn by 90 degrees clockwise, mirror 3rd axis
            extent = [
                -mm_phase / 2 - phase_offset,
                mm_phase / 2 - phase_offset,
                -mm_read / 2 - read_offset,
                mm_read / 2 - read_offset,
            ]
        elif specimen == 'other':
            # Conventional image display: turn by 90 degrees clockwise, mirror 3rd axis
            extent = [
                -mm_phase / 2 + phase_offset,
                mm_phase / 2 + phase_offset,
                -mm_read / 2 + read_offset,
                mm_read / 2 + read_offset,
            ]

    elif BrukerExp.method['PVM_SPackArrSliceOrient'] == 'sagittal':
        if specimen == 'rodent':
            # Conventional image display: turn by 90 degrees clockwise, mirror 2nd axis, 3rd axis
            extent = [
                -mm_phase / 2 - phase_offset,
                mm_phase / 2 - phase_offset,
                -mm_read / 2 - read_offset,
                mm_read / 2 - read_offset,
            ]
        elif specimen == 'other':
            # Conventional image display: turn by 90 degrees clockwise, mirror 2nd axis, 3rd axis
            extent = [
                -mm_read / 2 + read_offset,
                mm_read / 2 + read_offset,
                -mm_phase / 2 + phase_offset,
                mm_phase / 2 + phase_offset,
            ]

    return extent


def add_colorbar(
    image=None,
    boundaries=None,
    title=None,
    mappable=None,
    color='k',
    fontsize=12,
    box=[0.075, -0.05, 0.2, 0.05],
    shrink=1.0,
    aspect=4,
    pad=-0.075,
    tickpad=0.5,
    titlepad=None,
    xtext=None,
    ytext=0.15,
    strformat='%.2f',
    orientation='horizontal',
    side=False,
):
    '''
    Add colorbar to an image,
    provide 'image' or 'boundaries' (dictionary with vmin and vmax).

    Parameters
    ----------
    image: 2D-array, optional
        Input image.
    boundaries: dictionary, optional
        Dictionary that contains vmin and vmax limits.
    title: string, optional
        Colorbar title.
    mappable: plt.mappable, optional
        Mappable of the image to add the colorbar.
    color: string, optional
        Edge and text color.
    fontsize: int, optional
        Text font size.
    box: list, optional
        Subaxis location [start_x, start_y, width, height].
    shrink: float, optional
        Colorbar shrink.
    aspect: float, optional
        Colorbar aspect ratio: horizontal/vertical
    pad, tickpad, titlepad: float, optional
        Pad value for the colorbar, ticks, title.
    xtext, ytext: float, optional
        Position of the tick labels.
    strformat: string, optional
        Tick label number format.
    orientation: string, optional
        Colorbar orientation: 'horizontal' or 'vertical'
    side: bool, optional
        Add limit tick labels on the colorbar side.
    '''

    if not boundaries:
        assert type(image) != type(None)
        boundaries = dict(vmin=np.percentile(image, 0), vmax=np.percentile(image, 100))

    cbax = plt.gca().inset_axes(box)

    if side:
        cbar = plt.colorbar(
            cax=cbax,
            shrink=shrink,
            aspect=aspect,
            pad=pad,
            format=strformat,
            orientation=orientation,
            mappable=mappable,
        )
               
        # cbar = plt.colorbar(ticks=[], cax=cbax, shrink=shrink, aspect=aspect,
        # pad=pad, format=strformat, orientation=orientation, mappable=mappable)
        # cbar.ax.tick_params(labelsize=fontsize, colors='none', direction='out', pad=tickpad)
        # if xtext is None:
        # xtext=[-0.3,1.05]
        # cbar.ax.annotate(strformat %boundaries['vmin'], xy=(0, 0), c=color, xycoords='axes fraction', xytext=(xtext[0], ytext),  textcoords='axes fraction')
        # cbar.ax.annotate(strformat %boundaries['vmax'], xy=(0, 0), c=color, xycoords='axes fraction', xytext=(xtext[1], ytext),  textcoords='axes fraction')

    else:
        cbar = plt.colorbar(
            ticks=[boundaries['vmin'], boundaries['vmax']],
            cax=cbax,
            shrink=shrink,
            aspect=aspect,
            pad=pad,
            format=strformat,
            orientation=orientation,
            mappable=mappable,
        )
        cbar.ax.tick_params(
            labelsize=fontsize, colors=color, direction='out', pad=tickpad
        )

    if title:
        if titlepad:
            cbar.set_label(title, labelpad=titlepad, color=color, fontsize=fontsize)
        else:
            cbar.ax.set_title(title, color=color, fontsize=fontsize)

    cbar.outline.set_color(color)


def add_scalebar(
    px,
    ax=None,
    units='mm',
    fixed_value=None,
    color='k',
    box_alpha=0.0,
    box_color='w',
    location='lower right',
    frameon=False,
    pad=0.075,
    length_fraction=None,
    border_pad=0.075,
    fontsize=12,
):
    '''
    Add scalebar to image, given the pixel size 'px'.
    '''

    if not ax:
        ax = plt.gca()
    ax.add_artist(
        ScaleBar(
            px,
            units=units,
            fixed_value=fixed_value,
            color=color,
            box_alpha=box_alpha,
            box_color=box_color,
            location=location,
            frameon=frameon,
            pad=pad,
            length_fraction=length_fraction,
            border_pad=border_pad,
            font_properties={'size': fontsize},
        )
    )


def add_patch(
    image,
    patchcenter=None,
    patchbox=None,
    ax=None,
    sl=None,
    color='r',
    lw=2,
    fc='none',
    out=False,
):
    '''
    Add rectangular patch to an image.
    If no other variables are passed, adds a patch of the size of the image.

    Parameters
    ----------
    image: 2D-array
        Input image.
    patchcenter: (2,) tuple, optional
        Center of the rectangle.
    patchbox: tuple of ints, optional
        Rectangle side dimensions in pixels: (dy, dx).
    ax: plt.axis, optional
        Axis of the image to add the patch.
    sl: slice, optional
        Slice/ROI of the patch (i.e. np.s_[:,...]).
    color: string, optional
        Patch color.
    lw: int, optional
        Patch line width.
    fc: string, optional
        Patch face color.
    out: bool, optional
        If true returns numpy slice.

    Returns
    -------
    sl : slice
        If out is True returns the slice.
    '''

    sh = image.shape

    if type(patchbox) == int:
        patchbox = [patchbox, patchbox]

    if type(sl) == type(None):
        if patchbox and patchcenter:
            sl = np.s_[
                patchcenter[0] - patchbox[0] // 2 : patchcenter[0] + patchbox[0] // 2,
                patchcenter[1] - patchbox[1] // 2 : patchcenter[1] + patchbox[1] // 2,
            ]
        elif patchbox[0]:
            sl = np.s_[
                sh[0] // 2 - patchbox[0] // 2 : sh[0] // 2 + patchbox[0] // 2,
                sh[1] // 2 - patchbox[1] // 2 : sh[1] // 2 + patchbox[1] // 2,
            ]
        else:
            sl = np.s_[0 : sh[0], 0 : sh[1]]

    if ax is None:
        ax = plt.gca()
    ax.add_patch(
        patches.Rectangle(
            (sl[1].start, sl[0].start),
            sl[1].stop - sl[1].start,
            sl[0].stop - sl[0].start,
            linewidth=lw,
            edgecolor=color,
            facecolor=fc,
        )
    )
    if out == True:
        return sl


def plane_x(n, r0, y, z):
    return -(n[1] * (y - r0[1]) + n[2] * (z - r0[2])) / n[0] + r0[0]


def plane_y(n, r0, x, z):
    return -(n[0] * (x - r0[0]) + n[2] * (z - r0[2])) / n[1] + r0[1]


def plane_z(n, r0, x, y):
    return -(n[0] * (x - r0[0]) + n[1] * (y - r0[1])) / n[2] + r0[2]


def get_slice_boundaries(
    spec_BrukerExp,
    Anatomical_BrukerExp,
    RARE_slice_number,
    specimen='rodent',
    db=True,
):
    # Image stack
    image_stack = Anatomical_BrukerExp.seq2d

    # Image shape
    image_shape = image_stack.shape[:-1]

    # Pixel size [mm]
    pixel_size_mm = Anatomical_BrukerExp.method['PVM_SpatResol']

    # RARE slice centers [mm]
    RARE_slice_offset_mm = Anatomical_BrukerExp.method['PVM_SPackArrSliceOffset']
    RARE_slice_centers_mm = (
        # Anatomical_BrukerExp.method['PVM_SliceOffset'] - 2 * RARE_slice_offset_mm
        Anatomical_BrukerExp.method['PVM_SliceOffset'] + 0 * RARE_slice_offset_mm
    )

    # spec slice thickness [mm]
    spec_slice_thickness_mm = spec_BrukerExp.method.get('SliceThick', None)
    if spec_slice_thickness_mm is None:
        spec_slice_thickness_mm = spec_BrukerExp.method.get('PVM_SliceThick', None)
        if spec_slice_thickness_mm is None:
            raise ValueError('Unable to determine slicethickness!')

    # spec slice normal vector [mm]
    n = spec_BrukerExp.method.get('SliceVec'[0], None)
    # if parameter SliceVec not defined:
    if n is None:
        # Get  acquisition matrix:
        n = spec_BrukerExp.acqp.get('ACQ_grad_matrix', None)
        # if acquisition matrix was defined:
        if n is not None:
            n = n[0][2]  # 3rd row of matrix == slice normal vector
        else:
            raise ValueError('Not able to find normal Vector!')
    if db:
        print('spec slice normal vector [mm]:\t{}'.format(np.round(n, 2)))

    # spec slice translation vector [mm]
    spec_slice_offset = spec_BrukerExp.method.get('SliceOffset', None)
    if spec_slice_offset is None:
        spec_slice_offset = spec_BrukerExp.method.get('PVM_SliceOffset', None)
        if spec_slice_offset is None:
            raise ValueError('Not able to determine spec slice offset!')

    t = spec_slice_offset * n

    # Coordinate system
    if Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'axial':
        # Conventional image display: turn by 90 degrees counterclockwise
        pixel_size_mm[[0, 1]] = pixel_size_mm[[1, 0]]

        # print('spec slice translation vector [mm]:\t{}'.format(np.round(t, 2)))

        # spec slice boundaries centers
        spec_slice_boundary_1_center = t - spec_slice_thickness_mm / 2 * n
        spec_slice_boundary_2_center = t + spec_slice_thickness_mm / 2 * n
    
        # Calculate two points of both slice boundaries in the given RARE slice number
        sliceboundary_1_r = np.array([-50, 50])
        sliceboundary_2_r = np.array([-50, 50])

        if specimen == 'rodent':
            sliceboundary_1_p = -plane_y(
                n,
                spec_slice_boundary_1_center,
                -sliceboundary_1_r,
                np.repeat(RARE_slice_centers_mm[RARE_slice_number], 2),
            )
            sliceboundary_2_p = -plane_y(
                n,
                spec_slice_boundary_2_center,
                -sliceboundary_2_r,
                np.repeat(RARE_slice_centers_mm[RARE_slice_number], 2),
            )

        elif specimen == 'other':
            sliceboundary_1_p = plane_y(
                n,
                spec_slice_boundary_1_center,
                -sliceboundary_1_r,
                np.repeat(RARE_slice_centers_mm[RARE_slice_number], 2),
            )
            sliceboundary_2_p = plane_y(
                n,
                spec_slice_boundary_2_center,
                -sliceboundary_2_r,
                np.repeat(RARE_slice_centers_mm[RARE_slice_number], 2),
            )

            if np.any(sliceboundary_1_p == np.inf):
                sliceboundary_1_p = np.array([-50, 50])
                sliceboundary_2_p = np.array([-50, 50])

                sliceboundary_1_r = plane_x(
                    n,
                    spec_slice_boundary_1_center,
                    sliceboundary_1_p,
                    np.repeat(RARE_slice_centers_mm[RARE_slice_number], 2),
                )
                sliceboundary_2_r = plane_x(
                    n,
                    spec_slice_boundary_2_center,
                    sliceboundary_2_p,
                    np.repeat(RARE_slice_centers_mm[RARE_slice_number], 2),
                )

    elif Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'coronal':
        # Conventional image display: turn by 90 degrees clockwise, mirror 3rd axis
        pixel_size_mm[[0, 1]] = pixel_size_mm[[1, 0]]

        # print('spec slice translation vector [mm]:\t{}'.format(np.round(t, 2)))

        # spec slice boundaries centers
        spec_slice_boundary_1_center = t - spec_slice_thickness_mm / 2 * n
        spec_slice_boundary_2_center = t + spec_slice_thickness_mm / 2 * n

        # Calculate two points of both slice boundaries in the given RARE slice number
        sliceboundary_1_r = np.array([-50, 50])
        sliceboundary_2_r = np.array([-50, 50])

        if specimen == 'rodent':
            sliceboundary_1_p = plane_z(
                n,
                spec_slice_boundary_1_center,
                sliceboundary_1_r,
                -np.repeat(RARE_slice_centers_mm[RARE_slice_number], 2),
            )
            sliceboundary_2_p = plane_z(
                n,
                spec_slice_boundary_2_center,
                sliceboundary_2_r,
                -np.repeat(RARE_slice_centers_mm[RARE_slice_number], 2),
            )

        elif specimen == 'other':
            sliceboundary_1_p = plane_z(
                n,
                spec_slice_boundary_1_center,
                sliceboundary_1_r,
                np.repeat(RARE_slice_centers_mm[RARE_slice_number], 2),
            )
            sliceboundary_2_p = plane_z(
                n,
                spec_slice_boundary_2_center,
                sliceboundary_2_r,
                np.repeat(RARE_slice_centers_mm[RARE_slice_number], 2),
            )

            if np.any(sliceboundary_1_p == np.inf):
                sliceboundary_1_p = np.array([-50, 50])
                sliceboundary_2_p = np.array([-50, 50])

                sliceboundary_1_r = plane_x(
                    n,
                    spec_slice_boundary_1_center,
                    np.repeat(RARE_slice_centers_mm[RARE_slice_number], 2),
                    sliceboundary_1_p,
                )
                sliceboundary_2_r = plane_x(
                    n,
                    spec_slice_boundary_2_center,
                    np.repeat(RARE_slice_centers_mm[RARE_slice_number], 2),
                    sliceboundary_2_p,
                )

    elif Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'sagittal':
        # Conventional image display: turn by 90 degrees clockwise, mirror 2nd axis, mirror 3rd axis
        pixel_size_mm[[0, 1]] = pixel_size_mm[[1, 0]]

        # print('spec slice translation vector [mm]:\t{}'.format(np.round(t, 2)))

        # spec slice boundaries centers
        spec_slice_boundary_1_center = t - spec_slice_thickness_mm / 2 * n
        spec_slice_boundary_2_center = t + spec_slice_thickness_mm / 2 * n

        # Calculate two points of both slice boundaries in the given RARE slice number
        sliceboundary_1_r = np.array([-50, 50])
        sliceboundary_2_r = np.array([-50, 50])

        if specimen == 'rodent':
            sliceboundary_1_p = -plane_y(
                n,
                spec_slice_boundary_1_center,
                -np.repeat(RARE_slice_centers_mm[RARE_slice_number], 2),
                sliceboundary_1_r,
            )
            sliceboundary_2_p = -plane_y(
                n,
                spec_slice_boundary_2_center,
                -np.repeat(RARE_slice_centers_mm[RARE_slice_number], 2),
                sliceboundary_2_r,
            )

        elif specimen == 'other':
            sliceboundary_1_p = plane_y(
                n,
                spec_slice_boundary_1_center,
                np.repeat(RARE_slice_centers_mm[RARE_slice_number], 2),
                sliceboundary_1_r,
            )
            sliceboundary_2_p = plane_y(
                n,
                spec_slice_boundary_2_center,
                np.repeat(RARE_slice_centers_mm[RARE_slice_number], 2),
                sliceboundary_2_r,
            )

            if np.any(sliceboundary_1_p == np.inf):
                sliceboundary_1_p = np.array([-50, 50])
                sliceboundary_2_p = np.array([-50, 50])

                sliceboundary_1_r = plane_z(
                    n,
                    spec_slice_boundary_1_center,
                    np.repeat(RARE_slice_centers_mm[RARE_slice_number], 2),
                    sliceboundary_1_p,
                )
                sliceboundary_2_r = plane_z(
                    n,
                    spec_slice_boundary_2_center,
                    np.repeat(RARE_slice_centers_mm[RARE_slice_number], 2),
                    sliceboundary_2_p,
                )
    if db:
        print('t: ', t)
        print('n: ', n)
        print(
            'slice boundary coordinates ([r11, r12], [r21, r22], [p11, p12], [p21, p22]) [mm]: ',
            sliceboundary_1_r,
            sliceboundary_2_r,
            sliceboundary_1_p,
            sliceboundary_2_p,
        )

    return t, sliceboundary_1_r, sliceboundary_2_r, sliceboundary_1_p, sliceboundary_2_p


def plot_image_stack(
    spec_BrukerExp,
    Anatomical_BrukerExp,
    animal_ID,
    image_stack=None,
    specimen='rodent',
    cell_line=None,
    scale_bar=None,
    vmin=0,
    vmax=120,
    initial_slice_number=0,
    line_color='dodgerblue',
    patchcenterlist=[(0, 0)],
    patchboxlist=[(10, 10)],
    plotsavepath='',
    plotname='',
    saveplot=False,
    plot_params={},
    file_format='png',
):
    '''
    Plots the position of spec_BrukerExp onto the Anatomical_BrukerExp. Slices can be selected using a slider.
    If plot_params['plot_all_in_one'] = True is passed, all (or plot_params['slice_range']) slice will be plotted in
    one figure.
    Parameters
    ----------
    spec_BrukerExp: brukerexp object
        Image of which the position will be plotted onto Anatomical_BrukerExp.
    Anatomical_BrukerExp:brukerexp object
        The anatomical to show the position of spec_BrukerExp.
    animal_ID
    specimen
    cell_line
    scale_bar
    vmin: float, optional
        Minimum colorvalue of the anatomical image that will be displayed. (None=auto)
    vmax: float, optional
        Maximum colorvalue of the anatomical image that will be displayed. (None=auto)
    initial_slice_number
    line_color
    patchcenterlist
    patchboxlist
    plotsavepath: str, optional
        Where the figure will be saved
    plotname: str, optional
        Name of the figure to be saved
    saveplot: bool, optional
        Toggles the saving of the plotted figure
    plot_params: dict, optional
        Contains plot settings, such as plot_all_in_one (plots all slices in one figure), slice_range (which slices to
        plot)
    file_format: str, optional
        Format in which the figure will be saved

    Returns
    -------

    '''
    
    if np.any(image_stack):
        image_stack = image_stack
        seq2d_shape = image_stack.shape
    
    else:
        try:
            image_stack = np.squeeze(np.transpose(
                Anatomical_BrukerExp.seq2d_oriented, (2, 3, 1, 0, 4, 5)
            ))[:, ::-1]
            seq2d_shape = image_stack.shape
        except:
            try:            
                seq2d_shape = Anatomical_BrukerExp.seq2d.shape
                
                image_stack = np.reshape(
                    Anatomical_BrukerExp.seq2d, (seq2d_shape[0], seq2d_shape[1], seq2d_shape[-1])
                )
            except:
                image_stack = np.reshape(
                    Anatomical_BrukerExp.seq2d[:, :, 0, :], (seq2d_shape[0], seq2d_shape[1], seq2d_shape[-1])
                )

            

    # Slice number
    slice_number = image_stack.shape[-1]
    
    # try:        
        # slice_number = Anatomical_BrukerExp.Nslices
    # except:
        # slice_number = image_stack.shape[-1]

    # Pixel size [mm]
    pixel_size_mm = Anatomical_BrukerExp.method['PVM_SpatResol']

    # Coordinate system
    # er = Anatomical_BrukerExp.method['PVM_SPackArrGradOrient'][0][0]
    # ep = Anatomical_BrukerExp.method['PVM_SPackArrGradOrient'][0][1]
    # es = Anatomical_BrukerExp.method['PVM_SPackArrGradOrient'][0][2]

    if Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'axial':
        er = np.array([1, 0, 0])
        ep = np.array([0, -1, 0])
        es = np.array([0, 0, -1])

        ylabel = 'y [mm]'
        xlabel = 'x [mm]'

        if specimen == 'rodent':
            # Conventional image display: turn by 90 degrees counterclockwise
            image_stack = np.swapaxes(
                np.reshape(
                    image_stack,
                    (seq2d_shape[0], seq2d_shape[1], seq2d_shape[-1]),
                ),
                0,
                1,
            )[::-1, :]

        elif specimen == 'other':
            # Conventional image display: turn by 90 degrees counterclockwise
            image_stack = np.swapaxes(
                np.reshape(
                    image_stack,
                    (seq2d_shape[0], seq2d_shape[1], seq2d_shape[-1]),
                ),
                0,
                1,
            )

        pixel_size_mm[[0, 1]] = pixel_size_mm[[1, 0]]

    elif Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'coronal':
        er = np.array([0, 0, -1])
        ep = np.array([1, 0, 0])
        es = np.array([0, -1, 0])

        ylabel = 'z [mm]'
        xlabel = 'x [mm]'

        if specimen == 'rodent':
            # Conventional image display: turn by 90 degrees clockwise, mirror 3rd axis
            image_stack = np.swapaxes(
                np.reshape(
                    image_stack,
                    (seq2d_shape[0], seq2d_shape[1], seq2d_shape[-1]),
                ),
                1,
                2,
            )[:, ::-1, ::-1]
            image_stack = np.swapaxes(
                    image_stack,
                0,
                1,
            )
        elif specimen == 'other':
            # Conventional image display: turn by 90 degrees clockwise, mirror 3rd axis
            image_stack = np.swapaxes(
                np.reshape(
                    image_stack,
                    (seq2d_shape[0], seq2d_shape[1], seq2d_shape[-1]),
                ),
                0,
                1,
            )[:, ::-1]

        pixel_size_mm[[0, 1]] = pixel_size_mm[[1, 0]]

    elif Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'sagittal':
        er = np.array([0, -1, 0])
        ep = np.array([0, 0, -1])
        es = np.array([1, 0, 0])

        ylabel = 'z [mm]'
        xlabel = 'y [mm]'

        if specimen == 'rodent':
            # Conventional image display: turn by 90 degrees clockwise, mirror 2nd axis, mirror 3rd axis
            # image_stack = np.swapaxes(np.reshape(Anatomical_BrukerExp.seq2d, (seq2d_shape[0], seq2d_shape[1], seq2d_shape[-1])), 0, 1)[:, :, ::-1]
            image_stack = np.reshape(
                image_stack, (seq2d_shape[0], seq2d_shape[1], seq2d_shape[-1])
            )[::-1, :, ::-1]

        elif specimen == 'other':
            # Conventional image display: turn by 90 degrees clockwise, mirror 2nd axis, mirror 3rd axis
            # image_stack = np.swapaxes(np.reshape(Anatomical_BrukerExp.seq2d, (seq2d_shape[0], seq2d_shape[1], seq2d_shape[-1])), 0, 1)[:, :, ::-1]
            image_stack = np.reshape(
                image_stack, (seq2d_shape[0], seq2d_shape[1], seq2d_shape[-1])
            )[:, ::-1, :]

        pixel_size_mm[[0, 1]] = pixel_size_mm[[1, 0]]

    # Image shape
    image_shape = image_stack.shape[:-1]

    # Maximum and minimum signal value
    max_signal = image_stack.max()
    min_signal = image_stack.min()

    vmin_anat = plot_params.get('vmin_anat', vmin)
    vmax_anat = plot_params.get('vmax_anat', vmax)
    cmap_anat = plot_params.get('cmap_anat', 'bone')
    line_label = plot_params.get('line_label', 'CSI slice boundaries')
    linecolor = plot_params.get('linecolor', line_color)
    linewidth = plot_params.get('linewidth', 1)
    linestyle = plot_params.get('linestyle', 'dashed')
    figsize = plot_params.get('figsize', (1.9 * 2.0, 1.9 * 3.5))
    showticks = plot_params.get('showticks', True)
    showtime = plot_params.get('showtime', False)
    showcolorbar = plot_params.get('showcolorbar', False)
    showspectroscopy_slice = plot_params.get('showspectroscopy_slice', True)
    showtitle = plot_params.get('showtitle', True)
    leg_distance_to_plot = plot_params.get('leg_distance_to_plot', 1.2)
    plot_all_in_one = plot_params.get('plot_all_in_one', False)
    flip_rows_columns = plot_params.get('flip_rows_columns', False)
    scalebar_factor = plot_params.get('scalebar_factor', 1.5)
    CSI_anatomicals = plot_params.get('CSI_anatomicals', False)
    extent = plot_params.get('extent', get_extent(Anatomical_BrukerExp, specimen=specimen))

    slice_range = plot_params.get('slice_range', range(slice_number))
    if len(slice_range) != slice_number:
        slice_number = len(slice_range)

    if plot_all_in_one:
        from ..plot.utils_plot import subplot_auto_size

        grid_rows, grid_cols = subplot_auto_size(num_subplots=slice_number)
        # Create subplots, flipping rows and columns if specified.
        if flip_rows_columns:
            fig, ax = plt.subplots(grid_cols, grid_rows, figsize=figsize)
        else:
            fig, ax = plt.subplots(grid_rows, grid_cols, figsize=figsize)
        ax = ax.flatten()
    else:
        # Plot image stack with slider specifying the slice number
        fig, ax = plt.subplots(figsize=figsize)

    def plot_image_stack_single(slice):
            
        ax.clear()
        
        try:
            cb_ax.clear()
            cb_ax.set_visible(False)
        except:
            pass
    
        subplot = ax.imshow(
            image_stack[:, :, slice],
            cmap=cmap_anat,
            vmin=vmin_anat,
            vmax=vmax_anat,
            extent=extent,
        )
        # extent=(-0.5*pixel_size_mm[0], (image_shape[1]-0.5)*pixel_size_mm[0], (image_shape[0]-0.5)*pixel_size_mm[1], -0.5*pixel_size_mm[1])) # (xleft, xright, ybottom, ytop)
        # aspect=pixel_size_mm[1]/pixel_size_mm[0]) # dy/dx

        # print('image extent [mm]: ', get_extent(Anatomical_BrukerExp, specimen=specimen))

        if showspectroscopy_slice:
            (
                t,
                sliceboundary_1_x,
                sliceboundary_2_x,
                sliceboundary_1_y,
                sliceboundary_2_y,
            ) = get_slice_boundaries(
                spec_BrukerExp, Anatomical_BrukerExp, slice, specimen=specimen, db=False
            )
    
            sliceboundary_1 = ax.plot(
                sliceboundary_1_x,
                sliceboundary_1_y,
                color=linecolor,
                linewidth=linewidth,
                linestyle=linestyle,
                label=line_label,
            )
            sliceboundary_2 = ax.plot(
                sliceboundary_2_x,
                sliceboundary_2_y,
                color=linecolor,
                linewidth=linewidth,
                linestyle=linestyle,
            )

        # if Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'axial':
            # ax.scatter(t[0], t[1], color=linecolor)
        # if Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'coronal':
            # ax.scatter(t[0], t[2], color=linecolor)
        # if Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'sagittal':
            # ax.scatter(t[2], t[1], color=linecolor)

        ax.set_ylim(
            extent[2],
            extent[3],
        )
        ax.set_xlim(
            extent[0],
            extent[1],
        )

        # ax.set_ylabel(ylabel)
        # ax.set_xlabel(xlabel)

        if not showticks:
            ax.axis('off')

        # if Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'axial':

        # ax.quiver(0, 0, 1, 0, color='red')
        # ax.quiver(0, 0, 0, 1, color='green')
        
        try:
            time_s = (slice+1) * Anatomical_BrukerExp.method['Scan_RepetitionTime'] * Anatomical_BrukerExp.method['PVM_SPackArrNSlices'] * 1e-3
        except:
            time_s = (slice+1) * Anatomical_BrukerExp.method['PVM_ScanTime'] * 1e-3 / slice_number
        
        # text_x = get_extent(Anatomical_BrukerExp)[0] + 1 / 10 * (
        # get_extent(Anatomical_BrukerExp)[1] - get_extent(Anatomical_BrukerExp)[0]
        # )
        # text_y = get_extent(Anatomical_BrukerExp)[3] - 1 / 10 * (
        # get_extent(Anatomical_BrukerExp)[3] - get_extent(Anatomical_BrukerExp)[2]
        # )

        # plt.text(text_x, text_y, 't = {1:02d} min {2:02d} s'.format(int(time_s//3600), int((time_s%3600)//60), int(time_s%60), int(((time_s%60)*1000)%1000)), color='white', bbox=dict(fill=False, edgecolor=None, linewidth=0), fontsize=10)

        # if Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'axial' or Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'coronal' or Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'sagittal':

        # add_colorbar(
        # image=image_stack[..., slice],
        # # image=None,
        # # boundaries=None,
        # boundaries={'vmin': vmin, 'vmax': vmax},
        # title=r'signal [a. u.]',
        # mappable=subplot,
        # color='k',
        # fontsize=12,
        # box=[0.3, -0.1, 0.2, 0.05],
        # shrink=1.0,
        # aspect=4,
        # pad=-0.075,
        # tickpad=0.5,
        # titlepad=0.075,
        # xtext=None,
        # ytext=0.15,
        # strformat='%d',
        # orientation='horizontal',
        # side=False
        # )

        # add_colorbar(
                     # image=image_stack[..., slice],
                     # boundaries=None,
                     # # boundaries={'vmin': min_signal, 'vmax': max_signal},
                     # title=r'signal [a. u.]',
                     # mappable=subplot,
                     # color='k',
                     # fontsize=12,
                     # box=[1.01, 0.0, 0.05, 1.0],
                     # shrink=1.0,
                     # aspect=4,
                     # pad=0.0,
                     # tickpad=0.5,
                     # titlepad=5,
                     # xtext=None,
                     # ytext=0.3,
                     # strformat='%d',
                     # orientation='vertical',
                     # side=True
                     # )

        # plot_colorbar(fig, ax, image_stack[..., slice], label=r'signal [a. u.]')

        if scale_bar:
            add_scalebar(
                px=1e-1, # pixel_size_mm[0] * scalebar_factor * 1e-1,
                units='cm',
                fixed_value=1,
                color='w',
                box_alpha=0.0,
                box_color='w',
                location='lower right',
                frameon=True,
                pad=0.075,
                length_fraction=None,
                border_pad=0.075,
                fontsize=12,
            )
        else:
            pass

        if showcolorbar:
            divider = make_axes_locatable(ax)
            cb_ax   = divider.append_axes('right', size='5%', pad=0.05)
            cbar    = fig.colorbar(subplot, cax=cb_ax)
            cbar.set_label(r'signal [a. u.]')
            cbar.set_ticks(ticks=[vmin_anat, (vmax_anat-vmin_anat)//2, vmax_anat])
            
        insetslicelist = []
        colorlist = ['r']

        # for i, patchcenter in enumerate(patchcenterlist):
        # insetslicelist.append(add_patch(
        # image=image_stack[..., slice],
        # patchcenter=patchcenter,
        # patchbox=patchboxlist[i],
        # ax=None,
        # sl=None,
        # color=colorlist[i],
        # lw=1,
        # fc='none',
        # out=True
        # ))

        # axin = ax.inset_axes([1.05, 0.0, (0.6*image_shape[0]*patchboxlist[0][1]/patchboxlist[0][0])/image_shape[1], 0.6])

        # inset = axin.imshow(image_stack[:, :, slice],
        # vmin=vmin, vmax=vmax,
        # cmap='gray',
        # extent=get_extent(Anatomical_BrukerExp)) # (xleft, xright, ybottom, ytop)
        # # aspect=pixel_size_mm[1]/pixel_size_mm[0]) # dy/dx

        # axin.set_xlim([insetslicelist[0][1].start, insetslicelist[0][1].stop])
        # axin.set_ylim([insetslicelist[0][0].start, insetslicelist[0][0].stop])

        # axin.set_xticks([])
        # axin.set_yticks([])

        # axin.spines['bottom'].set_color(colorlist[0])
        # axin.spines['top'].set_color(colorlist[0])
        # axin.spines['left'].set_color(colorlist[0])
        # axin.spines['right'].set_color(colorlist[0])

        # add_scalebar(
        # px=5*pixel_size_mm[0],
        # ax=axin,
        # units='mm',
        # fixed_value=1,
        # color='w',
        # box_alpha=0.0,
        # box_color='w',
        # location='lower right',
        # frameon=True,
        # pad=0.075,
        # length_fraction=None,
        # border_pad=0.075,
        # fontsize=12
        # )

        handles, labels = ax.get_legend_handles_labels()
        # handles.append(patches.Patch(color='none', label=animal_ID))
        # handles.append(patches.Patch(color='none', label=cell_line))
        # leg = ax.legend(loc='center left', bbox_to_anchor=(0, 1.2), handles=handles, title=animal_ID + '\n' + cell_line)
        if showtime:
            leg = ax.legend(
                loc='center left',
                bbox_to_anchor=(0, leg_distance_to_plot),
                handles=handles,
                title=animal_ID + '\n' + cell_line
                + '\n'
                + 't = {1:02d} min {2:02d} s'.format(
                int(time_s // 3600),
                int((time_s % 3600) // 60),
                int(time_s % 60),
                int(((time_s % 60) * 1000) % 1000),
                ),
            )
            leg.get_frame().set_edgecolor('black')

        elif cell_line:
            leg = ax.legend(
                loc='center left',
                bbox_to_anchor=(0, leg_distance_to_plot),
                handles=handles,
                title=animal_ID + '\n' + cell_line
            )
            leg.get_frame().set_edgecolor('black')

        else:
            pass
            
        plt.tight_layout()
                
        # print('Figure size [in]: ', fig.get_size_inches())

    if plot_all_in_one:
        for s, slice in enumerate(slice_range):
            subplot = ax[s].imshow(
                image_stack[:, :, slice],
                cmap=cmap_anat,
                vmin=vmin_anat,
                vmax=vmax_anat,
                extent=get_extent(Anatomical_BrukerExp, specimen=specimen),
            )
            # extent=(-0.5*pixel_size_mm[0], (image_shape[1]-0.5)*pixel_size_mm[0], (image_shape[0]-0.5)*pixel_size_mm[1], -0.5*pixel_size_mm[1])) # (xleft, xright, ybottom, ytop)
            # aspect=pixel_size_mm[1]/pixel_size_mm[0]) # dy/dx

            (
                t,
                sliceboundary_1_x,
                sliceboundary_2_x,
                sliceboundary_1_y,
                sliceboundary_2_y,
            ) = get_slice_boundaries(
                spec_BrukerExp, Anatomical_BrukerExp, slice, specimen=specimen, db=False
            )

            if not CSI_anatomicals:
                sliceboundary_1 = ax[s].plot(
                    sliceboundary_1_x,
                    sliceboundary_1_y,
                    color=linecolor,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    label=line_label,
                )
                sliceboundary_2 = ax[s].plot(
                    sliceboundary_2_x,
                    sliceboundary_2_y,
                    color=linecolor,
                    linewidth=linewidth,
                    linestyle=linestyle,
                )
            # print(t)
            # if Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'axial':
            #     ax[s].scatter(t[0], t[1], color=linecolor)
            # if Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'coronal':
            #     ax[s].scatter(t[0], t[2], color=linecolor)
            # if Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'sagittal':
            #     ax[s].scatter(t[2], t[1], color=linecolor)

            ax[s].set_ylim(
                get_extent(Anatomical_BrukerExp, specimen=specimen)[2],
                get_extent(Anatomical_BrukerExp, specimen=specimen)[3],
            )
            ax[s].set_xlim(
                get_extent(Anatomical_BrukerExp, specimen=specimen)[0],
                get_extent(Anatomical_BrukerExp, specimen=specimen)[1],
            )
            
            if scale_bar:
                add_scalebar(
                    px=1e-1, # pixel_size_mm[0] * 1.5 * 1e-1,
                    units='cm',
                    fixed_value=1,
                    color='w',
                    box_alpha=0.0,
                    box_color='w',
                    location='lower right',
                    frameon=True,
                    pad=0.075,
                    length_fraction=None,
                    border_pad=0.075,
                    fontsize=12,
                )
            else:
                pass

            handles, labels = ax[s].get_legend_handles_labels()
            # handles.append(patches.Patch(color='none', label=animal_ID))
            # handles.append(patches.Patch(color='none', label=cell_line))
            # leg = ax.legend(loc='center left', bbox_to_anchor=(0, 1.2), handles=handles, title=animal_ID + '\n' + cell_line)
            if cell_line:
                leg = ax[s].legend(
                    loc='center left',
                    bbox_to_anchor=(0, leg_distance_to_plot),
                    handles=handles,
                    title=animal_ID + '\n' + cell_line
                    # + '\n'
                    # + 't = {1:02d} min {2:02d} s'.format(
                    # int(time_s // 3600),
                    # int((time_s % 3600) // 60),
                    # int(time_s % 60),
                    # int(((time_s % 60) * 1000) % 1000),
                    # ),
                )
                leg.get_frame().set_edgecolor('black')

            else:
                pass

            if not showticks:
                if not CSI_anatomicals:
                    ax[s].axis('off')               
                else: 
                    if s in np.arange(slice_number//2 - spec_BrukerExp.method['PVM_SliceThick']//Anatomical_BrukerExp.method['PVM_SliceThick']//2, slice_number//2 + spec_BrukerExp.method['PVM_SliceThick']//Anatomical_BrukerExp.method['PVM_SliceThick']//2 + 1):
                        ax[s].set_xticks([])
                        ax[s].set_yticks([])
        
                        ax[s].spines["bottom"].set_color(linecolor)
                        ax[s].spines["top"].set_color(linecolor)
                        ax[s].spines["left"].set_color(linecolor)
                        ax[s].spines["right"].set_color(linecolor)
                        
                        ax[s].spines["bottom"].set_linewidth(2)
                        ax[s].spines["top"].set_linewidth(2)
                        ax[s].spines["left"].set_linewidth(2)
                        ax[s].spines["right"].set_linewidth(2)                        
                    else:
                        ax[s].axis('off') 
                            
            if showtitle:
                ax[s].set_title(slice)

        # Turn off unused subplots
        for i in range(slice_number, len(ax)):
            ax[i].set_axis_off()

        plt.tight_layout()

        if saveplot:
            plt.savefig(
                os.path.join(
                    plotsavepath,
                    plotname
                    + '_'
                    + Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient']
                    + '.'
                    + str(file_format),
                )
            )

    elif saveplot:

        for slice in tqdm(range(slice_number)):
            ax.clear()
            plot_image_stack_single(slice)            
            plt.savefig(
                os.path.join(
                    plotsavepath,
                    plotname
                    + Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient']
                    + '_slice_{:03d}'.format(slice)
                    + '.'
                    + str(file_format),
                )
            )

    else:
    
        @widgets.interact(slice=(0, (slice_number - 1), 1))
        def plot_image_stack_with_slice_slider(slice=initial_slice_number):
            ax.clear()
            plot_image_stack_single(slice)


def save_image_stack(
    spec_BrukerExp,
    Anatomical_BrukerExp,
    animal_ID,
    cell_line,
    vmin=0,
    vmax=120,
    initial_slice_number=16,
    patchcenterlist=[(0, 0)],
    patchboxlist=[(10, 10)],
    plotsavepath='',
    plotname='',
):
    # Image stack
    seq2d_shape = Anatomical_BrukerExp.seq2d.shape
    image_stack = np.reshape(
        Anatomical_BrukerExp.seq2d, (seq2d_shape[0], seq2d_shape[1], seq2d_shape[-1])
    )

    # Slice number
    slice_number = image_stack.shape[-1]

    # Pixel size [mm]
    pixel_size_mm = Anatomical_BrukerExp.method['PVM_SpatResol']

    # Coordinate system
    if Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'axial':
        er = np.array([1, 0, 0])
        ep = np.array([0, -1, 0])
        es = np.array([0, 0, -1])

        ylabel = 'y [mm]'
        xlabel = 'x [mm]'

        # Conventional image display: turn by 90 degrees counterclockwise
        image_stack = np.swapaxes(
            np.reshape(
                Anatomical_BrukerExp.seq2d, (seq2d_shape[0], seq2d_shape[1], seq2d_shape[-1])
            ),
            0,
            1,
        )[::-1, :]
        pixel_size_mm[[0, 1]] = pixel_size_mm[[1, 0]]

    elif Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'coronal':
        er = np.array([0, 0, -1])
        ep = np.array([1, 0, 0])
        es = np.array([0, -1, 0])

        ylabel = 'z [mm]'
        xlabel = 'x [mm]'

        # Conventional image display: turn by 90 degrees clockwise, mirror 3rd axis
        image_stack = np.swapaxes(
            np.reshape(
                Anatomical_BrukerExp.seq2d, (seq2d_shape[0], seq2d_shape[1], seq2d_shape[-1])
            ),
            0,
            1,
        )[:, ::-1, ::-1]
        pixel_size_mm[[0, 1]] = pixel_size_mm[[1, 0]]

    elif Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'sagittal':
        er = np.array([0, -1, 0])
        ep = np.array([0, 0, -1])
        es = np.array([1, 0, 0])

        ylabel = 'z [mm]'
        xlabel = 'y [mm]'

        # Conventional image display: turn by 90 degrees clockwise, mirror 2nd axis, mirror 3rd axis
        # image_stack = np.swapaxes(np.reshape(Anatomical_BrukerExp.seq2d, (seq2d_shape[0], seq2d_shape[1], seq2d_shape[-1])), 0, 1)[:, :, ::-1]
        image_stack = np.reshape(
            Anatomical_BrukerExp.seq2d, (seq2d_shape[0], seq2d_shape[1], seq2d_shape[-1])
        )[::-1, :, ::-1]
        pixel_size_mm[[0, 1]] = pixel_size_mm[[1, 0]]

    # Image shape
    image_shape = image_stack.shape[:-1]

    # Maximum and minimum signal value
    max_signal = image_stack.max()
    min_signal = image_stack.min()

    # Plot image stack with slider specifying the slice number
    fig, ax = plt.subplots(figsize=(1.9 * 2.0, 1.9 * 3.5))

    for slice in tqdm(range(slice_number)):
        ax.clear()

        subplot = ax.imshow(
            image_stack[:, :, slice],
            cmap='gray',
            vmin=vmin,
            vmax=vmax,
            extent=get_extent(Anatomical_BrukerExp),
        )
        # extent=(-0.5*pixel_size_mm[0], (image_shape[1]-0.5)*pixel_size_mm[0], (image_shape[0]-0.5)*pixel_size_mm[1], -0.5*pixel_size_mm[1])) # (xleft, xright, ybottom, ytop)
        # aspect=pixel_size_mm[1]/pixel_size_mm[0]) # dy/dx

        (
            sliceboundary_1_x,
            sliceboundary_2_x,
            sliceboundary_1_y,
            sliceboundary_2_y,
        ) = get_slice_boundaries(spec_BrukerExp, Anatomical_BrukerExp, slice)

        sliceboundary_1 = ax.plot(
            sliceboundary_1_x,
            sliceboundary_1_y,
            color='dodgerblue',
            linewidth=1,
            linestyle='dashed',
        )
        sliceboundary_2 = ax.plot(
            sliceboundary_2_x,
            sliceboundary_2_y,
            color='dodgerblue',
            linewidth=1,
            linestyle='dashed',
        )

        ax.set_ylim(get_extent(Anatomical_BrukerExp)[2], get_extent(Anatomical_BrukerExp)[3])
        ax.set_xlim(get_extent(Anatomical_BrukerExp)[0], get_extent(Anatomical_BrukerExp)[1])

        # ax.set_ylabel(ylabel)
        # ax.set_xlabel(xlabel)

        ax.axis('off')

        # if Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'axial':

        # ax.quiver(0, 0, 1, 0, color='red')
        # ax.quiver(0, 0, 0, 1, color='green')

        time_s = (slice+1) * Anatomical_BrukerExp.method['PVM_ScanTime'] * 1e-3 / slice_number
        text_x = get_extent(Anatomical_BrukerExp)[0] + 1 / 10 * (
            get_extent(Anatomical_BrukerExp)[1] - get_extent(Anatomical_BrukerExp)[0]
        )
        text_y = get_extent(Anatomical_BrukerExp)[3] - 1 / 10 * (
            get_extent(Anatomical_BrukerExp)[3] - get_extent(Anatomical_BrukerExp)[2]
        )

        # plt.text(text_x, text_y, 't = {1:02d} min {2:02d} s'.format(int(time_s//3600), int((time_s%3600)//60), int(time_s%60), int(((time_s%60)*1000)%1000)), color='white', bbox=dict(fill=False, edgecolor=None, linewidth=0), fontsize=10)

        if (
            Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'axial'
            or Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'coronal'
            or Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'sagittal'
        ):
            add_colorbar(
                image=image_stack[..., slice],
                # image=None,
                # boundaries=None,
                boundaries={'vmin': vmin, 'vmax': vmax},
                title=r'signal [a. u.]',
                mappable=subplot,
                color='k',
                fontsize=12,
                box=[0.3, -0.1, 0.2, 0.05],
                shrink=1.0,
                aspect=4,
                pad=-0.075,
                tickpad=0.5,
                titlepad=0.075,
                xtext=None,
                ytext=0.15,
                strformat='%d',
                orientation='horizontal',
                side=False,
            )

        # add_colorbar(
        # image=image_stack[..., slice],
        # boundaries=None,
        # # boundaries={'vmin': min_signal, 'vmax': max_signal},
        # title=r'signal [a. u.]',
        # mappable=subplot,
        # color='k',
        # fontsize=12,
        # box=[1.01, 0.0, 0.03, 1.0],
        # shrink=1.0,
        # aspect=4,
        # pad=0.0,
        # tickpad=0.5,
        # titlepad=5,
        # xtext=None,
        # ytext=0.3,
        # strformat='%d',
        # orientation='vertical',
        # side=True
        # )

        # plot_colorbar(fig, ax, image_stack[..., slice], label=r'signal [a. u.]')

        # divider = make_axes_locatable(ax)
        # cb_ax   = divider.append_axes('right', size='5%', pad=0.05)
        # cbar    = fig.colorbar(subplot, cax=cb_ax)
        # cbar.set_label(r'signal [a. u.]')

        add_scalebar(
            px=1e-1,  # 3.2*pixel_size_mm[1], # 5*pixel_size_mm[0],
            units='mm',
            fixed_value=None,
            color='w',
            box_alpha=0.0,
            box_color='w',
            location='lower right',
            frameon=True,
            pad=0.075,
            length_fraction=None,
            border_pad=0.075,
            fontsize=12,
        )

        handles, labels = ax.get_legend_handles_labels()
        # handles.append(patches.Patch(color='none', label=animal_ID))
        # handles.append(patches.Patch(color='none', label=cell_line))
        # leg = ax.legend(loc='center left', bbox_to_anchor=(0, 1.2), handles=handles, title=animal_ID + '\n' + cell_line)
        leg = ax.legend(
            loc='center left',
            bbox_to_anchor=(0, 1.3),
            handles=handles,
            title=animal_ID
            + '\n'
            + cell_line
            + '\n'
            + 't = {1:02d} min {2:02d} s'.format(
                int(time_s // 3600),
                int((time_s % 3600) // 60),
                int(time_s % 60),
                int(((time_s % 60) * 1000) % 1000),
            ),
        )
        leg.get_frame().set_edgecolor('black')

        insetslicelist = []
        colorlist = ['r']

        # for i, patchcenter in enumerate(patchcenterlist):
        # insetslicelist.append(add_patch(
        # image=image_stack[..., slice],
        # patchcenter=patchcenter,
        # patchbox=patchboxlist[i],
        # ax=None,
        # sl=None,
        # color=colorlist[i],
        # lw=1,
        # fc='none',
        # out=True
        # ))

        # axin = ax.inset_axes([1.05, 0.0, (0.6*image_shape[0]*patchboxlist[0][1]/patchboxlist[0][0])/image_shape[1], 0.6])

        # inset = axin.imshow(image_stack[:, :, slice],
        # vmin=vmin, vmax=vmax,
        # cmap='gray',
        # extent=get_extent(Anatomical_BrukerExp)) # (xleft, xright, ybottom, ytop)
        # # aspect=pixel_size_mm[1]/pixel_size_mm[0]) # dy/dx

        # axin.set_xlim([insetslicelist[0][1].start, insetslicelist[0][1].stop])
        # axin.set_ylim([insetslicelist[0][0].start, insetslicelist[0][0].stop])

        # axin.set_xticks([])
        # axin.set_yticks([])

        # axin.spines['bottom'].set_color(colorlist[0])
        # axin.spines['top'].set_color(colorlist[0])
        # axin.spines['left'].set_color(colorlist[0])
        # axin.spines['right'].set_color(colorlist[0])

        # add_scalebar(
        # px=5*pixel_size_mm[0],
        # ax=axin,
        # units='mm',
        # fixed_value=1,
        # color='w',
        # box_alpha=0.0,
        # box_color='w',
        # location='lower right',
        # frameon=True,
        # pad=0.075,
        # length_fraction=None,
        # border_pad=0.075,
        # fontsize=12
        # )

        plt.tight_layout()

        plt.savefig(
            plotsavepath
            + r'/'
            + plotname
            + Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient']
            + '_slice_{}'.format(slice)
        )


def make_gif(frame_folder, gif_name, duration=50):

    # frames = [
        # Image.open(image)
        # for image in sorted(glob.glob(f'{frame_folder}/*.png'), key=os.path.getmtime)
    # ]
    # frame_one = frames[0]
    # # frame_one.save(gif_name + '.gif', format='GIF', append_images=frames, save_all=True, duration=250, loop=0)
    # frame_one.save(
        # gif_name + '.gif',
        # format='GIF',
        # append_images=frames,
        # save_all=True,
        # duration=duration,
        # loop=0,
    # )
    
    with imageio.get_writer(gif_name + '.gif', mode='I', duration=duration*1e-3) as writer:
        for filename in sorted(glob.glob(f'{frame_folder}/*.png')):
            image = imageio.imread(filename)
            writer.append_data(image)


def plot_slice_orientation(
    spec_BrukerExp, Anatomical_BrukerExp, plotsavepath, plotname, figsize=None
):
    # Coordinate system
    if Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'axial':
        er = np.array([1, 0, 0])
        ep = np.array([0, -1, 0])
        es = np.array([0, 0, -1])

    elif Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'coronal':
        er = np.array([0, 0, -1])
        ep = np.array([1, 0, 0])
        es = np.array([0, -1, 0])

    elif Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] == 'sagittal':
        er = np.array([0, -1, 0])
        ep = np.array([0, 0, -1])
        es = np.array([1, 0, 0])

    # spec slice normal vector [mm]
    n = spec_BrukerExp.method.get('SliceVec'[0], None)
    # if parameter SliceVec not defined:
    if n is None:
        # Get  acquisition matrix:
        n = spec_BrukerExp.acqp.get('ACQ_grad_matrix', None)
        # if acquisition matrix was defined:
        if n is not None:
            n = n[0][2]
        else:
            raise ValueError('Not able to find normal Vector!')

    # 3D plot
    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(111, projection='3d')

    ax.quiver(
        0,
        0,
        0,
        er[0],
        er[2],
        er[1],
        color='red',
        label=Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] + ' image read dir',
    )
    ax.quiver(
        0,
        0,
        0,
        ep[0],
        ep[2],
        ep[1],
        color='green',
        label=Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] + ' image phase dir',
    )
    ax.quiver(
        0,
        0,
        0,
        es[0],
        es[2],
        es[1],
        color='blue',
        label=Anatomical_BrukerExp.method['PVM_SPackArrSliceOrient'] + ' image slice dir',
    )
    ax.quiver(0, 0, 0, n[0], n[2], n[1], color='black', label='normal dir')

    # spec sample of slice orientation
    zz, xx = np.meshgrid(
        np.linspace(-0.5, 0.5, 1000), np.linspace(-0.5, 0.5, 1000), indexing='xy'
    )

    yy = plane_y(n, np.zeros(3), xx, zz)
    # print(yy)

    if np.any(yy == np.inf):
        xx, yy = np.meshgrid(
            np.linspace(-0.5, 0.5, 1000), np.linspace(-0.5, 0.5, 1000), indexing='xy'
        )
        zz = plane_z(n, np.zeros(3), xx, yy)
        # print(zz)

        if np.any(zz == np.inf):
            yy, zz = np.meshgrid(
                np.linspace(-0.5, 0.5, 1000),
                np.linspace(-0.5, 0.5, 1000),
                indexing='xy',
            )
            xx = plane_x(n, np.zeros(3), yy, zz)
            # print(xx)

    ax.plot_surface(xx, zz, np.where(np.abs(yy) > 1, np.nan, yy), alpha=0.5)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    ax.dist = 13

    leg = ax.legend(framealpha=1)
    # leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    leg.get_frame().set_edgecolor('black')

    plt.tight_layout()
    print('Figure size [in]: ', fig.get_size_inches())

    try:
        plt.savefig(
            os.path.join(
                plotsavepath,
                plotname
                + '.png',
            )
        )
    except:
        plt.savefig(
            os.path.join(
                plotsavepath,
                '_' + plotname,
            )
        )

