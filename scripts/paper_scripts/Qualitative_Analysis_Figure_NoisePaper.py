# Standard imports
import matplotlib.pyplot as plt
import pathlib
import imageio
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Torch imports
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

# Lion imports
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_benchmarking_experiments as ct_benchmarking
from ts_algorithms import fdk, sirt, tv_min, nag_ls
from LION.CTtools.ct_utils import make_operator

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import os

from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np



def plot_imgs(file_list, rows, savename, width=None, vmin=None, vmax=None):
    if width is None:
        width = plt.rcParams["figure.figsize"][0]

    rows = rows
    cols = int(np.ceil(len(file_list)/rows))

    fig = plt.figure(figsize=(width, rows / cols * width))

    grid = ImageGrid(
        fig,
        111,                # similar to subplot(111)
        nrows_ncols=(rows, cols),
        axes_pad=0.025,       # pad between axes in inch.
        label_mode="all",   # all axes get a label (which we remove later)
    )

    experiment_labels = ["Full Data", "Limited 120", "Limited 90", "Limited 60",
    "Sparse 360", "Sparse 120", "Sparse 60", "Low Dose", "Beam Hardening"]

    algo_names = ["FBP", "NAG", "CHP",
    "FBPUNet", "FBPMSDNet", "FBPDnCNN",
    "LG", "LGTV", "LPD",
    "AR", "TDV", "ACR",
    "DnCNN-PnP", "DRUNet-PnP", "GS-PnP"]

    for i, file_name in enumerate(file_list):
        img = np.load(file_name)
        #if i < 9:
        #    label = experiment_labels[i] #f"Img {i}"
        #else:
        #    label = ""

        ax = grid[i]
        ax.axis('off')
        for s in ax.spines.values():
            s.set_visible(False)

        #ax.set_title(label)
        #if i%9 == 0:
        #    print(algo_names[int(i/9)])
        #    ax.set_ylabel(algo_names[int(i/9)])
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)

    plt.savefig(savename, dpi=300)
    plt.tight_layout()
    plt.show()


def add_zoom_bubble(axes_image,
                    inset_center=(0.75, 0.75),
                    inset_radius=0.2,
                    roi=(0.2, 0.2),
                    zoom=2,
                    edgecolor="red",
                    linewidth=2,
                    alpha=1.0,
                    **kwargs):
    """Add a zoom bubble to an AxesImage

    All coordinates are in (x,y) form where the lowerleft corner is (0, 0)
    and the topright corner is (1,1).

    :param axes_image: `matplotlib.image.AxesImage`

        A return value of `plt.imshow`

    :param inset_center: `(float, float)`

        The center of the inset bubble.

    :param inset_radius: `float`

        The radius of the inset bubble.

    :param roi: `(float, float)`

        The center of the region of interest.

    :param zoom: `float`

        The zoom factor by which the region of interest is magnified.
        The bubble of the region of interest is `zoom` times smaller
        than the bubble of the inset.

    :returns: None
    :rtype: NoneType

    """
    ax = axes_image.axes
    data = axes_image.get_array()
    roi_radius = inset_radius / zoom

    opts = dict(facecolor="none")
    opts["edgecolor"] = edgecolor
    opts["linewidth"] = linewidth
    opts["alpha"] = alpha

    # Make axis for inset bubble.
    axins = ax.inset_axes(
        [
            inset_center[0] - inset_radius,
            inset_center[1] - inset_radius,
            2 * inset_radius,
            2 * inset_radius
        ],
        transform=ax.transAxes,
    )
    axins.axis('off')

    im_inset = axins.imshow(
        data,
        cmap=axes_image.get_cmap(),
        norm=axes_image.norm,
        aspect="auto",            # unknown..
        interpolation="nearest",
        alpha=1.0,
        #vmin=axes_image.get_clim()[0],
        #vmax=axes_image.get_clim()[1],
        origin=axes_image.origin,
        extent=axes_image.get_extent(),
        filternorm=axes_image.get_filternorm(),
        filterrad=axes_image.get_filterrad(),
        # imlim=None,             # imlim is Deprecated
        resample=None,          # No clue..
        url=None,               # No clue..
        data=None,              # This is another way to present args..
    )
    # Show region of interest of the original image
    # This must be in data coordinates.
    axis_to_data = ax.transAxes + ax.transData.inverted()
    lower_left = axis_to_data.transform(np.array(roi) - roi_radius)
    top_right = axis_to_data.transform(np.array(roi) + roi_radius)
    axins.set_xlim(lower_left[0], top_right[0])
    axins.set_ylim(lower_left[1], top_right[1])

    # Clip inset axis to circle and show circle
    patch = patches.Circle(
        inset_center,
        radius=inset_radius,
        transform=ax.transAxes,
        zorder=axins.get_zorder() + 1,               # Show above inset
        **opts,
    )
    im_inset.set_clip_path(patch)
    ax.add_patch(patch)

    # Show bubble at region of interest
    ax.add_patch(
        patches.Circle(
            roi,
            radius=roi_radius,
            transform=ax.transAxes,
            **opts,
        )
    )

    # Draw connection between the two bubbles:
    inset_center = np.array(inset_center)
    roi_center = np.array(roi)
    v = inset_center - roi_center
    d = np.linalg.norm(v)

    ax.add_patch(
        patches.ConnectionPatch(
            # edge of roi bubble
            roi_center + roi_radius / d * v,
            # edge of inset bubble
            roi_center + (d - inset_radius) / d * v,
            'axes fraction', 'axes fraction',
            axesA=ax, axesB=ax, arrowstyle="-",
            **opts
        )
    )


"""
Main function part
"""

device = torch.device("cuda:0")

# Load .npy files

savepath = "/export/scratch3/mbk/LION/noise_slices/"

algo_names = ["UNet_ExpNoise", "UNet_ArtNoise",
"MSD_ExpNoise", "MSD_ArtNoise",
"FBPUNet_ExpNoise", "FBPUNet_ArtNoise",
"FBPMSDNet_ExpNoise", "FBPMSDNet_ArtNoise"]

experiment_names = ["ExpNoise_test", "ArtNoise_test"]

slice_names = []

slice_indices = [205,366,408] #182,220,257

for algo in algo_names:
    for index in slice_indices:
        slice_name = str(savepath+algo+"_ExpNoise_test_slice"+str(index)+".npy")
        slice_names.append(slice_name)
        slice_name = str(savepath+algo+"_ArtNoise_test_slice"+str(index)+".npy")
        slice_names.append(slice_name)

file_list = slice_names

plot_imgs(
    file_list,
    rows = 8,
    vmin=0,
    vmax=0.010,
    width=20,
    savename = "/export/scratch3/mbk/LION/noise_slices/Qualitative_Analysis_Figure_NoisePaper.pdf"
)

import matplotlib.patches as patches

axes = plt.gcf().axes
for ax in axes[:len(file_list)]:
    add_zoom_bubble(ax.images[0], roi=(.325, .45), zoom=2,linewidth=0.5,
    inset_center=(0.8, 0.2), inset_radius=.2)

plt.savefig("/export/scratch3/mbk/LION/noise_slices/Qualitative_Analysis_Figure_NoisePaper.pdf", dpi=300)
