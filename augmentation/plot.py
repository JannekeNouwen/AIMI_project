import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from constants import *
from matplotlib.patches import Patch


def make_image_mask_seg_plot(
    images,
    masks,
    gt_segmentations,
    title,
    file_prefix,
    save_plots,
    slices_to_show,
):
    logging.info(f"Showing or saving images: {title}")
    plt.figure(figsize=(20, 25))

    NUM_IMAGES_TO_SHOW = len(images)
    SLICES_TO_SHOW = len(slices_to_show)

    for i, img in enumerate(images[:NUM_IMAGES_TO_SHOW]):
        for j in range(SLICES_TO_SHOW):
            slice_number_to_show = slices_to_show[j]

            plt.subplot(
                NUM_IMAGES_TO_SHOW * 3, SLICES_TO_SHOW, i * SLICES_TO_SHOW * 3 + j + 1
            )
            plt.imshow(img[slice_number_to_show], cmap="gray")
            plt.imshow(
                np.ma.masked_where(
                    masks[i][slice_number_to_show] == 0, masks[i][slice_number_to_show]
                ),
                cmap="autumn",
                alpha=0.5,
            )
            plt.imshow(
                np.ma.masked_where(
                    gt_segmentations[i][slice_number_to_show] == 0,
                    gt_segmentations[i][slice_number_to_show],
                ),
                cmap="spring",
                alpha=0.5,
            )
            plt.title(f"{title} + Overlays\nSlice {slice_number_to_show}")
            plt.axis("off")

        plt.subplot(
            NUM_IMAGES_TO_SHOW * 3,
            SLICES_TO_SHOW,
            (i * SLICES_TO_SHOW * 3) + SLICES_TO_SHOW + 1,
        )
        plt.text(
            0.5,
            0.5,
            "Ground Truth Segmentation",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )
        plt.axis("off")

        for j in range(SLICES_TO_SHOW):
            slice_number_to_show = slices_to_show[j]

            plt.subplot(
                NUM_IMAGES_TO_SHOW * 3,
                SLICES_TO_SHOW,
                i * SLICES_TO_SHOW * 3 + SLICES_TO_SHOW + j + 1,
                facecolor="black",
            )
            plt.imshow(
                gt_segmentations[i][slice_number_to_show], cmap="spring", alpha=0.5
            )
            plt.title(f"GT Slice {slice_number_to_show}", color="white")
            plt.axis("off")

        plt.subplot(
            NUM_IMAGES_TO_SHOW * 3,
            SLICES_TO_SHOW,
            (i * SLICES_TO_SHOW * 3) + 2 * SLICES_TO_SHOW + 1,
        )
        plt.text(
            0.5,
            0.5,
            "Masks",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )
        plt.axis("off")

        for j in range(SLICES_TO_SHOW):
            slice_number_to_show = slices_to_show[j]

            plt.subplot(
                NUM_IMAGES_TO_SHOW * 3,
                SLICES_TO_SHOW,
                i * SLICES_TO_SHOW * 3 + 2 * SLICES_TO_SHOW + j + 1,
                facecolor="black",
            )
            plt.imshow(masks[i][slice_number_to_show], cmap="autumn", alpha=0.5)
            plt.title(f"Mask Slice {slice_number_to_show}", color="white")
            plt.axis("off")

    legend_elements = [
        Patch(facecolor="red", edgecolor="r", label="Mask", alpha=0.5),
        Patch(facecolor="green", edgecolor="g", label="Ground Truth", alpha=0.5),
    ]

    plt.suptitle(title)
    plt.figlegend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1))

    if save_plots:
        plt.savefig(
            os.path.join(
                SCRIPT_PLOTS_PATH,
                f"{file_prefix}.png",
            ),
            bbox_inches="tight",
        )
    else:
        plt.show()
    plt.close()
