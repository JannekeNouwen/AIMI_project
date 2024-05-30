import os

import matplotlib.pyplot as plt
import numpy as np
from constants import N_SLICES_TO_SHOW, SCRIPT_PLOTS_PATH
from matplotlib.colors import LinearSegmentedColormap

yellow_cmap = LinearSegmentedColormap.from_list("yellow_cmap", ["yellow", "yellow"])


def make_image_mask_seg_plot(
    images,
    masks,
    augmented_images,
    augmented_masks,
    transformation_done,
    title,
    file_prefix,
    save_plots,
):
    num_images = len(images)

    for img_idx in range(num_images):
        fig, axes = plt.subplots(2, N_SLICES_TO_SHOW, figsize=(15, 6))
        fig.suptitle(f"{title} - Image {img_idx + 1}", fontsize=16)

        mask_sizes = np.array([np.sum(mask) for mask in masks[img_idx]])

        sorted_indices = np.argsort(mask_sizes)[::-1]
        selected_slices = []
        min_distance = 5

        for idx in sorted_indices:
            if len(selected_slices) >= N_SLICES_TO_SHOW:
                break
            if all(abs(idx - s) >= min_distance for s in selected_slices):
                selected_slices.append(idx)

        selected_slices = sorted(selected_slices)

        for idx, slice_idx in enumerate(selected_slices):

            img = images[img_idx][slice_idx]
            mask = masks[img_idx][slice_idx]
            transformation = transformation_done[img_idx][slice_idx]
            augmented_img = augmented_images[img_idx][slice_idx]
            augmented_mask = augmented_masks[img_idx][slice_idx]
            axes[0, idx].imshow(img, cmap="gray")
            axes[0, idx].imshow(
                np.ma.masked_where(mask == 0, mask), cmap=yellow_cmap, alpha=0.5
            )
            axes[0, idx].axis("off")
            axes[0, idx].set_title(f"Original Slice {slice_idx + 1}")

            axes[1, idx].imshow(augmented_img, cmap="gray")
            axes[1, idx].imshow(
                np.ma.masked_where(augmented_mask == 0, augmented_mask),
                cmap=yellow_cmap,
                alpha=0.5,
            )
            axes[1, idx].axis("off")

            transformation_str = "\n".join(transformation)
            axes[1, idx].set_title(f"Augmentations:\n{transformation_str}")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_plots:
            os.makedirs(SCRIPT_PLOTS_PATH, exist_ok=True)
            plt.savefig(
                os.path.join(
                    SCRIPT_PLOTS_PATH, f"{file_prefix}_image_{img_idx + 1}.png"
                )
            )
