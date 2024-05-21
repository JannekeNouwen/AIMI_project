import logging
import os
import pickle

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from constants import *

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ImageDataset:
    def __init__(self, input_path, gt_segmentations_path):
        self.input_path = input_path
        self.gt_segmentations_path = gt_segmentations_path

    def load_npz(self, filename):
        logging.info(f"Loading NPZ file: {filename}")
        data = np.load(filename)
        return data["data"], data["seg"]

    def load_pkl(self, filename):
        logging.info(f"Loading PKL file: {filename}")
        with open(filename, "rb") as file:
            metadata = pickle.load(file)
        print(metadata)
        return metadata

    def save_pkl(self, metadata, filename):
        logging.info(f"Saving PKL file: {filename}")
        with open(filename, "wb") as file:
            pickle.dump(metadata, file)

    def load_nii(self, filename):
        logging.info(f"Loading NII file: {filename}")
        nii = nib.load(filename)
        return nii.get_fdata()

    def save_nii(self, data, filename):
        logging.info(f"Saving NII file: {filename}")
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, filename)

    def show_or_save_images(
        self, images, title, file_prefix, save_plots, is_mask=False
    ):
        logging.info(f"Showing or saving images: {title}")
        plt.figure(figsize=(10, 10))
        for i, img in enumerate(images[:NUM_IMAGES_TO_SHOW]):

            # if img.ndim == 3 and img.shape[-1] not in [1, 3]:
            #     img = img[..., img.shape[-1] // 2]
            # if img.ndim == 4 and img.shape[-1] == 1:
            #     img = img[..., 0, 0]

            plt.subplot(1, NUM_IMAGES_TO_SHOW, i + 1)
            plt.imshow(img, cmap="gray" if is_mask else None)
            plt.axis("off")
        plt.suptitle(title)

        if save_plots:
            plt.savefig(
                os.path.join(
                    SCRIPT_PLOTS_PATH,
                    f"{file_prefix}_{title.replace(' ', '_').lower()}.png",
                )
            )
        else:
            plt.show()
        plt.close()

    def get_file_paths(self, extension):
        file_paths = []
        for root, _, files in os.walk(self.input_path):
            for fname in files:
                if fname.endswith(extension):
                    file_paths.append(os.path.join(root, fname))
        return file_paths
