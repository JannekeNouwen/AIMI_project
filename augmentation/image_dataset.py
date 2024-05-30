import logging
import os
import pickle
import random

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

    def get_file_paths(self, extension):
        file_paths = []
        for root, _, files in os.walk(self.input_path):
            for fname in files:
                if fname.endswith(extension):
                    file_paths.append(os.path.join(root, fname))
        random.shuffle(file_paths)
        if NUM_IMAGES_TO_PROCESS != -1:
          file_paths = file_paths[:NUM_IMAGES_TO_PROCESS]
        return file_paths

