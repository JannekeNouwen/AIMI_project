import logging

from constants import *
from utils import (
    augment_and_copy_images,
    compare_directory_structure,
    copy_over_input_images,
    copy_over_jsons,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


if __name__ == "__main__":
    logging.info("Starting 2D image processing")

    augment_and_copy_images(INPUT_PATH_2D, OUTPUT_PATH_2D, OUTPUT_PATH_SEG)

    logging.info("Copying 3D full-resolution images")

    copy_over_input_images(INPUT_PATH_3D, OUTPUT_PATH_3D)

    copy_over_jsons(
        json_files=["dataset.json", "dataset_fingerprint.json", "nnUNetPlans.json"]
    )

    compare_directory_structure(INPUT_PATH_BASE, OUTPUT_PATH_BASE)
