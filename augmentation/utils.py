import logging
import os
import shutil

from augmenter import Augmenter2D
from constants import (
    AUGMENTATION_CONFIG,
    GT_SEGMENTATIONS_PATH,
    INPUT_PATH_BASE,
    OUTPUT_PATH_BASE,
    SAVE_PLOTS,
)
from image_dataset import ImageDataset


def compare_directory_structure(dir1, dir2):
    dir1_files = sorted(
        [
            os.path.relpath(os.path.join(root, file), dir1)
            for root, _, files in os.walk(dir1)
            for file in files
        ]
    )
    dir2_files = sorted(
        [
            os.path.relpath(os.path.join(root, file), dir2)
            for root, _, files in os.walk(dir2)
            for file in files
        ]
    )
    assert (
        dir1_files == dir2_files
    ), f"Directory structures do not match: {dir1_files} vs {dir2_files}"


def copy_over_jsons(
    json_files=["dataset.json", "dataset_fingerprint.json", "nnUNetPlans.json"]
):
    for json_file in json_files:
        src_file = os.path.join(INPUT_PATH_BASE, json_file)
        dst_file = os.path.join(OUTPUT_PATH_BASE, json_file)
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        shutil.copy(src_file, dst_file)
        logging.info(f"Copied {src_file} to {dst_file}")


def copy_over_input_images(input_path, output_path_2d):
    for root, _, files in os.walk(input_path):
        for file_name in files:
            src_file = os.path.join(root, file_name)
            dst_file = os.path.join(
                output_path_2d, os.path.relpath(src_file, input_path)
            )
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            shutil.copy(src_file, dst_file)
            logging.info(f"Copied {src_file} to {dst_file}")


def augment_and_copy_images(input_path, output_path_img, output_path_seg):
    dataset_2d = ImageDataset(input_path, GT_SEGMENTATIONS_PATH)
    augmenter_2d = Augmenter2D(AUGMENTATION_CONFIG)
    augmenter_2d.process_and_augment_images(
        dataset_2d, input_path, output_path_img, output_path_seg, SAVE_PLOTS
    )
