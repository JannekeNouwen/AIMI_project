import logging
import os
from copy import deepcopy

import albumentations as A
import numpy as np
from constants import *
from plot import make_image_mask_seg_plot

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

CACHE_FILE_NAME_PATH = "cache_filenames.npy"


def get_tranformations_from_replay(replay):
    transformations = []
    for replay_item in replay["transforms"]:
        if not replay_item["applied"]:
            continue
        transformations.append(replay_item["__class_fullname__"])
    return transformations


class Augmenter:
    def __init__(self, augmentation_config):
        self.transform = A.ReplayCompose(augmentation_config)

    def augment_image_and_segmentation(self, image, segmentation):
        logging.info("Starting augmentation")

        assert image.ndim == 3, "Image must be 3D"
        assert segmentation.ndim == 3, "Segmentation must be 3D"
        assert image.size > 0, "Image is empty"
        assert segmentation.size > 0, "Segmentation is empty"

        logging.info(f"Image shape before conversion: {image.shape}")
        logging.info(f"Segmentation shape before conversion: {segmentation.shape}")

        transformations_applied = []

        for i in range(image.shape[0]):
            transformation_done = False
            while not transformation_done:
                augmented = self.transform(image=image[i], mask=segmentation[i])
                augmented_image = augmented["image"]
                augmented_segmentation = augmented["mask"]
                replay = augmented["replay"]

                image[i] = augmented_image
                segmentation[i] = augmented_segmentation
                transformations_applied_item = get_tranformations_from_replay(replay)
                if len(transformations_applied_item) > 0:
                    transformations_applied.append(transformations_applied_item)
                    transformation_done = True

        assert image.ndim == 3, "Image must be 3D after augmentation"
        assert segmentation.ndim == 3, "Segmentation must be 3D after augmentation"

        assert image.size > 0, "Image is empty after augmentation"
        assert segmentation.size > 0, "Segmentation is empty after augmentation"
        assert (
            image.shape == segmentation.shape
        ), f"Image, segmentation shape mismatch: {image.shape} vs {segmentation.shape}, after augmentation"

        logging.info(f"Image shape after conversion: {image.shape}")
        logging.info(f"Segmentation shape after conversion: {segmentation.shape}")

        logging.info(f"Augmentation complete: {image.shape}, {segmentation.shape}")
        return image, segmentation, transformations_applied

    def process_and_augment_images(
        self, dataset, input_path, output_path_img, output_path_seg, save_plots
    ):
        raise NotImplementedError("This method should be implemented by subclasses")


class Augmenter2D(Augmenter):
    def process_and_augment_images(
        self, dataset, input_path, output_path_img, output_path_seg, save_plots, dimension = "2D"
    ):
        logging.info(f"Processing {dimension} images")
        npz_file_paths = dataset.get_file_paths(".npz")
        original_images = []
        augmented_images = []
        original_masks = []
        augmented_masks = []
        transformations = []

        # Load cached filenames
        if os.path.exists(CACHE_FILE_NAME_PATH):
            with open(CACHE_FILE_NAME_PATH, "rb") as f:
                cached_filenames = set(np.load(f))
        else:
            cached_filenames = set()

        logging.info(f"Processing {len(npz_file_paths)} files")
        for i, file_path in enumerate(npz_file_paths):
            if file_path in cached_filenames:
                logging.info(f"Skipping cached file: {file_path}")
                continue

            data, seg = dataset.load_npz(file_path)
            original_image = data[0]
            original_images.append(original_image)
            original_mask = seg[0]
            original_masks.append(original_mask)
            gt_segmentation_file = os.path.join(
                GT_SEGMENTATIONS_PATH,
                os.path.basename(file_path).replace(".npz", ".nii.gz"),
            )
            segmentation_gt = dataset.load_nii(gt_segmentation_file)

            augmented_image, augmented_mask, transformations_applied = (
                self.augment_image_and_segmentation(
                    deepcopy(original_image),
                    deepcopy(original_mask),
                )
            )

            augmented_images.append(augmented_image)
            augmented_masks.append(augmented_mask)
            transformations.append(transformations_applied)

            final_image = {
                "data": np.expand_dims(augmented_image, axis=0),
                "seg": np.expand_dims(augmented_mask, axis=0),
            }
            output_file_path = os.path.join(
                output_path_img, os.path.relpath(file_path, input_path)
            )
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            np.savez(
                output_file_path, data=np.expand_dims(final_image, axis=0), seg=seg
            )

            output_segmentation_gt_path = os.path.join(
                output_path_seg,
                os.path.relpath(gt_segmentation_file, GT_SEGMENTATIONS_PATH),
            )
            dataset.save_nii(
                segmentation_gt,
                output_segmentation_gt_path,
            )

            pkl_file_path = file_path.replace(".npz", ".pkl")
            if os.path.exists(pkl_file_path):
                metadata = dataset.load_pkl(pkl_file_path)
                metadata["transformations_applied"] = transformations_applied
                output_pkl_path = os.path.join(
                    output_path_img, os.path.relpath(pkl_file_path, input_path)
                )
                os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
                dataset.save_pkl(metadata, output_pkl_path)

            # Update cache
            cached_filenames.add(file_path)

        # Save updated cache
        with open(CACHE_FILE_NAME_PATH, "wb") as f:
            np.save(f, list(cached_filenames))

        make_image_mask_seg_plot(
            original_images,
            original_masks,
            augmented_images,
            augmented_masks,
            transformations,
            "Original and Augmented Images and Masks",
            f"augmentations_{dimension}",
            save_plots,
        )
