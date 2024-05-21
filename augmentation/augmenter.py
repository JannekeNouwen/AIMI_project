import logging
import os

import albumentations as A
import numpy as np
from constants import *

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Augmenter:
    def __init__(self, augmentation_config):
        self.transform = A.Compose(augmentation_config)

    def augment_image_and_segmentation(self, image, segmentation):
        logging.info("Starting augmentation")

        assert image.size > 0, "Image is empty"
        assert segmentation.size > 0, "Segmentation is empty"
        assert (
            image.shape[:2] == segmentation.shape[:2]
        ), f"Image and segmentation shape mismatch: {image.shape[:2]} vs {segmentation.shape[:2]}"

        if segmentation.ndim == 3:
            segmentation = segmentation[..., 0]

        logging.info(f"Segmentation shape after conversion: {segmentation.shape}")

        augmented = self.transform(image=image, mask=segmentation)
        augmented_image = augmented["image"]
        augmented_segmentation = augmented["mask"]

        if augmented_image.shape[-1] == 1:
            augmented_image = augmented_image[..., 0]

        logging.info(
            f"Augmentation complete: {augmented_image.shape}, {augmented_segmentation.shape}"
        )
        return augmented_image, augmented_segmentation

    def process_and_augment_images(
        self, dataset, input_path, output_path_img, output_path_seg, save_plots
    ):
        raise NotImplementedError("This method should be implemented by subclasses")


class Augmenter2D(Augmenter):
    def process_and_augment_images(
        self, dataset, input_path, output_path_img, output_path_seg, save_plots
    ):
        logging.info("Processing 2D images")
        npz_file_paths = dataset.get_file_paths(".npz")
        original_images = []
        augmented_images = []
        original_masks = []
        augmented_masks = []

        for i, file_path in enumerate(npz_file_paths[:NUM_IMAGES_TO_SHOW]):
            data, seg = dataset.load_npz(file_path)
            original_image = data[0][0]
            original_images.append(original_image)
            original_mask = seg[0][0]
            # gt_segmentation_file = os.path.join(
            #     GT_SEGMENTATIONS_PATH,
            #     os.path.basename(file_path).replace(".npz", ".nii.gz"),
            # )
            # segmentation = (
            #     dataset.load_nii(gt_segmentation_file)
            #     if os.path.exists(gt_segmentation_file)
            #     else np.zeros_like(original_image)
            # )
            original_masks.append(original_mask)

        dataset.show_or_save_images(
            original_images, "Original Images", "original_images", save_plots
        )
        dataset.show_or_save_images(
            original_masks, "Original Masks", "original_masks", save_plots, is_mask=True
        )

        for file_path in npz_file_paths:
            data, seg = dataset.load_npz(file_path)
            original_image = data[0][0]

            logging.info(f"Original image shape: {original_image.shape}")

            gt_segmentation_file = os.path.join(
                GT_SEGMENTATIONS_PATH,
                os.path.basename(file_path).replace(".npz", ".nii.gz"),
            )
            segmentation = (
                dataset.load_nii(gt_segmentation_file)
                if os.path.exists(gt_segmentation_file)
                else np.zeros_like(original_image)
            )

            logging.info(
                f"Segmentation shape before augmentation: {segmentation.shape}"
            )

            augmented_image, augmented_segmentation = (
                self.augment_image_and_segmentation(original_image, segmentation)
            )

            logging.info(f"Augmented image shape: {augmented_image.shape}")

            if len(augmented_images) < NUM_IMAGES_TO_SHOW:
                augmented_images.append(augmented_image)
                augmented_masks.append(augmented_segmentation)

            output_file_path = os.path.join(
                output_path_img, os.path.relpath(file_path, input_path)
            )
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            np.savez(
                output_file_path, data=np.expand_dims(augmented_image, axis=0), seg=seg
            )

            output_segmentation_path = os.path.join(
                output_path_seg,
                os.path.relpath(gt_segmentation_file, GT_SEGMENTATIONS_PATH),
            )
            os.makedirs(os.path.dirname(output_segmentation_path), exist_ok=True)
            dataset.save_nii(augmented_segmentation, output_segmentation_path)

            pkl_file_path = file_path.replace(".npz", ".pkl")
            if os.path.exists(pkl_file_path):
                metadata = dataset.load_pkl(pkl_file_path)
                output_pkl_path = os.path.join(
                    output_path_img, os.path.relpath(pkl_file_path, input_path)
                )
                os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
                dataset.save_pkl(metadata, output_pkl_path)

        dataset.show_or_save_images(
            augmented_images, "Augmented Images", "augmented_images", save_plots
        )
        dataset.show_or_save_images(
            augmented_masks,
            "Augmented Masks",
            "augmented_masks",
            save_plots,
            is_mask=True,
        )

        for file_path in npz_file_paths[:NUM_IMAGES_TO_SHOW]:
            data, seg = dataset.load_npz(file_path)
            original_image = data[0][0]
            augmented_image = augmented_images.pop(0)
            assert (
                original_image.shape == augmented_image.shape
            ), f"Shape mismatch after augmentation: {original_image.shape} vs {augmented_image.shape}"
            assert (
                original_image.dtype == augmented_image.dtype
            ), "Type mismatch after augmentation"
