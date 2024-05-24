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


class Augmenter:
    def __init__(self, augmentation_config):
        self.transform = A.Compose(augmentation_config)

    def augment_image_and_segmentation(self, image, segmentation, segmentation_gt):
        logging.info("Starting augmentation")

        assert image.ndim == 3, "Image must be 3D"
        assert segmentation.ndim == 3, "Segmentation must be 3D"
        assert segmentation_gt.ndim == 3, "Segmentation GT must be 3D"
        assert image.size > 0, "Image is empty"
        assert segmentation.size > 0, "Segmentation is empty"
        assert segmentation_gt.size > 0, "Segmentation GT is empty"
        assert (
            image.shape == segmentation.shape == segmentation_gt.shape
        ), f"Image, segmentation, segmentation_gt shape mismatch: {image.shape[:2]} vs {segmentation.shape[:2]} vs {segmentation_gt.shape[:2]}"

        logging.info(f"Image shape before conversion: {image.shape}")
        logging.info(f"Segmentation shape before conversion: {segmentation.shape}")
        logging.info(f"Segmentation shape before conversion: {segmentation_gt.shape}")

        for i in range(image.shape[0]):
            augmented = self.transform(
                image=image[i], mask=segmentation[i], segmentation_gt=segmentation_gt[i]
            )
            augmented_image = augmented["image"]
            augmented_segmentation = augmented["mask"]
            augmented_segmentation_gt = augmented["segmentation_gt"]

            image[i] = augmented_image
            segmentation[i] = augmented_segmentation
            segmentation_gt[i] = augmented_segmentation_gt

        assert image.ndim == 3, "Image must be 3D after augmentation"
        assert segmentation.ndim == 3, "Segmentation must be 3D after augmentation"
        assert (
            segmentation_gt.ndim == 3
        ), "Segmentation GT must be 3D after augmentation"
        assert image.size > 0, "Image is empty after augmentation"
        assert segmentation.size > 0, "Segmentation is empty after augmentation"
        assert segmentation_gt.size > 0, "Segmentation GT is empty after augmentation"
        assert (
            image.shape == segmentation.shape == segmentation_gt.shape
        ), f"Image, segmentation, segmentation_gt shape mismatch: {image.shape} vs {segmentation.shape} vs {segmentation_gt.shape}, after augmentation"

        logging.info(f"Image shape after conversion: {image.shape}")
        logging.info(f"Segmentation shape after conversion: {segmentation.shape}")
        logging.info(f"Segmentation shape after conversion: {segmentation_gt.shape}")

        logging.info(
            f"Augmentation complete: {image.shape}, {segmentation.shape}, {segmentation_gt.shape}"
        )
        return image, segmentation, segmentation_gt

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
        original_gt_segmentations = []
        augmented_gt_segmentations = []
        logging.info(f"Processing {len(npz_file_paths)} files")
        for i, file_path in enumerate(npz_file_paths):
            # data and seg are lists of 2D images, shape 1, 128, 256, 256
            data, seg = dataset.load_npz(file_path)
            original_image = data[0]
            original_images.append(original_image)
            original_mask = seg[0]
            original_masks.append(original_mask)
            gt_segmentation_file = os.path.join(
                GT_SEGMENTATIONS_PATH,
                os.path.basename(file_path).replace(".npz", ".nii.gz"),
            )
            # segmentation_gt is exactly opposite
            segmentation_gt = dataset.load_nii(gt_segmentation_file)

            original_gt_segmentation = segmentation_gt[:, :, :, 0]
            logging.info(f"Segmentation GT shape: {original_gt_segmentation.shape}")
            original_gt_segmentation = np.moveaxis(original_gt_segmentation, -1, 0)
            logging.info(
                f"Segmentation GT shape after moveaxis: {original_gt_segmentation.shape}"
            )
            original_gt_segmentations.append(original_gt_segmentation)

            augmented_image, augmented_mask, augmented_gt_segmentation = (
                self.augment_image_and_segmentation(
                    deepcopy(original_image),
                    deepcopy(original_mask),
                    deepcopy(original_gt_segmentation),
                )
            )

            augmented_images.append(augmented_image)
            augmented_masks.append(augmented_mask)
            augmented_gt_segmentations.append(augmented_gt_segmentation)

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
            os.makedirs(os.path.dirname(output_segmentation_gt_path), exist_ok=True)
            logging.info(
                f"Flipping segmentation GT shape: {augmented_gt_segmentation.shape}"
            )
            augmented_gt_segmentation = np.moveaxis(augmented_gt_segmentation, 0, -1)
            logging.info(
                f"Flipped segmentation GT shape: {augmented_gt_segmentation.shape}"
            )
            dataset.save_nii(
                np.expand_dims(augmented_gt_segmentation, -1),
                output_segmentation_gt_path,
            )

            pkl_file_path = file_path.replace(".npz", ".pkl")
            if os.path.exists(pkl_file_path):
                metadata = dataset.load_pkl(pkl_file_path)
                output_pkl_path = os.path.join(
                    output_path_img, os.path.relpath(pkl_file_path, input_path)
                )
                os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
                dataset.save_pkl(metadata, output_pkl_path)
        slice_number_to_show = np.linspace(
            0, original_images[0].shape[0] - 1, SLICES_TO_SHOW, dtype=int
        )
        make_image_mask_seg_plot(
            original_images,
            original_masks,
            original_gt_segmentations,
            "Original Images",
            "original_images",
            save_plots,
            slice_number_to_show,
        )

        make_image_mask_seg_plot(
            augmented_images,
            augmented_masks,
            augmented_gt_segmentations,
            "Augmented Images",
            "augmented_images",
            save_plots,
            slice_number_to_show,
        )
