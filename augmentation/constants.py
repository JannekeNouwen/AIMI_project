import os

import albumentations as A

INPUT_PATH_BASE = "nnUnet_preprocessed/Dataset501_RadboudumcBone"
INPUT_PATH_2D = os.path.join(INPUT_PATH_BASE, "nnUNetPlans_2d")
INPUT_PATH_3D = os.path.join(INPUT_PATH_BASE, "nnUNetPlans_3d_fullres")
OUTPUT_PATH_BASE = "nnUnet_preprocessed_augment/Dataset501_RadboudumcBone"
GT_SEGMENTATIONS_PATH = os.path.join(INPUT_PATH_BASE, "gt_segmentations")
OUTPUT_PATH_2D = os.path.join(OUTPUT_PATH_BASE, "nnUNetPlans_2d")
OUTPUT_PATH_3D = os.path.join(OUTPUT_PATH_BASE, "nnUNetPlans_3d_fullres")
OUTPUT_PATH_SEG = os.path.join(OUTPUT_PATH_BASE, "gt_segmentations")
SCRIPT_PLOTS_PATH = "script_plots"
NUM_IMAGES_TO_SHOW = 3
SLICES_TO_SHOW = 5
SAVE_PLOTS = True
os.makedirs(OUTPUT_PATH_2D, exist_ok=True)
os.makedirs(OUTPUT_PATH_3D, exist_ok=True)
os.makedirs(OUTPUT_PATH_SEG, exist_ok=True)
os.makedirs(SCRIPT_PLOTS_PATH, exist_ok=True)
AUGMENTATION_CONFIG = [
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    A.Normalize(mean=(0.5,), std=(0.5,)),
]