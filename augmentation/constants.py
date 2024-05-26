import os

import albumentations as A

INPUT_PATH_BASE = "/home/ljulius/algorithm/nnunet/nnUNet_preprocessed/Dataset501_RadboudumcBone"
INPUT_PATH_2D = os.path.join(INPUT_PATH_BASE, "nnUNetPlans_2d")
INPUT_PATH_3D = os.path.join(INPUT_PATH_BASE, "nnUNetPlans_3d_fullres")
OUTPUT_PATH_BASE = "/home/ljulius/algorithm/nnunet/nnUNet_preprocessed_augment/Dataset501_RadboudumcBone"
GT_SEGMENTATIONS_PATH = os.path.join(INPUT_PATH_BASE, "gt_segmentations")
OUTPUT_PATH_2D = os.path.join(OUTPUT_PATH_BASE, "nnUNetPlans_2d")
OUTPUT_PATH_3D = os.path.join(OUTPUT_PATH_BASE, "nnUNetPlans_3d_fullres")
OUTPUT_PATH_SEG = os.path.join(OUTPUT_PATH_BASE, "gt_segmentations")
SCRIPT_PLOTS_PATH = "script_plots"
CACHE_FILE_NAME_PATH = "cache_filenames.npy"
NUM_IMAGES_TO_SHOW = 8
NUM_IMAGES_TO_PROCESS = -1
N_SLICES_TO_SHOW = 4
SAVE_PLOTS = True
os.makedirs(OUTPUT_PATH_2D, exist_ok=True)
os.makedirs(OUTPUT_PATH_3D, exist_ok=True)
os.makedirs(OUTPUT_PATH_SEG, exist_ok=True)
os.makedirs(SCRIPT_PLOTS_PATH, exist_ok=True)
AUGMENTATION_CONFIG = [
    A.ElasticTransform(),
    A.CoarseDropout(),
   # A.OpticalDistortion(),
    A.Sharpen(),
]
