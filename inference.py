import os

import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def main(nnUNet_raw, nnUNet_results):

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda", 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, "Dataset003_Liver/nnUNetTrainer__nnUNetPlans__3d_lowres"),
        use_folds=(0,),
        checkpoint_name="checkpoint_final.pth",
    )

    predictor.predict_from_files(
        join(nnUNet_raw, "Dataset003_Liver/imagesTs"),
        join(nnUNet_raw, "Dataset003_Liver/imagesTs_predlowres"),
        save_probabilities=False,
        overwrite=False,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )


if __name__ == "__main__":
    nnUNet_results = os.environ.get("nnUNet_results")
    nnUNet_results = (
        nnUNet_results if nnUNet_results.endswith("/") else f"{nnUNet_results}/"
    )
    print(f"Setting nnUNet_results to {nnUNet_results}")

    nnUNet_raw = os.environ.get("nnUNet_raw")
    nnUNet_raw = nnUNet_raw if nnUNet_raw.endswith("/") else f"{nnUNet_raw}/"
    print(f"Setting nnUNet_raw to {nnUNet_raw}")

    dataset_id = str(os.environ.get("dataset_id"))
    dataset_id = "0" * (3 - len(dataset_id)) + dataset_id
    print(f"Setting dataset_id to {dataset_id}")

    main(nnUNet_raw, nnUNet_results, dataset_id)
