import pickle

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IMAGES_PATH = '/home/ljulius/data/test/testimg_2d.png'
class Augmenter:
    def __init__(self):
        pass

    def augment(self, data):
        for filename in data:
            self.read_npz(filename)
            break

    def read_npz(self, filename):
        img = np.load(filename)
        # print(img.files)  # data, seg
        data = img["data"]
        seg = img["seg"]
        imgplot = plt.imshow(seg[0][50])
        plt.savefig(f"/home/ljulius/data/test/testimg_2d.png")

    def batch_npz_to_npy(self, file_names):
        imgs = []
        for filename in file_names:
            data, seg = self.npz_to_npy(filename)
            img = self.npy_to_image(data)
            print(data.shape, seg.shape)

            imgs.append(img)
        return imgs

    def npy_to_image(self, npy):
        img = Image.fromarray(npy[0][0]).convert('RGB')
        return img

    def npz_to_npy(self, filename):
        img = np.load(filename)
        data = img["data"]
        seg = img["seg"]
        return data, seg

		def display_image_grid(images_filepaths, predicted_labels=(), cols=5):
			rows = len(images_filepaths) // cols
			figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
			for i, image_filepath in enumerate(images_filepaths):
					image = cv2.imread(image_filepath)
					image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
					true_label = os.path.normpath(image_filepath).split(os.sep)[-2]
					predicted_label = predicted_labels[i] if predicted_labels else true_label
					color = "green" if true_label == predicted_label else "red"
					ax.ravel()[i].imshow(image)
					ax.ravel()[i].set_title(predicted_label, color=color)
					ax.ravel()[i].set_axis_off()
			plt.tight_layout()
			plt.show()
               
    def read_pkl(self, filename):
        with open(filename, "rb") as file:
            metadata = pickle.load(file)
        return metadata
        # print(data.keys())  # dict_keys(['sitk_stuff', 'spacing', 'shape_before_cropping', 'bbox_used_for_cropping', 'shape_after_cropping_and_before_resampling', 'class_locations'])


if __name__ == "__main__":
    augmenter = Augmenter()
    # augmenter.augment(
    #     [
    #         "/home/ljulius/algorithm/nnunet/nnUNet_preprocessed/Dataset501_RadboudumcBone/nnUNetPlans_2d/bone_00002_lesion_01.npz",
    #     ]
    # )

    # augmenter.read_pkl(
    #     "/home/ljulius/algorithm/nnunet/nnUNet_preprocessed/Dataset501_RadboudumcBone/nnUNetPlans_2d/bone_00002_lesion_01.pkl"
    # )
    file_names = [
        "/home/ljulius/algorithm/nnunet/nnUNet_preprocessed/Dataset501_RadboudumcBone/nnUNetPlans_2d/bone_00002_lesion_01.npz",
    ]
    images = augmenter.batch_npz_to_npy(file_names)
    images[0].save(IMAGES_PATH)
