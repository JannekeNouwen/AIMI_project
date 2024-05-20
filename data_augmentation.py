import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Augmenter:
	def __init__(self):
		pass

	def augment(self, data):
		for filename in data:
			self.rotate(filename)
			break

	def rotate(self, filename):
		img = np.load(filename)
		# print(img.files)  # data, seg
		data = img["data"]
		seg = img["seg"]
		for i in [1, 20, 50, 80, 100, 120]:
			imgplot = plt.imshow(seg[0][i])
			plt.savefig(f"/home/ljulius/data/test/testimg_seg_{i}.png")


if __name__ == "__main__":
	augmenter = Augmenter()
	augmenter.augment([
		"/home/ljulius/algorithm/nnunet/nnUNet_preprocessed/Dataset501_RadboudumcBone/nnUNetPlans_3d_fullres/bone_00002_lesion_01.npz",
		"/home/ljulius/algorithm/nnunet/nnUNet_preprocessed/Dataset501_RadboudumcBone/nnUNetPlans_3d_fullres/bone_00002_lesion_02.npz",
		"/home/ljulius/algorithm/nnunet/nnUNet_preprocessed/Dataset501_RadboudumcBone/nnUNetPlans_3d_fullres/bone_00002_lesion_03.npz",
	])

	# augmenter.augment([
	# 	"/home/ljulius/algorithm/nnunet/nnUNet_preprocessed/Dataset501_RadboudumcBone/nnUNetPlans_3d_fullres/bone_00002_lesion_01.pkl"
	# 	"/home/ljulius/algorithm/nnunet/nnUNet_preprocessed/Dataset501_RadboudumcBone/nnUNetPlans_3d_fullres/bone_00002_lesion_02.pkl"
	# 	"/home/ljulius/algorithm/nnunet/nnUNet_preprocessed/Dataset501_RadboudumcBone/nnUNetPlans_3d_fullres/bone_00002_lesion_03.pkl"
	# ])