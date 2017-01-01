import numpy as np
from scipy.misc import imread, imresize


def load_res_img(img_path):
	img_orig = (imread(img_path)[:, :, :3]).astype(np.float32)
	img_resize = imresize(img_orig, (227, 227))

	return img_resize