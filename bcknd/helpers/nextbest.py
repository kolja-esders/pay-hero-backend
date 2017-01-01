import numpy as np
import _pickle as pickle

from helpers.cnn import FeatureExtr
from helpers.config import *
from helpers.io import load_res_img

from obrec.models import Product, Picture


class NextBest(object):
	"""
	Finds the next best stuff given some features
	"""

	# Singleton stuff
	__instance = None

	def __new__(cls):
		if NextBest.__instance is None:
			NextBest.__instance = object.__new__(cls)
			#Setup only the first time the object is initiallized
			NextBest.__instance.setup()
		return NextBest.__instance


	def setup(self):

		self.pic_list = pickle.load(open(NEAREST_NEIGH_PATH + "pic_list.p", "rb"))
		self.kdt = pickle.load(open(NEAREST_NEIGH_PATH + "tree.p", "rb"))


	def find_next_best_product(self, img):

		feat_extr = FeatureExtr()

		img_feat = feat_extr.get_features([img])[0]
		img_feat = img_feat/ np.linalg.norm(img_feat)

		nn_dist, nn_ind = self.kdt.query([img_feat], k=2, return_distance=True)

		next_img_ind = nn_ind[0][1]
		extern_id = self.pic_list[next_img_ind][0]

		# prod = Product.objects.filter(external_id=extern_id)[0]
		# print(prod.name)

		return extern_id

