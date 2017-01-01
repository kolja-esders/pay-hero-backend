# import sqlalchemy
# import sqlite3
#
# engine = sqlalchemy.create_engine('sqlite:///:memory:', echo=True)

import _pickle as pickle
import numpy as np
import json
import requests
import os
import urllib

from scipy.misc import imread, imresize
from sklearn.neighbors import BallTree

os.environ["DJANGO_SETTINGS_MODULE"] = "bcknd.settings"
import django

django.setup()

from obrec.models import Product, Picture

from helpers.cnn import *
from helpers.nextbest import *
from helpers.config import *


def get_zalando_stuff():
	"""Downloads product details from zalando and stores them"""

	url = "https://api.zalando.com/articles"

	page_size = 10

	for i in range(0, 100):

		querystring = {"pageSize": str(page_size), "page": str(i + 1),
		               "fields": "id,name,brand.name,units.id,units.size,units.price.value,media.images.type,media.images.mediumUrl"}

		headers = {
			'accept-encoding': "gzip",
			'accept-language': 'de-DE',
			'cache-control': "no-cache",
		}

		response = requests.request("GET", url, headers=headers, params=querystring)

		z_data_dict = json.loads(response.text)

		j = 0
		for product in z_data_dict['content']:
			print("%07d" % (i * page_size + j))
			# print(product)
			# print("\n")

			prod_dir_path = TRAIN_IMG_PATH + "%07d" % (i * page_size + j) + "/"
			os.makedirs(prod_dir_path)

			with open(prod_dir_path + 'data.txt', 'w') as outfile:
				json.dump(product, outfile)

			for img_file in product['media']['images']:
				img_url = img_file['mediumUrl']
				img_name = img_url.split('/')[-1]
				urllib.request.urlretrieve(img_url, prod_dir_path + img_name)

			j += 1


def test_dj_database():
	"""Test the basic DB connection"""

	prd = Product(name="Monkey", brand="Capuchin", external_id="Brian", price=25.90, display_img_path="/home/david/me.png")
	prd.save()
	p = Product.objects.get(id=1)




def show_images_quad(images, clear=False, show=True, cmap=None, Title=None, path=None, scale_up=False):
	"""Display a list of images"""
	import matplotlib.pyplot as plt
	plt.ion()
	if clear:
		plt.close()
	n_ims = images.shape[0]
	n_sqre = int(np.ceil(np.sqrt(n_ims)))
	imgs_min_val = 0
	imgs_max_val = np.max(images)
	fig = plt.figure(Title)
	n = 1
	for image in images:
		a = fig.add_subplot(n_sqre, n_sqre, n)  # Make subplot
		if image.ndim == 2 and cmap is None:  # Is image grayscale?
			cmap = 'Greys_r'
		# plt.gray() # Only place in this blog you can't replace 'gray' with 'grey'
		plt.imshow(image, interpolation='nearest', cmap=cmap)
		plt.axis('off')
		n += 1
	if scale_up:
		fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
	if Title is not None:
		fig.suptitle(Title)
	if path is not None:
		fig.savefig(path)
	if show:
		plt.show()
		plt.pause(1.0)


def test_obects_nn():
	"""Tests the NN"""

	feat_extr = FeatureExtr()

	path = "/home/david/Documents/proj/tp/imgs/"

	imgs = []

	for file in os.listdir(path):
		if file.endswith(".jpg"):
			print(file)
			im1 = (imread(path + file)[:, :, :3]).astype(np.float32)
			# im1 = im1 - mean( im1 )

			# plt.imshow(im1)
			# plt.show()
			# plt.pause(0.5)

			imgs.append(im1)

	imgs_feats = []

	for img in imgs:
		img_feat = feat_extr.get_features([img])
		imgs_feats.append(img_feat[0] / np.linalg.norm(img_feat[0]))

	# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
	kdt = BallTree(imgs_feats, leaf_size=30, metric='euclidean')

	for i in range(len(imgs)):
		orig_img = imgs[i]
		img_feat = imgs_feats[i]
		nn_dist, nn_ind = kdt.query([img_feat], k=2, return_distance=True)

		next_img_ind = nn_ind[0][1]

		next_img = imgs[next_img_ind]

		show_images_quad(np.asarray([orig_img, next_img]) * -1, Title="best matches")

		print(nn_dist[0][1])


def put_zalando_stuff_in_db():
	"""Puts the downloaded Zalando stuff into the database etc"""

	feat_extr = FeatureExtr()

	pic_list = []
	X = []

	for file in os.listdir(TRAIN_IMG_PATH):
		if os.path.isdir(TRAIN_IMG_PATH + file):
			with open(TRAIN_IMG_PATH + file + "/data.txt") as data_file:
				prod_data = json.load(data_file)

			prd = Product(name=prod_data['name'], brand=prod_data['brand']["name"], external_id=prod_data['id'],
			              price=prod_data["units"][0]["price"]["value"], display_img_path="")

			prd.save()

			for img_file in prod_data['media']['images']:
				img_name = img_file['mediumUrl'].split('/')[-1]
				img_path = TRAIN_IMG_PATH + file + "/" + img_name

				img_orig = (imread(img_path)[:, :, :3]).astype(np.float32)
				img_resize = imresize(img_orig, (227, 227))

				img_feat = feat_extr.get_features([img_resize])[0]
				norm_img_feat = img_feat / np.linalg.norm(img_feat)

				prd.picture_set.create(img_type=img_file['type'], img_path=img_path, feature_array=norm_img_feat)

				pic_list.append((prd.external_id, norm_img_feat, img_path))
				X.append(norm_img_feat)


	kdt = BallTree(X, leaf_size=30, metric='euclidean')

	pickle.dump(pic_list, open(NEAREST_NEIGH_PATH + "pic_list.p", "wb"))
	pickle.dump(kdt, open(NEAREST_NEIGH_PATH + "tree.p", "wb"))


def find_nearest_neighbour(imgs):
	# img_path = "/media/david/547E65434FED0E83/z_data/0000000/SP611M018-Q11@3.1.jpg"


	feat_list = pickle.load(open(NEAREST_NEIGH_PATH + "pic_list.p", "rb"))
	kdt = pickle.load(open(NEAREST_NEIGH_PATH + "tree.p", "rb"))

	feat_extr = FeatureExtr()

	img_feats = feat_extr.get_features(imgs)

	i = 0
	for img_feat in img_feats:
		img_feat = img_feat / np.linalg.norm(img_feat)

		nn_dist, nn_ind = kdt.query([img_feat], k=2, return_distance=True)

		next_img_ind = nn_ind[0][1]
		extern_id = feat_list[next_img_ind][0]
		prod = Product.objects.filter(external_id=extern_id)[0]

		print(prod.name)

		next_best_img_path = feat_list[next_img_ind][2]
		next_best_img = (imread(next_best_img_path)[:, :, :3]).astype(np.float32)

		show_images_quad(np.asarray([imgs[i], imresize(next_best_img, (227, 227))]) * -1, Title="best matches")

		i += 1


def test_nearest_neighbour():

	path = TRAIN_IMG_PATH

	img_paths = []

	for root, dirs, files in os.walk(path):
		for file in files:
			if file.endswith(".jpg"):
				img_paths.append(root+"/"+file)

	img_subset = np.random.choice(img_paths, 100)

	imgs = []


	for img_path in img_subset:

		img_orig = (imread(img_path)[:, :, :3]).astype(np.float32)
		img_resize = imresize(img_orig, (227, 227))
		imgs.append(img_resize)

		nb = NextBest()
		nb.find_next_best_product(img_resize)

	find_nearest_neighbour(imgs)


# Create your tests here.
def test_upload_image():
	""" unit tests for testing image uploads using DRF"""
	path_to_image = "/home/david/Pictures/prophet/snapshot.png"
	url = "http://127.0.0.1:8000/obrec/"

	data = {
		'file': open(path_to_image, 'rb')
	}
	r = requests.post(url, files=data)
	print(r.text)

test_nearest_neighbour()


print("Done.")
