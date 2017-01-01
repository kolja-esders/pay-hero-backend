import ast
import numpy as np
from django.db import models

#pip3 django-picklefield
from picklefield.fields import PickledObjectField



# Create your models here.


class Product(models.Model):
	name = models.CharField(max_length=200)
	brand = models.CharField(max_length=200)
	external_id = models.CharField(max_length=200)
	price = models.FloatField()
	display_img_path = models.CharField(max_length=200)


class Picture(models.Model):
	product = models.ForeignKey(Product, on_delete=models.CASCADE)
	img_type = models.CharField(max_length=200, default="None")
	img_path = models.CharField(max_length=200)
	feature_array = PickledObjectField(default=[])



