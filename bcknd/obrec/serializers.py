from rest_framework import serializers

from obrec.models import Product

class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = ('id', 'name', 'brand', 'external_id', 'price', 'display_img_path')