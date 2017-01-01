import tempfile

from io import BufferedWriter, FileIO

from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view

from obrec.models import Product
from obrec.serializers import ProductSerializer

from helpers.nextbest import NextBest
from helpers.io import load_res_img



@api_view(['GET', 'POST', ])
def index(request):

	upload = request.FILES['file']

	fh = tempfile.NamedTemporaryFile(delete=False)
	extension = upload.name.split(".")[1]
	filename = "{}.{}".format(fh.name, extension)

	with BufferedWriter(FileIO(filename, "w")) as dest:
		for c in upload.chunks():
			dest.write(c)

	img = load_res_img(filename)

	nb = NextBest()
	nb_prod_id = nb.find_next_best_product(img)
	p = Product.objects.filter(external_id = nb_prod_id)[0]
	serializer = ProductSerializer(p)

	return Response(serializer.data)

