import json
from django.shortcuts import render
from .forms import ImageUploadForm
from .captcha_solver import predict_captcha_from_image
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64
from django.core.files.base import ContentFile

def upload_image(request):
    uploaded_image = None
    prediction = None

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.cleaned_data['image']

    else:
        form = ImageUploadForm()

    return render(request, 'solver/upload.html', {'form': form, 'uploaded_image': uploaded_image, 'prediction': prediction})


@csrf_exempt
def predict_captcha(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        format, imgstr = data['image'].split(';base64,')
        ext = format.split('/')[-1]
        data = ContentFile(base64.b64decode(imgstr), name='temp.' + ext)
        prediction = predict_captcha_from_image(data.read())
        return JsonResponse({'prediction': prediction})
