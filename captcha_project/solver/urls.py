from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_image, name='upload_image'),
     path('predict_captcha/', views.predict_captcha, name='predict_captcha'),
]
