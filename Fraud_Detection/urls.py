from django.urls import path, include

from . import views

urlpatterns = [
    path('', views.index, name='ind'),
    path('Fraud_Detection/home', views.home, name='home')
]