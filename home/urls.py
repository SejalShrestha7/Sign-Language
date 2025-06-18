from os import name
from django.urls import URLPattern, path
from . import views
from home import views



urlpatterns = [
path('', views.index, name='home'),
path('animation/',views.animation_view,name='animation'),
path('camera-feed/', views.camera_feed, name='camera_feed'),
path('camera-view/', views.camera_view, name='camera_view'),
path('perform-prediction/', views.perform_prediction, name='perform_prediction'),
]