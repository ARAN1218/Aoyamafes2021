from django.urls import path
from . import views

urlpatterns = [
    path('', views.entrance, name='entrace'),
    path('score/', views.score, name='score'),
    path('travel/score/score_detail/', views.score_detail, name='score_detail'),
    path('rent/', views.rent, name='rent'),
    path('travel/', views.travel, name='travel'),
]