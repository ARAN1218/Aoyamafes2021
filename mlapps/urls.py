from django.urls import path
from . import views

urlpatterns = [
    path('', views.entrance, name='entrace'),
    path('score/', views.score, name='score'),
    path('travel_detail/score/score_detail/', views.score_detail, name='score_detail'),
    path('rent/', views.rent, name='rent'),
    path('travel_detail/rent/rent_detail/', views.rent_detail, name='rent_detail'),
    path('travel/', views.travel, name='travel'),
    path('travel_detail/travel/travel_detail/', views.travel_detail, name='travel_detail'),
]