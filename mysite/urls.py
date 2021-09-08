from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    #path('admin/', admin.site.urls),
    path('', include('mlapps.urls')),
    path('score/', include('mlapps.urls')),
    path('rent/', include('mlapps.urls')),
    path('travel/', include('mlapps.urls')),
]
