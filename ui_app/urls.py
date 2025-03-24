# from django.urls import path
# from .views import home, start_detection

# urlpatterns = [
#     path('', home, name='home'),
#     path('start-detection/', start_detection, name='start_detection'),
# ]


from django.urls import path
from .views import home, video_feed

urlpatterns = [
    path('', home, name='home'),
    path('video_feed/', video_feed, name='video_feed'),
]


