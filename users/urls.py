from django.urls import path
from users.views import *

urlpatterns = [
    path('userhome/', userhome, name='userhome'),
    path('userlivetracker/', userlivetracker, name='userlivetracker'),
    path('api/live-sample/', get_live_tracker_sample, name='get_live_tracker_sample'),
    path('manuallivetrackers/', manuallivetrackers, name='manuallivetrackers'),
]