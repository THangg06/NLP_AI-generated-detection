from django.urls import path
from . import views

app_name = 'detector'

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict_page, name='predict'),
    path('quick-predict/', views.quick_predict, name='quick_predict'),
    path('history/', views.articles_history, name='history'),
    path('article/<int:pk>/', views.article_detail, name='article_detail'),
    path('statistics/', views.statistics, name='statistics'),
    
    # API endpoints
    path('api/predict/', views.api_predict, name='api_predict'),
]
