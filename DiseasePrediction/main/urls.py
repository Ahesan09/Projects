from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name='home'),
    path('logout/', views.logoutaccount,name='logout'),
    path('login/', views.loginaccount,name='login'),
    path('signup/', views.signupaccount,name='signup'),
    path("health_predictiontion/", views.health_prediction, name="health_prediction"),
    path("Diabetes_prediction/",views.Diabetes_prediction,name="Diabetes_prediction"),
    path("diabetesPage/",views.diabetesPage,name="diabetesPage"),
    path("lungCancerPage/",views.lungCancerPage,name="lungCancerPage"),
    path("lungCancerPrediction/",views.lungCancerPrediction,name="lungCancerPrediction"),
    path('heartPage/',views.heartPage,name='heartPage'),
    path("heartDiseasePrediction/",views.heartDiseasePrediction,name="heartDiseasePrediction"),

]