# -*- coding: utf-8 -*-
"""
Created on Mon May 18 02:30:40 2020

@author: Mustafiz
"""

from django.conf.urls import url
from django.views.generic import RedirectView
from django.views.generic import TemplateView
from portfolioapp import views


urlpatterns = [
    url(r'^$', views.HomePageView.as_view()),
    url(r'project/',views.ProjectPageView.as_view()),
    url(r'contact/',views.ContactPageView.as_view()),
]

