from django.shortcuts import render
from django.views.generic import TemplateView


# Create your views here.
class HomePageView(TemplateView):
    template_name =  "index.html"
    def post(self,request,**kwargs):
        return render(request,'index.html',context=None)

class ProjectPageView(TemplateView):
    template_name =  "project.html"

class ContactPageView(TemplateView):
    template_name =  "contact.html"