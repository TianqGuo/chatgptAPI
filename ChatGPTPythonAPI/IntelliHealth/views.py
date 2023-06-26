from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse


def test1(request):
    return HttpResponse("This is the test link. If you are seeing this, that means the server is working.")


def test2(request):
    return HttpResponse("another test")
