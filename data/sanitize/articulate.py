#!/usr/bin/env python3

from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
import subprocess

path = ''

def collect_names(dir_path):
    pages = subprocess.run(["ls", str(dir_path)], stdout=subprocess.PIPE,
        universal_newlines=True)
    html = pages.stdout
    html = [link for link in html.split('\n')]
    return html

def define_new_dir_path():
    new_path = input("Enter the new directory: ")
    return new_path

def content(HTML):
    HTML = bs(HTML.read(), 'html.parser')
    return HTML

def title():
    pass
