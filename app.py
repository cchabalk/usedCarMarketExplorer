from flask import Flask, render_template, request, redirect

#import pandas as pd
#import json
#import requests
#import datetime

#import sys
#sys.path.append('./myImports/')
#import carDDDASImports
#import importlib
#importlib.reload(carDDDASImports)

#import matplotlib.pyplot as plt
#import numpy as np
#import seaborn as sns
#import matplotlib.lines as mlines
#from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
#import sklearn


app = Flask(__name__)
app.vars = {}  #session data

@app.route('/')
def main():
  return redirect('/index')

@app.route('/index')
def index():
  return render_template('index.html')


@app.route("/faq")
def faq():
    return render_template('faq.html')


if __name__ == '__main__':
  #app.run(port=33507)
  app.run(port=33507)