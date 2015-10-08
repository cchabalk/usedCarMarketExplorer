from flask import Flask, render_template, request, redirect

import pandas as pd
import json
import requests
#import datetime

import sys
sys.path.append('./myImports/')
import carDDDASImports
import importlib
importlib.reload(carDDDASImports)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import sklearn


app = Flask(__name__)
app.vars = {}  #session data

@app.route('/')
def main():
  return redirect('/index')

@app.route('/index')
def index():
  return render_template('index.html')

@app.route('/graph', methods=['GET', 'POST'])
def graph():


    #get POST data
    make      = request.form['make'].title()
    model     = request.form['model'].title()
    yearBegin = request.form['yearBegin']
    yearEnd   = request.form['yearEnd']
    zipCode   = request.form['zipCode']
    

    file1 = request.form['file1']
    file2 = request.form['file2']
    file3 = request.form['file3']
    file4 = request.form['file4']

    features = request.form.getlist('features')
    

    params= {}
    #for item in features.keys():
    #    if item:
    #        params['sites'].append(key)
    if len(make)>0:
        params['Make'] = make
        params['Model'] = model
        params['beginYear'] = int(yearBegin)
        params['endYear'] = int(yearEnd)
        params['zip'] = zipCode
        params['sites'] = features
    else:
        params['files'] = []
        print(type(file1))
        print(file1)
        print(file1.lstrip())
        params['files'].append(file1.lstrip().rstrip())
        if len(file2)>0:
            params['files'].append(file2.lstrip().rstrip())
        if len(file3)>0:
            params['files'].append(file3.lstrip().rstrip())
        if len(file4)>0:
            params['files'].append(file4.lstrip().rstrip())

    #print(features)
    
    #now run the machinery
    #import os
    #from os import listdir
    #from os import getcwd
    #from os.path import isfile, join
    #onlyfiles = [ f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(),f)) ]
    
    #make = os.getcwd()

    #make = params['files']
    #make = onlyfiles
    #import pickle
    try:
        if file1[-2:] == '.p': #pickled dataframe
            file1 = file1.lstrip().rstrip() #remove extra spaces
            carsDF = pickle.load( open( "./myImports/"+file1, "rb" ) )
        else:
            carsDF = carDDDASImports.importData(params)
    except IOError as e:
            return render_template('errorPage.html', error=e)
    if carsDF is None:
            return render_template('errorPage.html', error='No cars found.  Did you enter all of the required data?')
    
    #simply load an existing dataframe
    
    #try without writing the pickle
    #carsDF = pickle.load( open( "./myImports/carsDFHondaAccord.p", "rb" ) )
    

    #REMOVE OUTLIERS
    years = carsDF['year'].unique()
    for i in years:
        tmp = carsDF[carsDF['year']==i]['price']
        carsDF[carsDF['year']==i] = carsDF[carsDF['year']==i][(np.abs(tmp-tmp.mean())<=(3*tmp.std()))]
    carsDF = carsDF.dropna(subset = ['price', 'year'])
    carsDF = carsDF.reset_index(drop=True)  #reindex in case dropna created gaps

    titleString = carsDF['make'].iloc[0] + ' ' + carsDF['model'].iloc[0]
    finalYear = int(carsDF['year'].unique().max())
    #carsSummary['count']=carsSummary.groupby('trim').count()
    #carsSummary = carsSummary['count']


    app.vars['carsDF'] = carsDF.to_json()
    #use this in lieu of session data
    #pickle.dump( carsDF, open( "tempPickle.p", "wb" ) )

    if 1==1:
        
        #histos = generateHistograms(carsDF)
        summaryPlot = yearsMilesDistancePlot()
        #print("called simple2")
        html = simple2()
        trimPlotPlot = trimPlot()
        #print("simple2 ran")
        return render_template('yearMilesDistancePlot.html',
				titleString = titleString,
                finalYear = finalYear,
                            tickerSymbol = [],
				features = [],
                                plottingData = html,
                                plottingData2 = summaryPlot,
                                plottingData3 = trimPlotPlot)

        #response = simple2()
        #return render_template('histograms.html',
#titleString = [],			
#carsDF1 = [],
#titles = ['na', 'A few examples', 'counts'],
#				features = None,
#                                plottingData = None,
#renderedPlot1 = [])


@app.route("/simple")
def simple2():
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    x = [1,2,3]
    y = [1,2,3]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x,y,color='k',s=80,linewidth=1.5,edgecolor=[0, 0, 0],\
                               alpha=0.2)

    xFit,yFit = decayFit(x,y)
    #plt.plot(x,y,'.')
    plt.hold
    ax.plot(xFit,yFit,'k')


    sio = BytesIO()

    fig.savefig(sio, format="png")

    html = """<html><body>
    <img src="data:image/png;base64,{}"/>
    </body></html>""".format(base64.encodebytes(sio.getvalue()).decode()) 
    return html



@app.route("/histograms")
def generateHistograms(carsDF):
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    #carsDF = session.get('carsDF', None)

    #titleString = carsDF['make'][0]+' '+carsDF['model'][0]
    #carsSummary = carsDF.groupby(['trim','year']).count()
    #carsSummary['count'] = carsSummary['origin']
    
    carsSummary = carsDF.groupby(['trim']).count()
    carsSummary['count'] = carsSummary['origin']
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax1 = carsSummary['count'].plot(kind='bar',legend=None,title="Counts of different option packages")
    ax=ax1
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.plot([1,2,3])

    sio = BytesIO()
    fig.savefig(sio, format="png")

    html = """<html><body>
    <img src="data:image/png;base64,{}"/>
    </body></html>""".format(base64.encodebytes(sio.getvalue()).decode()) 
    return html

@app.route("/yearMilesDistancePlot",methods=['GET','POST'])
def yearsMilesDistancePlot():

    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    #carsDF = session.get('carsDF', None)
    
    carsDF = pd.io.json.read_json(app.vars['carsDF'])
    #session['carsDF'] = carsDF.to_json() #resave it
    #carsDF = pickle.load( open( "tempPickle.p", "rb" ) )
    carsDF.dropna(subset = ['price', 'year'])
    carsDF.reset_index(drop=True)  #reindex in case dropna created gaps
    carsDF['year']=carsDF['year'].astype('int')
    
    #session['carsDF2'] = carsDF.to_json() #resave it

    plt.get_cmap('copper')
#color = [cmx.rainbow(0),cmx.rainbow(128),cmx.rainbow(240)]
#color = sns.color_palette("Set2", 9)
    #colorsBasis = sns.choose_colorbrewer_palette('qualitative')
    colorsBasis = [(0.89411765336990356, 0.10196078568696976, 0.1098039224743836), (0.21602460800432688, 0.49487120380588578, 0.71987698697576341), (0.30426760128900115, 0.68329106055054012, 0.29293349969620797), (0.60083047361934894, 0.30814303335021531, 0.63169552298153153), (1.0, 0.50591311045721454, 0.0031372549487094226), (0.99315647868549106, 0.98700499826786559, 0.19915417450315831), (0.65845446095747096, 0.34122261685483596, 0.17079585352364723), (0.95850826852461857, 0.50846600392285535, 0.7449288887136124), (0.60000002384185791, 0.60000002384185791, 0.60000002384185791)]

    color = colorsBasis+colorsBasis  #keep 18 colors

 #   color=[(0.89411765336990356, 0.10196078568696976, 0.1098039224743836),
 #(0.21602460800432688, 0.49487120380588578, 0.71987698697576341),
 #(0.30426760128900115, 0.68329106055054012, 0.29293349969620797)]

#color = [[0,102/255,255/255],[51/255,204/255,11/255],[200/255,20/255,20/255]]

    k = 0
    #fig,ax = plt.subplots()
    fig = plt.figure() #debug
    ax  = fig.add_subplot(111) #debug
    #ax = fig.add_axes([0,0,1,1])


    plotHandel = [0,0,0]
    year = carsDF['year'].unique() #extract unique years from carsDF
    year.sort()
    #print(year)
    for i in year:
        #print(i)
        x = carsDF[carsDF['year']==i]['miles']
        y = carsDF[carsDF['year']==i]['price']
        ax.scatter(x,y,color=color[k],s=50,linewidth=1.5,edgecolor=[0, 0, 0],\
                               alpha=0.2)
        plt.hold
        ax.hold
        #print(x)
        #print(y)
        xFit,yFit = decayFit(x,y)
        #print(yFit)
        #print(xFit)
        #ax.plot(xFit,yFit,color=color[k],linewidth=10)
        #ax.plot(xFit,yFit,color=color[k],linewidth=10,zorder=0)
        plt.plot(xFit,yFit,color=color[k],linewidth=2,zorder=1)
        #plt.plot(xFit,yFit,color=color[k],linewidth=10)
        #print(color[k])
        k = k+1;

    lines = [0]*len(year)
    nLines = len(year)
    for kk, item in enumerate(lines):
        lines[nLines-kk-1] = (mlines.Line2D([], [], color=color[kk], label=year[kk]))
    
    #line[1] = (mlines.Line2D([], [], color=color[1], label=str(year[1])))
    #line[0] = (mlines.Line2D([], [], color=color[2], label=str(year[2])))

    medianPrice = carsDF['price'].median()
    ax.set_xlim([0,100000])
    axisLimits = plt.axis() #[xMin xMax yMin yMax]
    ax.set_ylim([0,2*medianPrice])
    axisLimits = plt.axis() #[xMin xMax yMin yMax]


    plt.xlabel('miles',fontsize=14)
    plt.ylabel('price ($)',fontsize=14)
    #plt.tight_layout()
    plt.gcf().subplots_adjust(left=.15)
    plt.gcf().subplots_adjust(bottom=.15)
    plt.gcf().subplots_adjust(top=.90)
    #ax1 = add_subplot_axes(ax,[0.6,0.6,0.38,0.38],axisbg='w')
    ax1 = add_subplot_axes(ax,[0.6,0.6,0.38,0.38])
    k = 0
    for i in year:
        x = carsDF[carsDF['year']==i]['miles']
        y = carsDF[carsDF['year']==i]['price']       
        ax1.scatter(x,y,\
                               color=color[k],s=30,linewidth=1.5,edgecolor="black",\
                               alpha=0.5);plt.hold
        k = k+1;

    fig = plt.gcf()
    fig.set_size_inches(7,5)


    ax1.set_xlim([0,50000])
    
    ax1.set_ylim([0.95*medianPrice,1.05*medianPrice])
#ax.legend(reversed(plotHandel),reversed(year),loc=2)
#ax.legend(plotHandel,reversed(year),loc=2)
    titleString = carsDF['make'].iloc[0] + ' ' + carsDF['model'].iloc[0]
    ax.legend(handles = lines,loc=2)
    ax.set_title(titleString + ' sedan pricing',fontsize=14)
    kk = ax.get_xticklabels()
    for item in (ax.get_xticklabels() +ax.get_yticklabels()):
        item.set_fontsize(14)
    for item in (ax1.get_xticklabels() +ax1.get_yticklabels()):
        item.set_fontsize(10)


    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.plot([1,2,3])

    sio = BytesIO()
    fig.savefig(sio, format="png")

    html = """<html><body>
    <img src="data:image/png;base64,{}"/>
    </body></html>""".format(base64.encodebytes(sio.getvalue()).decode()) 
    return html

def trimPlot():
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout':True})
    
    carsDFWhole = pd.io.json.read_json(app.vars['carsDF'])
    #session['carsDF'] = carsDFWhole.to_json() #resave it
    #carsDFWhole= pickle.load( open( "tempPickle.p", "rb" ) )

    #print(carsDFWhole.head())
    finalYear = int(carsDFWhole['year'].unique().max())
    carsDF = carsDFWhole[carsDFWhole['year']==finalYear]
    carsDF.dropna(subset = ['price', 'year'])
    carsDF.reset_index(drop=True)  #reindex in case dropna created gaps
    # plt.subplots instead of plt.subplot
    # create a figure and two subplots side by side, they share the
    # x and the y-axis
    fig = plt.figure()
    #axes = fig.add_axes # two rows, one column, first plot
    axes = fig.add_subplot(111)
    
    #fig.add_axes
    #fig, axes = plt.subplots(ncols=2, sharey=True, sharex=True)
    #fig, axes = plt.axis()
    data = np.random.random([10, 2])

    trimList = carsDF['trim'].unique()

    nTrims = len(trimList)
    #sort trims in terms of B value
    BList = []
    #print("nTrims",len(trimList))
    for i in trimList:
            rowMask = np.logical_and(carsDF['year']==finalYear, carsDF['trim']==i)
            x = carsDF[rowMask]['miles']
            y = carsDF[rowMask]['price']
            B,M = decayCoeffs(x,y)
            BList.append(B)
    #print("nB",len(BList))
    #sort trimList in terms of B value
    BList, trimList = (list(t) for t in zip(*sorted(zip(BList, trimList)))) #ENABLE
   
    #print(trimList)
    #print(BList)
   
    trimListLabel = []
    for n,i in enumerate(trimList):
        if i=='':
            trimListLabel.append(' ')
        else:
            trimListLabel.append(i)

    # np.r_ instead of lists
    colors = np.r_[np.linspace(0.2, 1, nTrims), np.linspace(0.2, 1, nTrims)] 
    mymap = plt.get_cmap("Reds")
    # get the colors from the color map
    my_colors = mymap(colors)
    # here you give floats as color to scatter and a color map
    # scatter "translates" this
    #axes[1].scatter(data[:, 0], data[:, 1], s=40,
    #                c=colors, edgecolors='None',
    #                cmap=mymap)
    #print(nTrims)
    #print("emptyTrim",type(trimList[2]))
    #print(str(''==trimList[2]))
    for n in range(nTrims):
        # here you give a color to scatter
        
        X = carsDF[carsDF['trim']==trimList[n]]['miles']
        Y = carsDF[carsDF['trim']==trimList[n]]['price']
        #print(str(n),trimList[n],len(X))
        axes.scatter(X, Y, s=40,
                    color=my_colors[n], edgecolors='k',
                    label=trimListLabel[n],alpha=.65)

    # by default legend would show multiple scatterpoints (as you would normally
    # plot multiple points with scatter)
    # I reduce the number to one here
    #axes[1].legend(bbox_to_anchor=(0.5, -0.05))
    #plt.legend(scatterpoints=1,bbox_to_anchor=(1.55, 1.02))
    plt.xlabel('miles',fontsize=14)
    plt.ylabel('price ($)',fontsize=14)
    plt.legend(scatterpoints=1,bbox_to_anchor=(1.0, 1.00))
    fig = plt.gcf()
    fig.set_size_inches(8,5)
    
    plt.axis([0, 150000, 0, carsDF['price'].max()*1.1])
    for item in (axes.get_xticklabels() + axes.get_yticklabels()):
        item.set_fontsize(14)

    #plt.gcf().tight_layout()
    #plt.tight_layout()
    #plt.show()
    plt.tight_layout()
    plt.gcf().subplots_adjust(left=.15)
    plt.gcf().subplots_adjust(bottom=.15)
    plt.gcf().subplots_adjust(top=.90)
    sio = BytesIO()

    fig.savefig(sio, format="png")

    html = """<html><body>
    <img src="data:image/png;base64,{}"/>
    </body></html>""".format(base64.encodebytes(sio.getvalue()).decode()) 
    return html




#test code for fitting
def decayFit(x,y):

    import sklearn as sk
    from sklearn.linear_model import LinearRegression

    yTrans = np.log(y)
    xTrans = np.abs(x)
    #print(yTrans)
    #print(np.shape(yTrans))

    #filter
    idx = xTrans>-np.inf
    xTrans = xTrans[idx]
    #print("xTransHere1")
    #print(xTrans)
    yTrans = yTrans[idx]
    idx2 = yTrans>-np.inf

    #print("idx2")
    #print(idx2)
    #filter
    xTrans = xTrans[idx2]
    yTrans = yTrans[idx2]
    
    xTrans = xTrans.reshape((-1,1))
    yTrans = yTrans.reshape((-1,1))

    clf = LinearRegression(fit_intercept=True)
    linearFit = clf.fit(xTrans,yTrans)
    B = np.exp(linearFit.intercept_[0])
    M = linearFit.coef_[0]

    xFit = np.linspace(np.min(xTrans),np.max(xTrans),100)
    yFit = B*np.exp(M*xFit)

    return xFit, yFit


def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    #subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    subax = fig.add_axes([x,y,width,height])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax




@app.route("/updateWithPrediction", methods=['GET','POST'])
def updateWithPrediction():

    carsDF = pickle.load( open( "tempPickle.p", "rb" ) )
    #carsDF = pd.io.json.read_json(session['carsDF'])
    #print(carsDF.head())
    carsDF = carsDF.dropna(subset = ['price', 'year'])
    carsDF = carsDF.reset_index(drop=True)  #reindex in case dropna created gaps
    titleString = carsDF['make'].iloc[0] + ' ' + carsDF['model'].iloc[0]
    #print(titleString)
    finalYear = int(carsDF['year'].unique().max())
    #carsSummary['count']=carsSummary.groupby('trim').count()
    #carsSummary = carsSummary['count']

    #print(type(request.form['mileage']))
    milesInput     = request.form['mileage']
    priceInput     = request.form['price']
    
    predicted_group = 'Not Enough Data'
    if len(milesInput)==0 or len(priceInput)==0:
        predicted_group = "Please enter a valid price and mileage"
        #print(miles)
        #print(price)
    else:
        milesInput = int(milesInput)
        priceInput = int(priceInput)
        possibleTrims = carsDF['trim'].unique()
        testPoint = [milesInput, priceInput]  #miles, dollars
        #check is agains every best fit line
        #return which ever has the lowest difference
        bestTrim = possibleTrims[0]
        currentDiff = 2000000 #absurdly large
        for i in possibleTrims:
            rowMask = np.logical_and(carsDF['year']==finalYear, carsDF['trim']==i)
            x = carsDF[rowMask]['miles']
            y = carsDF[rowMask]['price']
            if len(x)==len(y) and len(x)>5:
                B,M = decayCoeffs(x,y)
                #print("hi")
                #print(len(testPoint))
                #print(B)
                #print(M)
                #print(testPoint)
                absDiff = np.abs(testPoint[1]-B*np.exp(M*testPoint[0]))
                if absDiff<currentDiff:
                    currentDiff = absDiff
                    predicted_group = i
        #print(absDiff)

    #print(miles)


    #print(miles)
    #print(price)

    #REMOVE OUTLIERS
    #years = carsDF['year'].unique()
    #for i in years:
    #    tmp = carsDF[carsDF['year']==i]['price']
    #    carsDF[carsDF['year']==i] = carsDF[carsDF['year']==i][(np.abs(tmp-tmp.mean())<=(3*tmp.std()))]
    #carsDF = carsDF.dropna(subset = ['price', 'year'])

        
    #histos = generateHistograms(carsDF)
    summaryPlot = yearsMilesDistancePlot()
    #print("called simple2")
    html = simple2()
    trimPlotPlot = trimPlot()
    #print("simple2 ran")
    return render_template('yearMilesDistancePlot.html',
                titleString = titleString,
                finalYear = finalYear,
                            tickerSymbol = [],
                features = [],
                                plottingData = html,
                                plottingData2 =summaryPlot,
                                plottingData3 = trimPlotPlot,
                                predicted_group = predicted_group,
                                scroll='groupings')


def decayCoeffs(x,y):

    import sklearn as sk
    from sklearn.linear_model import LinearRegression

    #xTrans = np.asarray(x)
    #yTrans = np.asarray(y)

    yTrans = np.log(y)
    xTrans = np.abs(x)
    #filter
    idx = xTrans>-np.inf
    xTrans = xTrans[idx]
    yTrans = yTrans[idx]
    idx2 = yTrans>-np.inf

    #filter
    xTrans = xTrans[idx2]
    yTrans = yTrans[idx2]
    xTrans = xTrans.reshape((-1,1))
    yTrans = yTrans.reshape((-1,1))

    clf = LinearRegression(fit_intercept=True)
    linearFit = clf.fit(xTrans,yTrans)
    B = np.exp(linearFit.intercept_[0])
    M = linearFit.coef_[0]

    xFit = np.linspace(np.min(xTrans),np.max(xTrans),100)
    yFit = B*np.exp(M*xFit)
    #print(B)
    return B, M
@app.route("/faq")
def faq():
    return render_template('faq2.html')















if __name__ == '__main__':
  app.run(port=33507)
