'''
Created on 22.06.2017

@author: khamidova
'''

import numpy as np
import seaborn as sns; #sns.set()
from pandas.plotting import parallel_coordinates,scatter_matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib import ticker
import math

def plotDataFrame(plt,df,order=None,figure_n=1,by_number=False,with_line=True,by_class=False):
    plt.figure(figure_n)
    print_df=df.transpose()
    #print print_df

    n=df.shape[0]
    #print n
    
    if order==None:
        order=range(0,n)

    if not by_number:
        plt.scatter(print_df.loc[0],print_df.loc[1],color='b')
    else:
        #add column to df
        colors=np.zeros(n)
        for i in range (1,n):
            current=order[i]
            colors[current]=i
        if by_class:
            #plt.scatter(print_df.loc[0,],print_df.loc[1],c=print_df.loc[2].values,cmap='spectral')
            df1=df.loc[df[2]==0,:].transpose()
            df2=df.loc[df[2]==1,:].transpose()
            df3=df.loc[df[2]==2,:].transpose()
            df4=df.loc[df[2]==3,:].transpose()
            plt.scatter(df1.loc[0],df1.loc[1],color='r')
            plt.scatter(df2.loc[0],df2.loc[1],color='g')
            plt.scatter(df3.loc[0],df3.loc[1],color='b')
            plt.scatter(df4.loc[0],df4.loc[1],color='black')
        else:
            plt.scatter(print_df.loc[0],print_df.loc[1],c=colors,cmap='inferno')      
        

    if with_line:
        
        for i in range (1,n):
            current=order[i]
            previous=order[i-1]
            x1=df.loc[current,0]
            x2=df.loc[previous,0]
            y1=df.loc[current,1]
            y2=df.loc[previous,1]
            plt.plot([x1,x2],[y1,y2],color='r')
        
        current=order[n-1]
        previous=order[0]
        x1=df.loc[current,0]
        x2=df.loc[previous,0]
        y1=df.loc[current,1]
        y2=df.loc[previous,1]
        
        plt.plot([x1,x2],[y1,y2],color='r')
    #plt.show()   
def visualizeData(viz_method,datafull,filename,ordering=None,index=1,class_column='',scale_columns=False):
    
    if viz_method=='heatmap':
        if class_column=='':
            data=datafull
        else:
            data=datafull.drop(class_column, 1) 
        
        if scale_columns:
            #print data.head()
            data=pd.DataFrame(MinMaxScaler().fit_transform(data),columns=data.columns,index=data.index)
        
        sns.plt.figure(1)
        sns.plt.clf()
        m=data.shape[1]
        xticks=False
        sns.heatmap(data,yticklabels=False,xticklabels=xticks,cbar=False,cmap='inferno')
        sns.plt.savefig(filename,bbox_inches='tight',pad_inches = 0,dpi=600)

    elif viz_method=='PC':
        plt.figure(1)
        plt.clf()
        if class_column=='':
            datafull['label']='all'
            class_column='label'

        parallel_coordinates(datafull,class_column)
        plt.savefig(filename)
    elif viz_method=='scaledPC':
        plt.figure(1)
        plt.clf()
        if class_column=='':
            datafull['label']='all'
            class_column='label'
        scaledPC(datafull, filename, class_column)

    elif viz_method=='SPLOM':
        sns.plt.figure(1)
        sns.plt.clf()
        if class_column=='':
            sns.pairplot(datafull)
        else:
            sns.pairplot(datafull,hue=class_column)
            
        sns.plt.savefig(filename)    
    elif viz_method=='':
        return 1 
    else:
        print 'Incorrect visualization method: ',viz_method
        return 0
    
    
# Initial source: http://benalexkeen.com/parallel-coordinates-in-matplotlib/
# Adapted version
def scaledPC(df,filename,class_column=''):

    if class_column=='':
        df['class']='all'
        class_column='class'
    
    df[class_column]=df[class_column].astype('category')
    
    number_cat=len(df[class_column].cat.categories)
    #print number_cat
    cols = df.columns.tolist()
    cols.remove(class_column)
    
    #['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    x = [i for i, _ in enumerate(cols)]
    #colours = ['#2e8ad8', '#cd3785', '#c64c00', '#889a00']
    colours = uniqueColors(number_cat)
    #print colours
    # create dict of categories: colours
    colours = {df[class_column].cat.categories[i]: colours[i] for i, _ in enumerate(df[class_column].cat.categories)}
    
    # Create (X-1) sublots along x axis
    fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(15,5))
    
    # Get min, max and range for each column
    # Normalize the data for each column
    min_max_range = {}
    for col in cols:
        min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
        df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))
      
    # Set the tick positions and labels on y axis for each plot
    # Tick positions based on normalized data
    # Tick labels are based on original data
    def set_ticks_for_axis(dim, ax, ticks):
        min_val, max_val, val_range = min_max_range[cols[dim]]
        step = val_range / float(ticks-1)
        tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]
        norm_min = df[cols[dim]].min()
        norm_range = np.ptp(df[cols[dim]])
        norm_step = norm_range / float(ticks-1)
        ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
        ax.yaxis.set_ticks(ticks)
        ax.set_yticklabels(tick_labels)
    
    for dim, ax in enumerate(axes):
        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        set_ticks_for_axis(dim, ax, ticks=6)
        ax.set_xticklabels([cols[dim]])        
    
    # Move the final axis' ticks to the right-hand side
    ax = plt.twinx(axes[-1])
    dim = len(axes)
    ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    set_ticks_for_axis(dim, ax, ticks=6)
    ax.set_xticklabels([cols[-2], cols[-1]])
    
    # Plot each row
    for i, ax in enumerate(axes):
        for idx in df.index:
            row_category = df.loc[idx, class_column]
            row_color=colours[row_category]
            #row_color=(1,0,0,0.2)
            ax.plot(x, df.loc[idx, cols], color=row_color)
        ax.set_xlim([x[i], x[i+1]])
    
    # Remove space between subplots
    plt.subplots_adjust(wspace=0)
    
    # Add legend to plot
    plt.legend(
        [plt.Line2D((0,1),(0,0), color=colours[cat]) for cat in df[class_column].cat.categories],
        df[class_column].cat.categories,
        bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)
    
    #plt.title("Iris Data Set")
    
    plt.savefig(filename)

def rgbColor(h, f):

    v = 1.0
    s = 1.0
    p = 0.0
    
    if h == 0:
        r=v
        g=f
        b=p
    elif h == 1:
        r=(1 - f)
        g=v
        b=p
    elif h == 2:
        r=p
        g=v
        b=f
    elif h == 3:
        r=p
        g=(1 - f)
        b=v
    elif h == 4:
        r=f
        g=p
        b=v
    elif h == 5:
        r=v
        g=p
        b=(1 - f)
    return (r,g,b,0.2)

def uniqueColors(n):
    hues = [360.0 / n * i for i in range(n)]
    hs = [math.floor(hue / 60) % 6 for hue in hues]
    fs = [hue / 60 - math.floor(hue / 60) for hue in hues]  
    rgb_list=[rgbColor(h, f) for h, f in zip(hs, fs)]

    return rgb_list