'''
Created on 20.06.2017

@author: khamidova
'''
from sklearn.cluster import DBSCAN

from time import gmtime, strftime
import pandas as pd; pd.options.mode.chained_assignment = None
import scipy as sc
from PIL import Image
import numpy as np

from scagnostics import scagnostics
import seaborn as sns 
from scipy.spatial.distance import pdist,squareform
from sklearn.metrics import mutual_info_score
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_distances

scagnostics_features=['outlying','skewed','clumpy','sparse','striated','convex','skinny','stringy','monotonic']


############# HEATMAP

# Input a pandas data frame 
def entropy(data):
    #convert df to series
    s_data=data.unstack()
    p_data= s_data.value_counts()/len(data) # calculates the probabilities
    entropy=sc.stats.entropy(p_data)  # input probabilities to get the entropy 
    return entropy
       
def calculateEntropyStddev(image,mode='full',grid_number=10,method='entropy'):
    color_image=image
    grey_image=color_image.convert('L')
    
    color_image=np.array(color_image)
    grey_image=np.array(grey_image)
    
    if mode=='full':
        #entropy of full image
        df=pd.DataFrame(grey_image)
        if method=='stddev':
            return df.stack().std()
        else:
            return entropy(df)
    elif mode=='grid':
        #grid_number=10
        image_shape=grey_image.shape
        image_entropy=np.zeros((grid_number,grid_number))
        
        grid_size_x=int(np.ceil(image_shape[0]/grid_number))
        grid_size_y=int(np.ceil(image_shape[1]/grid_number))
        
        #print grid_size_x,grid_size_y
        
        for row in range(grid_number):
            for col in range(grid_number):
                    Lx=np.max([0,row*grid_size_x])
                    Ux=np.min([image_shape[0],(row+1)*grid_size_x])
                    Ly=np.max([0,col*grid_size_y])
                    Uy=np.min([image_shape[1],(col+1)*grid_size_y])
                    region=grey_image[Lx:Ux,Ly:Uy] #we would do it in the procedure .flatten()
                    region_df=pd.DataFrame(region)
                    if method=='stddev':
                        image_entropy[row,col]=region_df.stack().std()
                    else:
                        image_entropy[row,col]=entropy(region_df)
        #print image_entropy
        return image_entropy.sum()

    elif mode=='region':
        region_size=5
        image_shape=grey_image.shape
        image_entropy=np.array(grey_image)
        for row in range(image_shape[0]):
            for col in range(image_shape[1]):
                    Lx=np.max([0,col-region_size])
                    Ux=np.min([image_shape[1],col+region_size])
                    Ly=np.max([0,row-region_size])
                    Uy=np.min([image_shape[0],row+region_size])
                    region=grey_image[Ly:Uy,Lx:Ux] #we would do it in the procedure .flatten()
                    region_df=pd.DataFrame(region)
                    if method=='stddev':
                        image_entropy[row,col]=region_df.stack().std()
                    else:
                        image_entropy[row,col]=entropy(region_df)
                    

        return image_entropy.sum()

############## PC

#creates 2D dataFrame from 1 and last column
def create2Dfirstlast_df(data):
    n,m=data.shape
    data.columns=range(m)
    
    df=pd.concat([data[0],data[m-1]],axis=1)
    df.columns=[0,1]
    return df

def create2Dfirstlast_np(data):
    n,m=data.shape
    
    arr=np.concatenate((data[:,0:1],data[:,m-1:]),axis=1)
    #df.columns=[0,1]
    return arr
    

def calculateLineCrossing(data):
    data_shape=data.shape
    m=data_shape[1]
    n=data_shape[0]
    
    data.columns=range(m)
    
    line_crossing=np.zeros(m)
    
    if n>2500:
        data_values=data.values
        for col in range(0,m-1):
            #calculate line crossing between two dimensions
            #print 'Col:',col
            crossed_lines=0.0
            for row1 in range(0,n):
                
                for row2 in range(row1+1,n):
                    x1=data_values[row1,col]
                    y1=data_values[row1,col+1]
                    x2=data_values[row2,col]
                    y2=data_values[row2,col+1] 
                    
                    if (x1>x2 and y1<y2) or (x1<x2 and y1>y2):
                        crossed_lines+=1
            line_crossing[col]=crossed_lines/n/n
    else:
        for col in range(0,m):
            if col==m-1:
                df=create2Dfirstlast_df(data)
            else:
                df=data.iloc[:,col:col+2]
            df.columns=['X1','Y1']
            df['label']=0
            df2=df.copy()
            df2.columns=['X2','Y2','label']
            merged_df=df.merge(df2,how='outer')
            line_crossing[col]=float(merged_df.loc[(merged_df.X1<merged_df.X2) & (merged_df.Y1>merged_df.Y2)].shape[0])/n/n
    
    #last column
    
    return line_crossing.sum()/m
    
def calculateOutliers_DBSCAN(data):
    
    data_shape=data.shape
    m=data_shape[1]
    n=data_shape[0]
    
    outliers_number=np.zeros(m)
    
    for col in range(0,m):
        #calculate number of outliers
        if col==m-1:
            X=create2Dfirstlast_df(data).values
        else:
            X=data.loc[:,col:col+1]
        X.columns=['x','y']
        #print X
        db = DBSCAN(eps=0.3, min_samples=10).fit(X.values)
    
        labels = db.labels_
        outliers=[i for i in labels if i==-1]
        outliers_number[col]=0.0+float(len(outliers))/n
    
    return outliers_number

def calculateOutliers(data,threshold=0.01):
    
    data_shape=data.shape
    m=data_shape[1]
    n=data_shape[0]
    
    scaled_data=preprocessing.MinMaxScaler().fit_transform(data.values)
    #print scaled_data
    outliers_number=np.zeros(m)
    
    for col in range(0,m):
        #print 'Col:',col,
        #calculate number of outliers
        if col==m-1:
            twoD=create2Dfirstlast_np(scaled_data)
        else:
            twoD=scaled_data[:,col:col+2]
        #twoD.columns=['x','y']
        #print X
        #print twoD
        dist_matrix=squareform(pdist(twoD))
        
        for i in range(n):
            dist_matrix[i,i]=10
            
        #print dist_matrix    
        min_distance=np.min(dist_matrix,axis=0)
        #print min_distance
        outliers_number[col]=(min_distance>threshold).sum()
    
    return np.sum(outliers_number)/m/n   
    
def calculateMutualInformation(data,n_bins=30):
    
    m=data.shape[1]
    sum_mi=0.0
    
    for i in range(m):
        if i==m-1:
            x=data[i]
            y=data[0]
        else:
            x=data[i]
            y=data[i+1]
        c_xy = np.histogram2d(x, y, n_bins)[0]
        sum_mi+=mutual_info_score(None, None, contingency=c_xy)
    return sum_mi   
  
############### SPLOM
  
def calculateScagnostics_matrix(data):
    m=data.shape[1]  
    data.columns=range(m)
    result_dict={}
    for feature in scagnostics_features:
        result_dict[feature]=np.zeros((m,m))
    for row in range(m):
        #print row,strftime("%Y-%m-%d %H:%M:%S", gmtime())
        for col in range(m):
            #print row,col,strftime("%Y-%m-%d %H:%M:%S", gmtime())
            if row==col:
                continue
            else:
                x=data[row]
                y=data[col]
                try:
                    scags=scagnostics(x,y)    
                    for feature in scagnostics_features:
                        result_dict[feature][row,col]=scags[feature]
                except:
                    print 'Error for row col:',row,col 
    
    return result_dict

def calculateScagnostics_diagonal(data):
    m=data.shape[1]
    data.columns=range(m)
    result_dict={}

    for feature in scagnostics_features:
        result_dict[feature]=np.zeros(m-1)
    for row in range(m-1):
        x=data[row]
        y=data[row+1]
        scags=scagnostics(x,y)    
        for feature in scagnostics_features:
            result_dict[feature][row]=scags[feature]
    result_mean={}
    for feature in scagnostics_features:
        result_mean[feature]=[np.mean(result_dict[feature])]          
    return result_dict,result_mean

def calculatePearson_matrix(data):
    m=data.shape[1]
    result=np.zeros((m,m))
    for row in range(m):
        #print row,strftime("%Y-%m-%d %H:%M:%S", gmtime())
        for col in range(m):
            if row==col:
                continue
            x=data.iloc[:,row].values
            y=data.iloc[:,col].values
            (pearson,p)=sc.stats.pearsonr(x,y)
            result[row,col]=pearson
    return result

def calculatePearson_diagonal(data):
    m=data.shape[1]
    result=np.zeros(m-1)
    for row in range(m-1):
        x=data.iloc[:,row].values
        y=data.iloc[:,row+1].values
        (pearson,p)=sc.stats.pearsonr(x,y)
        result[row]=pearson
    return result

def calculatePearson_Peng(data,threshold=0.005):
    m=data.shape[1]
    result=np.zeros((m,m))
    
    pearson_matrix=calculatePearson_matrix(data)

    sum=0
    for row in range(1,m):
        for col in range(row):
            pearson=pearson_matrix[row,col]
            for i in range(row+1,m):
                for j in range(col+1,row):
                    pearson2=pearson_matrix[i,j]
                    if np.abs(pearson-pearson2)<=threshold:
                        #add distance to sum
                        sum+=np.sqrt((i-row)**2+(j-col)**2)

    return sum
   
def calculateScagnostics_lambda(data):
    #calculate all matrix
    scags_dict=calculateScagnostics_matrix(data)
    result_dict={}
    for feature in scagnostics_features:
        current_matrix=scags_dict[feature]
        result_dict[feature]=calculate_lambda(current_matrix)
    return result_dict       
        
def calculate_lambda(matrix):
    m=matrix.shape[0]
    w=int(np.ceil((0.043*m+0.3)/2))
    #print w
    
    #we need to extent matrix (adding w columns and rows at top, bottom, 
    ext_matrix=extend_matrix(matrix, w)
    
    #sum differences over matrix
    sum=0.0
    for x in range(w,m+w):
        for y in range(w,m+w):
            for i in range(-w,w):
                for j in range(-w,w):
                    if i==j:
                        continue
                    else:
                        sum+=(ext_matrix[x,y]-ext_matrix[x+i,y+j])**2
    return sum
    
def extend_matrix(matrix,w):
    (n,m)=matrix.shape
    extended_matrix=np.zeros((n+w*2,m+w*2))   
    extended_matrix[w:(n+w),w:(m+w)]=matrix
    extended_matrix[0:w,:]=extended_matrix[n:n+w,:]
    extended_matrix[n+w:n+w*2,:]=extended_matrix[w:w*2,:]
    extended_matrix[:,0:w]=extended_matrix[:,m:m+w]
    extended_matrix[:,m+w:m+w*2]=extended_matrix[:,w:w*2]
    
    return extended_matrix

    
def calculatePearson_lambda(data):  
    pearson_matrix=calculatePearson_matrix(data)
    return calculate_lambda(pearson_matrix)   

        
def visualizeScagnostics(scagnostics_dict,filename):
    sns.set()
    
    m=scagnostics_dict[scagnostics_features[0]].shape[0]
    
    #set diagonal values to mean
    for feature in scagnostics_features:
        feature_mean=np.mean(scagnostics_dict[feature])
            #print feature_mean
        for row in range(m):
            scagnostics_dict[feature][row,row]=feature_mean
    
    for index,feature in enumerate(scagnostics_features):
        sns.plt.subplot(3,3,index+1)
        sns.plt.title(feature)
        #sns.heatmap(scagnostics_dict[feature],cbar=False, vmin=0,vmax=1)
        sns.heatmap(scagnostics_dict[feature],cbar=False, xticklabels=False,yticklabels=False,cmap='inferno')
    sns.plt.savefig(filename)
    
def saveScagnostics(scagnostics_dict,filename):
    writer=pd.ExcelWriter(filename)
    for index,feature in enumerate(scagnostics_features):
        df=pd.DataFrame(scagnostics_dict[feature])
        df.to_excel(writer, feature)
    writer.save()
 
 
#################### data   
def calculateFOM(data,class_column):
    n=data.shape[0]
    #print data.head()
    
    #print data.head()
    #print class_column
    df=data[class_column]
    fom=0.0
    
    for i in range(n-1):
        #print data.iloc[i][class_column], data.iloc[i+1][class_column]
        
        if data.iloc[i][class_column]!=data.iloc[i+1][class_column]:
            fom=fom+1
            
    return fom/(n-1)
  
def calculateStress(data):
    
    m=data.shape[1]
    n=data.shape[0]
    
    nparray=data.values
    
    left_part=nparray[:,:m-1]
    right_part=nparray[:,1:m]
    top_part=nparray[:n-1,:]  
    bottom_part=nparray[1:n,:]
    
    stress=np.linalg.norm(left_part-right_part)*2+np.linalg.norm(top_part-bottom_part)*2
    
    return stress   
   
 
def calculateTourLength(data,tour,distance='euclidean'):
    
    dist=distanceMatrix(data, distance)
    sum=dist[tour[0],tour[len(tour)-1]]
    for index in range(1,len(tour)):
        sum+=dist[tour[index],tour[index-1]]

    return sum

def distanceMatrix(data,distance='euclidian'):
    if distance=='manhattan':
        result = pairwise_distances(data,metric='manhattan')
    else:
        result = pairwise_distances(data,metric='euclidean')    

    return result
    
def calculateMetric(data,metric_name,viz_method,filename='',class_column='',parameter=''):
    
    data_shape=data.shape
    m=data_shape[1]
    n=data_shape[0]
    
    if viz_method=='heatmap':
        if metric_name=='image_full_entropy':
            image=Image.open(filename)
            return calculateEntropyStddev(image,mode='full',method='entropy')
        elif metric_name=='image_full_stddev':
            image=Image.open(filename)
            return calculateEntropyStddev(image,mode='full',method='stddev')
        elif  metric_name=='grid_entropy':
            image=Image.open(filename)
            if parameter=='':
                parameter=10
            return calculateEntropyStddev(image,mode='grid',grid_number=parameter,method='entropy')
        elif  metric_name=='grid_stddev':
            image=Image.open(filename)
            if parameter=='':
                parameter=10
            return calculateEntropyStddev(image,mode='grid',grid_number=parameter,method='stddev')
    elif viz_method=='SPLOM':
        if metric_name=='scagnostics_matrix':
            return calculateScagnostics_matrix(data)
        elif metric_name=='scagnostics_diagonal':
            return calculateScagnostics_diagonal(data)[1] #mean values
        elif metric_name=='pearson_matrix':    
            return calculatePearson_matrix(data)   
        elif metric_name=='pearson_peng':
            return calculatePearson_Peng(data)
        elif metric_name=='pearson_diagonal':
            return calculatePearson_diagonal(data)      
        
        #used metrics
        elif metric_name=='pearson_lambda':
            return calculatePearson_lambda(data)          
        elif metric_name=='scagnostics_lambda':
            return calculateScagnostics_lambda(data) #mean values
        elif metric_name=='lambda':
            #print 'Scagnostics',strftime("%Y-%m-%d %H:%M:%S", gmtime())
            scags=calculateScagnostics_lambda(data)
            #print 'Pearson',strftime("%Y-%m-%d %H:%M:%S", gmtime())
            scags['pearson']=calculatePearson_lambda(data)
            
            return scags
             
    elif viz_method=='PC' or viz_method=='scaledPC':
        if metric_name=='line_crossing':
            return calculateLineCrossing(data)
            
        elif metric_name=='outliers':
            if parameter=='':
                parameter=0.01
            return calculateOutliers(data,threshold=parameter)
        
        elif metric_name=='mutual_information':
            return calculateMutualInformation(data)
        
    if metric_name=='fom':
        return calculateFOM(data,class_column)
    elif metric_name=='path_length_euclidian':
        return calculateTourLength(data, range(n))
    elif metric_name=='path_length_manhattan':
        return calculateTourLength(data, range(n),distance='manhattan')
    elif metric_name=='neumann_stress':
        return calculateStress(data)