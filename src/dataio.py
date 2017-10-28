'''
Created on 12.06.2017

@author: khamidova
'''

import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs,make_circles
from sklearn import preprocessing

def normalizeData(data):
    x = data.values #returns a numpy array
    x_normilized = preprocessing.scale(x)
    result = pd.DataFrame(x_normilized)
    return result

#load data from csv file
#
# delimiter in csv file, defaul - ','
# numeric - if True then all columns containing not numeric data will be deleted
# header_column - set True if dataset has a header column (for instanse, row_id) which you do not need to load
# header - column names will be loaded from the file (True) or set to range 0,1,2...
# class_column - name of the column, which contain class information. Won't be deleted even if not numeric
# delete_class - set True if class column has numeric type and you do not want to load this from file
def loadData(filename,delimiter=',',numeric=True,header_column=False, header=False, class_column='',delete_class=False):
    
    #load column names from 1st row
    if header:
        header_param=0
    else:
        #no headers
        header_param=None
    
    df=pd.read_csv(filename, sep=delimiter, header = header_param)
    
    
    if header_column:
        df=df.iloc[:,1:]
        
    #if there is no header, set column indexes
    if not header:
        df.columns=range(0,df.shape[1])
    else:
        cols=pd.read_csv(filename, sep=delimiter, header = None,nrows=1)
        #print df.head()
    cols=[]
    #print df.head()
    if numeric:
        #exclude not numeric data
        for i in range(0,len(df.columns)):
            #print df.columns[i]
            if df.dtypes[i]==np.int64 or df.dtypes[i]==np.float64 or df.columns[i]==class_column:
                cols.append(df.columns[i])
                #print 'Deleted'
        #print cols        
        df=df[cols]
    
    #print df.head()
    #print df.columns
    if delete_class and class_column!='' and class_column in df.columns:
        df.drop(class_column,axis=1,inplace=True)
    
    return df

#save dataset to a csv file
#if order is given, the dataset is first reordered and then saved
def saveData(data,filename,delimiter=',',order=None,header=False):
    if order!=None:
        data=getOrderedDF(data, order)
    data.to_csv(filename, sep=delimiter, header=header, index=False, encoding='utf-8')

def getOrderedList(data,order):
    result=[]
    for index in range(0,len(order)):
        result.append(data.iloc[order[index]][0])
    return result

def convertOrderToIndex(order):
    #return list with position of the element concerning order
    n=len(order)
    new_order=[-1 for i in range(n)]
    
    for i in range(n):
        new_order[order[i]]=i
        
    #print order
    #print new_order
    return new_order

def getOrderedDF(data,order):

    converted_order=convertOrderToIndex(order)
    order_df=pd.DataFrame({'order':converted_order},index=data.index.tolist())
    
    #print order_df

    output_df=pd.concat([data,order_df],axis=1)
    output_df.sort_values('order', inplace=True)
    
    #print output_df.head()
    
    output_df.drop('order',axis=1,inplace=True)
    
    #print output_df.head()
    
    return output_df

def generateCircles(size):
    n=size//2
    r_3=60
    r_4=80
    arr_3=np.radians(np.random.uniform(0,360,n))
    arr_4=np.radians(np.random.uniform(0,360,n))
    
    noise_x=np.random.rand(size)
    noise_y=np.random.rand(size)
    
    noise_x*=2
    noise_y*=2
    
    #print noise_x
    
    arr_3_x=r_3*np.cos(arr_3)
    arr_3_y=r_3*np.sin(arr_3)
    arr_4_x=r_4*np.cos(arr_4)
    arr_4_y=r_4*np.sin(arr_4)
    

    arr_x=np.concatenate((arr_3_x,arr_4_x))+noise_x
    arr_y=np.concatenate((arr_3_y,arr_4_y))+noise_y
    df=pd.DataFrame({0:arr_x,1:arr_y})
    #df_4=pd.DataFrame({'0':arr_4_x,'1':arr_4_y})
                           
    return df

def makeCircles(size):
    noisy_circles,y = make_circles(n_samples=size, factor=.5,noise=.05)
    print noisy_circles
    data=pd.DataFrame(noisy_circles)
    return data
    
def makeBlobs(size):
    centers=[[0, 0], [5, 5], [0, 5],[5,0]]
    X, labels_true = make_blobs(n_samples=size, centers=centers, cluster_std=1,random_state=0)
    df=pd.DataFrame(X)
    return df,labels_true    

sources_dict={}
datasets_folder='E:/Master-thesis/7. source code/LiClipse/HDDR-viz/datasets/'

#unused
#sources_dict['gas_sensors']={'filename':datasets_folder+'gas_sensors.csv','scale_columns':True,'class_column':'id','header':True,'header_column':False,'delimiter':',','delete_class':True}
#sources_dict['traffic']={'filename':datasets_folder+'traffic.csv','scale_columns':False,'class_column':'','header':True,'header_column':True,'delimiter':','}


sources_dict['iris']={'filename':datasets_folder+'iris.txt','scale_columns':True,'class_column':'species','header':True,'header_column':False,'delimiter':','}
sources_dict['wdbc']={'filename':datasets_folder+'wdbc.txt','scale_columns':True,'class_column':'diagnosis','header':True,'header_column':True,'delimiter':','}
sources_dict['cars']={'filename':datasets_folder+'cars.csv','scale_columns':True,'class_column':'','header':True,'header_column':False,'delimiter':','}
sources_dict['olive']={'filename':datasets_folder+'oils.csv','scale_columns':True,'class_column':'region','header':True,'header_column':False,'delimiter':','}
sources_dict['yeast']={'filename':datasets_folder+'yeast.csv','scale_columns':True,'class_column':'class','header':True,'header_column':False,'delimiter':','}
sources_dict['wine']={'filename':datasets_folder+'wine.txt','scale_columns':True,'class_column':'class','header':True,'header_column':False,'delimiter':',','delete_class':True}
sources_dict['aaup']={'filename':datasets_folder+'aaup.csv','scale_columns':True,'class_column':'type','header':True,'header_column':True,'delimiter':','}
sources_dict['auto']={'filename':datasets_folder+'auto_converted.csv','scale_columns':True,'class_column':'make','header':True,'header_column':False,'delimiter':','}
sources_dict['parkinson']={'filename':datasets_folder+'parkinsons.txt','scale_columns':True,'class_column':'status','header':True,'header_column':True,'delimiter':',','delete_class':True}
sources_dict['community']={'filename':datasets_folder+'communities.txt','scale_columns':True,'class_column':'','header':True,'header_column':True,'delimiter':','}
sources_dict['income']={'filename':datasets_folder+'census_income.txt','scale_columns':True,'class_column':'','header':True,'header_column':False,'delimiter':' '}
sources_dict['abalone']={'filename':datasets_folder+'abalone_male.csv','scale_columns':True,'class_column':'rings','header':True,'header_column':False,'delimiter':',','delete_class':True}

#LARGE
sources_dict['traffic']={'filename':datasets_folder+'traffic3000.csv','scale_columns':False,'class_column':'','header':True,'header_column':True,'delimiter':','}
sources_dict['bike']={'filename':datasets_folder+'bike_sharing.csv','scale_columns':True,'class_column':'','header':True,'header_column':True,'delimiter':','}
sources_dict['magic']={'filename':datasets_folder+'magic.txt','scale_columns':True,'class_column':'class','header':True,'header_column':False,'delimiter':','}
sources_dict['waveform']={'filename':datasets_folder+'waveform.data','scale_columns':True,'class_column':'class','header':True,'header_column':False,'delimiter':',','delete_class':True}
sources_dict['noisy_wave']={'filename':datasets_folder+'waveform_noise.data','scale_columns':True,'class_column':'class','header':True,'header_column':False,'delimiter':',','delete_class':True}
sources_dict['subway']={'filename':datasets_folder+'subway.csv','scale_columns':False,'class_column':'','header':True,'header_column':False,'delimiter':','}
sources_dict['alon']={'filename':datasets_folder+'alon.csv','scale_columns':True,'class_column':'class','header':True,'header_column':False,'delimiter':',','delete_class':True}
sources_dict['golub']={'filename':datasets_folder+'golub.csv','scale_columns':False,'class_column':'class','header':True,'header_column':False,'delimiter':',','delete_class':True}

#TSP
#sources_dict['ch130']={'filename':datasets_folder+'ch130.txt','scale_columns':False,'class_column':'','header':True,'header_column':True,'delimiter':' '}
#sources_dict['d657']={'filename':datasets_folder+'d657.tsp','scale_columns':False,'class_column':'','header':True,'header_column':True,'delimiter':' '}
#sources_dict['fl417']={'filename':datasets_folder+'fl417.tsp','scale_columns':False,'class_column':'','header':True,'header_column':True,'delimiter':' '}
#sources_dict['pr107']={'filename':datasets_folder+'pr107.txt','scale_columns':False,'class_column':'','header':True,'header_column':True,'delimiter':' '}
#sources_dict['lin318']={'filename':datasets_folder+'lin318.tsp','scale_columns':False,'class_column':'','header':True,'header_column':True,'delimiter':' '}


def makeDataset(params):
    data_source=params['data_source']
    
    if data_source=='circle':
        data=makeCircles(100)
        datafull=data
    elif data_source=='blobs':
        data=makeBlobs(100)
        datafull=data
        
    return data,datafull

def loadDataset(params):
    data_source=params['data_source']
    
    if data_source in sources_dict.keys():
        source_params=sources_dict[data_source]
        scale_columns=source_params['scale_columns']
        class_column=source_params['class_column']
        header=source_params['header']
        header_column=source_params['header_column']
        delimiter=source_params['delimiter']
        filename=source_params['filename']
        '''
        if 'delete_class' in source_params.keys():
            #delete class having numeric values
            delete_class=source_params['delete_class']
            data=loadData(filename=filename,delimiter=delimiter,header_column=header_column, header=header, \
                          delete_class=delete_class, class_column=class_column)
        else:    
            data=loadData(filename=filename,delimiter=delimiter,header_column=header_column, header=header)
        '''
        loaded_df=loadData(filename=filename,delimiter=delimiter,header_column=header_column, header=header, class_column=class_column)
        #print class_column
        if class_column=='':
            data=loaded_df.copy()
            datafull=loaded_df.copy()
            if data_source in ['traffic','traffic3000']:
                data=data/120
                datafull=datafull/120
        else:
            if 'delete_class' in source_params.keys():
            #delete class having numeric values
                delete_class=source_params['delete_class']
                data=loadData(filename=filename,delimiter=delimiter,header_column=header_column, header=header, \
                          delete_class=delete_class, class_column=class_column)
            else:    
                data=loadData(filename=filename,delimiter=delimiter,header_column=header_column, header=header)

            datafull=pd.concat([data,loaded_df[class_column]],axis=1)
    else:
        print 'Incorrect data source for loading: ',data_source
        return 0,0,params
    
    params['scale_columns']=scale_columns
    params['class_column']=class_column
    params['header']=header
    params['header_column']=header_column
    params['delimiter']=delimiter

    return data, datafull, params

def setDatasetParams(params):
    data_source=params['data_source']
    
    if data_source in sources_dict.keys():
        source_params=sources_dict[data_source]
        params['scale_columns']=source_params['scale_columns']
        params['class_column']=source_params['class_column']
        params['header']=source_params['header']
        params['header_column']=source_params['header_column']
        params['delimiter']=source_params['delimiter']
        
    else:
        print 'Incorrect data source for loading: ',data_source
        return params


    return params