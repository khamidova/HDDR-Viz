'''
Created on 28.06.2017

@author: khamidova
'''

import dataio
import viz
import pandas as pd
import ordering
import os
import metrics
import json
import argparse




def processData(params):
    
    if 'ordering_direction' in params.keys():
        ordering_direction=params['ordering_direction']
    else:
        print 'You must specify ordering direction: rows or cols'
        return 0
     
    '''    
    if 'tasks' in params.keys():
        tasks=params['tasks']
    else:
        tasks=['ordering','viz','metrics']
    '''
   
    data_source_type=params['data_source_type']
    data_source=params['data_source']
    
    scale_columns=False
    
    if 'delimiter' in params.keys():
        delimiter=params['delimiter']
    elif data_source_type!='load':
        print 'You must specify delimiter parameter!'
        return 0
    
    if 'header' in params.keys():
        header=params['header']
    elif data_source_type=='file':
        print 'You must specify header parameter!'
        return 0    
    
    if 'header_column' in params.keys():
        header_column=params['header_column']
    elif data_source_type=='file':
        print 'You must specify header_column parameter!'
        return 0  
     
    if 'class_column' in params.keys():
        class_column=params['class_column']
    else:
        class_column=''
    
    ordering_methods=params['ordering_methods']
    output_name=params['output_name']
    viz_method=params['viz_method']
    
    #if 'ordering' in tasks:
    
    #load data from data source
    if data_source_type=='file':
        datafull=dataio.loadData(filename=data_source, delimiter=delimiter,header_column=header_column, header=header, class_column=class_column)
        data=dataio.loadData(filename=data_source, delimiter=delimiter,header_column=header_column, header=header)  
        #print data.head()   
    elif data_source=='make':
        data, datafull=dataio.makeDataset(params)
        
    elif data_source_type=='load':
        data,datafull,params=dataio.loadDataset(params)
        scale_columns=params['scale_columns']
        class_column=params['class_column']
        header=params['header']
        header_column=params['header_column']
        delimiter=params['delimiter']
            
    else:
        print 'Incorrect data source type: ',data_source_type
        return 0


    if ordering_direction=='rows' and viz_method!='heatmap':
        print 'Error: you would like to reorder rows for ',viz_method,'visualization!'
        return 0
    
    #transpose data for PC and SPLOM (reordering of examples is useless)
    
    if ordering_direction=='rows':
        data_transposed=False
    else:
        data_transposed=True
        column_names=datafull.columns
        data=data.transpose()
        datafull=datafull.transpose()
        #reset index
        
        data.reset_index(drop=True,inplace=True)
        datafull.reset_index(drop=True,inplace=True)
 
        #print datafull
        #print column_names
        #datafull.reset_index(inplace=True)
    
    output_data=[]
    
    #check directory and create
    filename=output_name+'.txt'
    dirname=os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    #print ordering_methods
    #order data and save results
    for ordering_method in ordering_methods:
        
        print 'Processing',ordering_direction,'using reordering method: ',ordering_method
        
        if ordering_method=='original':
            output_order=data.index.tolist()
        elif ordering_method=='random':
            output_order=ordering.shuffleData(data)
        elif ordering_method=='EM':
            output_order=ordering.EMordering_standard(data)
        elif ordering_method=='EMmanhattan':
            output_order=ordering.EMordering_manhattan(data)    
        elif ordering_method=='LK':
            output_order=ordering.LKHordering(data)    
        elif ordering_method=='TSPmeans':
            output_order=ordering.TSPmeansordering(data)       
        elif ordering_method=='MDS':          
            output_order=ordering.MDSordering(data).tolist()
        elif ordering_method=='HColo':
            output_order=ordering.HCOLOordering(data)
        elif ordering_method=='TSNE':          
            output_order=ordering.TSNEordering(data).tolist()
     
        else:
            print 'Incorrect ordering method: ',ordering_method
            return 0
        
        if class_column!='' and data_transposed:
            #print type(output_order)
            output_order.append(len(output_order))
        output_df=dataio.getOrderedDF(datafull, output_order) 

        
        if data_transposed:
            output_df=output_df.transpose()
            #print column_names
            #print output_order
            new_column_names=[]
            for order_item in output_order:
                new_column_names.append(column_names[order_item])
                
            output_df.columns=new_column_names
        #return column names
            
        #print output_df
        
        output_data.append(output_df)
        
        #save results to .csv
        outcsv = os.path.join(output_name+'_'+ordering_method+'_'+ordering_direction+'.csv')
        dataio.saveData(output_df, outcsv, ',',header=header)
        #save reordered data
    '''
    else:
        #set params for further visualization
        if data_source_type=='load':
            params=dataio.setDatasetParams(params)
            scale_columns=params['scale_columns']
            class_column=params['class_column']
            header=params['header']
            header_column=params['header_column']
            delimiter=params['delimiter']
    '''    
            
    #if 'viz' in tasks:
           
        #if 'ordering' in tasks:
    for index,output_df in enumerate(output_data):
        
        print 'Visualizing data: ',ordering_methods[index],viz_method
                
        outpng = os.path.join(output_name+'_'+ordering_methods[index]+'_'+viz_method+'.png')
        viz.visualizeData(viz_method,output_df,outpng,ordering_methods[index],index,class_column,scale_columns)
    ''' 
        else:
            #we need to read data from csv
            for index,ordering_method in enumerate(ordering_methods):
                print 'Visualizing and saving data: ',ordering_method,viz_method
                outcsv = os.path.join(output_name+'_'+ordering_method+'_'+ordering_direction+'.csv') 
                #out_df=dataio.loadData(outcsv,header=True)
                out_df_full=dataio.loadData(outcsv,header=True,class_column=class_column) 
                outpng = os.path.join(output_name+'_'+ordering_method+'_'+viz_method+'.png')
                viz.visualizeData(viz_method,out_df_full,outpng,ordering_method,index,class_column,scale_columns) 
    '''
     
    #if 'metrics' in tasks:
        #print 'metrics'
    """
        if 'ordering' in tasks:
            
            for index,output_df in enumerate(output_data):
                
                print 'Calculating metrics: ',ordering_methods[index]
                        
                outpng = os.path.join(output_name+'_'+ordering_methods[index]+'_'+viz_method+'.png')
                viz.visualizeData(viz_method,output_df,outpng,ordering_methods[index],index,class_column,scale_columns)
    """ 
    
    #which metrics to calculate?
    if viz_method=='heatmap':
        metrics_list=['fom','path_length_euclidian','neumann_stress','grid_entropy','grid_stddev']
        parameters=[5,10,15,25,30]
        
    elif viz_method=='PC' or viz_method=='scaledPC':
        metrics_list=['line_crossing','mutual_information','outliers']
        parameters=[0.1,0.075,0.05,0.025,0.01]
    elif viz_method=='SPLOM':
        metrics_list=['lambda']
        
    
        
    #calculate metrics and save as csv file
    metrics_text='dataset,ordering,metric,parameter,value'
    
    for index,output_df in enumerate(output_data):
    
        print 'Calculating metrics for ordering results: ',ordering_methods[index]
        
        if ordering_direction=='rows':
            csv_rows = output_name+'_'+ordering_methods[index]+'_rows.csv'    
            png_heatmap = output_name+'_'+ordering_methods[index]+'_heatmap.png'
            df_rows=dataio.loadData(csv_rows,header=True,class_column=class_column,delete_class=True)
            df_full_rows=dataio.loadData(csv_rows,header=True,class_column=class_column)
            for metric in metrics_list:
                if metric=='fom':
                    if class_column!='':
                        value=metrics.calculateMetric(df_full_rows, metric, 'data',class_column=class_column)
                        metrics_text=metrics_text+'\n'+','.join((data_source,ordering_methods[index],metric,'no_param',str(value)))
                    else:
                        continue
                elif metric=='grid_entropy' or metric=='grid_stddev':
                    for grid_number in parameters:
                        value=metrics.calculateMetric(df_rows, metric, 'heatmap',filename=png_heatmap,parameter=grid_number)
                        metrics_text=metrics_text+'\n'+','.join((data_source,ordering_methods[index],metric,str(grid_number),str(value)))
                else:
                    value=metrics.calculateMetric(df_rows, metric, 'data')
                    metrics_text=metrics_text+'\n'+','.join((data_source,ordering_methods[index],metric,'no_param',str(value)))
                
                
        else:
            csv_cols = output_name+'_'+ordering_methods[index]+'_cols.csv'
            df_cols=dataio.loadData(csv_cols,header=True,class_column=class_column,delete_class=True)
            #df_full_cols=dataio.loadData(csv_cols,header=True,class_column=class_column)
            for metric in metrics_list:
                if metric=='outliers':
                    for thresh in parameters:
                        value=metrics.calculateMetric(df_cols, metric, 'PC',parameter=thresh)
                        metrics_text=metrics_text+'\n'+','.join((data_source,ordering_methods[index],metric,str(thresh),str(value)))
                elif  metric=='lambda':
                    lambda_value=metrics.calculateMetric(df_cols, 'lambda', 'SPLOM')
                    for feature in lambda_value.keys():
                        metrics_text=metrics_text+'\n'+','.join((data_source,ordering_methods[index],'lambda',feature,str(lambda_value[feature])))
                else:
                    value=metrics.calculateMetric(df_cols, metric, 'PC')
                    metrics_text=metrics_text+'\n'+','.join((data_source,ordering_methods[index],metric,'no_param',str(value)))
            
    with open(output_name+'_metrics_'+viz_method+'.txt', 'w') as dest:
        dest.write(metrics_text)
        
    print 'Creating combined Excel file'    
    #read metrics and rewrite them as Excel sheet (pivot)
    df_metrics=pd.read_csv(output_name+'_metrics_'+viz_method+'.txt')
    data=pd.pivot_table(df_metrics,values='value',index=['dataset','metric','parameter'],columns='ordering')
    writer=pd.ExcelWriter(output_name+'_metrics_'+viz_method+'.xlsx')
    data.to_excel(writer)
    writer.close()
    
    print 'Processing done.'
    return 1




if __name__ == '__main__':

    options={
        'data_source_type':['file','load'],
        'data_source':['circles','blobs','iris'],
        'viz':['PC','heatmap','SPLOM','scaledPC'],
        'ordering':['original', 'random', 'MDS','LK','EM','HColo','TSNE','EMmanhattan'],
        #'metrics':['scagnostics','image_full_entropy','image_entropy'],
        #'tasks':['ordering','viz','metrics'],
        'ordering_direction':['rows','cols']
        }            
    """   
    params={
        'data_source_type':'load',
        'data_source':'iris',
        'delimiter':',',
        'header_column':False,
        'header':True,
        'class_column':'species',
        'viz_method':'PC',
        'ordering_methods':['random','MDS','TSNE','HColo','EM','TSPmeans','LK'],
        'output_name':'E:/Master-thesis/output/demo/iris_heatmap/iris',
        #'tasks':['ordering','viz','metrics'],
        'ordering_direction':'cols'
        }
    
    with open('demo1.json', 'w') as fp:
        json.dump(params, fp)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-parameters', type=argparse.FileType('r'))
    args = parser.parse_args()

    with args.parameters as file:
        params=json.load(file)
    
    processData(params)