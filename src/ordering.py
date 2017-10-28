'''
Created on 16.06.2017

@author: khamidova
'''
from sklearn.manifold import MDS,TSNE
import pandas as pd
import matlab.engine
import EMordering
from TSPmeans import TSPmeans
from LKH import solveTSPLKH

def MDSordering(data):
    mds=MDS(n_components=1,n_jobs=4)
    projected_df = pd.DataFrame(mds.fit_transform(data).T).transpose()
    sorted_df=projected_df.sort_values(by=0)
    return sorted_df.index.values

def TSNEordering(data):
    tsne=TSNE(n_components=1)
    projected_df = pd.DataFrame(tsne.fit_transform(data).T).transpose()
    sorted_df=projected_df.sort_values(by=0)
    return sorted_df.index.values
    
def HCOLOordering(data): 
    #call MATLAB implementation of HC-olo
    eng = matlab.engine.start_matlab()
    matlab_data=matlab.double(data.values.tolist())
    distance_matrix = eng.pdist(matlab_data);
    tree = eng.linkage(distance_matrix,'average');
    leaf_order = eng.optimalleaforder(tree,distance_matrix);
    
    order=[]
    for index in leaf_order[0]:
        order.append(int(index-1))
    
    return order

def EMordering_standard(data):
    return EMordering.EMordering(data)

def EMordering_manhattan(data):
    return EMordering.EMordering(data,distance='manhattan')

def shuffleData(data):
    data=data.sample(frac=1)
    return data.index.tolist()

def LKHordering(data):
    points=data.shape[0]
    if points>3:
        return solveTSPLKH(data)['tour']
    else:
        print 'No LKH neeeded for',points,'examples'
        return range(points)

def TSPmeansordering(data):
    result, order = TSPmeans(data)
    return order