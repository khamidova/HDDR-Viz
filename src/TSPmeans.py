'''
Created on 01.06.2017

@author: khamidova
'''
from time import gmtime, strftime
from sklearn.cluster import KMeans

import numpy as np
#import pandas as pd
import tree
import os
import LKH
#import dataio
#import viz
#import matplotlib.pyplot as plt
import spectral
import metrics

################################################################################
################################ FUNCTIONS #####################################
################################################################################

    
def dfMean(data):
    return data.mean().to_frame().transpose().as_matrix()[0]

def kmeansCluster(data,max_iter=3000,distance='euclidean'):
    
    fast_mode=True
    #initarray=data.loc[0:1,:].values
    
    
    n=data.shape[0]
    """
    n_init=n//2
    if n_init<10:
        n_init=10
    """    
    if fast_mode:
        max_iter=300
        n_init=10
    else:
        n_init=100
    
    #print 'Data len:', n, strftime("%Y-%m-%d %H:%M:%S", gmtime())
    
    if distance=='manhattan':
        labels_arr,centers = spectral.kmeans(np.array([data.values]),nclusters=2,max_iterations=max_iter,distance=spectral.clustering.L1)
        labels=labels_arr[0]
    else:
        kmeans = KMeans(n_clusters=2,max_iter=max_iter,n_init=n_init).fit(data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
    
    if labels[0]==0:
        outdf1 = data[labels==0]
        outdf2 = data[labels==1]
        mean1 = centers[0]
        mean2 = centers[1]
    else:
        outdf1 = data[labels==1]
        outdf2 = data[labels==0]
        mean1 = centers[1]
        mean2 = centers[0]

    
 
    #print type(kmeans.cluster_centers_[0])
 
    return outdf1, outdf2, mean1, mean2

def recursiveKMeans(data,root,parent_node=None,distance='euclidian'):
    if parent_node==None:
        #for the first call parent is root
        parent_node = root
    n=data.shape[0]    
    #print data   
    #raw_input("Press Enter to continue...") 
    #cluster the set and call function recursively
    if n>1:
        out1,out2,mean1,mean2=kmeansCluster(data,distance=distance)
        
        #print
        if False:
            print 'Full data:',data
            print 'Cluster 1:', out1
            print 'Cluster 2:', out2
            print ' '
        
        if out1.empty or out2.empty:
            #df contain same values
            #split it manually
            center_row=n//2
            out1=data.iloc[:center_row,:]
            out2=data.iloc[center_row:,:]
            mean1=dfMean(out1)
            mean2=dfMean(out2)
        
        if out1.shape[0]==1:
            #add Node
            node1=tree.addNode(mean1,parent_node,index=out1.index.tolist()[0])
        else:
            #add centroids
            node1=tree.addNode(mean1,parent_node)
            
        if out2.shape[0]==1:
            #add Node
            node2=tree.addNode(mean2,parent_node,index=out2.index.tolist()[0])
        else:
            #add centroids
            node2=tree.addNode(mean2,parent_node)
        #recursive call
        recursiveKMeans(out1, root, node1)
        recursiveKMeans(out2, root, node2)
    '''
    else:

        out1=data.iloc[0:1,:]
        mean1=dfMean(out1)
        node1=tree.addNode(mean1,parent_node,index=out1.index.tolist()[0])
        if n>1:
            out2=data.iloc[1:2,:]
            mean2=dfMean(out2)
            node2=tree.addNode(mean2,parent_node,index=out2.index.tolist()[0])
    '''    

def reconstructTree(root_node,l):
    #delete all levels except each l-th
    
    #set current nodes
    current_nodes=root_node.children
    to_delete=[]
    
    
    for i in range(1,root_node.height+1):
        #print 'Level', i
        #print current_nodes
        
        new_nodes=()
        for n in current_nodes:
            #add all children for the next step
            new_nodes=new_nodes+n.children
            if i%l==0:
                #if current level / l = 0 -> rearrange to l-parent
                tree.makeOlder(n, l-1)
            else:
                if n.is_leaf:
                    #if leaf rearrabge to (ostatok ot delenija na l) parent
                    tree.makeOlder(n,i%l-1)
                else:
                    to_delete.append(n)
        
        current_nodes=new_nodes
     
    #print to_delete
    for n in to_delete:
        tree.deleteNode(n, False)   

def solveTSP(nodes_list,left,right,distance='euclidean'):
    
    #print len(nodes_list)
    values_list=tree.nodesToList(nodes_list)
    #print values_list
    n_values=len(values_list)
    
    if n_values<=3:
        return nodes_list
    
    #solve TSP for this list
    thisdir = os.path.abspath(os.path.dirname(__file__))
    #print thisdir
    
    if left and right:
        neighbours=True
    else:
        neighbours=False
    
    
    outf = os.path.join(thisdir, "test.tsp")
    with open(outf, 'w') as dest:
        #for two neighbours we would add fixed edge and start from 0
        dest.write(LKH.dumpMatrix(metrics.distanceMatrix(values_list,distance=distance), neighbours))
    #for only left neighbour we would start from 0
    tsp_result = LKH.runLKH(outf,0) #start 0

    order_list=tsp_result['tour']

    
    #for only right neighbour we would end with n
    if not left and right:
        # rotate to the beginning of the route
        while order_list[n_values-1] != n_values-1:
            order_list = order_list[1:] + order_list[:1]
    
    if left  and (order_list[0]!=0) or right and order_list[n_values-1]!=(n_values-1):
        print 'Attention: something wrong in the order'
    
    #print order_list
    #print len(nodes_list)
    
    #add nodes in a new order
    new_nodes_list=[]
    for order in order_list:
        new_nodes_list.append(nodes_list[order])
    
    #resulting tour
    return new_nodes_list

def traverseTree(root,debug=False,distance='euclidian'):
    nodes_list=[root]
    list_len=1
    not_all_leaves=True
    level=0
    
    lkh_inc=0
    
    
    while not_all_leaves:
        
        #print 'Level', level, strftime("%Y-%m-%d %H:%M:%S", gmtime())
        
        if debug:
            print 'Level', level
            print 'Current list:', tree.nodesToList(nodes_list)
            
        not_all_leaves=False
        
        new_nodes_list=[]
        for index,current_node in enumerate(nodes_list):
            if not current_node.is_leaf:
                #add children arranged by LKH-tsp
                nodes_tsp=[]
                start=0
                end=0
                
                if index>0:
                    #we append last ordered node from previous TSP run
                    nodes_tsp.append(new_nodes_list[len(new_nodes_list)-1])
                    #nodes_tsp.append(nodes_list[index-1]) #TODO current left node
 
                    start=1
                    end=1
                    neighbour_left=True
                else:
                    neighbour_left=False
                for child_node in current_node.children:
                    nodes_tsp.append(child_node)
                    end=end+1

                if index<list_len-1:
                    nodes_tsp.append(nodes_list[index+1])
                    neighbour_right=True
                else:
                    neighbour_right=False
                
                if debug:        
                    print 'TSP-solver for:', tree.nodesToList(nodes_tsp)
                    print 'Neighbours:',neighbour_left,neighbour_right
                    print 'TSP-solver for indices', tree.nodesToIndexList(nodes_tsp)
                nodes_tsp=solveTSP(nodes_tsp,neighbour_left,neighbour_right,distance=distance)
                
                lkh_inc+=1
                #print 'LKH #', lkh_inc, 'number of nodes: ', len(nodes_tsp)
                
                if debug:
                    print 'Result:', tree.nodesToList(nodes_tsp)
                    print 'Result indices:',tree.nodesToIndexList(nodes_tsp)
                    
                
                #add nodes to list
                for k in range(start,end):
                    new_nodes_list.append(nodes_tsp[k])
                not_all_leaves=True
                
                if debug:
                    print 'New list indices',tree.nodesToIndexList(new_nodes_list)
            else:
                new_nodes_list.append(current_node)
        
        nodes_list=new_nodes_list
        list_len=len(nodes_list)
        level=level+1
        
        tour=tree.nodesToIndexList(nodes_list)
        result=tree.nodesToList(nodes_list)
    return result, tour

def TSPmeans(data,level=None,distance='euclidian',debug=False):
    root_node = tree.addNode(dfMean(data))
    
    #print 'Tree construction',strftime("%Y-%m-%d %H:%M:%S", gmtime())
    
    recursiveKMeans(data, root_node,distance=distance)
    
    if debug:
        print 'Binary tree:'
        tree.printTree(root_node)
        
    if level==None:
        n=data.shape[0]
        level=int(np.ceil(np.log2(n)/2))
        #print level
    
    #print 'Tree reconstruction', strftime("%Y-%m-%d %H:%M:%S", gmtime())
    reconstructTree(root_node, level)
    
    if debug:
        print 'Reconstructed tree:'
        tree.printTree(root_node)
    
    #print 'Tour calculation', strftime("%Y-%m-%d %H:%M:%S", gmtime())
    return traverseTree(root_node,debug,distance=distance)
 


################################################################################
################################### TESTS ######################################
################################################################################




################################################################################
################################## TESTS #######################################
################################################################################
''' 
def testReconstruction():
    root=tree.addNode(1)
    node2=tree.addNode(2,root)
    node3=tree.addNode(3,root)
    node4=tree.addNode(4,node2)
    node5=tree.addNode(5,node2)
    node6=tree.addNode(6,node3)
    node7=tree.addNode(7,node3)
    node8=tree.addNode(8,node4)
    node9=tree.addNode(9,node4)
    node10=tree.addNode(10,node5)
    node11=tree.addNode(11,node5)
    
    node14=tree.addNode(14,node7)
    node15=tree.addNode(15,node7)
    node16=tree.addNode(16,node8)
    node17=tree.addNode(17,node8)
    node18=tree.addNode(18,node9)
    node19=tree.addNode(19,node9)
    node22=tree.addNode(22,node11)
    node23=tree.addNode(23,node11)
    
    tree.printTree(root)
    
    reconstructTree(root, 2)
    tree.printTree(root)

################################################################################
################################### MAIN #######################################
################################################################################
   

       
def test():
    
    #df=dataio.loadData('test.txt',delimiter=' ',header_column=True)
    
    #df=pd.DataFrame.from_dict({'0':[1,5,5,5,16,23,4,34,25,18,38,27,62,14,22,7,57,9,70,44,39,17,29,78,36,45,34,22,1,17,90]})
    #df2=pd.DataFrame.from_dict({'0':[90,62,57,44,45,70,78,38.5,34.6,28,23,14,17,1,5.83]})
    #df=pd.DataFrame.from_dict({'0':[1,5,9,70,44,39],'1':[3,12,6,7,14,2]})
    
    #print df
    
    #df=dataio.makeCircles(500)
    df=dataio.makeBlobs(20)
    print df
    #viz.plotDataFrame(plt,df,None,3,by_number=False,with_line=False)
    
    #print dumpDataFrame(df)
    
    #result=solveTSPLKHfile('tsp/pr107.tsp')
    result=LKH.solveTSPLKH(df)
    tourLKH=result['tour']
    print tourLKH
    print 'Tour length', metrics.calculateTourLength(df, tourLKH)
    #print dataio.getOrderedList(df2, tourLKH)
    #print calculateTourLength(df2, tourLKH)
    
    df2=dataio.getOrderedDF(df, tourLKH)
    print df2
    
    
    
    viz.plotDataFrame(plt,df,tourLKH,by_number=True,with_line=False)
    viz.plotDataFrame(plt,df2,None,3,by_number=True,with_line=False)
    
    plt.show()
    """
    result, tour =TSPmeans(df,debug=False)
    #print 'Result:', result
    print 'Tour length', tour
    
    #print result
    print calculateTourLength(df, tour)  
    print 'Number of examples', len(tour)
    
    
    viz.plotDataFrame(plt,df,tour,2,by_number=True,with_line=False) 
    
    plt.show()
    
    
    for i in range(0,10):
        out1,out2,mean1,mean2=kmeansCluster(df,max_iter=20000)
        print 'Cluster 1:', len(out1)
        print 'Cluster 2:', len(out2)
        print ' '
    
    
    circles=dataio.generateCircles(200)
    print circles
    viz.plotDataFrame(plt,circles,None,2,by_number=False,with_line=False) 
    
    plt.show()   
    """
#test()

def dumpDataFrame(data):
    template = """NAME: {name}
TYPE: TSP
COMMENT: {name}
DIMENSION: {n_cities}
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
{matrix_s}
{edges_s}EOF"""

    name='LKH'
    n_cities = data.shape[0]
    matrix_s = ""
        
    for index, row in data.iterrows():
        matrix_s += str(index+1)+" "+str(row[0])+" 0"
        matrix_s += "\n"
    
    fixed_edges_s=''

    return template.format(**{'name': name,
                              'n_cities': n_cities,
                              'matrix_s': matrix_s,
                              'edges_s':fixed_edges_s})
      
'''
