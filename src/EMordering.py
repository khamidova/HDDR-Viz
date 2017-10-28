'''
Created on 01.06.2017

@author: khamidova
'''
from time import gmtime, strftime
import dataio
import TSPmeans
import pandas as pd
import numpy as np


def EMordering(data,level=None,distance='euclidean',debug=False,fast_mode=True):
    
    #print 'EM ordering initialization', strftime("%Y-%m-%d %H:%M:%S", gmtime())
    #print 'Normalize data', strftime("%Y-%m-%d %H:%M:%S", gmtime())
    normalized_data=dataio.normalizeData(data)

    epsilon = 0.001
    #print 'calculate sigma', strftime("%Y-%m-%d %H:%M:%S", gmtime())
    #sigma=initializeSigma(normalized_data)
    sigma=calculateSigma(normalized_data)
    
    #print 'calculate entropy', strftime("%Y-%m-%d %H:%M:%S", gmtime())
    entropy=calculateEntropy(normalized_data,sigma)
    
    entropy_list=[entropy]
    shift_list=[0]
    iteration=1
    
    if debug:
        print 'Data:',data
        print 'Normalized data:',normalized_data
        print 'Sigma:',sigma
        print 'Entropy:',entropy
        
    
    while True:
        
        old_entropy=entropy
        scaled_data=scaleData(normalized_data,sigma)
        reordered_data, order = TSPmeans.TSPmeans(scaled_data,distance=distance)
       
        sigma=calculateSigma(normalized_data,order)
        
        entropy=calculateEntropy(normalized_data,sigma,order)
        
        shift=np.absolute(old_entropy-entropy)
        
        iteration+=1
        
        
        if debug:
            
            print 'Iteration:',iteration
            print 'Scaled data:',scaled_data
            print 'Sigma:',sigma
            print 'Entropy:',entropy
            print 'Shift:',shift
        entropy_list.append(entropy)
        shift_list.append(shift)
        

        if debug:
            print 'Last 3 iteration', shift_list[iteration-2:]
        max_shift=np.max(shift_list[iteration-2:])
        
        if fast_mode:
            if iteration>5: #convergence condition on Entropy
                break
        else:
            if max_shift<0.01 or iteration>300: #convergence condition on Entropy
                break
    
    return order#,entropy,entropy_list,shift_list

def initializeSigma(data):
    #standart deviation of features
    sigma=data.std(axis=0).to_frame().transpose()
    
    return sigma

def scaleData(data,sigma):
    
    return data.div(sigma.iloc[0],axis='columns')

def calculateSigma(data,order=None):

    n=data.shape[0]
    m=data.shape[1]
    
    if order==None:
        order=range(0,n)
        
    #print 'Order',order
    
    
    data_ordered=dataio.getOrderedDF(data, order)
    values_ordered=data_ordered.values
    
    if n<2:
        print 'Cannot calculate Sigmas with one example'
        return np.zeros(m)
    
    diff_arr=(values_ordered[0:n-1,:]-values_ordered[1:n,:])
    sigma_squared=np.sum(diff_arr**2,axis=0)
    sigma=np.sqrt(sigma_squared)/(n-1)
    
    mean_sigma=np.mean(sigma)
    sigma[sigma==0]=mean_sigma
    
    sigmadf=pd.DataFrame([sigma],columns=range(0,m))
    
    #sigmadf.transpose().to_csv('sigma.csv')
    return sigmadf   
       
def calculateEntropy(data,sigma,order=None):
  
    n=data.shape[0]
    m=data.shape[1]
    
    if order==None:
        order=range(0,n)
        
    values_ordered=dataio.getOrderedDF(data,order).values
    diff_arr=((values_ordered[0:n-1,:]-values_ordered[1:n,:])/sigma.values)**2
    sum_r=np.sum(diff_arr)/(2*(n-1))

    sum_l=np.sum(np.log(sigma.values))
    sum_l+=m*np.log(2*np.pi)
    sum_l=n*sum_l/(2*(n-1))
    
    return sum_r+sum_l
