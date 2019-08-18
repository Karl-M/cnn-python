#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 19:40:43 2019

@author: konstantin
"""
# collection of functions

#import os
#import shutil
import numpy as np
#import matplotlib.pyplot as plt



# convolution operation with randomnly initialized filters
def convolute(image, filter_matrix):
    
    if len(filter_matrix.shape) < 3:
        number_filters = 1
    else: 
        number_filters = filter_matrix.shape[0]
    
    height, width = image.shape
    feature_map = np.zeros(shape=(height - 3 + 1, width - 3 + 1, number_filters))

    for k in range(number_filters):
       for i in range(height - 3 + 1):
            for j in range(width - 3 + 1):
                res = image[i:(i + 3), j:(j + 3)] * filter_matrix[k]
                feature_map[i, j, k] = np.sum(res) 
    
    #    intermediates = {"num_fil": number_filters, 
    #                     "height": height,
    #                     "width": width
    #                     }
    #    
    return feature_map, filter_matrix


def maxpool(feature_map):
    
    if len(feature_map.shape) < 3:
        number_filters = 1
        height, width = feature_map.shape
        feature_map = np.reshape((1, height, width), feature_map)
    else: 
         height, width, number_filters = feature_map.shape
        
    pooling_map = np.zeros(shape=(height // 2, width // 2,number_filters))
    
    # need indices from max for backprop
#    index = np.full(feature_map.shape, False) # index array
    k = 0
    for k in range(number_filters):
       for i in range(height // 2):
            for j in range(width // 2):
                res = feature_map[i*2:i*2 + 2, j*2:(j*2 + 2), k]
                pooling_map[i, j, k] = np.amax(res)                
#                where = np.where(res == np.amax(res))
#                m = where[0][0]
#                n = where[1][0]
#             #   print(m, n)
#                index[i*2:i*2 + 2, j*2:(j*2 + 2), k][m, n]  = True
#    print("regions maxpool: ", k)           
    return pooling_map#, index

# softmax

def softmax(output_maxpool, weight_matrix, bias_vector):
    
    n_classes = weight_matrix.shape[1]
    num_filter, height, width = output_maxpool.shape
  #  input_len = num_filter * height * width
    output_maxpool_flattened = output_maxpool.flatten()
    
  #  bias_vector = np.zeros(shape=10)
  #  weight_matrix = np.ones(shape=(1014, 10))
    input_softmax = output_maxpool_flattened.dot(weight_matrix) + bias_vector#+ 10
    
    exponentials = np.exp(input_softmax)
    sum_exponentials = np.sum(exponentials)
    probabilities = exponentials / sum_exponentials
    
    intermediates = {"exp": exponentials, 
                             "sum_exp": sum_exponentials,
                             "input_softmax": input_softmax,
                             "weight_matrix": weight_matrix,
                             "output_maxpool": output_maxpool,
                             "bias_vector": bias_vector,
                             "n_classes": n_classes
                             }
    return probabilities, intermediates


def backprop_softmax(inter_soft, probabilities, label, learn_rate):
    ps = probabilities
    
    pooling_map_shape = inter_soft["output_maxpool"].shape
    # to implement backprop, we need intermediate results, 
    # e.g. the derivative of the loss
    
    # derivative of loss function with respect to output last layer
    dLoss_daL = np.zeros(inter_soft["n_classes"]) # dL / da
    dLoss_daL[label] = - 1 / ps[label]
  #  print("dLoss_daL: ", dLoss_daL)
    
    # derivative of softmax with respect to input 
    # (input =  - (output_maxpool.dot(weight_matrix) + bias_vector) )
 #   daL_dzL = np.zeros(inter_soft["n_classes"])
    exp = inter_soft["exp"]
  #  print("exp: ", exp)
    S = inter_soft["sum_exp"]
  #  print("sum: ", S)
#    print("sum_exp: ", S)
    daL_dzL = - (exp[label] * exp) / (S ** 2)
    
    daL_dzL[label] = (exp[label] *  (S - exp[label])) / ( S ** 2) 
    
#    print("daL_dzL: ", daL_dzL)
    # derivative of Loss with respect to bias vector in softmax
    #deltaL = np.zeros(inter_soft["n_classes"])
    #deltaL[label] = dLoss_daL[label] * daL_dzL[label]
    
    # new try
    deltaL = dLoss_daL[label] * daL_dzL
   # print("deltaL: ", deltaL)
    #dbL[label] = dLoss[label]
    dL_dbL = deltaL
 #   deltaL_cor = np.dot(deltaL, inter_soft["weight_matrix"].T)    # wrong format?
    #deltaL_cor1 = np.dot(inter_soft["weight_matrix"], deltaL)
    deltaL_cor = np.dot(deltaL, inter_soft["weight_matrix"].T)
    ### weil einträge von backPropMax in der falschen Reihenfolge befüllt werden,
    # mal mit transponieren probieren
    deltaL_cor = deltaL_cor.reshape(pooling_map_shape)
   # print("deltaL_cor: ", deltaL_cor)
#    print("deltaL_cor shape in softmax: ",  deltaL_cor.shape)
    # derivative with respect to weight matrix in softmax
    dzL_dwL = np.zeros(shape=(np.prod(pooling_map_shape), inter_soft["n_classes"]))
    dzL_dwL[:, label] = inter_soft["output_maxpool"].flatten()
    
    # derivative of Loss function with respect to weight matrix in softmax
    dL_dwL = np.zeros(shape=(np.prod(pooling_map_shape), inter_soft["n_classes"]))
    #dL_dwL[:, label] = dzL_dwL.dot(deltaL) # version from blog    
    dL_dwL = np.dot(inter_soft["output_maxpool"].flatten()[np.newaxis].T, deltaL[np.newaxis] ) # my version
    #print("dL_dwL: ", dL_dwL)
 #   print("sum dL_dwL: ", np.sum(dL_dwL, axis=(0)))
    #d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
    
    # Im only updating one column of weight matrix, the other guy all?
    # updating weights
    weight_matrix = inter_soft["weight_matrix"] - learn_rate * dL_dwL 
    bias_vector = inter_soft["bias_vector"] - learn_rate * dL_dbL
    
    intermediates = {"exp": exp,
                     "deltaL": deltaL,
                     "sum_exp": S,
                     "dLoss_daL": dLoss_daL,
                     "daL_dzL": daL_dzL}
    

    return weight_matrix, bias_vector, deltaL_cor

#
#def backprop_maxpool(feature_map, index_max, deltaL_cor):
#    
#    feature_map_back = np.zeros(feature_map.shape)
#    feature_map_back[index_max] =  deltaL_cor.flatten()
#    
#    return feature_map_back
#
#
#<<<<<<< HEAD
#def backprop_maxpool(feature_map, gradient):
#
#    h = feature_map.shape[1]
#    w = feature_map.shape[2]
#    num_filters = feature_map.shape[0]
#    dMP = np.zeros(shape=feature_map.shape)
#    for f in range(num_filters):
#        for i in range(h // 2):
#            for j in range(w // 2):
#                region = feature_map[f, 2*i:2*i+2, 2*j:2*j+2]
#                for m in range(2):
#                    for n in range(2):
#                        if region[m, n] == np.amax(region):
#                            dMP[f, 2*i:2*i+2, 2*j:2*j+2][m, n] = gradient[f, i, j]
#        
#    return dMP
#=======
#def backprop_maxpool(feature_map, index_max, deltaL_cor):
#    
#    feature_map_back = np.zeros(feature_map.shape)
#    feature_map_back[index_max] =  deltaL_cor
#    
#    return feature_map_back

def backprop_maxpool(feature_map, gradient):

    h = feature_map.shape[0]
    w = feature_map.shape[1]
    num_filters = feature_map.shape[2]
    dMP = np.zeros(shape=feature_map.shape)
  #  print(dMP.shape)
    for f in range(num_filters):
        for i in range(h // 2):
            for j in range(w // 2):
                region = feature_map[2*i:2*i+2, 2*j:2*j+2, f]
             #   print(region)
                for m in range(2):
                    for n in range(2):
                 #       print(f, i, j, m, n)
                        if region[m, n] == np.amax(region):
                            dMP[2*i:2*i+2, 2*j:2*j+2, f][m, n] = gradient[i, j, f]
                            #print(dMP)
    return dMP

#>>>>>>> filter_last


def backprop_conv(image, filter_conv, gradient, learn_rate):           
    #gradient = gradient.reshape(shape_outmax)
    dpool_dfilter = np.zeros(shape=filter_conv.shape)
    n_filters = dpool_dfilter.shape[0]
    n_rows = dpool_dfilter.shape[1]
    n_cols = dpool_dfilter.shape[2]
    
    for f in range(n_filters):
        #row_max, col_max = np.where(index_max[:, :, f] == True)
        row_max, col_max = np.where(gradient[:, :, f] != 0)
        for i in range(n_rows):
            for j in range(n_cols):
                for m, n in zip(row_max, col_max):
                    dpool_dfilter[f, i, j ] += image[m+i, n+j] * gradient[m, n, f]
                    
    filter_conv = filter_conv - learn_rate * dpool_dfilter

    return filter_conv#, dpool_dfilter

    return weights_conv, weights_soft, feature_map_back
    


