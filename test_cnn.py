#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 19:07:31 2019

@author: konstantin
"""

import numpy as np
import sys
#from pathlib import Path
import mnist
#import os
#path2 = "/home/konstantin/Documents/master_arbeit/nn_in_r/cnn-from-scratch"
#path = "C:/Users/D2GU53/Documents/master_arbeit/nn_in_r"
#path2 = "C:/Users/D2GU53/Documents/master_arbeit/nn_in_r/cnn-from-scratch"


#os.listdir(path2)


#import training as train
#import test_cnn as test


test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]


def debug_cnn(n_iter, version, learn_rate):

#    from importlib import reload 
    path = "/home/konstantin/Documents/master_arbeit/"
    sys.path.append(path)
    import functions as fun
    #import os
    
    if version == "changed":
        print("changed blog version is being used")
        path_blog_changed = "/home/konstantin/Documents/master_arbeit/cnn_python/cnn-from-scratch-changed/"
        sys.path.append(path_blog_changed)
        #os.listdir(path_blog_changed)    
        from conv import Conv3x3
        from maxpool import MaxPool2
        from softmax import Softmax
    
    if version == "original":
        print("original version is being used")
        path_blog_original = "/home/konstantin/Documents/master_arbeit/cnn_python/original_blog/cnn-from-scratch"
        sys.path.append(path_blog_original)
        from conv import Conv3x3
        from maxpool import MaxPool2
        from softmax import Softmax
#        Conv3x3 = reload(Conv3x3)
#        MaxPool2 = reload(MaxPool2)
#        Softmax = reload(Softmax)
 
        
        
    num_filters = 8
    np.random.seed(seed=444); own_filter_conv = np.random.randn(num_filters, 3, 3) / 9
    own_filter_conv = np.round(own_filter_conv)
    
    dim_maxpool = 13 * 13 *8
    np.random.seed(seed=666); own_weight_soft = (np.random.randn(dim_maxpool, 10) / dim_maxpool) 
    own_bias_soft = np.zeros(10)
    
    conv = Conv3x3(8)  # 28x28x1 -> 26x26x8
    pool = MaxPool2()  # 26x26x8 -> 13x13x8
    dim_maxpool = np.prod(13 * 13 * 8)
    softmax = Softmax(dim_maxpool, 10)
    
    for i in range(n_iter):

        image = test_images[i] / 255 - 0.5
        label = test_labels[i]
        
        own_feature_map, own_filter_conv = fun.convolute(image=image, filter_matrix=own_filter_conv)
        own_maxpool_map = fun.maxpool(feature_map=own_feature_map)
        own_probs, own_inter_soft = fun.softmax(own_maxpool_map, 
                                        weight_matrix=own_weight_soft, 
                                        bias_vector=own_bias_soft)
        own_weight_soft, own_bias_soft, own_gradient_soft = fun.backprop_softmax(inter_soft=own_inter_soft,
                                                           probabilities=own_probs,
                                                           label = label,
                                                           learn_rate=learn_rate) 
        own_gradient_max = fun.backprop_maxpool(feature_map=own_feature_map, 
                                                gradient=own_gradient_soft)
        own_filter_conv = fun.backprop_conv(image=image, filter_conv=own_filter_conv,
                                        gradient=own_gradient_max, learn_rate=learn_rate)
     
       
    # run model from blog with same data

        blog_out_conv = conv.forward(image)
        #print(out_conv)   
        blog_out_max = pool.forward(blog_out_conv)
        blog_out_soft = softmax.forward(blog_out_max) #
        #print(blog_out_soft)
        gradient_L = np.zeros(10)
        gradient_L[label] = -1 / blog_out_soft[label]
        blog_gradient_soft = softmax.backprop(
                gradient_L, learn_rate)
        blog_gradient_max = pool.backprop(blog_gradient_soft)
        conv.backprop(blog_gradient_max, learn_rate)
        
        
        
        ################## compare feedforward ####################################
        ###########################################################################
        print("This is iteration", i)
        if np.sum(own_feature_map == blog_out_conv) == np.prod(own_feature_map.shape):
            print("YEAAAH! FeatureMaps are the same")
        else:
            print("NOOOO! featuremaps are not the same")
            
       # conv.filters == filter_conv # 
        # after first iteration these are not the same anymore,
        # since they get updated
        if np.sum(own_maxpool_map == blog_out_max) == np.prod(blog_out_max.shape):
            print("YEAHHH! maxpool is the same")
        else:
            print("NOOOO! maxpool is not the same")
        if np.sum(own_probs == blog_out_soft) == np.prod(blog_out_soft.shape):
            print("YEAAAH! predicted probabilities are the same")
        else:
            print("NOOOO! predicted probabilities are not the same")
            print("Own probabilities")
            print(own_probs)
            print("Blog probabilities")
            print(blog_out_soft)
         #   break
        ######################### compare backprop #################################
        ############################################################################
         
        ######## softmax: gradients:    
        if np.sum(own_gradient_soft == blog_gradient_soft) == np.prod(blog_gradient_soft.shape):
            print("YEAHHHH! gradients softmax are the same")
        else:
            print("NOOOO! gradients softmax are not the same")
            
        ## weight updates weight matrix softmax layer
#        if np.sum(own_weight_soft == blog_weights_updated) == np.prod(own_weight_soft.shape):
#            print("Yeaaah! updated weightmatrix softmax is the same")
#        else:
#            print("updated weightmatrix softmax is not the same")
#        ## weight updates bias vector
#        if np.sum(own_bias_soft == blog_biases_updated) == np.prod(blog_biases_updated.shape):
#            print("Yeaaah! Updated bias vector softmax is the same")
#        else:
#            print("updated bias vector is not the same")
        #### maxpool
        if np.sum(own_gradient_max== blog_gradient_max) == np.prod(blog_gradient_max.shape):
            print("YEAHHHH! gradients maxpool layer are the same")
        else:
            print("NOOOO! updated gradients maxpool are not the same")
        ## conv
#        if np.sum(own_filter_conv == blog_filter_update) == np.prod(own_filter_conv.shape):
#            print("YEAAAHHH! updated filter convlayer are the same")
#        else:
#            print("NOOOOO! updated filter conv layer is not the same")
#            
            
        # So! After two runs the predicted probabilities are already different, why?
        
        
    return None
    