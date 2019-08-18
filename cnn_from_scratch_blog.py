# -*- coding: utf-8 -*-
"""
#Created on Mon Aug  5 15:26:32 2019

@author: d2gu53
"""

# cnn from scratch in python 
# https://victorzhou.com/blog/intro-to-cnns-part-1/

import sys
import mnist
path = "/home/konstantin/Documents/master_arbeit/cnn_python"
sys.path.append(path)
import numpy as np
import test_cnn as test
import functions as fun

test.debug_cnn(n_iter=11, version="changed", learn_rate=0.01)

test_images = mnist.test_images()[:2001]
test_labels = mnist.test_labels()[:2001]


def train(training_data, labels, n_iter, n_classes, n_filter, learn_rate, print_acc=True):
    
    input_dim = int((((training_data[0].shape[0] - 3 + 1) / 2) ** 2) * n_filter)
    np.random.seed(seed=30); own_filter_conv = np.random.randn(n_filter, 3, 3) / 9
    np.random.seed(seed=30); own_weight_soft = (np.random.randn(input_dim, n_classes) / input_dim)
    own_bias_soft = np.random.randn(n_classes)
    
    num_correct = 0
    
    for i in range(n_iter):
    
            image = training_data[i] / 255 - 0.5
            label = labels[i]
            
            own_feature_map, own_filter_conv = fun.convolute(image=image, filter_matrix= own_filter_conv)
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
            
            prediction = np.argmax(own_probs)
            acc = 1 if prediction == label else 0
            num_correct += acc
            
            if i % 100 == 0 and i != 0 and print_acc:
                accuracy = num_correct / i
                print(f"accuracy for the first {i} samples: {accuracy}")
                print(f"{num_correct} predictions for {i} samples were correct")
                
    return None



train(test_images, test_labels, n_iter=1001, n_classes=10, n_filter=8, learn_rate=0.01)

