# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 07:45:44 2019

@author: d2gu53
"""

import numpy as np
import os
import sys

path = "C:/Users/D2GU53/Documents/master_arbeit/nn_in_r"
sys.path.append(path)

os.listdir(path)
import functions as fun

np.random.seed(seed=666); image = np.random.randn(6, 6) * 3
image = np.round(image)
label = 1

num_filters = 2
np.random.seed(seed=666); filter_conv = np.random.randn(num_filters, 3, 3) / 9

## after transpsosing input to softmax, feedforward agrees,
# but maybe it would agree in backprop, because I put the num filters first,
# he put them last

out_ownconv, filter_conv, inter = fun.convolute((image / 255) - 0.5, filter_conv)
out_ownconv.T
out_maxown, index_maxown = fun.max_pool(out_ownconv)

np.random.seed(seed=666); weight_soft = np.random.randn(8, 2) / 8
np.random.seed(seed=666); bias_soft = np.zeros(2)

probabilities, intermediates = fun.softmax(out_maxown.T, weight_soft, bias_soft)

out_maxown[0]
out_maxown.T[:, :, 0]

################################## backprop ######################

weight_soft.shape

### agrees, after Transposing input!
# needed to transpose weight matrix as well, now gradients from softmax roughly agree
back_soft = fun.backprop_softmax(intermediates, probabilities, label = label)[3]

## gradients from backprop max are not the same, try with exactly the same gradients
# from backsoftmax now, since they are only rougghly the same
back_soft = np.array([ 0.02303161,  0.01477759, -0.02779495,  0.05881862,  0.09134293,
        0.09521715,  0.10948755,  0.00828537])
# gradients from back propagation softmax are the same

# but probably is indexing not working anymore in backprop maxpool,
# since I transposed the input into the softmax layer
grad_max = fun.backprop_maxpool(out_ownconv, index_maxown, back_soft)
grad_max[0, :, : ]
# hmmmmmm die indices sind an den richtigen Stellen ungleich 0, aber 
# die updatees habend ie falschen Werte

grad_max[grad_max != 0]

out_ownconv[[index_maxown]
## still disagree!

grad_max.shape

fun.backprop_max_conv(image, out_ownconv, index_maxown, back_soft, 0.01)




#########################
out_maxown
out_ownconv[index_maxown]











