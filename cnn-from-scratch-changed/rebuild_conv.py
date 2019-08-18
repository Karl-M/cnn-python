# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 07:37:49 2019

@author: d2gu53
"""

import numpy as np
import os
import sys

path = "C:/Users/D2GU53/Documents/master_arbeit/nn_in_r/cnn-from-scratch"
path2 = "C:/Users/D2GU53/Documents/master_arbeit/nn_in_r"
sys.path.append(path)
sys.path.append(path2)
import functions as fun
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax


image = np.random.randn(10, 8) * 3
image = np.round(image)
label = 1

conv = Conv3x3(3)  # 28x28x1 -> 26x26x8
pool = MaxPool2()  # 26x26x8 -> 13x13x8
softmax = Softmax(4 * 3 * 3, 2)  # 13x13x8 -> 10

num_filters = 3
np.random.seed(seed=666); filter_conv = np.random.randn(num_filters, 3, 3) / 9

###############################################################################
######################################## feedforward ##########################
###############################################################################


### forward pass is the same!
out_conv = conv.forward((image / 255) - 0.5)
out_convown, filter_conv, inter = fun.convolute((image / 255) - 0.5, filter_conv)

for i in range(num_filters):
    if np.sum(out_conv[:, :, i] == out_convown[i], axis=(0, 1)) == np.prod(out_conv[:, :, 0].shape):
        print("Yeah, it works!")

out_conv.shape
out_convown.shape
# maxpool as well
out_max = pool.forward(out_conv)
out_maxown, index_maxown = fun.max_pool(out_convown)

for i in range(num_filters):
    if np.sum(out_max[:, :, i] == out_maxown[i], axis=(0, 1)) == np.prod(out_max[:, :, 0].shape):
        print("Yeah, it works!")



## output of softmax is different, maybe because input (out_max) is transposed)
size = np.prod(out_max.shape)
np.random.seed(seed=666); weight_soft = np.random.randn(size, 2) / size
bias_soft = np.zeros(2)

out_soft = softmax.forward(out_max)
# hmmmm not quite the same, maybe take exact the same input?
# update: exactly the same, transposing of tensor was just wrong
# not the same again?
probabilities, intermediates = fun.softmax(out_maxown, weight_soft*10, bias_soft)
weight_soft * 10

out_soft == probabilities 


###############################################################################
####################### backpropagation #######################################
###############################################################################

# Up to deltaL everything is fine. Im multipliying deltaL with the Transpose 
# of the softmax weight matrix
# The other guy is multiplying it with a weight matrix as well, but it is
# for some reason a little different

# Backprop
gradient = np.zeros(2)
gradient[label] = -1 / out_soft[label]

# since the flattened versions oft the output of the maxpool layer are different,
# one would expect the gradients of softmax should be in different order as well,
# and not the same!
out_max.shape
out_maxown.shape
weight_soft.shape
out_max.flatten()
out_maxown.flatten()


gradient_soft, weights, deltaL = softmax.backprop(gradient, 0.01)
gradient_softown = fun.backprop_softmax(intermediates, out_maxown.shape, probabilities, label=label)[3]

# gradient should not be the same, but seem to be very similar, my bc of random
# weight_soft initialisations?
# try with weight_soft * 10

gradients_soft_oldweights = gradient_soft
gradient_softown.shape
gradient_soft.flatten() ==  gradient_softown.flatten() # are the same, but format is wrong

# not the same, yes!
# no only remaining problem has to be in backprop_conv
# maybe still problem in back_softmax, 

gradient_soft[:, :, 0]
gradient_softown[1]
gradient_max = pool.backprop(gradient_soft)
gradient_maxown = fun.backprop_maxpool(out_convown, index_maxown, gradient_softown.flatten())

gradient_maxown[1]
gradient_max[:, :, 1]

gradient_max.shape
gradient_maxown.shape



############################################################################
########################################## backprop maxpool and conv#########
############################################################################

## mabe just stop here and continue with other backprobs
# backprop_maxpool should work, I just pass wrong(?) values



out = conv.forward(image)
out = pool.forward(out)
out = softmax.forward(out)
#out, loss, acc = forward(im, label)

# Calculate initial gradient
gradient_L = np.zeros(10)
gradient_L[label] = -1 / out_soft[label]

  # Backprop
gradient_soft = softmax.backprop(gradient_L, 0.01)[0]
gradient_max = pool.backprop(gradient_soft)


# maybe set gradients to one, that are != to zero, to check if conv back 
# das richtige macht

gradient_max[gradient_max != 0] = 1
gradient_maxown[gradient_maxown != 0] = 1

gradient_conv, grad_max = conv.backprop(gradient_max, 0.01)

gradient_convown = fun.backprop_max_conv(image, filter_conv, index_maxown, gradient_maxown, learn_rate=1)

gradient_conv.shape
gradient_convown.shape
gradient_max[:, :, 0]
# was ist wenn wir testweise jeden Eintrag des Gradienten auf 1 setzen?
# dann sollte sichtbar sein, ob die Ableitung dConv_dFij funktioniert

gradient_max = np.ones(shape=gradient_max.shape)
gradient_maxown = np.ones(shape=gradient_maxown.shape)

gradient_conv = conv.backprop(gradient_max, 0.01)

gradient_convown = fun.backprop_max_conv(image, filter_conv, index_maxown, gradient_maxown, learn_rate=1)

image

