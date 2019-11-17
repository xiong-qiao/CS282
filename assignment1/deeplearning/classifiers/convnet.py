import numpy as np

from deeplearning.layers import *
from deeplearning.fast_layers import *
from deeplearning.layer_utils import *


class ConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  [conv - bn - relu] * 2 - 2x2 max pool - [conv - bn - relu] * 2 - 2x2 max pool 
  - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    self.params['W1'] = np.random.normal(scale=weight_scale, size=(num_filters, C, filter_size, filter_size))
    self.params['b1'] = np.zeros(num_filters)
    self.params['gamma1'] = np.ones(num_filters)
    self.params['beta1'] = np.zeros(num_filters)
    self.params['W2'] = np.random.normal(scale=weight_scale, size=(num_filters, num_filters, filter_size, filter_size))
    self.params['b2'] = np.zeros(num_filters)
    self.params['gamma2'] = np.ones(num_filters)
    self.params['beta2'] = np.zeros(num_filters)

    self.params['W3'] = np.random.normal(scale=weight_scale, size=(num_filters, num_filters, filter_size, filter_size))
    self.params['b3'] = np.zeros(num_filters)
    self.params['gamma3'] = np.ones(num_filters)
    self.params['beta3'] = np.zeros(num_filters)
    self.params['W4'] = np.random.normal(scale=weight_scale, size=(num_filters, num_filters, filter_size, filter_size))
    self.params['b4'] = np.zeros(num_filters)
    self.params['gamma4'] = np.ones(num_filters)
    self.params['beta4'] = np.zeros(num_filters)

    # the output size of conv layer should be (F, H', W'), then after 2x2 max pool,
    # the output become (F, H'/2, W'/2)
    # See below, here stride is 1 and pad = (s - 1)/2. So the output size is the same as input size.
    self.params['W5'] = np.random.normal(scale=weight_scale, size=(num_filters * H//4 * W//4, hidden_dim))
    self.params['b5'] = np.zeros(hidden_dim)
    self.params['W6'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
    self.params['b6'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    mode = 'test' if y is None else 'train'

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    W6, b6 = self.params['W6'], self.params['b6']
    gamma1, beta1 = self.params['gamma1'], self.params['beta1']
    gamma2, beta2 = self.params['gamma2'], self.params['beta2']
    gamma3, beta3 = self.params['gamma3'], self.params['beta3']
    gamma4, beta4 = self.params['gamma4'], self.params['beta4']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    # bn parameter
    bn_param = {'mode': mode}

    scores = None
    caches = {}
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    scores, cache_conv1 = conv_bn_relu_forward(X, W1, b1, conv_param, gamma1, beta1, bn_param)
    scores, cache_conv2 = conv_bn_relu_forward(scores, W2, b2, conv_param, gamma2, beta2, bn_param)
    scores, cache_pool1 = max_pool_forward_fast(scores, pool_param)

    scores, cache_conv3 = conv_bn_relu_forward(scores, W3, b3, conv_param, gamma3, beta3, bn_param)
    scores, cache_conv4 = conv_bn_relu_forward(scores, W4, b4, conv_param, gamma4, beta4, bn_param)
    scores, cache_pool2 = max_pool_forward_fast(scores, pool_param)

    scores, cache_hidden = affine_relu_forward(scores, W5, b5)
    scores, cache_softmax = affine_forward(scores, W6, b6)

    caches["cache_conv1"] = cache_conv1
    caches["cache_conv2"] = cache_conv2
    caches["cache_conv3"] = cache_conv3
    caches["cache_conv4"] = cache_conv4
    caches["cache_pool1"] = cache_pool1
    caches["cache_pool2"] = cache_pool2
    caches["cache_hidden"] = cache_hidden
    caches["cache_softmax"] = cache_softmax
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, der = softmax_loss(scores, y)

    # L2 regularization loss
    for i in range(1, 6):
        W_name = "W" + "%d" % i
        loss += 0.5 * self.reg * np.sum(self.params[W_name] ** 2)

    der, grads["W6"], grads["b6"] = affine_backward(der, caches["cache_softmax"])
    der, grads["W5"], grads["b5"] = affine_relu_backward(der, caches["cache_hidden"])

    der = max_pool_backward_fast(der, caches["cache_pool2"])
    der, grads["W4"], grads["b4"], grads["gamma4"], grads["beta4"] = conv_bn_relu_backward(der, caches["cache_conv4"])
    der, grads["W3"], grads["b3"], grads["gamma3"], grads["beta3"] = conv_bn_relu_backward(der, caches["cache_conv3"])

    der = max_pool_backward_fast(der, caches["cache_pool1"])
    der, grads["W2"], grads["b2"], grads["gamma2"], grads["beta2"] = conv_bn_relu_backward(der, caches["cache_conv2"])
    der, grads["W1"], grads["b1"], grads["gamma1"], grads["beta1"] = conv_bn_relu_backward(der, caches["cache_conv1"])

    for i in range(1, 6):
        W_name = "W" + "%d" % i
        grads[W_name] += self.reg * self.params[W_name]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
