from __future__ import division
import tensorflow as tf
from ops import *
from utils import *


def network(inputs, embedding_dim=2):

    def prelu(inputs, name=''):
        alpha = tf.get_variable(name, shape=inputs.get_shape(), initializer=tf.constant_initializer(0.0), dtype=inputs.dtype)
        return tf.maximum(alpha*inputs, inputs)

    def conv(inputs, filters, kernel_size, strides, w_init, padding='same', suffix='', scope=None):
        conv_name = 'conv'+suffix
        relu_name = 'relu'+suffix

        with tf.name_scope(name=scope):
            if w_init == 'xavier':   w_init = tf.contrib.layers.xavier_initializer(uniform=True)
            if w_init == 'gaussian': w_init = tf.contrib.layers.xavier_initializer(uniform=False)
            input_shape = inputs.get_shape().as_list()
            net = tf.layers.conv2d(inputs, filters, kernel_size, strides, padding=padding,
                                kernel_initializer=w_init, name=conv_name)
            output_shape=net.get_shape().as_list()
            print("=================================================================================")
            print("layer:%8s    input shape:%8s   output shape:%8s" %(conv_name, str(input_shape), str(output_shape)))
            print("---------------------------------------------------------------------------------")
            #net = prelu(net, name=relu_name)
            net = tf.nn.relu(net, name=relu_name)
            return net

    def resnet_block(net, blocks, suffix=''):
        n = len(blocks)
        for i in range(n):
            if n == 2 and i == 0: identity = net
            net = conv(inputs=net,
                        filters=blocks[i]['filters'],
                        kernel_size=blocks[i]['kernel_size'],
                        strides=blocks[i]['strides'],
                        w_init=blocks[i]['w_init'],
                        padding=blocks[i]['padding'],
                        suffix=suffix+'_'+blocks[i]['suffix'],
                        scope='conv'+suffix+'_'+blocks[i]['suffix'])

            if n == 3 and i == 0: identity = net
        return identity + net

    res1_3=[
        {'filters':64, 'kernel_size':3, 'strides':2, 'w_init':'xavier',   'padding':'same', 'suffix':'1'},
        {'filters':64, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'2'},
        {'filters':64, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'3'},
    ]

    res2_3=[
        {'filters':128, 'kernel_size':3, 'strides':2, 'w_init':'xavier',   'padding':'same', 'suffix':'1'},
        {'filters':128, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'2'},
        {'filters':128, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'3'},
    ]

    res2_5=[
        {'filters':128, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'4'},
        {'filters':128, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'5'},
    ]

    res3_3=[
        {'filters':256, 'kernel_size':3, 'strides':2, 'w_init':'xavier',   'padding':'same', 'suffix':'1'},
        {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'2'},
        {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'3'},
    ]

    res3_5=[
        {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'4'},
        {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'5'},
    ]

    res3_7=[
        {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'6'},
        {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'7'},
    ]

    res3_9=[
        {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'8'},
        {'filters':256, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'9'},
    ]

    res4_3=[
        {'filters':512, 'kernel_size':3, 'strides':2, 'w_init':'xavier',   'padding':'same', 'suffix':'1'},
        {'filters':512, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'2'},
        {'filters':512, 'kernel_size':3, 'strides':1, 'w_init':'gaussian', 'padding':'same', 'suffix':'3'},
    ]

    net = inputs
    for suffix, blocks in zip(('1','2','2','3','3','3','3','4'),
                                (res1_3,res2_3,res2_5,res3_3,res3_5,res3_7,res3_9,res4_3)):
        net = resnet_block(net, blocks, suffix=suffix)

    net = tf.layers.flatten(net)
    embeddings = tf.layers.dense(net, units=embedding_dim, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    return embeddings

def network2(inputs, reuse=False):
    '''
    input : 64,64,1
    conv_1 : 64,64,64
    conv_2 : 64,64,64
    pool_1 : 32,32,64
    conv_3 : 32,32,128
    pool_2 : 16,16,128
    conv_4 : 16,16,256
    pool_3 : 8,8,256
    feature layer : 512
    '''
    with tf.variable_scope("feature_extraction_network"):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.leaky_relu, weights_initializer=tf.truncated_normal_initializer(stddev=0.01)): 
            #inputs = slim.flatten(inputs)
            net = slim.conv2d(inputs=inputs, num_outputs=64, kernel_size=[3, 3], normalizer_fn=slim.batch_norm, scope='conv1') 
            net = slim.conv2d(inputs=net, num_outputs=64, kernel_size=[3, 3], normalizer_fn=slim.batch_norm, scope='conv2') 
            net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool1')
            net = slim.conv2d(inputs=net, num_outputs=128, kernel_size=[3, 3], normalizer_fn=slim.batch_norm, scope='conv3') 
            net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool2')
            net = slim.conv2d(inputs=net, num_outputs=256, kernel_size=[3, 3], normalizer_fn=slim.batch_norm, scope='conv4') 
            net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool3')
            net = slim.flatten(net, scope='flatten')
            net = slim.fully_connected(net, 512, activation_fn=tf.nn.tanh)
            net = tf.expand_dims(net, axis=2)
        return net


def Angular_Softmax_Loss(embeddings, labels, margin=4):
    """
    Note:(about the value of margin)
    as for binary-class case, the minimal value of margin is 2+sqrt(3)
    as for multi-class  case, the minimal value of margin is 3

    the value of margin proposed by the author of paper is 4.
    here the margin value is 4.
    """
    l = 0.
    embeddings_norm = tf.norm(embeddings, axis=1) # ||x|| : embeddings norm , tf.norm == l2 norm??
    N = 10
    C = 2
    with tf.variable_scope("softmax"):
        weights = tf.get_variable(name='embedding_weights',
                                    shape=[embeddings.get_shape().as_list()[-1], C], # shape=[embedding 크기, # of class] (?)
                                    initializer=tf.contrib.layers.xavier_initializer())
        
        weights = tf.nn.l2_normalize(weights, axis=0)  # output = x / sqrt(max(sum(x**2), epsilon))
        
        # cacualting the cos value of angles between embeddings and weights
        orgina_logits = tf.matmul(embeddings, weights) # W^(T)x (W:weights, x:embeddings) , origina_logits shape : batchsize X # of class
        
        #N = embeddings.get_shape()[0] # get batch_size
        
        

        ### change target value : cos(theta) -> cos(m*theta)
        single_sample_label_index = tf.stack([tf.constant(list(range(N)), tf.int64), labels], axis=1)
        # N = 128, labels = [5,4,...,9]
        # single_sample_label_index: [[idx,label], ..]
        # [ [0,5],
        #   [1,4],
        #   ....
        #   [128,9]]
        selected_logits = tf.gather_nd(orgina_logits, single_sample_label_index) #  (tf.gather_nd : 2nd input이 가리키는 위치의 1st input의 값을 return)
        cos_theta = tf.div(selected_logits, embeddings_norm) # elementwise x/y
        cos_theta_power = tf.square(cos_theta) # cos_theta^2
        cos_theta_biq = tf.pow(cos_theta, 4) # x^y (element wise) => cos_theta^4
        sign0 = tf.sign(cos_theta)
        sign3 = tf.multiply(tf.sign(2*cos_theta_power-1), sign0)
        sign4 = 2*sign0 + sign3 -3
        result=sign3*(8*cos_theta_biq-8*cos_theta_power+1) + sign4 # result = cos(4*theta) ( note : cos(4x) = 8cos^4(x) - 8cos^2(x) + 1)
        ###

        margin_logits = tf.multiply(result, embeddings_norm) # ||x||*cos(4*theta)

        ### cos(theta), cos(4*theta) combine
        f = 1.0/(1.0+l)
        ff = 1.0 - f
        #combined_logits = tf.add(orgina_logits, tf.scatter_nd(single_sample_label_index, tf.subtract(margin_logits, selected_logits), orgina_logits.get_shape()))
        #scatter = tf.scatter_nd(single_sample_label_index, tf.subtract(margin_logits, selected_logits), [10,2])
        combined_logits = tf.add(orgina_logits, tf.scatter_nd(single_sample_label_index, tf.subtract(margin_logits, selected_logits), [N,C]))
        updated_logits = ff*orgina_logits + f*combined_logits
        ###
        
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=updated_logits))
        
        pred_prob = tf.nn.softmax(logits=updated_logits)
        return pred_prob, loss, cos_theta, orgina_logits




def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target): # mae??? not mse??
    return tf.reduce_mean(tf.abs(in_-target))

def mse_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
